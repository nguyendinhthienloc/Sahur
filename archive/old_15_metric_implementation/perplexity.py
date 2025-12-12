"""
Perplexity and surprisal metrics using multiple LMs with batching and caching.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Optional
import hashlib

class PerplexityEvaluator:
    """
    Multi-model perplexity evaluator with batching and optional caching.
    """
    
    def __init__(self, models: List[str] = None, cache_dir: Optional[str] = None, 
                 batch_size: int = 8, max_length: int = 512):
        """
        Initialize PerplexityEvaluator with batching support.
        
        Args:
            models: List of model names to use
            cache_dir: Optional directory for disk caching
            batch_size: Batch size for processing multiple texts
            max_length: Maximum sequence length
        """
        if models is None:
            self.models = ['gpt2']
        else:
            self.models = models
        
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache = {}  # In-memory cache
        
        # Pre-load models to avoid repeated loading
        self._preload_models()
    
    def _preload_models(self):
        """Pre-load all models at initialization to avoid per-text loading."""
        for model_name in self.models:
            self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load model and tokenizer if not already loaded."""
        if model_name not in self.loaded_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Set padding token if not exists
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                model.eval()  # Set to evaluation mode
                
                self.loaded_models[model_name] = model
                self.loaded_tokenizers[model_name] = tokenizer
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return None, None
        return self.loaded_models[model_name], self.loaded_tokenizers[model_name]
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model combination."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_name}_{text_hash}"
    
    def compute_perplexity(self, text: str, model_name: str = 'gpt2') -> float:
        """
        Compute perplexity for a given text and model with caching.
        
        Args:
            text: Input text
            model_name: Model to use for perplexity calculation
            
        Returns:
            Perplexity score
        """
        # Check cache first
        cache_key = self._get_cache_key(text, model_name)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        model, tokenizer = self._load_model(model_name)
        if model is None:
            return float('nan')

        # Tokenize with truncation
        encodings = tokenizer(text, return_tensors='pt', truncation=True, 
                            max_length=self.max_length)
        max_length = model.config.n_positions
        stride = min(512, max_length // 2)
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        if not nlls:
            result = 0.0
        else:
            ppl = torch.exp(torch.stack(nlls).mean())
            result = ppl.item()
        
        # Cache result
        self.cache[cache_key] = result
        return result
    
    def compute_perplexity_batch(self, texts: List[str], model_name: str = 'gpt2') -> List[float]:
        """
        Compute perplexity for multiple texts in batch.
        
        Args:
            texts: List of input texts
            model_name: Model to use
            
        Returns:
            List of perplexity scores
        """
        model, tokenizer = self._load_model(model_name)
        if model is None:
            return [float('nan')] * len(texts)
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Check cache first
            batch_results = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                cache_key = self._get_cache_key(text, model_name)
                if cache_key in self.cache:
                    batch_results.append((j, self.cache[cache_key]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Process uncached texts
            if uncached_texts:
                encodings = tokenizer(uncached_texts, return_tensors='pt', 
                                    truncation=True, padding=True,
                                    max_length=self.max_length)
                input_ids = encodings.input_ids.to(self.device)
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    # Get per-sample loss
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    losses = []
                    for sample_idx in range(len(uncached_texts)):
                        sample_loss = loss_fct(
                            shift_logits[sample_idx].view(-1, shift_logits.size(-1)),
                            shift_labels[sample_idx].view(-1)
                        )
                        # Only consider non-padding tokens
                        mask = shift_labels[sample_idx] != tokenizer.pad_token_id
                        if mask.sum() > 0:
                            ppl = torch.exp(sample_loss[mask].mean()).item()
                        else:
                            ppl = 0.0
                        losses.append(ppl)
                        
                        # Cache result
                        cache_key = self._get_cache_key(uncached_texts[sample_idx], model_name)
                        self.cache[cache_key] = ppl
                        batch_results.append((uncached_indices[sample_idx], ppl))
            
            # Sort by original index and extract values
            batch_results.sort(key=lambda x: x[0])
            results.extend([score for _, score in batch_results])
        
        return results
    
    def compute_perplexity_gap(self, text: str) -> Dict[str, float]:
        """
        Compute perplexity gap across multiple models.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with perplexity scores and gap
        """
        results = {}
        ppls = []
        
        for model_name in self.models:
            ppl = self.compute_perplexity(text, model_name)
            results[f'ppl_{model_name}'] = ppl
            if not np.isnan(ppl):
                ppls.append(ppl)
        
        # Calculate statistics
        if len(ppls) >= 1:
            results['ppl_mean'] = float(np.mean(ppls))
            results['ppl_std'] = float(np.std(ppls)) if len(ppls) > 1 else 0.0
            results['ppl_min'] = float(np.min(ppls))
            results['ppl_max'] = float(np.max(ppls))
        
        # Calculate gap if we have at least 2 models
        if len(ppls) >= 2:
            results['ppl_gap'] = results['ppl_max'] - results['ppl_min']
        else:
            results['ppl_gap'] = 0.0
                
        return results
    
    def clear_cache(self):
        """Clear in-memory cache."""
        self.cache.clear()
    
    def compute_surprisal_variance(self, text: str, window_size: int = 50) -> float:
        """
        Compute variance of surprisal across the document.
        """
        # We need token-level surprisals. 
        # Re-using compute_perplexity logic but keeping individual losses.
        model, tokenizer = self._load_model(self.models[0]) # Use first model
        if model is None:
            return 0.0
        # Tokenize without truncation; then process in overlapping chunks to avoid
        # exceeding model max positions. We will compute per-token losses per chunk
        # and only keep each token's loss once (avoid double-counting from overlaps).
        encodings = tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.device)

        seq_len = input_ids.size(1)
        max_length = getattr(model.config, 'n_positions', None) or getattr(model.config, 'max_position_embeddings', None) or 1024
        max_length = int(max_length)
        stride = min(512, max_length // 2)

        surprisals = []
        prev_end = 0

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            input_ids_chunk = input_ids[:, begin_loc:end_loc]

            with torch.no_grad():
                outputs = model(input_ids_chunk, labels=input_ids_chunk)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = input_ids_chunk[..., 1:].contiguous()

                # per-token loss for this chunk
                losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                losses = losses.view(shift_labels.size())  # shape [1, L-1]
                losses_np = losses.cpu().numpy().flatten()

            # positions covered by these losses correspond to token indices: begin_loc+1 .. end_loc-1
            # only keep positions strictly greater than prev_end to avoid duplicates
            start_pos = max(prev_end + 1, begin_loc + 1)
            if start_pos >= end_loc:
                # nothing new in this chunk
                prev_end = end_loc - 1
                if end_loc == seq_len:
                    break
                else:
                    continue

            # compute slice indices into losses_np
            slice_start = start_pos - (begin_loc + 1)
            slice_end = (end_loc - 1) - (begin_loc + 1) + 1  # inclusive end -> exclusive
            slice_start = int(max(0, slice_start))
            slice_end = int(max(0, slice_end))

            new_losses = losses_np[slice_start:slice_end]
            if new_losses.size:
                surprisals.extend(new_losses.tolist())

            prev_end = end_loc - 1
            if end_loc == seq_len:
                break

        if len(surprisals) < 2:
            return 0.0

        return float(np.var(np.array(surprisals)))
