"""
Embedding-based metrics using sentence-transformers with improved batching.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
from scipy.spatial.distance import cosine
import spacy

class EmbeddingAnalyzer:
    """
    Sentence embedding analyzer for topical drift and centroids with batching support.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_spacy: bool = False):
        self.model = SentenceTransformer(model_name)
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
            except Exception:
                self.nlp = None
    
    def compute_topical_drift(self, text: str) -> float:
        """
        Compute cosine distance between embeddings of first and last paragraph.
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            return 0.0
        
        emb_first = self.model.encode(paragraphs[0])
        emb_last = self.model.encode(paragraphs[-1])
        
        return float(cosine(emb_first, emb_last))
    
    def compute_centroid_distance(self, text: str, 
                                  human_centroid: np.ndarray) -> float:
        """
        Compute distance to human-topic centroid embedding.
        """
        if human_centroid is None or len(human_centroid) == 0:
            return 0.0
            
        emb_text = self.model.encode(text)
        return float(cosine(emb_text, human_centroid))

    def compute_sentence_similarity(self, text: str, doc: Optional[object] = None) -> float:
        """
        Compute average cosine similarity between consecutive sentences.
        
        Args:
            text: Raw text
            doc: Optional pre-processed spaCy doc for sentence splitting
        """
        # Use spaCy doc if provided, otherwise fallback to simple splitting
        if doc is not None and hasattr(doc, 'sents'):
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        elif self.nlp is not None:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback to simple splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        # Batch encode all sentences at once
        embeddings = self.model.encode(sentences, batch_size=32, show_progress_bar=False)
        similarities = []
        
        for i in range(len(embeddings) - 1):
            # cosine distance is 1 - similarity
            # so similarity is 1 - cosine distance
            sim = 1 - cosine(embeddings[i], embeddings[i+1])
            similarities.append(sim)
            
        return float(np.mean(similarities)) if similarities else 0.0
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Batch encode multiple texts for efficiency.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings
        """
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
