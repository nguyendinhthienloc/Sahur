"""
spaCy parsing module with disk caching.

Handles:
- Loading spaCy pipeline (en_core_web_lg preferred, fallback to sm)
- Batch processing via nlp.pipe
- Disk caching of parsed documents (pickle format)
- Cache invalidation and management
"""

import pickle
import hashlib
from pathlib import Path
from typing import List, Optional, Iterator
import logging
import spacy

from .utils import setup_logger

logger = setup_logger(__name__)

# Global spaCy pipeline (lazy loaded)
_NLP_PIPELINE = None


def get_spacy_pipeline(model_name: str = "en_core_web_lg", 
                      disable: Optional[List[str]] = None) -> spacy.language.Language:
    """
    Get or load spaCy pipeline (singleton pattern).
    
    Parameters
    ----------
    model_name : str
        spaCy model name
    disable : List[str], optional
        Pipeline components to disable (e.g., ['ner', 'textcat'])
        
    Returns
    -------
    spacy.language.Language
        Loaded spaCy pipeline
    """
    global _NLP_PIPELINE
    
    if _NLP_PIPELINE is not None:
        return _NLP_PIPELINE
    
    disable = disable or []
    
    try:
        logger.info(f"Loading spaCy model: {model_name}")
        _NLP_PIPELINE = spacy.load(model_name, disable=disable)
        logger.info(f"Successfully loaded {model_name}")
    except OSError:
        # Try fallback models
        fallback_models = ["en_core_web_sm", "en_core_web_md"]
        logger.warning(f"Model {model_name} not found. Trying fallbacks: {fallback_models}")
        
        for fallback in fallback_models:
            try:
                _NLP_PIPELINE = spacy.load(fallback, disable=disable)
                logger.info(f"Loaded fallback model: {fallback}")
                break
            except OSError:
                continue
        
        if _NLP_PIPELINE is None:
            raise RuntimeError(
                f"Could not load spaCy model. Tried: {model_name}, {fallback_models}. "
                f"Install with: python -m spacy download en_core_web_lg"
            )
    
    return _NLP_PIPELINE


def compute_text_hash(text: str) -> str:
    """Compute a short hash of text for cache keys."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def get_cache_path(doc_id: str, cache_dir: Path) -> Path:
    """Get cache file path for a document."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{doc_id}.pkl"


def load_cached_doc(doc_id: str, cache_dir: Path) -> Optional[spacy.tokens.Doc]:
    """
    Load a cached spaCy Doc.
    
    Returns
    -------
    spacy.tokens.Doc or None
        Cached doc if available, None otherwise
    """
    cache_path = get_cache_path(doc_id, cache_dir)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            doc = pickle.load(f)
        return doc
    except Exception as e:
        logger.warning(f"Failed to load cache for {doc_id}: {e}")
        return None


def save_cached_doc(doc: spacy.tokens.Doc, doc_id: str, cache_dir: Path) -> None:
    """Save a spaCy Doc to cache."""
    cache_path = get_cache_path(doc_id, cache_dir)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(doc, f)
    except Exception as e:
        logger.warning(f"Failed to save cache for {doc_id}: {e}")


def parse_documents(texts: List[str], 
                   doc_ids: List[str],
                   cache_dir: Optional[Path] = None,
                   use_cache: bool = True,
                   batch_size: int = 32,
                   n_process: int = 1) -> List[spacy.tokens.Doc]:
    """
    Parse documents with spaCy, using cache when available.
    
    Parameters
    ----------
    texts : List[str]
        List of texts to parse
    doc_ids : List[str]
        List of document IDs (for caching)
    cache_dir : Path, optional
        Directory for cache files
    use_cache : bool, default=True
        Whether to use caching
    batch_size : int, default=32
        Batch size for nlp.pipe
    n_process : int, default=1
        Number of processes for parallel parsing
        
    Returns
    -------
    List[spacy.tokens.Doc]
        List of parsed spaCy documents
    """
    if len(texts) != len(doc_ids):
        raise ValueError(f"Length mismatch: {len(texts)} texts vs {len(doc_ids)} doc_ids")
    
    nlp = get_spacy_pipeline()
    docs = []
    texts_to_parse = []
    indices_to_parse = []
    
    # Check cache first
    if use_cache and cache_dir:
        logger.info(f"Checking cache in {cache_dir}")
        for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
            cached_doc = load_cached_doc(doc_id, cache_dir)
            if cached_doc is not None:
                docs.append(cached_doc)
            else:
                docs.append(None)  # Placeholder
                texts_to_parse.append(text)
                indices_to_parse.append(i)
        
        logger.info(f"Found {len(texts) - len(texts_to_parse)} cached, parsing {len(texts_to_parse)} new documents")
    else:
        texts_to_parse = texts
        indices_to_parse = list(range(len(texts)))
        docs = [None] * len(texts)
    
    # Parse uncached documents
    if texts_to_parse:
        logger.info(f"Parsing {len(texts_to_parse)} documents with batch_size={batch_size}, n_process={n_process}")
        
        parsed_docs = list(nlp.pipe(
            texts_to_parse, 
            batch_size=batch_size,
            n_process=n_process if n_process > 1 else 1
        ))
        
        # Store parsed docs and cache them
        for parsed_doc, idx in zip(parsed_docs, indices_to_parse):
            docs[idx] = parsed_doc
            
            if use_cache and cache_dir:
                save_cached_doc(parsed_doc, doc_ids[idx], cache_dir)
    
    logger.info(f"Parsing complete: {len(docs)} documents")
    
    return docs


def get_parsed(text: str, doc_id: str, cache_dir: Optional[Path] = None) -> spacy.tokens.Doc:
    """
    Parse a single document with caching.
    
    Convenience function for single-document parsing.
    """
    docs = parse_documents(
        texts=[text],
        doc_ids=[doc_id],
        cache_dir=cache_dir,
        use_cache=cache_dir is not None,
        batch_size=1,
        n_process=1
    )
    return docs[0]


def clear_cache(cache_dir: Path) -> None:
    """Delete all cached documents."""
    if not cache_dir.exists():
        return
    
    count = 0
    for cache_file in cache_dir.glob("*.pkl"):
        cache_file.unlink()
        count += 1
    
    logger.info(f"Cleared {count} cached documents from {cache_dir}")
