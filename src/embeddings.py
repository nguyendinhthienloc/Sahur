"""
Sentence embedding module using sentence-transformers.

Provides:
- Lightweight sentence embedding wrapper
- Caching per document
- Mean adjacent cosine similarity computation
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Dict
import logging

from .utils import setup_logger

logger = setup_logger(__name__)

# Global model (lazy loaded)
_EMBEDDING_MODEL = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load sentence-transformers model (singleton pattern).
    
    Parameters
    ----------
    model_name : str
        Model name from sentence-transformers
        
    Returns
    -------
    SentenceTransformer
        Loaded model
    """
    global _EMBEDDING_MODEL
    
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
    
    logger.info(f"Loading embedding model: {model_name}")
    _EMBEDDING_MODEL = SentenceTransformer(model_name)
    logger.info(f"Model loaded: {model_name}")
    
    return _EMBEDDING_MODEL


def embed_sentences(sentences: List[str], 
                   model_name: str = "all-MiniLM-L6-v2",
                   batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of sentences.
    
    Parameters
    ----------
    sentences : List[str]
        List of sentence texts
    model_name : str
        Model name
    batch_size : int
        Batch size for encoding
        
    Returns
    -------
    np.ndarray
        Embeddings of shape (n_sentences, embedding_dim)
    """
    if not sentences:
        return np.array([])
    
    model = get_embedding_model(model_name)
    
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    return embeddings


def mean_adjacent_cosine(embeddings: np.ndarray) -> float:
    """
    Compute mean cosine similarity between adjacent embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Array of embeddings, shape (n, dim)
        
    Returns
    -------
    float
        Mean cosine similarity
    """
    if len(embeddings) < 2:
        return 0.0
    
    similarities = []
    
    for i in range(len(embeddings) - 1):
        v1 = embeddings[i]
        v2 = embeddings[i + 1]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
            similarities.append(cosine_sim)
    
    return float(np.mean(similarities)) if similarities else 0.0


def get_cache_path(doc_id: str, cache_dir: Path) -> Path:
    """Get cache file path for document embeddings."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{doc_id}_embed.pkl"


def load_cached_embeddings(doc_id: str, cache_dir: Path) -> Optional[np.ndarray]:
    """Load cached embeddings for a document."""
    cache_path = get_cache_path(doc_id, cache_dir)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    except Exception as e:
        logger.warning(f"Failed to load embedding cache for {doc_id}: {e}")
        return None


def save_cached_embeddings(embeddings: np.ndarray, doc_id: str, cache_dir: Path) -> None:
    """Save embeddings to cache."""
    cache_path = get_cache_path(doc_id, cache_dir)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        logger.warning(f"Failed to save embedding cache for {doc_id}: {e}")


def create_embed_function(model_name: str = "all-MiniLM-L6-v2",
                         cache_dir: Optional[Path] = None,
                         doc_id: Optional[str] = None) -> callable:
    """
    Create an embedding function with optional caching.
    
    This function is designed to be passed to compute_s2s_cosine_similarity.
    
    Parameters
    ----------
    model_name : str
        Model name
    cache_dir : Path, optional
        Cache directory
    doc_id : str, optional
        Document ID for caching
        
    Returns
    -------
    callable
        Function that takes list of sentences and returns embeddings
    """
    def embed_fn(sentences: List[str]) -> Optional[np.ndarray]:
        # Try cache first
        if cache_dir and doc_id:
            cached = load_cached_embeddings(doc_id, cache_dir)
            if cached is not None and len(cached) == len(sentences):
                return cached
        
        # Compute embeddings
        embeddings = embed_sentences(sentences, model_name=model_name)
        
        # Save to cache
        if cache_dir and doc_id:
            save_cached_embeddings(embeddings, doc_id, cache_dir)
        
        return embeddings
    
    return embed_fn
