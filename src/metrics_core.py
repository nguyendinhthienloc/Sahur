"""
Core linguistic metrics for the 6-metric baseline.

Implements exactly 6 metrics:
1. mtld - Lexical diversity (Measure of Textual Lexical Diversity)
2. nominalization_density - Derived nominalizations per 1000 words
3. modal_epistemic_rate - Modal verbs and epistemic markers per 100 words
4. clause_complexity - Mean clausal dependencies per sentence
5. passive_voice_ratio - Proportion of passive voice sentences
6. s2s_cosine_similarity - Mean sentence-to-sentence cosine similarity

All metrics are deterministic and use spaCy parses.
The s2s_cosine metric requires sentence embeddings (handled by embeddings module).
"""

import spacy
import numpy as np
from typing import Dict, List, Optional, Callable
from lexicalrichness import LexicalRichness

from .utils import (
    setup_logger, 
    is_nominalization, 
    MODAL_EPISTEMIC_PHRASES,
    safe_divide,
    normalize_per_1k,
    normalize_per_100
)

logger = setup_logger(__name__)


def compute_mtld(doc: spacy.tokens.Doc) -> float:
    """
    Compute MTLD (Measure of Textual Lexical Diversity).
    
    Uses lexicalrichness library for docs with >=50 tokens.
    For shorter documents, uses TTR-based fallback scaled to MTLD range.
    
    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document
        
    Returns
    -------
    float
        MTLD score (or TTR-based approximation for short docs)
    """
    # Extract alpha tokens (no punctuation)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    
    if len(tokens) < 10:
        # Too short, return minimum value
        return 10.0
    
    if len(tokens) < 50:
        # Use TTR-based fallback for short documents
        # TTR = types/tokens, scale to approximate MTLD range (typically 20-100)
        types = len(set(tokens))
        ttr = types / len(tokens)
        # Scale: TTR ranges from ~0.3 to ~0.9, MTLD typically 20-100
        # Use linear scaling: MTLD ≈ TTR * 100
        fallback_mtld = ttr * 100
        return float(fallback_mtld)
    
    try:
        text = " ".join(tokens)
        lex = LexicalRichness(text)
        mtld_score = lex.mtld(threshold=0.72)
        return float(mtld_score)
    except Exception as e:
        logger.warning(f"MTLD computation failed: {e}, using TTR fallback")
        # Fallback to TTR even for longer docs if MTLD fails
        types = len(set(tokens))
        ttr = types / len(tokens)
        return float(ttr * 100)


def compute_nominalization_density(doc: spacy.tokens.Doc) -> float:
    """
    Compute nominalization density per 1000 words.
    
    Uses improved heuristic:
    - Token is NOUN (not PROPN)
    - Ends with nominalization suffix (ion, ment, ity, ance, ence, al)
    - Frequency floor: word must appear at least once
    
    Includes both lemma-based detection and surface form patterns.
    
    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document
        
    Returns
    -------
    float
        Nominalizations per 1000 words
    """
    word_count = sum(1 for token in doc if token.is_alpha)
    
    if word_count < 20:
        return 0.0
    
    nom_count = 0
    
    # Primary suffixes for nominalization (stricter list)
    primary_suffixes = ('ion', 'tion', 'sion', 'ment', 'ity', 'ance', 'ence', 'ness')
    
    for token in doc:
        if not token.is_alpha or token.pos_ != "NOUN":
            continue
            
        text_lower = token.text.lower()
        lemma_lower = token.lemma_.lower()
        
        # Check if word has nominalization suffix and meets basic criteria
        has_suffix = any(text_lower.endswith(suf) for suf in primary_suffixes)
        
        if has_suffix and len(text_lower) >= 6:
            # Count it if:
            # 1. Lemma differs from surface (derived form), OR
            # 2. It has a strong nominalization suffix (-tion, -ment, -ity)
            strong_suffixes = ('tion', 'sion', 'ment', 'ity', 'ance', 'ence')
            
            if lemma_lower != text_lower or any(text_lower.endswith(suf) for suf in strong_suffixes):
                nom_count += 1
    
    # Ensure we return a reasonable value even for texts without detected nominalizations
    # Most texts will have at least some nominalizations
    result = normalize_per_1k(nom_count, word_count)
    
    # Log warning if suspiciously low
    if result == 0.0 and word_count >= 100:
        logger.debug(f"No nominalizations detected in document with {word_count} words")
    
    return result


def compute_modal_epistemic_rate(doc: spacy.tokens.Doc) -> float:
    """
    Compute modal and epistemic marker rate per 100 words.
    
    Counts:
    - Modal verbs (can, should, must, etc.)
    - Epistemic adverbs (probably, possibly, certainly, etc.)
    
    Uses lemma matching for robustness.
    
    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document
        
    Returns
    -------
    float
        Modal/epistemic markers per 100 words
    """
    word_count = sum(1 for token in doc if token.is_alpha)
    
    if word_count == 0:
        return 0.0
    
    modal_count = 0
    for token in doc:
        lemma_lower = token.lemma_.lower()
        text_lower = token.text.lower()
        
        if lemma_lower in MODAL_EPISTEMIC_PHRASES or text_lower in MODAL_EPISTEMIC_PHRASES:
            modal_count += 1
    
    return normalize_per_100(modal_count, word_count)


def compute_clause_complexity(doc: spacy.tokens.Doc) -> float:
    """
    Compute mean clausal dependencies per sentence.
    
    Counts these dependency relations per sentence:
    - advcl (adverbial clause)
    - ccomp (clausal complement)
    - acl (adjectival clause)
    - xcomp (open clausal complement)
    - relcl (relative clause)
    
    Returns mean count across all sentences.
    
    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document
        
    Returns
    -------
    float
        Mean clausal dependencies per sentence
    """
    sentences = list(doc.sents)
    
    if not sentences:
        return 0.0
    
    clause_deps = {'advcl', 'ccomp', 'acl', 'xcomp', 'relcl'}
    
    total_clauses = 0
    for sent in sentences:
        sent_clauses = sum(1 for token in sent if token.dep_ in clause_deps)
        total_clauses += sent_clauses
    
    return safe_divide(total_clauses, len(sentences), 0.0)


def compute_passive_voice_ratio(doc: spacy.tokens.Doc) -> float:
    """
    Compute proportion of sentences containing passive voice.
    
    A sentence is considered passive if it contains 'nsubjpass' or 'auxpass' dependency.
    
    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document
        
    Returns
    -------
    float
        Proportion of passive sentences (0.0 to 1.0)
    """
    sentences = list(doc.sents)
    
    if not sentences:
        return 0.0
    
    passive_count = 0
    for sent in sentences:
        has_passive = any(token.dep_ in ('nsubjpass', 'auxpass') for token in sent)
        if has_passive:
            passive_count += 1
    
    return safe_divide(passive_count, len(sentences), 0.0)


def compute_s2s_cosine_similarity(doc: spacy.tokens.Doc, 
                                 embed_fn: Optional[Callable] = None) -> float:
    """
    Compute mean sentence-to-sentence cosine similarity.
    
    This metric requires sentence embeddings. If embed_fn is None,
    returns NaN (embeddings must be computed separately).
    
    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document
    embed_fn : Callable, optional
        Function that takes a list of sentence texts and returns embeddings
        Should return np.ndarray of shape (n_sentences, embedding_dim)
        
    Returns
    -------
    float
        Mean cosine similarity between adjacent sentences
        Returns NaN if embeddings not available
    """
    sentences = list(doc.sents)
    
    if len(sentences) < 2:
        return 0.0
    
    if embed_fn is None:
        # Embeddings not computed - return NaN as placeholder
        return np.nan
    
    # Extract sentence texts
    sent_texts = [sent.text.strip() for sent in sentences]
    
    try:
        # Get embeddings
        embeddings = embed_fn(sent_texts)
        
        if embeddings is None or len(embeddings) < 2:
            return np.nan
        
        # Compute adjacent cosine similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            v1 = embeddings[i]
            v2 = embeddings[i + 1]
            
            # Cosine similarity
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
                similarities.append(cosine_sim)
        
        if similarities:
            return float(np.mean(similarities))
        else:
            return 0.0
            
    except Exception as e:
        logger.warning(f"Failed to compute s2s cosine similarity: {e}")
        return np.nan


def extract_core_metrics(doc: spacy.tokens.Doc,
                        embed_fn: Optional[Callable] = None,
                        doc_id: Optional[str] = None) -> Dict[str, float]:
    """
    Extract all 6 core metrics from a parsed document.
    
    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document
    embed_fn : Callable, optional
        Embedding function for s2s_cosine_similarity
    doc_id : str, optional
        Document ID for logging
        
    Returns
    -------
    Dict[str, float]
        Dictionary with all 6 core metrics
    """
    # Check document validity
    word_count = sum(1 for token in doc if token.is_alpha)
    
    if word_count < 50:
        logger.warning(f"Document {doc_id or 'unknown'} too short ({word_count} words), returning NaN")
        return {
            'mtld': np.nan,
            'nominalization_density': np.nan,
            'modal_epistemic_rate': np.nan,
            'clause_complexity': np.nan,
            'passive_voice_ratio': np.nan,
            's2s_cosine_similarity': np.nan,
            'word_count': word_count
        }
    
    metrics = {
        'mtld': compute_mtld(doc),
        'nominalization_density': compute_nominalization_density(doc),
        'modal_epistemic_rate': compute_modal_epistemic_rate(doc),
        'clause_complexity': compute_clause_complexity(doc),
        'passive_voice_ratio': compute_passive_voice_ratio(doc),
        's2s_cosine_similarity': compute_s2s_cosine_similarity(doc, embed_fn),
        'word_count': word_count
    }
    
    return metrics


# Define metric names for schema generation
CORE_METRIC_NAMES = [
    'mtld',
    'nominalization_density',
    'modal_epistemic_rate',
    'clause_complexity',
    'passive_voice_ratio',
    's2s_cosine_similarity'
]
