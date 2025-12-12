"""
Shared utilities for the 6-metric linguistic baseline pipeline.

Provides:
- Tokenization helpers
- Nominalization suffix lists
- Modal and epistemic phrase lists
- Schema generation for metrics
- Common logging utilities
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Nominalization suffixes (from IRAL and existing implementation)
NOMINALIZATION_SUFFIXES = (
    "tion", "sion", "ment", "ence", "ance", "ity", "ness", "hood", "ship",
    "ism", "acy", "ery", "ary", "age", "dom", "al", "ure"
)

# Modal verbs
MODAL_VERBS = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']

# Epistemic markers
EPISTEMIC_MARKERS = [
    'probably', 'possibly', 'certainly', 'clearly', 'obviously', 'perhaps',
    'apparently', 'arguably', 'presumably', 'supposedly', 'definitely',
    'undoubtedly', 'likely', 'unlikely', 'maybe'
]

# Combined modal/epistemic list
MODAL_EPISTEMIC_PHRASES = MODAL_VERBS + EPISTEMIC_MARKERS


def tokenize_alpha(text: str) -> List[str]:
    """
    Simple alpha-only tokenization.
    
    Returns list of lowercase alphabetic tokens (no punctuation, no numbers).
    Suitable for lexical diversity and frequency analysis.
    """
    if not text:
        return []
    tokens = []
    for word in text.split():
        clean = ''.join(c for c in word if c.isalpha())
        if clean:
            tokens.append(clean.lower())
    return tokens


def is_nominalization(token_text: str, token_lemma: str, token_pos: str) -> bool:
    """
    Heuristic to determine if a token is a derived nominalization.
    
    Criteria:
    1. POS must be NOUN (not PROPN)
    2. Surface form ends with nominalization suffix
    3. Lemma differs from surface (filters base nouns like "station")
    
    Parameters
    ----------
    token_text : str
        Surface form of the token
    token_lemma : str
        Lemma of the token
    token_pos : str
        POS tag
        
    Returns
    -------
    bool
        True if likely a derived nominalization
    """
    if token_pos != "NOUN":
        return False
    
    text_lower = token_text.lower()
    
    # Must end with a nominalization suffix
    has_suffix = any(text_lower.endswith(suffix) for suffix in NOMINALIZATION_SUFFIXES)
    if not has_suffix:
        return False
    
    # Lemma should differ from surface (derivation indicator)
    if token_lemma.lower() == text_lower:
        return False
    
    return True


def write_metrics_schema(output_path: Path, core_metrics: List[str], 
                        optional_metrics: List[str] = None) -> None:
    """
    Write a JSON schema describing the metrics.
    
    Parameters
    ----------
    output_path : Path
        Where to write the schema
    core_metrics : List[str]
        List of core metric names
    optional_metrics : List[str], optional
        List of optional metric names
    """
    schema = {
        "version": "1.0.0",
        "pipeline": "6-metric-baseline",
        "core_metrics": core_metrics,
        "optional_metrics": optional_metrics or [],
        "description": "Core linguistic features for human vs AI text classification"
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def normalize_per_1k(count: int, word_count: int) -> float:
    """Normalize a count per 1000 words."""
    return safe_divide(count * 1000, word_count, 0.0)


def normalize_per_100(count: int, word_count: int) -> float:
    """Normalize a count per 100 words."""
    return safe_divide(count * 100, word_count, 0.0)
