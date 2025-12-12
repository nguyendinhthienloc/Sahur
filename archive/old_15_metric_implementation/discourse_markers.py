"""
Discourse marker and modal detection using spaCy PhraseMatcher and lemma-aware counts.

Improvements:
- Uses spaCy PhraseMatcher to detect multi-word markers.
- Uses token.lemma_ for modal detection to catch variations.
- Avoids counting extremely high-frequency function words (e.g., 'and') by default.
- Returns normalized counts per 100 words and a breakdown map for diagnostics.

API:
- compute_discourse_distribution(doc_or_tokens, word_count, nlp=None, exclude_high_freq=True)
- compute_modal_rate(doc_or_tokens, word_count, nlp=None)
"""

from typing import Dict, List, Union
from collections import Counter
import logging

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Multi-word and single-word markers derived from PDTB categories (small, extend as needed)
_DISCOURSE_MARKERS = {
    'temporal': ['then', 'after', 'before', 'while', 'when', 'since', 'at the same time'],
    'contingency': ['because', 'if', 'so', 'thus', 'consequently', 'therefore', 'as a result', 'due to'],
    'comparison': ['however', 'but', 'although', 'conversely', 'nevertheless', 'yet', 'in contrast', 'on the other hand'],
    # Note: 'and' removed from default expansion to reduce noise
    'expansion': ['also', 'moreover', 'furthermore', 'additionally', 'besides', 'in addition', 'as well as']
}

_MODAL_VERBS = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
_EPISTEMIC_MARKERS = [
    'probably', 'possibly', 'certainly', 'clearly', 'obviously', 'perhaps',
    'apparently', 'arguably', 'presumably', 'supposedly'
]


def _ensure_nlp(nlp):
    """Load a minimal spaCy model if not supplied."""
    if nlp is not None:
        return nlp
    if not _SPACY_AVAILABLE:
        raise RuntimeError("spaCy is required for discourse marker detection but is not available.")
    try:
        return spacy.load("en_core_web_sm", disable=['ner'])
    except Exception:
        # Try to load without download; caller must ensure model exists in environment
        return spacy.load("en_core_web_sm", disable=['ner'])


def _build_phrase_matcher(nlp):
    """Return a PhraseMatcher with discourse marker patterns registered."""
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    for cat, markers in _DISCOURSE_MARKERS.items():
        patterns = [nlp.make_doc(m) for m in markers]
        matcher.add(cat, patterns)
    return matcher


def compute_discourse_distribution(doc_or_tokens: Union[str, List[str], 'spacy.tokens.Doc'],
                                   word_count: int,
                                   nlp=None,
                                   exclude_high_freq: bool = True) -> Dict[str, float]:
    """
    Compute counts per PDTB-like categories normalized per 1k words.

    Accepts:
      - spaCy Doc
      - raw text (str) -> will be parsed with nlp
      - list of tokens (strings) -> treated as simple bag-of-tokens fallback

    Returns a dict with keys: discourse_{category}_per1k and a small breakdown map.
    """
    if word_count <= 0:
        # Defensive early return
        return {f'discourse_{k}_per1k': 0.0 for k in _DISCOURSE_MARKERS.keys()}

    if isinstance(doc_or_tokens, str):
        nlp = _ensure_nlp(nlp)
        doc = nlp(doc_or_tokens)
    elif _SPACY_AVAILABLE and hasattr(doc_or_tokens, "ents"):
        doc = doc_or_tokens  # assumed spaCy Doc
    else:
        # List of tokens fallback
        lower_tokens = [t.lower() for t in doc_or_tokens] if isinstance(doc_or_tokens, list) else []
        counts = {}
        for cat, markers in _DISCOURSE_MARKERS.items():
            # simple token match fallback (single-word only)
            c = 0
            for m in markers:
                if ' ' in m:
                    continue  # skip multi-word in fallback
                c += lower_tokens.count(m)
            counts[cat] = c
        # Normalize per 100 words (was per 1k)
        multiplier = 100.0 / word_count
        return {f'discourse_{k}_per1k': v * multiplier for k, v in counts.items()}

    # Build matcher and run
    matcher = _build_phrase_matcher(nlp)
    matches = matcher(doc)

    # Count matches per category
    cat_counts = Counter()
    for match_id, start, end in matches:
        cat = nlp.vocab.strings[match_id]
        cat_counts[cat] += 1

    # Optionally also count single-token occurrences for markers not covered by phrases
    # (this uses lemma to catch morphological variants)
    lemmas = [t.lemma_.lower() for t in doc]
    for cat, markers in _DISCOURSE_MARKERS.items():
        for marker in markers:
            if ' ' in marker:
                continue
            # skip common high-frequency function words if exclude_high_freq==True
            if exclude_high_freq and marker in ('and', 'or', 'but'):
                continue
            cat_counts[cat] += lemmas.count(marker)

    # Normalize per 100 words (was per 1k)
    multiplier = 100.0 / word_count
    result = {f'discourse_{k}_per1k': cat_counts.get(k, 0) * multiplier for k in _DISCOURSE_MARKERS.keys()}
    # Provide a small breakdown for debugging
    result['discourse_breakdown'] = dict(cat_counts)
    return result


def compute_modal_rate(doc_or_tokens: Union[str, List[str], 'spacy.tokens.Doc'],
                       word_count: int,
                       nlp=None) -> Dict[str, float]:
    """
    Count modal verbs and epistemic markers per 1k words.
    Accepts spaCy Doc, raw text, or token list.
    """
    if word_count <= 0:
        return {'modal_verbs_per1k': 0.0, 'epistemic_markers_per1k': 0.0}

    if isinstance(doc_or_tokens, str):
        nlp = _ensure_nlp(nlp)
        doc = nlp(doc_or_tokens)
    elif _SPACY_AVAILABLE and hasattr(doc_or_tokens, "ents"):
        doc = doc_or_tokens
    else:
        lower_tokens = [t.lower() for t in doc_or_tokens] if isinstance(doc_or_tokens, list) else []
        modal_count = sum(lower_tokens.count(m) for m in _MODAL_VERBS)
        epistemic_count = sum(lower_tokens.count(m) for m in _EPISTEMIC_MARKERS)
        # Normalize per 100 words (was per 1k)
        multiplier = 100.0 / word_count
        return {'modal_verbs_per1k': modal_count * multiplier, 'epistemic_markers_per1k': epistemic_count * multiplier}

    lemmas = [t.lemma_.lower() for t in doc]
    modal_count = sum(lemmas.count(m) for m in _MODAL_VERBS)
    epistemic_count = sum(lemmas.count(m) for m in _EPISTEMIC_MARKERS)
    # Normalize per 100 words (was per 1k)
    multiplier = 100.0 / word_count
    return {'modal_verbs_per1k': modal_count * multiplier, 'epistemic_markers_per1k': epistemic_count * multiplier}
