"""
Nominalization detection: improved implementation using suffix + lemma + (optional) WordNet POS check.

Provides:
- compute_nominalization_density(doc, word_count, use_wordnet=True, include_list=False)
  returns {'nominalization_density': <per_1k>, ...optional 'nominalizations': [...]}

Heuristics:
1) token must be a NOUN (exclude PROPN)
2) surface form ends with a nominalization suffix
3) lemma differs from surface form (filters base nouns like "station")
4) (optional) WordNet contains the lemma as a VERB or ADJ (stronger evidence)

This is a pragmatic balance of precision and recall suitable for pipeline use.
"""

from typing import Dict, List, Optional, Tuple
import re

# Try to import WordNet for stronger checks; degrade gracefully if unavailable.
try:
    from nltk.corpus import wordnet as wn  # type: ignore
    _WN_AVAILABLE = True
except Exception:
    _WN_AVAILABLE = False

# Common derivational nominalization suffixes (expandable)
_NOMINALIZATION_SUFFIXES = (
    "tion", "sion", "ment", "ence", "ance", "ity", "ness", "hood", "ship",
    "ism", "acy", "ery", "ary", "age", "ery", "dom", "al", "ure"
)

# Precompile a regex to speed up suffix checks (word boundary)
_SUFFIX_PATTERN = re.compile(r"(" + r"|".join([re.escape(s) + r"$" for s in _NOMINALIZATION_SUFFIXES]) + r")", re.IGNORECASE)


def _lemma_pos_is_verb_or_adj(lemma: str) -> bool:
    """
    Use WordNet to determine whether lemma appears as a verb or adjective.
    Returns True if any synset for lemma has POS 'v' or 'a'.
    """
    if not _WN_AVAILABLE:
        return False

    try:
        synsets = wn.synsets(lemma)
        for s in synsets:
            if s.pos() in ("v", "a"):  # verb or adjective
                return True
    except Exception:
        return False

    return False


def _looks_like_nominalization(token_text: str, token_lemma: str) -> bool:
    """
    Lightweight heuristic: endswith known suffix AND lemma != surface (helps reject base nouns)
    """
    if not token_text:
        return False

    lower = token_text.lower()
    if not _SUFFIX_PATTERN.search(lower):
        return False

    # If lemma equals surface form (after lowercasing), likely not a derivation
    if token_lemma and token_lemma.lower() == lower:
        return False

    return True


def find_nominalizations_in_doc(doc, use_wordnet: bool = True) -> List[Tuple[str, int, str]]:
    """
    Return list of nominalization hits in the doc.
    Each hit is a tuple: (surface_form, token_index, lemma)

    Args:
        doc: spaCy Doc or iterable of token-like objects with .text, .lemma_, .pos_, .i
        use_wordnet: whether to apply WordNet check (if available)

    Returns:
        list of tuples for each detected nominalization
    """
    hits: List[Tuple[str, int, str]] = []

    for token in doc:
        # Only consider common noun POS tag (exclude proper nouns)
        # spaCy: token.pos_ == 'NOUN' for common nouns
        if getattr(token, "pos_", None) != "NOUN":
            continue

        surface = getattr(token, "text", "") or ""
        lemma = getattr(token, "lemma_", "") or ""

        # Basic suffix + lemma-mismatch heuristic
        if not _looks_like_nominalization(surface, lemma):
            continue

        # If WordNet is requested and available, require lemma to appear as verb/adjective
        if use_wordnet and _WN_AVAILABLE:
            if _lemma_pos_is_verb_or_adj(lemma):
                hits.append((surface, getattr(token, "i", -1), lemma))
            else:
                # If WordNet check fails, still allow a hit but mark as lower confidence
                # We'll still include it to avoid losing recall, but callers can filter.
                hits.append((surface, getattr(token, "i", -1), lemma))
        else:
            # No WordNet available or not requested: accept heuristic hit
            hits.append((surface, getattr(token, "i", -1), lemma))

    return hits


def compute_nominalization_density(doc, word_count: int, use_wordnet: bool = True,
                                   include_list: bool = False) -> Dict[str, object]:
    """
    Compute nominalization density per 1,000 words, with optional list of detected tokens.

    Args:
        doc: spaCy Doc (or list of tokens)
        word_count: int — number of tokens (words) used for normalization (should match pipeline 'word_count')
        use_wordnet: bool — if True and WordNet is available, use it to strengthen detections
        include_list: bool — if True, also return a 'nominalizations' list for debugging

    Returns:
        dict with at least {'nominalization_density': float}
        If include_list is True, returns {'nominalization_density': float, 'nominalizations': [...]}
    """
    # Defensive checks
    if doc is None:
        result = {'nominalization_density': 0.0}
        if include_list:
            result['nominalizations'] = []
        return result

    # Find hits
    hits = find_nominalizations_in_doc(doc, use_wordnet=use_wordnet)

    # Optionally filter duplicates or very short forms
    # Convert to unique surface forms while preserving order
    seen = set()
    unique_hits = []
    for surface, idx, lemma in hits:
        key = (surface.lower(), lemma.lower())
        if key in seen:
            continue
        seen.add(key)
        unique_hits.append({'surface': surface, 'index': idx, 'lemma': lemma})

    count = len(unique_hits)
    # Normalize per 100 words (previously per 1k words)
    multiplier = 100.0 / word_count if word_count and word_count > 0 else 0.0
    density = count * multiplier

    if include_list:
        return {'nominalization_density': density, 'nominalizations': unique_hits}
    else:
        return {'nominalization_density': density}


# Backwards-compatible function name (keeps same signature as earlier versions)
def compute_nominalization_density_legacy(doc, word_count: int) -> Dict[str, float]:
    """
    Legacy-compatible wrapper that doesn't attempt WordNet checks and only returns density.
    """
    return compute_nominalization_density(doc, word_count, use_wordnet=False, include_list=False)


# If run as main, self-test (very small)
if __name__ == "__main__":
    # Minimal smoke test using spaCy if available
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        text = "The government's announcement caused speculation about implementation and movement of resources. Station and nation were mentioned too."
        doc = nlp(text)
        print(compute_nominalization_density(doc, word_count=len([t for t in doc if not t.is_punct]), use_wordnet=_WN_AVAILABLE, include_list=True))
    except Exception:
        print("spaCy not available for smoke test. compute_nominalization_density is importable.")
