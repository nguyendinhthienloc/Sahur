"""
Entity specificity scoring.

Improvements:
- Weighted entity counts by entity type (PERSON, ORG, GPE, DATE, CARDINAL, etc.).
- Uniqueness boost for distinct entities (more unique named entities -> higher specificity).
- Length / span factor: multi-token entities are weighted more.
- Returns 'entity_specificity' normalized per 1k words and an optional breakdown.

API:
    compute_entity_specificity(doc, word_count, include_breakdown=False)
"""

from typing import Dict
from collections import Counter
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Tunable weights per spaCy entity label.
# Values chosen as reasonable defaults; adjust to taste/data.
_ENTITY_TYPE_WEIGHTS = {
    'PERSON': 1.5,
    'ORG': 1.3,
    'GPE': 1.2,
    'LOC': 1.1,
    'PRODUCT': 1.0,
    'EVENT': 1.1,
    'WORK_OF_ART': 0.9,
    'LAW': 1.0,
    'LANGUAGE': 0.8,
    'DATE': 0.6,
    'TIME': 0.5,
    'MONEY': 0.8,
    'QUANTITY': 0.5,
    'CARDINAL': 0.4,
    'ORDINAL': 0.4,
    # default weight for unknown labels
    'DEFAULT': 0.7
}


def _entity_weight(label: str) -> float:
    return _ENTITY_TYPE_WEIGHTS.get(label, _ENTITY_TYPE_WEIGHTS['DEFAULT'])


def compute_entity_specificity(doc, word_count: int, include_breakdown: bool = False) -> Dict[str, object]:
    """
    Compute a scored entity specificity per 100 words (previously per 1k words).

    Score components:
      - type weight (label importance)
      - uniqueness factor (unique entity surface forms)
      - span factor (longer multi-token entities get a small boost)

    Returns:
      {'entity_specificity': float} or with breakdown:
      {'entity_specificity': float, 'entity_breakdown': {...}}
    """
    if doc is None or word_count <= 0:
        result = {'entity_specificity': 0.0}
        if include_breakdown:
            result['entity_breakdown'] = {}
        return result

    ents = list(getattr(doc, "ents", []))
    if not ents:
        result = {'entity_specificity': 0.0}
        if include_breakdown:
            result['entity_breakdown'] = {}
        return result

    # Count entities and surfaces for uniqueness
    surface_counter = Counter([ent.text.lower() for ent in ents])
    type_counter = Counter([ent.label_ for ent in ents])

    total_score = 0.0
    breakdown = {}

    for ent in ents:
        label = ent.label_ or 'DEFAULT'
        weight = _entity_weight(label)
        surface = ent.text.lower()
        uniqueness_factor = 1.0 + (1.0 / surface_counter[surface])  # more unique -> closer to 2.0
        span_length = len(ent.text.split())
        span_factor = 1.0 + 0.1 * (span_length - 1)  # small boost for multi-token entities

        ent_score = weight * uniqueness_factor * span_factor
        total_score += ent_score

        # breakdown per type
        breakdown.setdefault(label, {'count': 0, 'raw_score': 0.0})
        breakdown[label]['count'] += 1
        breakdown[label]['raw_score'] += ent_score

    # Normalize per 100 words (reduced scale for interpretability)
    multiplier = 100.0 / word_count
    specificity = total_score * multiplier

    result = {'entity_specificity': specificity}
    if include_breakdown:
        # also return counts and per-type normalized scores
        normalized_breakdown = {
            lab: {
                'count': v['count'],
                'score_per1k': (v['raw_score'] * multiplier)
            } for lab, v in breakdown.items()
        }
        result['entity_breakdown'] = normalized_breakdown

    return result
