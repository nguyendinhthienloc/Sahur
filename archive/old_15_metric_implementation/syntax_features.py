"""
Syntactic complexity features from dependency parsing.
"""

from typing import Dict

def compute_clause_density(doc, word_count: int) -> Dict[str, float]:
    """
    Count advcl, ccomp, acl, parataxis per 1k words.
    """
    clause_types = {
        'advcl': 0,  # Adverbial clause
        'ccomp': 0,  # Clausal complement
        'acl': 0,    # Adjectival clause
        'parataxis': 0,
        'xcomp': 0   # Open clausal complement
    }
    
    for token in doc:
        if token.dep_ in clause_types:
            clause_types[token.dep_] += 1
    
    # Normalize per 100 words (was per 1k; reduced scale for interpretability)
    multiplier = 100.0 / word_count if word_count > 0 else 0
    return {
        f'{k}_per1k': v * multiplier 
        for k, v in clause_types.items()
    }

def compute_passive_ratio(doc) -> float:
    """
    Calculate % of sentences using passive structure.
    """
    sentences = list(doc.sents)
    passive_count = 0
    
    for sent in sentences:
        for token in sent:
            if token.dep_ == "nsubjpass":
                passive_count += 1
                break
    
    return passive_count / len(sentences) if sentences else 0.0
