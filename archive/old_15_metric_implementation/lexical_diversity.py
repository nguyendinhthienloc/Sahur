"""
Lexical diversity metrics using lexicalrichness library.
Implements MTLD and HD-D as replacements for TTR.
"""

from lexicalrichness import LexicalRichness
from typing import List, Dict

def compute_lexical_diversity(tokens: List[str]) -> Dict[str, float]:
    """
    Compute MTLD and HD-D lexical diversity.
    
    Parameters
    ----------
    tokens : List[str]
        Tokens without punctuation
    
    Returns
    -------
    Dict containing:
        - mtld: MTLD score
        - hdd: HD-D score
        - vocd: Optional voc-D score
    
    Notes
    -----
    - MTLD requires minimum ~100 tokens for reliable scores
    - HD-D works with shorter texts but may be less stable
    - Returns 0.0 for very short texts (<20 tokens)
    """
    if len(tokens) < 20:
        return {'mtld': 0.0, 'hdd': 0.0, 'vocd': 0.0}
    
    text = " ".join(tokens)
    lex = LexicalRichness(text)
    
    try:
        # MTLD may fail on short texts
        mtld_score = lex.mtld(threshold=0.72) if len(tokens) >= 50 else 0.0
    except Exception:
        mtld_score = 0.0
    
    try:
        hdd_score = lex.hdd(draws=42)
    except Exception:
        hdd_score = 0.0
    
    try:
        vocd_score = lex.vocd() if len(tokens) >= 50 else 0.0
    except Exception:
        vocd_score = 0.0
    
    return {
        'mtld': mtld_score,
        'hdd': hdd_score,
        'vocd': vocd_score
    }
