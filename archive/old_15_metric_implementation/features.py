"""
Feature extraction module.
"""

from typing import Dict, Any, List
from .lexical_diversity import compute_lexical_diversity
from .pos_tools import extract_dependency_depths

def compute_basic_metrics(doc) -> Dict[str, Any]:
    """
    Compute basic linguistic metrics for a document.
    
    Parameters
    ----------
    doc : spacy.tokens.Doc
        Processed spaCy document
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of computed metrics
    """
    # Extract tokens without punctuation for lexical diversity
    tokens = [token.text for token in doc if not token.is_punct]
    
    # Compute lexical diversity (MTLD, HD-D)
    lex_div = compute_lexical_diversity(tokens)
    
    # Compute dependency depths
    dep_depths = extract_dependency_depths(doc)
    
    metrics = {
        'word_count': len(tokens),
        'sentence_count': len(list(doc.sents)),
        **lex_div,
        **dep_depths
    }
    
    return metrics
