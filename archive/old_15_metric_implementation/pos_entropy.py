"""
POS sequence entropy calculations.
"""

from collections import Counter
import math
from typing import List, Dict

def compute_pos_ngram_entropy(pos_sequence: List[str], n: int = 2) -> float:
    """
    Calculate entropy of POS n-gram distribution.
    
    H = -Σ p(ngram) * log2(p(ngram))
    
    Parameters
    ----------
    pos_sequence : List[str]
        POS tag sequence
    n : int
        N-gram size (2=bigram, 3=trigram)
    """
    if len(pos_sequence) < n:
        return 0.0
    
    # Extract n-grams
    ngrams = []
    for i in range(len(pos_sequence) - n + 1):
        ngram = tuple(pos_sequence[i:i+n])
        ngrams.append(ngram)
    
    # Compute probabilities
    total = len(ngrams)
    counts = Counter(ngrams)
    
    # Calculate entropy
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    
    return entropy

def compute_pos_transition_markov(pos_sequence: List[str], 
                                  baseline_matrix: Dict) -> float:
    """
    Compute KL divergence of POS transition probabilities
    vs human baseline matrix.
    """
    if len(pos_sequence) < 2:
        return 0.0
        
    # Build transition matrix
    transitions = Counter()
    for i in range(len(pos_sequence) - 1):
        transitions[(pos_sequence[i], pos_sequence[i+1])] += 1
    
    total_transitions = sum(transitions.values())
    if total_transitions == 0:
        return 0.0
        
    kl_div = 0.0
    epsilon = 1e-10 # Smoothing for unseen transitions in baseline
    
    for transition, count in transitions.items():
        p = count / total_transitions
        # Get q from baseline, default to epsilon if not found
        # Baseline keys should be tuples or string representations
        q = baseline_matrix.get(transition, epsilon)
        if q == 0: q = epsilon
        
        kl_div += p * math.log2(p / q)
        
    return kl_div
