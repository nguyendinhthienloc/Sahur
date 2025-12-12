"""
Optional IRAL-style lexical explainability module.

Provides:
- Per-topic Log-Odds ratio with Haldane-Anscombe correction
- Bigram collocations with PMI scoring
- Top-k frequency lists per group
- MapReduce-friendly implementation

This module is OPTIONAL and enabled via --enable_irral_lexical flag.
Outputs are for interpretation/appendix only, not mixed into main statistical tests.
"""

import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

from .utils import setup_logger, tokenize_alpha

logger = setup_logger(__name__)

EPSILON = 1e-9
PSEUDOCOUNT = 0.5  # Haldane-Anscombe correction


def compute_log_odds(count_a: int, total_a: int, 
                    count_b: int, total_b: int,
                    pseudocount: float = PSEUDOCOUNT) -> float:
    """
    Compute log-odds ratio with Haldane-Anscombe correction.
    
    Formula:
    log_odds = log((count_a + pseudocount) / (total_a - count_a + pseudocount)) - 
               log((count_b + pseudocount) / (total_b - count_b + pseudocount))
    
    Parameters
    ----------
    count_a : int
        Count in group A
    total_a : int
        Total tokens in group A
    count_b : int
        Count in group B
    total_b : int
        Total tokens in group B
    pseudocount : float
        Smoothing pseudocount
        
    Returns
    -------
    float
        Log-odds ratio (positive = more frequent in A)
    """
    odds_a = (count_a + pseudocount) / (total_a - count_a + pseudocount + EPSILON)
    odds_b = (count_b + pseudocount) / (total_b - count_b + pseudocount + EPSILON)
    
    log_odds = math.log(odds_a + EPSILON) - math.log(odds_b + EPSILON)
    
    return log_odds


def extract_token_counts(texts: List[str]) -> Tuple[Counter, int]:
    """
    Extract token counts from a list of texts.
    
    Returns
    -------
    tuple
        (Counter of tokens, total token count)
    """
    all_tokens = []
    for text in texts:
        tokens = tokenize_alpha(text)
        all_tokens.extend(tokens)
    
    return Counter(all_tokens), len(all_tokens)


def compute_group_log_odds(group_a_texts: List[str],
                          group_b_texts: List[str],
                          min_count: int = 5,
                          top_k: int = 100) -> pd.DataFrame:
    """
    Compute log-odds ratios for tokens between two groups.
    
    Parameters
    ----------
    group_a_texts : List[str]
        Texts from group A
    group_b_texts : List[str]
        Texts from group B
    min_count : int
        Minimum token count threshold
    top_k : int
        Number of top tokens to return
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: token, count_a, count_b, log_odds
        Sorted by absolute log-odds (descending)
    """
    # Extract counts
    counts_a, total_a = extract_token_counts(group_a_texts)
    counts_b, total_b = extract_token_counts(group_b_texts)
    
    # Get union of tokens meeting min_count in either group
    all_tokens = set()
    for token, count in counts_a.items():
        if count >= min_count:
            all_tokens.add(token)
    for token, count in counts_b.items():
        if count >= min_count:
            all_tokens.add(token)
    
    # Compute log-odds for each token
    results = []
    for token in all_tokens:
        count_a = counts_a.get(token, 0)
        count_b = counts_b.get(token, 0)
        
        lor = compute_log_odds(count_a, total_a, count_b, total_b)
        
        results.append({
            'token': token,
            'count_a': count_a,
            'count_b': count_b,
            'log_odds': lor
        })
    
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        return df
    
    # Sort by absolute log-odds
    df['abs_log_odds'] = df['log_odds'].abs()
    df = df.sort_values('abs_log_odds', ascending=False)
    df = df.head(top_k)
    df = df.drop(columns=['abs_log_odds'])
    
    return df


def extract_bigram_collocations(texts: List[str],
                                min_count: int = 5,
                                top_k: int = 50) -> pd.DataFrame:
    """
    Extract bigram collocations using PMI.
    
    PMI(x,y) = log(P(x,y) / (P(x) * P(y)))
    
    Parameters
    ----------
    texts : List[str]
        List of texts
    min_count : int
        Minimum bigram frequency
    top_k : int
        Number of top collocations
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bigram, count, pmi
    """
    # Tokenize
    all_tokens = []
    for text in texts:
        tokens = tokenize_alpha(text)
        all_tokens.extend(tokens)
    
    # Count unigrams and bigrams
    unigram_counts = Counter(all_tokens)
    total_unigrams = len(all_tokens)
    
    bigrams = []
    for i in range(len(all_tokens) - 1):
        bigrams.append((all_tokens[i], all_tokens[i + 1]))
    
    bigram_counts = Counter(bigrams)
    total_bigrams = len(bigrams)
    
    if total_bigrams == 0:
        return pd.DataFrame(columns=['bigram', 'count', 'pmi'])
    
    # Compute PMI
    results = []
    for bigram, count in bigram_counts.items():
        if count < min_count:
            continue
        
        word1, word2 = bigram
        
        p_xy = count / total_bigrams
        p_x = unigram_counts[word1] / total_unigrams
        p_y = unigram_counts[word2] / total_unigrams
        
        if p_x > 0 and p_y > 0:
            pmi = math.log2((p_xy + EPSILON) / (p_x * p_y + EPSILON))
            
            results.append({
                'bigram': f"{word1} {word2}",
                'count': count,
                'pmi': pmi
            })
    
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        return df
    
    df = df.sort_values('pmi', ascending=False)
    df = df.head(top_k)
    
    return df


def compute_top_k_frequencies(texts: List[str],
                             k: int = 100) -> pd.DataFrame:
    """
    Compute top-k most frequent tokens.
    
    Parameters
    ----------
    texts : List[str]
        List of texts
    k : int
        Number of top tokens
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: token, count, frequency
    """
    counts, total = extract_token_counts(texts)
    
    top_tokens = counts.most_common(k)
    
    results = []
    for token, count in top_tokens:
        results.append({
            'token': token,
            'count': count,
            'frequency': count / total if total > 0 else 0
        })
    
    return pd.DataFrame(results)


def compute_irral_outputs(df: pd.DataFrame,
                         group_col: str = 'label',
                         text_col: str = 'text',
                         output_dir: Path = None,
                         min_count: int = 5,
                         top_k: int = 100) -> Dict[str, pd.DataFrame]:
    """
    Compute all IRAL lexical explainability outputs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with text and labels
    group_col : str
        Column name for group labels
    text_col : str
        Column name for text
    output_dir : Path, optional
        Directory to save outputs
    min_count : int
        Minimum frequency threshold
    top_k : int
        Number of top items to extract
        
    Returns
    -------
    dict
        Dictionary of output dataframes
    """
    logger.info("Computing IRAL lexical explainability outputs")
    
    # Get unique groups
    groups = df[group_col].unique()
    
    if len(groups) < 2:
        logger.warning("Need at least 2 groups for log-odds comparison")
        return {}
    
    group_a_val, group_b_val = groups[0], groups[1]
    
    group_a_texts = df[df[group_col] == group_a_val][text_col].tolist()
    group_b_texts = df[df[group_col] == group_b_val][text_col].tolist()
    
    outputs = {}
    
    # Log-odds
    logger.info("Computing log-odds ratios")
    log_odds_df = compute_group_log_odds(
        group_a_texts, group_b_texts,
        min_count=min_count, top_k=top_k
    )
    outputs['log_odds'] = log_odds_df
    
    # Collocations for each group
    logger.info("Extracting collocations")
    collocations_a = extract_bigram_collocations(
        group_a_texts, min_count=min_count, top_k=top_k // 2
    )
    collocations_b = extract_bigram_collocations(
        group_b_texts, min_count=min_count, top_k=top_k // 2
    )
    outputs[f'collocations_group_{group_a_val}'] = collocations_a
    outputs[f'collocations_group_{group_b_val}'] = collocations_b
    
    # Top-k frequencies
    logger.info("Computing top-k frequencies")
    freq_a = compute_top_k_frequencies(group_a_texts, k=top_k)
    freq_b = compute_top_k_frequencies(group_b_texts, k=top_k)
    outputs[f'top_freq_group_{group_a_val}'] = freq_a
    outputs[f'top_freq_group_{group_b_val}'] = freq_b
    
    # Save outputs if directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df_out in outputs.items():
            if len(df_out) > 0:
                path = output_dir / f"{name}.csv"
                df_out.to_csv(path, index=False)
                logger.info(f"Saved {name} to {path}")
    
    logger.info("IRAL lexical outputs complete")
    
    return outputs
