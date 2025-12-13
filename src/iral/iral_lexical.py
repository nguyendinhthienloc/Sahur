"""
Optional IRAL-style lexical explainability module (preferred spelling: IRAL).

Provides:
- Per-topic Log-Odds ratio with Haldane-Anscombe correction
- Bigram collocations with PMI scoring
- Top-k frequency lists per group

This module is OPTIONAL and can be used separately from the main pipeline.
"""

import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

from ..utils import setup_logger, tokenize_alpha

logger = setup_logger(__name__)

EPSILON = 1e-9
PSEUDOCOUNT = 0.5  # Haldane-Anscombe correction


def compute_log_odds(count_a: int, total_a: int, 
                    count_b: int, total_b: int,
                    pseudocount: float = PSEUDOCOUNT) -> float:
    odds_a = (count_a + pseudocount) / (total_a - count_a + pseudocount + EPSILON)
    odds_b = (count_b + pseudocount) / (total_b - count_b + pseudocount + EPSILON)
    log_odds = math.log(odds_a + EPSILON) - math.log(odds_b + EPSILON)
    return log_odds


def extract_token_counts(texts: List[str]) -> Tuple[Counter, int]:
    all_tokens = []
    for text in texts:
        tokens = tokenize_alpha(text)
        all_tokens.extend(tokens)
    return Counter(all_tokens), len(all_tokens)


def compute_group_log_odds(group_a_texts: List[str], group_b_texts: List[str], min_count: int = 5, top_k: int = 100) -> pd.DataFrame:
    counts_a, total_a = extract_token_counts(group_a_texts)
    counts_b, total_b = extract_token_counts(group_b_texts)
    all_tokens = set([t for t,c in counts_a.items() if c >= min_count] + [t for t,c in counts_b.items() if c >= min_count])
    results = []
    for token in all_tokens:
        count_a = counts_a.get(token, 0)
        count_b = counts_b.get(token, 0)
        lor = compute_log_odds(count_a, total_a, count_b, total_b)
        results.append({'token': token, 'count_a': count_a, 'count_b': count_b, 'log_odds': lor})
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df['abs_log_odds'] = df['log_odds'].abs()
    df = df.sort_values('abs_log_odds', ascending=False).head(top_k).drop(columns=['abs_log_odds'])
    return df


def extract_bigram_collocations(texts: List[str], min_count: int = 5, top_k: int = 50) -> pd.DataFrame:
    all_tokens = []
    for text in texts:
        tokens = tokenize_alpha(text)
        all_tokens.extend(tokens)
    unigram_counts = Counter(all_tokens)
    total_unigrams = len(all_tokens)
    bigrams = [(all_tokens[i], all_tokens[i+1]) for i in range(len(all_tokens)-1)]
    bigram_counts = Counter(bigrams)
    total_bigrams = len(bigrams)
    if total_bigrams == 0:
        return pd.DataFrame(columns=['bigram','count','pmi'])
    results = []
    for bigram, count in bigram_counts.items():
        if count < min_count:
            continue
        w1, w2 = bigram
        p_xy = count / total_bigrams
        p_x = unigram_counts[w1] / total_unigrams
        p_y = unigram_counts[w2] / total_unigrams
        if p_x > 0 and p_y > 0:
            pmi = math.log2((p_xy + EPSILON) / (p_x * p_y + EPSILON))
            results.append({'bigram': f"{w1} {w2}", 'count': count, 'pmi': pmi})
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values('pmi', ascending=False).head(top_k)


def compute_top_k_frequencies(texts: List[str], k: int = 100) -> pd.DataFrame:
    counts, total = extract_token_counts(texts)
    top_tokens = counts.most_common(k)
    return pd.DataFrame([{'token':t,'count':c,'frequency': c/total if total>0 else 0} for t,c in top_tokens])


def compute_iral_outputs(df: pd.DataFrame, group_col: str = 'label', text_col: str = 'text', output_dir: Path = None, min_count: int = 5, top_k: int = 100):
    logger.info("Computing IRAL lexical explainability outputs")
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError(f"compute_iral_outputs is only safe for exactly 2 groups (legacy 1v1 IRAL). Got {len(groups)}: {groups}. Use the orchestrator for multi-corpus analysis.")
    a, b = groups[0], groups[1]
    a_texts = df[df[group_col]==a][text_col].tolist()
    b_texts = df[df[group_col]==b][text_col].tolist()
    outputs = {}
    outputs['log_odds'] = compute_group_log_odds(a_texts, b_texts, min_count=min_count, top_k=top_k)
    outputs[f'collocations_group_{a}'] = extract_bigram_collocations(a_texts, min_count=min_count, top_k=top_k//2)
    outputs[f'collocations_group_{b}'] = extract_bigram_collocations(b_texts, min_count=min_count, top_k=top_k//2)
    outputs[f'top_freq_group_{a}'] = compute_top_k_frequencies(a_texts, k=top_k)
    outputs[f'top_freq_group_{b}'] = compute_top_k_frequencies(b_texts, k=top_k)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df_out in outputs.items():
            if len(df_out) > 0:
                (output_dir / f"{name}.csv").write_text(df_out.to_csv(index=False))
    return outputs
