import pandas as pd
def compare_one_vs_many(
    df: pd.DataFrame,
    metrics: list,
    group_col: str,
    reference_group: str
) -> pd.DataFrame:
    """
    Compare one reference group (e.g., Human) vs each other group (e.g., models) for each metric.
    Returns a DataFrame with all results and multiple-testing correction.
    """
    import numpy as np
    results = []
    cmp_groups = [g for g in df[group_col].unique() if g != reference_group]
    for metric in metrics:
        if metric not in df.columns:
            continue
        ref_vals = df[df[group_col] == reference_group][metric].values
        ref_vals = ref_vals[~np.isnan(ref_vals)]
        for cmp_group in cmp_groups:
            cmp_vals = df[df[group_col] == cmp_group][metric].values
            cmp_vals = cmp_vals[~np.isnan(cmp_vals)]
            if len(ref_vals) < 2 or len(cmp_vals) < 2:
                continue
            ref_mean = np.mean(ref_vals)
            cmp_mean = np.mean(cmp_vals)
            mean_diff = ref_mean - cmp_mean
            t_stat, t_pval = welch_ttest(ref_vals, cmp_vals)
            u_stat, u_pval = mann_whitney_u(ref_vals, cmp_vals)
            d = cohen_d(ref_vals, cmp_vals)
            results.append({
                'metric': metric,
                'reference': reference_group,
                'comparison': cmp_group,
                'ref_mean': ref_mean,
                'cmp_mean': cmp_mean,
                'mean_diff': mean_diff,
                't_stat': t_stat,
                't_pvalue': t_pval,
                'u_stat': u_stat,
                'u_pvalue': u_pval,
                'cohens_d': d,
                'n_ref': len(ref_vals),
                'n_cmp': len(cmp_vals)
            })
    results_df = pd.DataFrame(results)
    # Multiple testing correction across all comparisons
    if len(results_df) > 0:
        results_df['t_p_holm'] = holm_bonferroni(results_df['t_pvalue'].tolist())
        results_df['t_p_fdr'] = benjamini_hochberg(results_df['t_pvalue'].tolist())
        results_df['u_p_holm'] = holm_bonferroni(results_df['u_pvalue'].tolist())
        results_df['u_p_fdr'] = benjamini_hochberg(results_df['u_pvalue'].tolist())
    return results_df
"""
Statistical analysis module adapted from IRAL replication.

Implements:
- Normality checks (Shapiro-Wilk)
- Variance tests (Levene)
- Two-sample tests (Welch t-test, Mann-Whitney U)
- Effect sizes (Cohen's d with pooled std)
- Multiple testing correction (Holm, Benjamini-Hochberg)
- Export to statistical_tests.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

from .utils import setup_logger

logger = setup_logger(__name__)


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size with pooled standard deviation.
    
    d = (mean_a - mean_b) / pooled_std
    
    where pooled_std = sqrt(((n_a-1)*var_a + (n_b-1)*var_b) / (n_a + n_b - 2))
    """
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    # Remove NaN
    group_a = group_a[~np.isnan(group_a)]
    group_b = group_b[~np.isnan(group_b)]
    
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan
    
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    
    n_a = len(group_a)
    n_b = len(group_b)
    
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean_a - mean_b) / pooled_std
    
    return d


def welch_ttest(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
    """
    Perform Welch's t-test (unequal variances).
    
    Returns
    -------
    tuple
        (t_statistic, p_value)
    """
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    # Remove NaN
    group_a = group_a[~np.isnan(group_a)]
    group_b = group_b[~np.isnan(group_b)]
    
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan, np.nan
    
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    return float(t_stat), float(p_val)


def mann_whitney_u(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test (non-parametric).
    
    Returns
    -------
    tuple
        (u_statistic, p_value)
    """
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    # Remove NaN
    group_a = group_a[~np.isnan(group_a)]
    group_b = group_b[~np.isnan(group_b)]
    
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan, np.nan
    
    u_stat, p_val = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
    
    return float(u_stat), float(p_val)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni correction.
    
    Parameters
    ----------
    p_values : List[float]
        List of p-values
        
    Returns
    -------
    List[float]
        Adjusted p-values
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Compute adjusted p-values
    adjusted_p = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted_p[i] = min(p * (n - i), 1.0)
    
    # Enforce monotonicity
    for i in range(1, n):
        adjusted_p[i] = max(adjusted_p[i], adjusted_p[i - 1])
    
    # Restore original order
    restored_p = np.zeros(n)
    restored_p[sorted_indices] = adjusted_p
    
    return restored_p.tolist()


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Parameters
    ----------
    p_values : List[float]
        List of p-values
        
    Returns
    -------
    List[float]
        Adjusted p-values (q-values)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    if n == 0:
        return []
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Compute adjusted p-values
    adjusted_p = np.zeros(n)
    for i in range(n - 1, -1, -1):
        adjusted_p[i] = min(sorted_p[i] * n / (i + 1), 1.0)
    
    # Enforce monotonicity (reverse)
    for i in range(n - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])
    
    # Restore original order
    restored_p = np.zeros(n)
    restored_p[sorted_indices] = adjusted_p
    
    return restored_p.tolist()


def compare_groups(df: pd.DataFrame, 
                  metrics: List[str],
                  group_col: str = 'label',
                  group_a_value: any = 0,
                  group_b_value: any = 1) -> pd.DataFrame:
    """
    Compare two groups across multiple metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with metrics and group labels
    metrics : List[str]
        List of metric column names
    group_col : str
        Column name for group labels
    group_a_value : any
        Value indicating group A
    group_b_value : any
        Value indicating group B
        
    Returns
    -------
    pd.DataFrame
        Results table with statistics for each metric
    """
    results = []
    
    group_a_df = df[df[group_col] == group_a_value]
    group_b_df = df[df[group_col] == group_b_value]
    
    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in dataframe")
            continue
        
        group_a_vals = group_a_df[metric].values
        group_b_vals = group_b_df[metric].values
        
        # Remove NaN
        group_a_clean = group_a_vals[~np.isnan(group_a_vals)]
        group_b_clean = group_b_vals[~np.isnan(group_b_vals)]
        
        if len(group_a_clean) < 2 or len(group_b_clean) < 2:
            logger.warning(f"Insufficient data for metric '{metric}'")
            continue
        
        # Descriptive stats
        mean_a = np.mean(group_a_clean)
        mean_b = np.mean(group_b_clean)
        std_a = np.std(group_a_clean, ddof=1)
        std_b = np.std(group_b_clean, ddof=1)
        
        # Welch t-test
        t_stat, t_pval = welch_ttest(group_a_clean, group_b_clean)
        
        # Mann-Whitney U
        u_stat, u_pval = mann_whitney_u(group_a_clean, group_b_clean)
        
        # Cohen's d
        d = cohen_d(group_a_clean, group_b_clean)
        
        # Levene's test for variance
        try:
            levene_stat, levene_pval = stats.levene(group_a_clean, group_b_clean)
        except:
            levene_stat, levene_pval = np.nan, np.nan
        
        results.append({
            'metric': metric,
            'group_a_mean': mean_a,
            'group_a_std': std_a,
            'group_a_n': len(group_a_clean),
            'group_b_mean': mean_b,
            'group_b_std': std_b,
            'group_b_n': len(group_b_clean),
            'mean_diff': mean_a - mean_b,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'u_statistic': u_stat,
            'u_pvalue': u_pval,
            'cohens_d': d,
            'levene_statistic': levene_stat,
            'levene_pvalue': levene_pval
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing corrections
    if len(results_df) > 0:
        results_df['t_pvalue_holm'] = holm_bonferroni(results_df['t_pvalue'].tolist())
        results_df['t_pvalue_bh'] = benjamini_hochberg(results_df['t_pvalue'].tolist())
        results_df['u_pvalue_holm'] = holm_bonferroni(results_df['u_pvalue'].tolist())
        results_df['u_pvalue_bh'] = benjamini_hochberg(results_df['u_pvalue'].tolist())
    
    return results_df


def export_statistical_tests(results_df: pd.DataFrame, output_path: Path) -> None:
    """
    Export statistical test results to CSV.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from compare_groups()
    output_path : Path
        Output CSV path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Round numeric columns for readability
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_cols] = results_df[numeric_cols].round(4)
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"Statistical tests exported to {output_path}")
