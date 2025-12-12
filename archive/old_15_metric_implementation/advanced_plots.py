"""
Advanced visualization helpers — updated for large-scale runs.

Improvements:
- Optional plotting (enable/disable via `enable_plots`).
- Subsampling (sample_frac or sample_size) to avoid OOM on large datasets.
- Defensive checks for missing metrics, single-class labels, NaNs.
- Safe defaults: functions return quietly if not enough data.
- All functions accept a DataFrame and explicit metric list; they do not attempt
  to infer metrics from the DataFrame automatically (avoid surprises).

Usage:
    create_violin_boxplot(df, metric, label_col, output_path,
                          enable_plots=True, sample_frac=0.1, min_rows=50)
"""

from typing import List, Optional
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional import - UMAP can be heavy, keep as optional import at runtime
try:
    import umap  # type: ignore
    _UMAP_AVAILABLE = True
except Exception:
    _UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], sample_size: Optional[int]) -> pd.DataFrame:
    """Return a sampled copy of df according to fraction or fixed size."""
    if df is None or df.empty:
        return df
    if sample_size is not None and sample_size > 0:
        if len(df) <= sample_size:
            return df.copy()
        return df.sample(n=sample_size, random_state=42)
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        return df.sample(frac=sample_frac, random_state=42)
    return df.copy()


def _ensure_metric_exists(df: pd.DataFrame, metric: str) -> bool:
    if metric not in df.columns:
        logger.warning("Metric '%s' not found in DataFrame columns.", metric)
        return False
    if not np.issubdtype(df[metric].dtype, np.number):
        logger.warning("Metric '%s' is not numeric.", metric)
        return False
    return True


def create_violin_boxplot(df: pd.DataFrame,
                          metric: str,
                          label_col: str,
                          output_path: str,
                          enable_plots: bool = True,
                          sample_frac: Optional[float] = 0.1,
                          sample_size: Optional[int] = None,
                          min_rows: int = 30) -> None:
    """Combined violin + box plot comparing metric across label groups."""
    if not enable_plots:
        return

    if df is None or df.empty:
        logger.info("Empty DataFrame; skipping violin plot for %s.", metric)
        return

    if not _ensure_metric_exists(df, metric) or label_col not in df.columns:
        return

    df_sample = _maybe_sample(df[[metric, label_col]].dropna(), sample_frac, sample_size)
    if len(df_sample) < min_rows:
        logger.info("Not enough rows (%d) for violin plot of %s; skipping.", len(df_sample), metric)
        return

    try:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_sample, x=label_col, y=metric, inner=None)
        sns.boxplot(data=df_sample, x=label_col, y=metric, width=0.2, color="white")
        plt.title(f'Distribution of {metric} by {label_col}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.exception("Failed to create violin plot for %s: %s", metric, e)


def create_correlation_heatmap(df: pd.DataFrame,
                               metrics: List[str],
                               output_path: str,
                               enable_plots: bool = True,
                               sample_frac: Optional[float] = 0.05,
                               sample_size: Optional[int] = None,
                               min_features: int = 3) -> None:
    """Correlation heatmap for a selected list of numeric metrics."""
    if not enable_plots:
        return

    if df is None or df.empty:
        logger.info("Empty DataFrame; skipping correlation heatmap.")
        return

    metrics_filtered = [m for m in metrics if m in df.columns and np.issubdtype(df[m].dtype, np.number)]
    if len(metrics_filtered) < min_features:
        logger.info("Too few numeric metrics (%d) for heatmap; skipping.", len(metrics_filtered))
        return

    df_sample = _maybe_sample(df[metrics_filtered].dropna(), sample_frac, sample_size)
    try:
        corr = df_sample.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(max(8, len(metrics_filtered)*0.5), max(6, len(metrics_filtered)*0.4)))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Metric Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.exception("Failed to create correlation heatmap: %s", e)


def create_radar_chart(means_human: np.ndarray,
                       means_ai: np.ndarray,
                       metrics: List[str],
                       output_path: str,
                       enable_plots: bool = True) -> None:
    """Radar chart comparing two averaged metric profiles (expects arrays normalized 0..1)."""
    if not enable_plots:
        return
    if len(metrics) < 3:
        logger.info("Need at least 3 metrics for radar chart; skipping.")
        return

    try:
        import numpy as _np
        N = len(metrics)
        angles = [_np.linspace(0, 2*_np.pi, N, endpoint=False)]
        angles = list(_np.linspace(0, 2*_np.pi, N, endpoint=False))
        angles += angles[:1]

        values_human = list(means_human) + [means_human[0]]
        values_ai = list(means_ai) + [means_ai[0]]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        plt.xticks(angles[:-1], metrics, color='grey', size=8)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)

        ax.plot(angles, values_human, linewidth=1, linestyle='solid', label="Human")
        ax.fill(angles, values_human, alpha=0.1)

        ax.plot(angles, values_ai, linewidth=1, linestyle='solid', label="AI")
        ax.fill(angles, values_ai, alpha=0.1)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Metric Profile Comparison')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.exception("Failed to create radar chart: %s", e)


def create_pca_scatter(df: pd.DataFrame,
                       label_col: str,
                       metrics: List[str],
                       output_path: str,
                       enable_plots: bool = True,
                       sample_frac: Optional[float] = 0.05,
                       sample_size: Optional[int] = None,
                       min_rows: int = 10) -> None:
    """PCA scatter of selected metrics colored by label_col."""
    if not enable_plots:
        return
    if df is None or df.empty:
        return
    if label_col not in df.columns:
        logger.warning("Label column '%s' not found for PCA scatter.", label_col)
        return

    metrics_filtered = [m for m in metrics if m in df.columns and np.issubdtype(df[m].dtype, np.number)]
    if not metrics_filtered:
        logger.info("No numeric metrics for PCA scatter; skipping.")
        return

    df_sample = _maybe_sample(df[[label_col] + metrics_filtered].dropna(), sample_frac, sample_size)
    if len(df_sample) < min_rows:
        logger.info("Not enough rows (%d) for PCA scatter; skipping.", len(df_sample))
        return

    try:
        x = df_sample[metrics_filtered].values
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        components = pca.fit_transform(x)
        plot_df = pd.DataFrame({'PC1': components[:, 0], 'PC2': components[:, 1], label_col: df_sample[label_col].values})
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue=label_col, alpha=0.7)
        plt.title(f'PCA of Linguistic Metrics (PC1 {pca.explained_variance_ratio_[0]:.2%})')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.exception("Failed to create PCA scatter: %s", e)


def create_umap_scatter(df: pd.DataFrame,
                       label_col: str,
                       metrics: List[str],
                       output_path: str,
                       enable_plots: bool = True,
                       sample_frac: Optional[float] = 0.05,
                       sample_size: Optional[int] = None,
                       min_rows: int = 50) -> None:
    """UMAP scatter (optional)."""
    if not enable_plots:
        return

    if not _UMAP_AVAILABLE:
        logger.info("UMAP not available; skipping UMAP scatter.")
        return

    if df is None or df.empty:
        return
    if label_col not in df.columns:
        logger.warning("Label column '%s' not found for UMAP scatter.", label_col)
        return

    metrics_filtered = [m for m in metrics if m in df.columns and np.issubdtype(df[m].dtype, np.number)]
    if not metrics_filtered:
        logger.info("No numeric metrics for UMAP scatter; skipping.")
        return

    df_sample = _maybe_sample(df[[label_col] + metrics_filtered].dropna(), sample_frac, sample_size)
    if len(df_sample) < min_rows:
        logger.info("Not enough rows (%d) for UMAP plot; skipping.", len(df_sample))
        return

    try:
        x = df_sample[metrics_filtered].values
        x = StandardScaler().fit_transform(x)
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(x)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=df_sample[label_col], alpha=0.7)
        plt.title('UMAP Projection of Linguistic Metrics')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.exception("Failed to create UMAP scatter: %s", e)


def create_paired_difference_plot(df: pd.DataFrame,
                                  metrics: List[str],
                                  label_col: str,
                                  output_path: str,
                                  enable_plots: bool = True,
                                  sample_frac: Optional[float] = 0.05,
                                  sample_size: Optional[int] = None) -> None:
    """
    Barplot showing percent difference between two classes for selected metrics.
    This function explicitly checks for binary labels and will skip otherwise.
    """
    if not enable_plots:
        return
    if df is None or df.empty:
        return
    if label_col not in df.columns:
        logger.warning("Label column '%s' not found for paired difference plot.", label_col)
        return

    metrics_filtered = [m for m in metrics if m in df.columns and np.issubdtype(df[m].dtype, np.number)]
    if not metrics_filtered:
        logger.info("No numeric metrics for paired difference plot; skipping.")
        return

    df_sample = _maybe_sample(df[[label_col] + metrics_filtered].dropna(), sample_frac, sample_size)
    groups = df_sample[label_col].unique()
    if len(groups) != 2:
        logger.info("Paired difference plot requires exactly 2 classes; found %d; skipping.", len(groups))
        return

    try:
        means = df_sample.groupby(label_col)[metrics_filtered].mean()
        baseline = means.iloc[0]
        comparison = means.iloc[1]
        diff_pct = ((comparison - baseline) / (baseline.replace({0: np.nan}))) * 100
        diff_df = diff_pct.reset_index().melt(id_vars=['index'], var_name='Metric', value_name='PercentDifference')
        # The above melting step will be simplified to a clean barplot
        diff_df = pd.DataFrame({'Metric': metrics_filtered, 'PercentDifference': (comparison - baseline) / (baseline.replace({0: np.nan})) * 100})
        diff_df = diff_df.sort_values('PercentDifference')
        plt.figure(figsize=(12, 8))
        sns.barplot(data=diff_df, x='PercentDifference', y='Metric')
        plt.title(f'Percentage Difference: {groups[1]} vs {groups[0]}')
        plt.axvline(0, color='k', linestyle='--')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.exception("Failed to create paired difference plot: %s", e)


def create_feature_importance_plot(importances: List[float], feature_names: List[str], output_path: str, enable_plots: bool = True) -> None:
    """Top feature importances (XGBoost/SHAP style)."""
    if not enable_plots:
        return
    try:
        df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df = df.sort_values('Importance', ascending=False).head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df, x='Importance', y='Feature')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.exception("Failed to create feature importance plot: %s", e)
