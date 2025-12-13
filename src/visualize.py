"""
Visualization module using IRAL plotting styles.

Provides high-quality publication-ready figures:
- Violin plots for metric distributions
- Radar charts for metric profiles
- Keyword/token bar plots
- Per-topic stability heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import logging

from .utils import setup_logger

logger = setup_logger(__name__)

# Set IRAL journal style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 1.0,
})


def create_violin_plot(df: pd.DataFrame,
                      metrics: List[str],
                      group_col: str = 'label',
                      output_path: Path = None) -> None:
    """
    Create violin plots for metrics by group.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics and group labels
    metrics : List[str]
        List of metric column names
    group_col : str
        Column for grouping
    output_path : Path, optional
        Where to save the plot
    """
    # For each metric, create a separate violin plot with all 7 groups
    import os
    palette = sns.color_palette("colorblind", 7)
    group_order = sorted(df[group_col].unique(), key=lambda x: (x != "Human_story", x))
    label_map = {"Human_story": "Human", "gemma-2-9b": "Gemma-2.9B", "mistral-7B": "Mistral-7B", "qwen-2-72B": "Qwen-2-72B", "llama-8B": "Llama-8B", "accounts/yi-01-ai/models/yi-large": "Yi-Large", "GPT_4-o": "GPT-4o"}
    for metric in metrics:
        plt.figure(figsize=(10, 6), dpi=150)
        ax = sns.violinplot(data=df, x=group_col, y=metric, order=group_order, palette=palette, inner="box")
        ax.set_title(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticklabels([label_map.get(lbl.get_text(), lbl.get_text()) for lbl in ax.get_xticklabels()], rotation=35, ha="right", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        if output_path:
            outdir = output_path.parent
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"violin_{metric}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved violin plot to {fname}")
        plt.close()


def create_radar_chart(df: pd.DataFrame,
                      metrics: List[str],
                      group_col: str = 'label',
                      output_path: Path = None) -> None:
    """
    Create a radar chart comparing metric profiles.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics
    metrics : List[str]
        Metrics to include
    group_col : str
        Grouping column
    output_path : Path, optional
        Output path
    """
    # Normalize metrics to 0-1 scale for radar chart
    import os
    df_norm = df.copy()
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val > min_val:
            df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
        else:
            df_norm[metric] = 0.5
    # Compute mean per group
    group_means = df_norm.groupby(group_col)[metrics].mean()
    group_order = sorted(group_means.index, key=lambda x: (x != "Human_story", x))
    label_map = {"Human_story": "Human", "gemma-2-9b": "Gemma-2.9B", "mistral-7B": "Mistral-7B", "qwen-2-72B": "Qwen-2-72B", "llama-8B": "Llama-8B", "accounts/yi-01-ai/models/yi-large": "Yi-Large", "GPT_4_o": "GPT-4o", "GPT_4-o": "GPT-4o"}
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    # Use 7 distinct color-blind-safe colors
    import seaborn as sns
    palette = sns.color_palette("colorblind", len(group_order))
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=150)
    for idx, group_name in enumerate(group_order):
        row = group_means.loc[group_name]
        values = row.tolist() + [row.tolist()[0]]
        label = label_map.get(group_name, group_name)
        ax.plot(angles, values, 'o-', linewidth=2.5, label=label, color=palette[idx], zorder=10)
        ax.fill(angles, values, color=palette[idx], alpha=0.18, zorder=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11, title="Model", title_fontsize=12)
    ax.set_title('Metric Profile Comparison', pad=20, fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved radar chart to {output_path}")
    plt.close()

def create_heatmap(df: pd.DataFrame, metrics: list, group_col: str, output_path: Path = None) -> None:
    """
    Create a heatmap (models x metrics) for mean metric values (normalized 0-1).
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    label_map = {"Human_story": "Human", "gemma-2-9b": "Gemma-2.9B", "mistral-7B": "Mistral-7B", "qwen-2-72B": "Qwen-2-72B", "llama-8B": "Llama-8B", "accounts/yi-01-ai/models/yi-large": "Yi-Large", "GPT_4_o": "GPT-4o", "GPT_4-o": "GPT-4o"}
    df_norm = df.copy()
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val > min_val:
            df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
        else:
            df_norm[metric] = 0.5
    group_means = df_norm.groupby(group_col)[metrics].mean()
    group_means.index = [label_map.get(idx, idx) for idx in group_means.index]
    plt.figure(figsize=(10, 5), dpi=150)
    sns.heatmap(group_means, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5, fmt=".2f", annot_kws={"fontsize":10})
    plt.title("Normalized Metric Values by Model", fontsize=13, fontweight='bold')
    plt.ylabel("Model")
    plt.xlabel("Metric")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved heatmap to {output_path}")
    plt.close()
    # Optionally: create one radar per metric (not typical, but for completeness)
    # for metric in metrics:
    #     ...


def create_keyword_barplot(keywords_df: pd.DataFrame,
                          output_path: Path,
                          title: str = "Top Keywords",
                          top_n: int = 30) -> None:
    """
    Create horizontal bar plot for keywords/tokens (IRAL style).
    
    Parameters
    ----------
    keywords_df : pd.DataFrame
        DataFrame with 'token' and 'log_odds' (or similar) columns
    output_path : Path
        Output path
    title : str
        Plot title
    top_n : int
        Number of top items to show
    """
    if len(keywords_df) == 0:
        logger.warning("No keywords to plot")
        return
    
    # Determine score column
    score_col = None
    for col in ['log_odds', 'pmi', 'score', 'frequency']:
        if col in keywords_df.columns:
            score_col = col
            break
    
    if score_col is None:
        logger.warning("Could not find score column in keywords dataframe")
        return
    
    # Take top N
    df_plot = keywords_df.head(top_n).copy()
    df_plot = df_plot.sort_values(score_col)
    
    fig, ax = plt.subplots(figsize=(7, max(8, top_n * 0.25)), dpi=150)
    
    # Horizontal bar plot
    y_pos = np.arange(len(df_plot))
    ax.barh(y_pos, df_plot[score_col], color='#A0A0A0', edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['token'], fontsize=9)
    ax.set_xlabel(score_col.replace('_', ' ').title(), fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Grid
    ax.grid(axis='x', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved keyword plot to {output_path}")
    
    plt.close()


def create_all_figures(df: pd.DataFrame,
                      metrics: List[str],
                      group_col: str,
                      output_dir: Path,
                      lexical_outputs: Optional[Dict] = None) -> None:
    """
    Generate all standard figures.
    
    Parameters
    ----------
    df : pd.DataFrame
        Features dataframe
    metrics : List[str]
        Metric column names
    group_col : str
        Grouping column
    output_dir : Path
        Output directory
    lexical_outputs : dict, optional
        Optional lexical outputs from iral_lexical module
    """
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating visualization figures")
    
    # Violin plots (one per metric)
    create_violin_plot(
        df, metrics, group_col,
        output_path=figures_dir / 'violin_dummy.png'  # dummy, real files are violin_{metric}.png
    )

    # Radar chart (all 7 overlays, 7 colors)
    create_radar_chart(
        df, metrics, group_col,
        output_path=figures_dir / 'radar_chart.png'
    )

    # Heatmap (models x metrics)
    create_heatmap(
        df, metrics, group_col,
        output_path=figures_dir / 'heatmap.png'
    )
    
    # Lexical plots if available
    if lexical_outputs:
        if 'log_odds' in lexical_outputs:
            log_odds_df = lexical_outputs['log_odds']
            if len(log_odds_df) > 0:
                create_keyword_barplot(
                    log_odds_df.sort_values('log_odds', ascending=False),
                    output_path=figures_dir / 'keywords_log_odds_positive.png',
                    title='Keywords with Positive Log-Odds (Group A)'
                )
                
                create_keyword_barplot(
                    log_odds_df.sort_values('log_odds', ascending=True),
                    output_path=figures_dir / 'keywords_log_odds_negative.png',
                    title='Keywords with Negative Log-Odds (Group B)'
                )
    
    logger.info(f"All figures saved to {figures_dir}")
