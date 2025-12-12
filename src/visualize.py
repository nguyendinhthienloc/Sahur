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
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=150)
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create violin plot
        sns.violinplot(data=df, x=group_col, y=metric, ax=ax,
                      palette=['#E8E8E8', '#A0A0A0'], inner='box')
        
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=10)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Grid
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved violin plot to {output_path}")
    
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
    
    # Number of metrics
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'), dpi=150)
    
    colors = ['#4472C4', '#ED7D31']
    
    for idx, (group_name, row) in enumerate(group_means.iterrows()):
        values = row.tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=str(group_name), color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=9)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Metric Profile Comparison', pad=20, fontsize=12, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved radar chart to {output_path}")
    
    plt.close()


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
        Optional lexical outputs from irral_lexical module
    """
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating visualization figures")
    
    # Violin plots
    create_violin_plot(
        df, metrics, group_col,
        output_path=figures_dir / 'violin_plots.png'
    )
    
    # Radar chart
    create_radar_chart(
        df, metrics, group_col,
        output_path=figures_dir / 'radar_chart.png'
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
