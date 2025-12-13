"""
IRAL-specific visualization module for creating publication figures.
Generates 3 figures matching Zhang (2024) IRAL article style.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


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


def create_flowchart(outpath):
    """
    Create Figure 1: Flowchart showing the analysis pipeline.
    
    Parameters
    ----------
    outpath : str
        Output file path for the flowchart
    """
    fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    color_input = '#E8E8E8'
    color_process = '#D0D0D0'
    color_analysis = '#B8B8B8'
    color_output = '#A0A0A0'
    
    # Box dimensions
    box_width = 6
    box_height = 0.8
    x_center = 5
    
    # Define process steps with positions and colors
    steps = [
        (11.5, "Raw Data\n(CSV or Text Files)", color_input),
        (10.3, "↓", None),
        (9.8, "Data Ingestion\n(ingest.py)", color_process),
        (9.0, "↓", None),
        (8.5, "Text Cleaning\n(clean.py)\nRemove citations, references", color_process),
        (7.5, "↓", None),
        (7.0, "Tokenization & POS Tagging\n(pos_tools.py)\nspaCy / NLTK", color_process),
        (6.0, "↓", None),
        (5.5, "Feature Computation\n(features.py)\nLexical metrics, POS ratios", color_analysis),
        (4.5, "↓", None),
        (4.0, "Nominalization Detection\n(nominalization.py)\nLemma-based + Suffix-based", color_analysis),
        (3.0, "↓", None),
        (2.5, "Collocation & Keyword Extraction\n(collocations.py)\nPMI, Log-odds ratio", color_analysis),
        (1.5, "↓", None),
        (1.0, "Statistical Analysis\n(stats_analysis.py)\nt-test, Mann-Whitney, Cohen's d", color_analysis),
        (0.0, "↓", None),
        (-0.5, "Visualization & Export\n(plots.py)\nFigures, tables, CSV", color_output),
    ]
    
    # Draw boxes and arrows
    for y_pos, text, color in steps:
        if text == "↓":
            # Draw arrow
            ax.annotate('', xy=(x_center, y_pos + 0.2), xytext=(x_center, y_pos + 0.5),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        else:
            # Draw box
            box = FancyBboxPatch(
                (x_center - box_width/2, y_pos - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.1",
                linewidth=1.5,
                edgecolor='black',
                facecolor=color
            )
            ax.add_patch(box)
            
            # Add text
            ax.text(x_center, y_pos, text,
                   ha='center', va='center',
                   fontsize=9, fontweight='normal',
                   multialignment='center')
    
    # Add title
    ax.text(x_center, 12.3, 'Figure 1. IRAL Text Analysis Pipeline',
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Save figure
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Created Figure 1: {outpath}")


def create_keyword_dotplot(keywords, outpath, title, corpus_name, top_n=40):
    """
    Create Figure 2/3: Horizontal dot plot matching IRAL article style.
    Simple dots without stems, clean gridlines, matching exact IRAL formatting.
    
    Parameters
    ----------
    keywords : list of tuple
        List of (word, log_odds_score) tuples
    outpath : str
        Output file path
    title : str
        Figure title (e.g., "Figure 2: Keywords in human-produced texts.")
    corpus_name : str
        Name of corpus (e.g., "Human", "GPT")
    top_n : int, default=40
        Number of top keywords to display
    """
    if not keywords:
        print(f"⚠ No keywords to plot for {outpath}")
        return
    
    # Take top N keywords and sort by absolute value for display
    keywords = keywords[:top_n]
    words, scores = zip(*keywords)
    
    # Sort by score (lowest to highest for human, highest to lowest for AI)
    sorted_pairs = sorted(zip(words, scores), key=lambda x: x[1])
    words, scores = zip(*sorted_pairs)
    
    fig, ax = plt.subplots(figsize=(7, 10), dpi=300)
    
    # Create horizontal dot plot - SIMPLE DOTS ONLY (no stems)
    y_pos = np.arange(len(words))
    
    # Plot dots - simple black filled circles
    ax.scatter(scores, y_pos, s=60, c='black', edgecolors='black', 
              linewidths=0.5, zorder=3, alpha=0.9)
    
    # Customize appearance - match IRAL style exactly
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=9)
    ax.set_xlabel('Association strength (OddsRatio)', fontsize=10, fontweight='normal')
    ax.set_ylabel('Token', fontsize=10, fontweight='normal')
    
    # Title styling - match IRAL
    ax.set_title(title, pad=10, fontsize=10, fontweight='bold', loc='left')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Add subtle grid - vertical lines only like IRAL
    ax.grid(axis='x', alpha=0.4, linestyle='-', linewidth=0.5, color='#CCCCCC')
    ax.grid(axis='y', alpha=0)
    ax.set_axisbelow(True)
    
    # Adjust tick parameters to match IRAL
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8, length=4)
    
    # Set reasonable x-axis limits
    abs_max = max(abs(min(scores)), abs(max(scores)))
    margin = abs_max * 0.1
    ax.set_xlim(min(scores) - margin, max(scores) + margin)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Created {title}: {outpath}")


def create_three_iral_figures(keywords_group_0, keywords_group_1, outdir, 
                               label_names=None):
    """
    Create the 3 main IRAL publication figures.
    
    Parameters
    ----------
    keywords_group_0 : list of tuple
        Keywords for group 0 (human)
    keywords_group_1 : list of tuple
        Keywords for group 1 (AI)
    outdir : str
        Output directory for figures
    label_names : dict, optional
        Mapping of labels to display names
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Default names
    if label_names is None:
        label_names = {0: "Human", 1: "AI"}
    
    group_0_name = label_names.get(0, "Group 0")
    group_1_name = label_names.get(1, "Group 1")
    
    # Figure 1: Flowchart
    flowchart_path = os.path.join(outdir, "figure_1_flowchart.png")
    create_flowchart(flowchart_path)
    
    # Figure 2: Human corpus keywords (negative log-odds = human-specific)
    keywords_0_path = os.path.join(outdir, "figure_2_keywords_human.png")
    create_keyword_dotplot(
        keywords_group_0,
        keywords_0_path,
        f"Figure 2:  Keywords in {group_0_name.lower()}-produced texts.",
        group_0_name,
        top_n=40
    )
    
    # Figure 3: AI corpus keywords (positive log-odds = AI-specific)
    keywords_1_path = os.path.join(outdir, "figure_3_keywords_ai.png")
    create_keyword_dotplot(
        keywords_group_1,
        keywords_1_path,
        f"Figure 3:  Keywords in {group_1_name}-produced texts.",
        group_1_name,
        top_n=40
    )
    
    print("\n" + "="*60)
    print("✓ All 3 IRAL figures created successfully!")
    print("="*60)
    print(f"\nFigures saved to: {outdir}")
    print("  - figure_1_flowchart.png")
    print("  - figure_2_keywords_human.png")
    print("  - figure_3_keywords_ai.png")


def cleanup_old_figures(outdir):
    """Remove old individual metric plots, keep only the 3 main figures."""
    import glob
    
    # Patterns for old figures to remove
    old_patterns = [
        '*_barplot.png',
        '*_boxplot.png'
    ]
    
    removed = 0
    for pattern in old_patterns:
        files = glob.glob(os.path.join(outdir, pattern))
        for file in files:
            if not file.endswith(('figure_1_flowchart.png', 
                                 'figure_2_keywords_human.png',
                                 'figure_3_keywords_ai.png')):
                try:
                    os.remove(file)
                    removed += 1
                except:
                    pass
    
    if removed > 0:
        print(f"\n✓ Cleaned up {removed} old figure files")
