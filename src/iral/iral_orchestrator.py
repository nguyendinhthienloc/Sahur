"""
IRAL Orchestrator: Multi-corpus IRAL lexical analysis for 1-vs-N design.
"""
import pandas as pd
from pathlib import Path
from .iral_lexical import compute_group_log_odds, extract_bigram_collocations, compute_top_k_frequencies
import logging

def run_iral_pipeline(df: pd.DataFrame, group_col: str = 'label', text_col: str = 'text', output_dir: Path = None, reference_group: str = 'Human_story', min_count: int = 5, top_k: int = 100):
    # --- Presentation Layer: Aggregate per-corpus frequency and collocation tables ---
    # This does NOT change any computation or remove any outputs.
    # It creates two new summary CSVs for human readability and publication.
    if output_dir:
        import glob
        import pandas as pd
        # Aggregate frequency tables
        freq_files = glob.glob(str(output_dir / '*_freq.csv'))
        freq_tables = []
        for fpath in freq_files:
            df_freq = pd.read_csv(fpath)
            # Extract corpus name from filename
            corpus = Path(fpath).stem.replace('_freq','')
            df_freq.insert(0, 'corpus', corpus)
            freq_tables.append(df_freq)
        if freq_tables:
            freq_summary = pd.concat(freq_tables, ignore_index=True, sort=False)
            freq_summary.to_csv(output_dir / 'lexical_frequency_summary.csv', index=False)

        # Aggregate collocation tables
        colloc_files = glob.glob(str(output_dir / '*_colloc.csv'))
        colloc_tables = []
        for fpath in colloc_files:
            df_colloc = pd.read_csv(fpath)
            corpus = Path(fpath).stem.replace('_colloc','')
            df_colloc.insert(0, 'corpus', corpus)
            colloc_tables.append(df_colloc)
        if colloc_tables:
            colloc_summary = pd.concat(colloc_tables, ignore_index=True, sort=False)
            colloc_summary.to_csv(output_dir / 'lexical_collocation_summary.csv', index=False)
    logger = logging.getLogger("iral_orchestrator")
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    import re
    def sanitize(name):
        return re.sub(r'[^A-Za-z0-9._-]', '_', str(name))

    # Prepare summary data
    freq_rows = []
    colloc_rows = []
    logodds_rows = []
    all_groups = list(df[group_col].unique())
    ref_texts = df[df[group_col] == reference_group][text_col].tolist() if reference_group in all_groups else []

    # Create top-level figures directory
    figures_dir = output_dir / "figures" if output_dir else None
    if figures_dir:
        figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate pipeline flowchart ONCE
    try:
        from . import iral_plots
        if figures_dir:
            iral_plots.create_flowchart(str(figures_dir / "iral_pipeline_flowchart.png"))
    except Exception as e:
        logger.warning(f"Could not generate pipeline flowchart: {e}")

    # For each corpus, generate metrics, plots, and word cloud
    for group in all_groups:
        texts = df[df[group_col] == group][text_col].tolist()
        safe_group = sanitize(group)
        group_dir = figures_dir / safe_group if figures_dir else None
        if group_dir:
            group_dir.mkdir(parents=True, exist_ok=True)
        # Frequency (append to in-memory summary only)
        freq = compute_top_k_frequencies(texts, k=top_k)
        for _, row in freq.iterrows():
            freq_rows.append({"corpus": group, **row.to_dict()})
        # Collocations (append to in-memory summary only)
        colloc = extract_bigram_collocations(texts, min_count=min_count, top_k=top_k//2)
        for _, row in colloc.iterrows():
            colloc_rows.append({"corpus": group, **row.to_dict()})
        # Log-odds (if reference group exists and is not this group)
        if ref_texts and group != reference_group:
            logodds = compute_group_log_odds(ref_texts, texts, min_count=min_count, top_k=top_k)
            if output_dir:
                logodds.to_csv(output_dir / f"log_odds_{reference_group}_vs_{safe_group}.csv", index=False)
            for _, row in logodds.iterrows():
                logodds_rows.append({"reference_group": reference_group, "comparison_group": group, **row.to_dict()})
        # --- Plotting ---
        try:
            # Frequency plot (bar chart)
            if not freq.empty and group_dir:
                freq_top = freq.head(20)
                plt_path = str(group_dir / "frequency_bar.png")
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8,4), dpi=300)
                plt.bar(freq_top['token'], freq_top['count'], color='skyblue')
                plt.xticks(rotation=60, ha='right', fontsize=8)
                plt.title(f"Top 20 Tokens: {group}", fontsize=12)
                plt.tight_layout()
                plt.savefig(plt_path, bbox_inches='tight')
                plt.close()
            # Collocation plot (bar chart)
            if not colloc.empty and group_dir:
                colloc_top = colloc.head(20)
                plt_path = str(group_dir / "collocation_bar.png")
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8,4), dpi=300)
                plt.bar(colloc_top['bigram'], colloc_top['pmi'], color='orange')
                plt.xticks(rotation=60, ha='right', fontsize=8)
                plt.title(f"Top 20 Bigrams (PMI): {group}", fontsize=12)
                plt.tight_layout()
                plt.savefig(plt_path, bbox_inches='tight')
                plt.close()
            # Word cloud
            if not freq.empty and group_dir:
                from wordcloud import WordCloud
                word_freq = {row['token']: row['count'] for _, row in freq.iterrows()}
                wc = WordCloud(width=800, height=400, background_color='white', colormap='tab20', prefer_horizontal=1.0)
                wc.generate_from_frequencies(word_freq)
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 4), dpi=300)
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title(f"Word Cloud: {group}", fontsize=12, fontweight='bold', loc='left')
                wc_path = str(group_dir / "wordcloud.png")
                plt.savefig(wc_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
        except Exception as e:
            logger.warning(f"Could not generate figures for {group}: {e}")

    # Write exactly two summary CSVs for descriptive metrics
    if output_dir:
        import pandas as pd
        freq_df = pd.DataFrame(freq_rows)
        if not freq_df.empty:
            freq_df.to_csv(output_dir / "lexical_frequency_summary.csv", index=False)
        colloc_df = pd.DataFrame(colloc_rows)
        if not colloc_df.empty:
            colloc_df.to_csv(output_dir / "lexical_collocation_summary.csv", index=False)
    logger.info(f"IRAL pipeline complete. Outputs in {output_dir}")
