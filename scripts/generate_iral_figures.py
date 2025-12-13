"""Generate IRAL figures (3 figures) for each results/iral run and samples.

Usage:
  python scripts/generate_iral_figures.py

Saves figures to results/.../figures/ as PNG files.
"""
from pathlib import Path
import pandas as pd

from src.iral.iral_plots import create_three_iral_figures

ROOT = Path('results/iral')
if not ROOT.exists():
    print(f"No IRAL results directory found at {ROOT}")
    raise SystemExit(1)

runs = [p for p in ROOT.iterdir() if p.is_dir()]
if not runs:
    print("No runs found under results/iral/")
    raise SystemExit(1)

for run in runs:
    # include full run (run/) and any sample subfolders
    subfolders = [run] + sorted([d for d in run.iterdir() if d.is_dir()])
    for sub in subfolders:
        # Accept either lexical/ subfolder or CSVs directly under the run folder
        lexical_dir = sub / 'lexical'
        if lexical_dir.exists():
            log_odds_csv = lexical_dir / 'log_odds.csv'
        else:
            log_odds_csv = sub / 'log_odds.csv'
        if not log_odds_csv.exists():
            print(f"Skipping {sub}: missing log_odds.csv")
            continue
        df = pd.read_csv(log_odds_csv)
        # Expect columns: token, log_odds
        if 'token' not in df.columns or 'log_odds' not in df.columns:
            print(f"Unexpected columns in {log_odds_csv}: {df.columns.tolist()}")
            continue
        # Select top human-specific (negative log_odds) and AI-specific (positive)
        human_df = df[df['log_odds'] < 0].sort_values('log_odds').head(40)
        ai_df = df[df['log_odds'] > 0].sort_values('log_odds', ascending=False).head(40)
        keywords_group_0 = list(zip(human_df['token'].astype(str).tolist(), human_df['log_odds'].astype(float).tolist()))
        keywords_group_1 = list(zip(ai_df['token'].astype(str).tolist(), ai_df['log_odds'].astype(float).tolist()))

        outdir = sub / 'figures'
        print(f"Creating figures for {sub} -> {outdir}")
        create_three_iral_figures(keywords_group_0, keywords_group_1, str(outdir))

print("Done.")
