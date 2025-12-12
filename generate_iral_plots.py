"""
Generate IRAL-style plots for an existing feature analysis.

Usage:
    python generate_iral_plots.py <path_to_output_directory>

Example:
    python generate_iral_plots.py results/gpt4_100pairs_analysis
"""

import sys
from pathlib import Path
import pandas as pd
from src.visualize import create_all_figures
from src.metrics_core import CORE_METRIC_NAMES

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_iral_plots.py <path_to_output_directory>")
        print("Example: python generate_iral_plots.py results/gpt4_100pairs_analysis")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"Error: Directory does not exist: {output_dir}")
        sys.exit(1)
    
    features_file = output_dir / 'all_features.parquet'
    if not features_file.exists():
        # Try CSV version
        features_file = output_dir / 'all_features.csv'
        if not features_file.exists():
            print(f"Error: No features file found in {output_dir}")
            print("Expected: all_features.parquet or all_features.csv")
            sys.exit(1)
    
    print(f"Loading features from {features_file}...")
    if features_file.suffix == '.parquet':
        df = pd.read_parquet(features_file)
    else:
        df = pd.read_csv(features_file)
    
    print(f"Loaded {len(df)} documents with {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for topic column
    topic_col = 'topic' if 'topic' in df.columns else None
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nGenerating IRAL-style plots...")
    print(f"Output directory: {figures_dir}")
    
    create_all_figures(
        df=df,
        metrics=CORE_METRIC_NAMES,
        output_dir=figures_dir,
        group_col='label',
        topic_col=topic_col
    )
    
    print(f"\n✓ All figures generated successfully!")
    print(f"✓ Saved to: {figures_dir}")
    print(f"\nGenerated files:")
    for file in sorted(figures_dir.glob('*.png')):
        print(f"  - {file.name}")

if __name__ == '__main__':
    main()
