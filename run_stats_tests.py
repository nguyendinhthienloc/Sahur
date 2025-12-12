"""
Manually run statistical tests on gpt4_100pairs_analysis
"""
import pandas as pd
from pathlib import Path
from src.stats_analysis import compare_groups, export_statistical_tests
from src.metrics_core import CORE_METRIC_NAMES

# Load data
output_dir = Path('results/gpt4_100pairs_analysis')
df = pd.read_csv(output_dir / 'all_features.csv')

print(f"Loaded {len(df)} documents")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

# Get unique label values
labels = df['label'].unique()
print(f"Unique labels: {labels}\n")

# Run statistical tests with correct label values
stats_results = compare_groups(
    df=df,
    metrics=CORE_METRIC_NAMES,
    group_col='label',
    group_a_value='human',  # Explicitly specify human
    group_b_value='ai'       # Explicitly specify AI
)

print(f"Statistical tests computed for {len(stats_results)} metrics\n")
print("Results preview:")
print(stats_results[['metric', 'group_a_mean', 'group_b_mean', 'mean_diff', 'cohens_d', 't_pvalue']].to_string())

# Export results
stats_dir = output_dir / 'tables'
export_statistical_tests(stats_results, stats_dir / 'statistical_tests.csv')

print(f"\n✓ Statistical tests saved to {stats_dir / 'statistical_tests.csv'}")
