Moved to docs/metrics_reference.md

Full content copied to `docs/metrics_reference.md` and top-level file replaced with this pointer to keep repository tidy.
df = pd.read_parquet(output_dir / 'all_features.parquet')

# Generate all figures
create_all_figures(
    df=df,
    metrics=CORE_METRIC_NAMES,
    output_dir=output_dir / 'figures',
    group_col='label',
    topic_col='topic'  # Optional: set to None if no topic column
)

print(f"✓ Figures saved to {output_dir / 'figures'}")
```

**Example:**
```python
from pathlib import Path
import pandas as pd
from src.visualize import create_all_figures
from src.metrics_core import CORE_METRIC_NAMES

output_dir = Path('results/full_analysis')
df = pd.re<PATH_TO_OUTPUT_DIRECTORY>/all_features.parquet \
  --output <PATH_TO_OUTPUT_DIRECTORY> \
  --enable-iral-lexical
```

**Example:**
```bash
python -m src.cli analyze \
  --input ad_parquet(output_dir / 'all_features.parquet')

create_all_figures(
    df=df,
    metrics=CORE_METRIC_NAMES,
    output_dir=output_dir / 'figures',
    group_col='label',
    topic_col='topic'
)

print(f"✓ Figures saved to {output_dir / 'figures'}")
```

**Outputs:**
- `figures/violin_plots.png` - Metric distributions by group
- `figures/radar_chart.png` - Metric profile comparison
- `figures/correlation_heatmap.png` - Metric correlations
- `figures/pca_biplot.png` - PCA visualization (if enough metrics)

---

### Step 4: IRAL Lexical Analysis (Optional)

```bash
python -m src.cli analyze \
  --input results/full_analysis/all_features.parquet \
  --output results/full_analysis \
  --enable-iral-lexical
```

**Outputs:**
- `lexical/log_odds.csv` - Keywords ranked by log-odds ratio
- `lexical/collocations_human.csv` - Bigram collocations for human text
- `lexical/collocations_ai.csv` - Bigram collocations for AI text
- `lexical/top_freq_human.csv` - Most frequent tokens (human)
- `lexical/top_freq_ai.csv` - Most frequent tokens (AI)
- `figures/keywords_top20.png` - Top keywords visualization

---

## Output Files Reference

### Core Outputs

```
results/
├── all_features.parquet      # Main feature file (compressed)
├── all_features.csv           # Human-readable version
├── metrics_schema.json        # Metric definitions and formulas
└── cache/                     # Cached parse data (reusable)
    ├── shard_0/
    │   └── parsed_docs.pkl
    ├── shard_1/
    └── shard_0_embeds/        # Embedding cache
        └── embeddings.npy
```

### Statistical Analysis Outputs

```
results/
└── tables/
    └── statistical_tests.csv  # Complete statistical test results
```

**Columns in statistical_tests.csv:**
- `metric` - Metric name
- `mean_human`, `std_human`, `n_human` - Human group statistics
- `mean_ai`, `std_ai`, `n_ai` - AI group statistics
- `welch_t`, `welch_p` - Welch's t-test results
- `mannwhitney_u`, `mannwhitney_p` - Mann-Whitney U test results
- `cohens_d` - Effect size (pooled std)
- `levene_stat`, `levene_p` - Variance homogeneity test
- `holm_reject`, `fdr_reject` - Multiple testing corrections

### Visualization Outputs

```
results/
└── figures/
    ├── violin_plots.png       # Metric distributions (IRAL style)
    ├── radar_chart.png        # Metric profiles
    ├── correlation_heatmap.png
    └── pca_biplot.png
```

### IRAL Lexical Outputs (Optional)

```
results/
├── lexical/
│   ├── log_odds.csv           # Keywords by log-odds ratio
│   ├── collocations_human.csv
│   ├── collocations_ai.csv
│   ├── top_freq_human.csv
│   └── top_freq_ai.csv
└── figures/
    └── keywords_top20.png     # Keyword bar plot
```

---

## Statistical Tests Explained

### Welch's t-test
**What it tests:** Whether the means of two groups differ significantly  
**When to use:** Primary test for comparing human vs AI metrics  
**Assumptions:** Normal distribution, unequal variances OK  
**Interpretation:** p < 0.05 = significant difference

### Mann-Whitney U Test
**What it tests:** Whether two groups have different distributions  
**When to use:** Non-parametric alternative when normality fails  
**Assumptions:** None (distribution-free)  
**Interpretation:** p < 0.05 = significant difference

### Cohen's d
**What it measures:** Effect size (magnitude of difference)  
**Formula:** `d = (mean_1 - mean_2) / pooled_std`  
**Interpretation:**
- |d| < 0.2 = small effect
- 0.2 ≤ |d| < 0.5 = small to medium
- 0.5 ≤ |d| < 0.8 = medium to large
- |d| ≥ 0.8 = large effect

### Levene's Test
**What it tests:** Whether variances are equal between groups  
**Interpretation:** p < 0.05 = unequal variances (use Welch's t-test)

### Multiple Testing Corrections
**Holm-Bonferroni:** Conservative correction for family-wise error rate  
**Benjamini-Hochberg (FDR):** Less conservative, controls false discovery rate  
**Why needed:** Testing multiple metrics increases Type I error risk

---

## Troubleshooting

### "No statistical tests found"
**Cause:** You ran `python -m src.cli run` but didn't run `analyze`  
**Solution:** Run `python -m src.cli analyze --input results/*/all_features.parquet --output results/*`

### "No figures directory found"
**Cause:** Visualization not run automatically  
**Solution:** Use the Python code in Step 3 above to generate plots

### "s2s_cosine_similarity is all NaN"
**Cause:** Embeddings not enabled  
**Solution:** Re-run with `--enable-embeddings` flag

### "Parsing is very slow"
**Cause:** Low batch size or single worker  
**Solution:** Increase `--batch-size 64 --workers 4`

### "Out of memory"
**Cause:** Too many embeddings in memory  
**Solution:** Increase `--shards` to process in smaller chunks

### "Cache not being used"
**Cause:** Cache directory changed or cleared  
**Solution:** Use same `--cache-dir` across runs

---

## Example: Complete Analysis Workflow

```bash
# 1. Extract features with embeddings
python -m src.cli run \
  --input data/gsingh1-train/train.csv \
  --output results/complete_analysis \
  --enable-embeddings \
  --enable-iral-lexical \
  --shards 8 \
  --workers 2 \
  --batch-size 64

# 2. Run statistical tests
python -m src.cli analyze \
  --input results/complete_analysis/all_features.parquet \
  --output results/complete_analysis

# 3. Generate plots (in Python)
python << EOF
from pathlib import Path
import pandas as pd
from src.visualize import create_all_figures
from src.metrics_core import CORE_METRIC_NAMES

output_dir = Path('results/complete_analysis')
df = pd.read_parquet(output_dir / 'all_features.parquet')

create_all_figures(
    df=df,
    metrics=CORE_METRIC_NAMES,
    output_dir=output_dir / 'figures',
    group_col='label'
)
print("✓ All figures generated")
EOF

# 4. View results
ls results/complete_analysis/tables/statistical_tests.csv
ls results/complete_analysis/figures/
ls results/complete_analysis/lexical/
```

**Expected outputs:**
- ✓ `all_features.parquet` (feature matrix)
- ✓ `tables/statistical_tests.csv` (statistical analysis)
- ✓ `figures/*.png` (IRAL-style visualizations)
- ✓ `lexical/*.csv` (keyword/collocation analysis)

---

**For more details, see [README.md](README.md) and [REFACTOR_NOTES.md](REFACTOR_NOTES.md).**
