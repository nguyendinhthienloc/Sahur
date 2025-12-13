# Human vs AI Text Classification: 6-Metric Linguistic Baseline

**Publication-ready linguistic feature extraction pipeline for distinguishing human and AI-generated text.**

This project provides a clean, maintainable implementation of six core linguistic metrics proven to differentiate human from AI writing, with optional IRAL-style lexical explainability features.

## ✨ Key Features

- **6 Core Metrics**: Focused, scientifically-grounded linguistic features
- **IRAL Integration**: Optional lexical explainability (Log-Odds, collocations)
- **Publication-Ready Figures**: IRAL journal-style visualizations
- **Robust Statistics**: Welch t-test, Mann-Whitney U, Cohen's d, FDR corrections
- **Efficient Processing**: Disk caching, parallel processing, resumable runs
- **Clean Architecture**: 10 modular files with single-responsibility design

---

## 🎯 Six Core Metrics (Default Mode)

1. **MTLD** - Lexical diversity (Measure of Textual Lexical Diversity)
2. **Nominalization Density** - Derived nominalizations per 1000 words
3. **Modal/Epistemic Rate** - Modal verbs and epistemic markers per 100 words
4. **Clause Complexity** - Mean clausal dependencies per sentence
5. **Passive Voice Ratio** - Proportion of passive voice sentences
6. **S2S Cosine Similarity** - Mean sentence-to-sentence semantic similarity *(requires embeddings)*

**Why these 6?** Each metric captures a distinct linguistic dimension with proven discriminative power, low redundancy, and computational efficiency.


## 📦 Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

### Core Dependencies

### 🌐 Cloud Environments (Colab/Kaggle)

For running on Google Colab or Kaggle notebooks:

```python
# Automated setup in one command
!git clone https://github.com/YOUR_USERNAME/AIvsHuman.git
%cd AIvsHuman
!python setup_environment.py
```

📖 **See [CLOUD_SETUP.md](CLOUD_SETUP.md) for complete cloud setup guide**


## ⚡ Quick Start & Complete Workflow

### Step 1: Extract Features

```bash
# Basic run: 6 core metrics without embeddings
python -m src.cli run \
  --input <PATH_TO_YOUR_INPUT_CSV> \
  --output <PATH_TO_OUTPUT_DIRECTORY> \
  --shards 4 \
  --workers 2
```

**Example:**
```bash
python -m src.cli run \
  --input data/gsingh1-train/train.csv \
  --output results/baseline_run \
  --shards 4 \
  --workers 2
```

**With embeddings (enables s2s_cosine_similarity):**
```bash
python -m src.cli run \
  --input data/gsingh1-train/train.csv \
  --output results/with_embeddings \
  --enable-embeddings \
  --shards 4 \
  --workers 2
```

**Outputs:**


### Step 2: Run Statistical Tests

⚠️ **Statistical tests are NOT automatic** - you must run them separately:

```bash
python -m src.cli analyze \
  --input <PATH_TO_OUTPUT_DIRECTORY>/all_features.parquet \
  --output <PATH_TO_OUTPUT_DIRECTORY>
```

**Example:**
```bash
python -m src.cli analyze \
  --input results/baseline_run/all_features.parquet \
  --output results/baseline_run
```

**Outputs:**


### Step 3: (Optional) Plots

Plots are no longer produced by the repository helper script. Use the CSV outputs in the `lexical/` folder to create figures locally with your preferred plotting tools.

Example: load `results/.../lexical/log_odds.csv` or `collocations_group_*.csv` into a notebook and generate visuals as needed.


### Step 4: IRAL Lexical Analysis (Optional)

For keyword extraction and log-odds analysis:

```bash
python -m src.cli analyze \
  --input <PATH_TO_OUTPUT_DIRECTORY>/all_features.parquet \
  --output <PATH_TO_OUTPUT_DIRECTORY> \
  --enable-iral-lexical
```

**Example:**
```bash
python -m src.cli analyze \
  --input results/baseline_run/all_features.parquet \
  --output results/baseline_run \
  --enable-iral-lexical
```

**Outputs:**


### Complete Example Workflow

```bash
# 1. Extract features with embeddings
python -m src.cli run \
  --input data/gsingh1-train/train.csv \
  --output results/full_analysis \
  --enable-embeddings \
  --shards 8 \
  --workers 2

# 2. Run statistical tests + IRAL lexical analysis
python -m src.cli analyze \
  --input results/full_analysis/all_features.parquet \
  --output results/full_analysis \
  --enable-iral-lexical

# 3. Generate plots (Manual)
# Use CSVs under `results/full_analysis/lexical/` to build figures locally.

# 4. View results
ls results/full_analysis/tables/
ls results/full_analysis/figures/
ls results/full_analysis/lexical/
```


### Quick Test Run

For testing with small sample:

```bash
python -m src.cli run \
  --input data/gsingh1-train/train.csv \
  --output results/test \
  --max-rows 100 \
  --shards 1
```


## 🔧 Command-Line Options

### `python -m src.cli run` (Main Pipeline)

| Option | Description | Default |
|--------|-------------|---------|
| `--input, -i` | Input CSV file path | *required* |
| `--output, -o` | Output directory | *required* |
| `--text-col` | Text column name | auto-detect |
| `--label-col` | Label column name | auto-detect |
| `--topic-col` | Topic column name | auto-detect |
| `--max-rows` | Limit rows (for testing) | all |
| `--shards` | Number of processing shards | 1 |
| `--workers` | spaCy parallel workers | 1 |
| `--batch-size` | Parsing batch size | 32 |
| `--enable-embeddings` | Enable sentence embeddings | off |
| `--enable-iral-lexical` | Enable lexical analysis | off |
| `--cache-dir` | Cache directory | `{output}/cache` |
| `--dry-run` | Preview without executing | off |
| `--debug` | Debug logging | off |

### `python -m src.cli analyze` (Re-analyze Features)

Analyze pre-computed features without re-parsing:

```bash
python -m src.cli analyze \
  --input results/baseline_run/all_features.parquet \
  --output results/reanalysis \
  --enable-iral-lexical
```


## 📊 Output Structure

```
results/
├── all_features.parquet      # Main output (compressed)
├── all_features.csv           # Human-readable version
├── metrics_schema.json        # Metric definitions
├── cache/                     # Parse & embedding cache
│   └── shard_*/
├── tables/
│   └── statistical_tests.csv # Welch, Mann-Whitney, Cohen's d, p-values
├── figures/
│   ├── violin_plots.png       # Metric distributions
│   ├── radar_chart.png        # Metric profiles
│   └── keywords_*.png         # (if --enable-iral-lexical)
└── lexical/                   # (if --enable-iral-lexical)
    ├── log_odds.csv
    ├── collocations_group_*.csv
    └── top_freq_group_*.csv
```


## 🧪 Testing

### Run Unit Tests

```bash
# All tests
pytest tests/test_core_metrics.py -v

# Specific test class
pytest tests/test_core_metrics.py::TestMTLD -v

# Fast mode (skip slow tests)
pytest tests/ -v -m "not slow"
```

### Integration Test

```bash
python -m src.cli run \
  --input data/gsingh1-train/train.csv \
  --output results/integration_test \
  --max-rows 50 \
  --shards 1

# Verify outputs
ls results/integration_test/all_features.parquet
ls results/integration_test/tables/statistical_tests.csv
ls results/integration_test/figures/
```


## 📚 Documentation



## 🏗️ Architecture

The pipeline consists of 10 modular Python files in `src/`:

1. **`cli.py`** - Command-line interface
2. **`ingest.py`** - Data loading and column detection
3. **`parse_and_cache.py`** - spaCy parsing with caching
4. **`metrics_core.py`** - Six core metric implementations
5. **`embeddings.py`** - Sentence embedding wrapper
6. **`stats_analysis.py`** - Statistical testing
7. **`iral_lexical.py`** - Optional lexical explainability
8. **`visualize.py`** - IRAL-style plotting
9. **`pipeline.py`** - Orchestration and shard processing
10. **`utils.py`** - Shared utilities

**Design principles:**


## 🔬 Statistical Analysis

The pipeline automatically computes:


All results exported to `tables/statistical_tests.csv`.


## 🎨 Visualization

IRAL journal-style figures (serif fonts, grayscale, 300 DPI):


Figures saved to `{output}/figures/` in PNG format.


## ⚙️ Configuration

Edit `config/metrics_config.yaml` to customize:



## 🚀 Performance Tips

### For Large Datasets (>10K documents)
```bash
```

### For Small Datasets (<1K documents)
```bash
```

### GPU Acceleration

### Caching


## 📖 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{6metric_baseline_2025,
  title={6-Metric Linguistic Baseline for Human vs AI Text Classification},
  author={Dinh Thien Loc Nguyen},
  year={2025},
  url={https://github.com/nguyendinhthienloc/AIvsHuman}
}
```

For IRAL lexical features, also cite the original IRAL paper.


## 🔄 Migration from 15-Metric Pipeline

The old 15-metric implementation is archived in `archive/old_15_metric_implementation/`. See [REFACTOR_NOTES.md](REFACTOR_NOTES.md) for migration guide.

**Key changes:**


## 🐛 Troubleshooting

### "spaCy model not found"
```bash
python -m spacy download en_core_web_lg
```

### "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

### "No text column found"
Specify column explicitly:
```bash
```

### Slow parsing
Increase batch size and workers:
```bash
```

### Out of memory
Reduce batch size or use more shards:
```bash
```


## 📝 License

MIT License


## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request


## 📞 Contact

For questions or issues, please open a GitHub issue.


**Built with ❤️ for linguistic analysis and human vs AI text classification research.**
