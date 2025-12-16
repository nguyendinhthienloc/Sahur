## Running the Pipeline (Terminal Instructions)

To run the main pipeline, statistical tests (1-vs-N), and IRAL lexical analysis on a dataset, use:

```
python -m src.cli run --input data/cleaned_by_topic/environment.csv --output results/environment_allcorpora_test --max-rows 10 --enable-embeddings --shards 2 --workers 2
```

- This will process the first 10 rows of environment.csv, extract features, run 1-vs-N statistical tests (Human vs all models), and run IRAL lexical analysis (frequencies, collocations, log-odds) for all corpora.
- Results will be saved in the specified output folder (e.g., `results/environment_allcorpora_test`).
- Figures, statistical tables, and IRAL outputs will be in subfolders.

**Requirements:** Activate your virtual environment and ensure all dependencies are installed before running.

```
.\.venv\Scripts\activate
```

Then run the pipeline command above.
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

# Human vs AI Text Classification: 6-Metric Linguistic Baseline

**Publication-ready linguistic feature extraction pipeline for distinguishing human and AI-generated text.**

This project provides a clean, maintainable implementation of six core linguistic metrics proven to differentiate human from AI writing, with optional IRAL-style lexical explainability features.

---

## ✨ Key Features

- **6 Core Metrics**: Scientifically-grounded linguistic features
- **IRAL Integration**: Optional lexical explainability (Log-Odds, collocations)
- **Publication-Ready Figures**: IRAL journal-style visualizations
- **Robust Statistics**: Welch t-test, Mann-Whitney U, Cohen's d, FDR corrections
- **Efficient Processing**: Disk caching, parallel processing, resumable runs
- **Clean Architecture**: Modular, single-responsibility design

---

## 🎯 Six Core Metrics

1. **MTLD** - Lexical diversity (Measure of Textual Lexical Diversity)
2. **Nominalization Density** - Derived nominalizations per 1000 words
3. **Modal/Epistemic Rate** - Modal verbs and epistemic markers per 100 words
4. **Clause Complexity** - Mean clausal dependencies per sentence
5. **Passive Voice Ratio** - Proportion of passive voice sentences
6. **S2S Cosine Similarity** - Mean sentence-to-sentence semantic similarity *(requires embeddings)*

---

## 📦 Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

---

## ⚡ Quick Start & Complete Workflow

### Step 1: Extract Features

```bash
python -m src.cli run \
  --input <PATH_TO_YOUR_INPUT_CSV> \
  --output <PATH_TO_OUTPUT_DIRECTORY> \
  --shards 4 \
  --workers 2
```

**With embeddings:**
```bash
python -m src.cli run \
  --input <PATH_TO_YOUR_INPUT_CSV> \
  --output <PATH_TO_OUTPUT_DIRECTORY> \
  --enable-embeddings \
  --shards 4 \
  --workers 2
```

### Step 2: Run Statistical Tests

```bash
python -m src.cli analyze \
  --input <PATH_TO_OUTPUT_DIRECTORY>/all_features.parquet \
  --output <PATH_TO_OUTPUT_DIRECTORY>
```

### Step 3: IRAL Lexical Analysis (Optional)

```bash
python -m src.cli analyze \
  --input <PATH_TO_OUTPUT_DIRECTORY>/all_features.parquet \
  --output <PATH_TO_OUTPUT_DIRECTORY> \
  --enable-iral-lexical
```

---

## 🔧 Command-Line Options

### `python -m src.cli run`

| Option                  | Description                        | Default      |
|-------------------------|------------------------------------|--------------|
| `--input, -i`           | Input CSV file path                | *required*   |
| `--output, -o`          | Output directory                   | *required*   |
| `--text-col`            | Text column name                   | auto-detect  |
| `--label-col`           | Label column name                  | auto-detect  |
| `--topic-col`           | Topic column name                  | auto-detect  |
| `--max-rows`            | Limit rows (for testing)           | all          |
| `--shards`              | Number of processing shards        | 1            |
| `--workers`             | spaCy parallel workers             | 1            |
| `--batch-size`          | Parsing batch size                 | 32           |
| `--enable-embeddings`   | Enable sentence embeddings         | off          |
| `--enable-iral-lexical` | Enable lexical analysis            | off          |
| `--cache-dir`           | Cache directory                    | `{output}/cache` |
| `--dry-run`             | Preview without executing          | off          |
| `--debug`               | Debug logging                      | off          |

### `python -m src.cli analyze`

```bash
python -m src.cli analyze \
  --input <PATH_TO_OUTPUT_DIRECTORY>/all_features.parquet \
  --output <PATH_TO_OUTPUT_DIRECTORY> \
  --enable-iral-lexical
```

---

## 📊 Output Structure

```
results/
├── all_features.parquet      # Main output (compressed)
├── all_features.csv          # Human-readable version
├── metrics_schema.json       # Metric definitions
├── cache/                    # Parse & embedding cache
│   └── shard_*/
├── tables/
│   └── statistical_tests.csv # Statistical results
├── figures/
│   ├── violin_plots.png
│   ├── radar_chart.png
│   └── keywords_*.png        # (if --enable-iral-lexical)
└── lexical/                  # (if --enable-iral-lexical)
    ├── log_odds.csv
    ├── collocations_group_*.csv
    └── top_freq_group_*.csv
```

---

## 🧪 Testing

```bash
# All tests
pytest tests/test_core_metrics.py -v

# Specific test class
pytest tests/test_core_metrics.py::TestMTLD -v

# Fast mode (skip slow tests)
pytest tests/ -v -m "not slow"
```

---

## 🏗️ Architecture

The pipeline consists of 10 modular Python files in `src/`:

1. **cli.py** - Command-line interface
2. **ingest.py** - Data loading and column detection
3. **parse_and_cache.py** - spaCy parsing with caching
4. **metrics_core.py** - Six core metric implementations
5. **embeddings.py** - Sentence embedding wrapper
6. **stats_analysis.py** - Statistical testing
7. **iral_lexical.py** - Optional lexical explainability
8. **visualize.py** - IRAL-style plotting
9. **pipeline.py** - Orchestration and shard processing
10. **utils.py** - Shared utilities

---

## 🐛 Troubleshooting

### "spaCy model not found"
```bash
python -m spacy download en_core_web_lg
```

### "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

---

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

---

## 📝 License

MIT License

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## 📞 Contact

For questions or issues, please open a GitHub issue.

**Built with ❤️ for linguistic analysis and human vs AI text classification research.**
pytest tests/test_core_metrics.py -v
