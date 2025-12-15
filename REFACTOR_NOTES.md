
# Refactoring Notes: 6-Metric Baseline Pipeline

## Overview

Successfully refactored the research codebase from a 15-metric exploration pipeline into a clean, maintainable 6-metric linguistic baseline with optional IRAL lexical explainability.

**Branch:** `refactor/6-metric-baseline`  
**Date:** December 12, 2025  
**Version:** 1.0.0

---

## Architecture Changes

### Core Module Structure (10 files in src/)

The new implementation follows a clean single-responsibility design:

1. **`cli.py`** - Command-line interface with `run` and `analyze` commands
2. **`ingest.py`** - Data loading, column detection, minimal cleaning, shard creation
3. **`parse_and_cache.py`** - spaCy pipeline management with disk caching
4. **`metrics_core.py`** - Six core metric implementations
5. **`embeddings.py`** - Sentence embedding wrapper for s2s_cosine
6. **`stats_analysis.py`** - Statistical tests (Welch, Mann-Whitney, Cohen's d, FDR corrections)
7. **`iral_lexical.py`** - Optional lexical explainability (Log-Odds, collocations, frequencies)
8. **`visualize.py`** - IRAL-style publication figures (violin, radar, keyword plots)
9. **`pipeline.py`** - Shard-local orchestrator, idempotent execution
10. **`utils.py`** - Shared helpers (tokenization, suffix lists, schema generation)

### Six Core Metrics (Default Mode)

All metrics are deterministic, based on spaCy parses, and robust to edge cases:

1. **`mtld`** - Measure of Textual Lexical Diversity (lexicalrichness library)
2. **`nominalization_density`** - Derived nominalizations per 1000 words (heuristic: suffix + lemma difference)
3. **`modal_epistemic_rate`** - Modal verbs + epistemic markers per 100 words
4. **`clause_complexity`** - Mean clausal dependencies per sentence (advcl, ccomp, acl, xcomp, relcl)
5. **`passive_voice_ratio`** - Proportion of sentences with passive voice
6. **`s2s_cosine_similarity`** - Mean cosine similarity between adjacent sentences (requires embeddings)

### Optional Features

- **Embeddings** (off by default): Enable with `--enable-embeddings` for s2s_cosine metric
- **IRAL Lexical** (off by default): Moved to `src/iral` as a separate module for log-odds, collocations, top-k frequencies

---

## Files Changed/Created

### New Files
- `src/cli.py` - CLI entrypoint
- `src/ingest.py` - Data ingestion
- `src/parse_and_cache.py` - Parsing with caching
- `src/metrics_core.py` - Core metrics
- `src/embeddings.py` - Embedding wrapper (replaced old version)
- `src/stats_analysis.py` - Statistical analysis
- `src/iral/iral_lexical.py` - Lexical explainability (moved to `src/iral`)
- `src/visualize.py` - Visualization
- `src/pipeline.py` - Pipeline orchestrator
- `src/utils.py` - Shared utilities
- `src/__init__.py` - Module exports
- `tests/test_core_metrics.py` - Unit tests for core metrics
- `config/metrics_config.yaml` - Pipeline configuration (replaced old version)

### Archived Files (moved to `archive/old_15_metric_implementation/`)
- `src/run_pipeline.py` - Old 15-metric pipeline
- `src/features.py` - Old feature extraction
- `src/lexical_diversity.py` - Superseded by metrics_core
- `src/syntax_features.py` - Superseded by metrics_core
- `src/pos_entropy.py` - Not needed in 6-metric baseline
- `src/perplexity.py` - Heavy metric, not in baseline
- `src/embeddings_old.py` - Old implementation
- `src/discourse_markers.py` - Superseded by metrics_core
- `src/function_words.py` - Not needed
- `src/nominalization.py` - Superseded by utils + metrics_core
- `src/pos_tools.py` - Not needed
- `src/advanced_plots.py` - Superseded by visualize
- Old test files (6 files)

---

## Usage Examples

### Basic Pipeline Run (Default Mode)

Process the canonical training dataset with 6 core metrics:

```bash
python -m src.cli run \
	--input data/gsingh1-train/train.csv \
	--output results/baseline_run \
	--shards 8 \
	--workers 2
```

**Output:**
- `results/baseline_run/all_features.parquet` - All documents with extracted metrics
- `results/baseline_run/tables/statistical_tests.csv` - Statistical comparison
- `results/baseline_run/figures/` - Violin plots, radar chart

### With Embeddings Enabled

Enable embeddings to compute s2s_cosine_similarity:

```bash
python -m src.cli run \
	--input data/gsingh1-train/train.csv \
	--output results/with_embeddings \
	--enable-embeddings \
	--shards 4
```

### With IRAL Lexical Explainability

IRAL lexical analysis has been moved to `src/iral` and is no longer executed by default from the main CLI.

To run IRAL lexical analysis separately, use the module directly in a small script or an ad-hoc call:

```python
from src.iral import compute_iral_outputs
outputs = compute_iral_outputs(df, group_col='label', text_col='text', output_dir=Path('results/lexical'))
```

---

## Data Assumptions

### Column Names

The pipeline auto-detects common column names:

- **Text column**: `text`, `content`, `document`, `human_story`
- **Label column**: `label`, `class`, `category`, `is_ai`, `is_human`
- **Topic column**: `topic`, `subject`, `domain`, `category`, `prompt`

### Minimum Document Length

Documents with <20 words return NaN for all metrics (configurable in `config/metrics_config.yaml`).

---

## Dependencies

Core dependencies (from `requirements.txt`):

```
spacy>=3.0
lexicalrichness>=0.5.0
sentence-transformers>=2.2.0  # Optional, for embeddings
pandas>=1.5
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
click>=8.0
pyyaml>=6.0
```

Install all:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

---

## Testing

### Unit Tests

Run core metric tests:
```bash
pytest tests/test_core_metrics.py -v
```

### Integration Test

Run on a small subset:
```bash
python -m src.cli run \
	--input data/gsingh1-train/train.csv \
	--output results/integration_test \
	--max-rows 50 \
	--shards 1
```

---

## Reproducing IRAL Figures

The visualization module (`src/visualize.py`) adopts IRAL plotting styles; IRAL lexical analysis now lives in `src/iral`.

