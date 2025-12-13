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
- **IRAL Lexical** (off by default): Enable with `--enable-iral-lexical` for Log-Odds, collocations, top-k frequencies

---

## Files Changed/Created

### New Files
- `src/cli.py` - CLI entrypoint
- `src/ingest.py` - Data ingestion
- `src/parse_and_cache.py` - Parsing with caching
- `src/metrics_core.py` - Core metrics
- `src/embeddings.py` - Embedding wrapper (replaced old version)
- `src/stats_analysis.py` - Statistical analysis
- `src/iral_lexical.py` - Lexical explainability
# Refactor notes moved to `docs/REFACTOR_NOTES.md`

Full refactor notes were moved into the `docs/` directory to organize documentation.

See [docs/REFACTOR_NOTES.md](docs/REFACTOR_NOTES.md) for the complete content.
```

For small datasets (<1K documents):
```bash
--shards 1 --workers 1 --batch-size 32
```

---

## Backwards Compatibility

### Re-enabling Heavy Metrics

The old 15-metric implementation is archived but can be restored if needed:

1. Copy archived files back to `src/`
2. Create a `--enable-heavy-metrics` flag in CLI
3. Import old feature extraction functions

### Data Format

Output format is backwards compatible:
- Parquet files can be read by old analysis scripts
- CSV output matches previous schema (with subset of metrics)

---

## Known Limitations

1. **train.csv format**: The provided train.csv has multiple model output columns. You may need to transform it to long format or specify a single text column.

2. **Embedding GPU**: s2s_cosine runs on CPU by default. For GPU acceleration, ensure CUDA-compatible PyTorch is installed.

3. **spaCy model**: Pipeline prefers `en_core_web_lg` but falls back to `en_core_web_sm`. Download lg for better accuracy.

4. **Short documents**: Documents <20 words return NaN metrics. Filter these before analysis if needed.

---

## Future Enhancements

Potential improvements (not implemented):

1. **Mixed-effects models**: Add statsmodels/R integration for `metric ~ label + (1|topic)` models
2. **Heavy metrics toggle**: Add `--enable-heavy-metrics` for perplexity, POS entropy, etc.
3. **Batch embedding optimization**: Use GPU batching for large-scale embedding computation
4. **Resume from checkpoint**: More granular checkpointing within shards
5. **Web interface**: Streamlit dashboard for interactive exploration

---

## Citation & Attribution

This refactoring integrates:
- **IRAL replication code** (plotting styles, statistical tests) - adapted with attribution
- **Original 15-metric pipeline** (archived) - retained for reference
- **New modular architecture** - designed for maintainability and extensibility

When using this pipeline, please cite:
- The IRAL paper (if using IRAL lexical features)
- This codebase (6-metric baseline implementation)

---

## Contact & Support

For issues or questions:
1. Check `README.md` for basic usage
2. Review `config/metrics_config.yaml` for settings
3. Run with `--debug` flag for detailed logging
4. Check archived code in `archive/old_15_metric_implementation/` for old implementations

---

## Checklist: Acceptance Criteria

- [x] Branch created: `refactor/6-metric-baseline`
- [x] 10 files in src/ (clean module structure)
- [x] 6 core metrics implemented (mtld, nominalization, modal, clause, passive, s2s)
- [x] CLI with `run` and `analyze` commands
- [x] Optional IRAL lexical module (--enable-iral-lexical)
- [x] Statistical analysis (Welch, Mann-Whitney, Cohen's d, FDR)
- [x] IRAL-style visualizations (violin, radar, keyword plots)
- [x] Unit tests (test_core_metrics.py)
- [x] Configuration file (metrics_config.yaml)
- [x] Documentation (REFACTOR_NOTES.md)
- [x] Old files archived (12 src files + 6 test files)
- [ ] Example run on train.csv (pending - requires data transformation or column spec)
- [ ] CI integration advice (run with --max-rows 100 for fast tests)

---

**End of Refactoring Notes**
