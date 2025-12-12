# SC203 Linguistic Metrics Pipeline (15‑Metric Core Version)

This project extracts 15 scientifically robust linguistic metrics that best distinguish human and AI writing.  
These metrics exclude stylistic noise, unstable features, and redundant signals.

## 🚀 What the Pipeline Does
- Cleans and preprocesses text
- Runs dependency parsing (spaCy)
- Computes 15 high‑signal linguistic metrics
- Generates visualizations (PCA, radar, heatmaps)
- Outputs a unified feature matrix for analysis or modeling

## 🧩 Final 15 Linguistic Metrics

### Lexical Diversity (2)
1. **MTLD**
2. **HD‑D**

### Syntactic Complexity (4)
3. **Dependency Tree Depth Percentiles**
4. **Clause‑Type Density**
5. **Passive Voice Ratio**
6. **POS N‑gram Entropy**

### Semantic & Discourse (3)
7. **Modal & Epistemic Marker Rate**
8. **Discourse Marker Distribution**
9. **Entity Specificity Score**

### Embedding & Coherence (3)
10. **Topical Drift**
11. **Sentence‑to‑Sentence Semantic Similarity**
12. **Embedding‑Centroid Distance**

### Language‑Model‑Based (3)
13. **Perplexity Gap**
14. **Surprisal Variance**
15. **Nominalization Density**

## 📦 Installation
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## ▶️ Quick Start
```bash
python -m src.run_pipeline --input data.csv --output results/

# For fast mode (skip heavy transformer models):
python -m src.run_pipeline --input data.csv --output results/ --disable-heavy
```

## 🧪 Running Tests

This project includes a comprehensive test suite with fast unit tests and optional slow integration tests.

### Fast Tests (Default - No Model Downloads)
Run all fast, deterministic unit tests without downloading large models:
```bash
pytest -q
```

### Slow Tests (Full Integration)
Run tests that require transformer/sentence-transformer models:
```bash
pytest -q -m slow
```

### Test Coverage
- `test_discourse_markers.py` - Discourse and modal marker extraction
- `test_embeddings.py` - Embedding-based metrics (with dummy models)
- `test_lexical_diversity.py` - MTLD and HD-D computation
- `test_perplexity.py` - Perplexity evaluation (monkeypatched)
- `test_pipeline_endtoend.py` - Full pipeline with 15-metric validation
- `test_performance.py` - Performance benchmarks (marked as slow)

All tests use deterministic fixtures and avoid network calls by default.

## 📊 Visualizations
See **DIAGRAMS.md** for a clean list of supported plots and examples.

## 📚 Full Metric Definitions
See **METRICS_REFERENCE.md**.
