# Metrics Reference & Pipeline Guide

**Complete reference for the 6 core linguistic metrics and step-by-step pipeline usage.**

---

## Table of Contents

1. [Six Core Metrics Explained](#six-core-metrics-explained)
2. [Pipeline Quick Start](#pipeline-quick-start)
3. [Complete Pipeline Workflow](#complete-pipeline-workflow)
4. [Output Files Reference](#output-files-reference)
5. [Statistical Tests Explained](#statistical-tests-explained)
6. [Troubleshooting](#troubleshooting)

---

## Six Core Metrics Explained

### 1. MTLD (Measure of Textual Lexical Diversity)

**What it measures:**  
How varied the vocabulary is across the text.

**How it is calculated:**
1. Tokenize the text into alphabetic words (no punctuation)
2. Scan left to right while computing a running Type–Token Ratio (TTR = unique_words / total_words)
3. Each time TTR drops below a threshold (0.72), record a "factor"
4. Continue scanning until the end
5. MTLD = total_tokens / number_of_factors

**Fallback for short texts (< 50 tokens):**
```
MTLD = (unique_words / total_words) * 100
```

**Why it matters:**  
AI tends to produce artificially high lexical diversity because RLHF (Reinforcement Learning from Human Feedback) discourages repetition. Humans naturally repeat words more.

**Typical values:**
- Human: 40–120  
- AI: 60–150  
- Higher = more diverse vocabulary

**Implementation:**  
Uses the `lexicalrichness` library for documents with ≥50 tokens. For shorter documents, uses TTR-based fallback scaled to approximate MTLD range.

---

### 2. Nominalization Density

**What it measures:**  
The number of **nominalized nouns** per 1000 words.

Common nominalization endings:  
- `-tion`, `-sion`, `-ment`, `-ity`, `-ness`, `-ance`, `-ence`

Examples: *creation, development, complexity, awareness, importance, reference*

**How it is calculated:**
1. For each token that is tagged as NOUN (not proper noun):
   - Check if lemma or surface form ends with nominalization suffix
   - Check if word length ≥ 6 characters
   - For strong suffixes (-tion, -sion, -ment, -ity, -ance, -ence), count directly
   - For others, only count if lemma differs from surface form
2. Normalize:
```
nominalization_density = (num_nominalizations / total_words) * 1000
```

**Why it matters:**  
AI strongly prefers dense noun-heavy academic phrasing. AI models compress ideas into abstract nominalizations, while humans use more varied syntactic structures.

**Typical values:**
- Human: 20–60 per 1000 words
- AI: 30–150 per 1000 words
- Higher = more abstract/formal style

**Implementation:**  
Uses spaCy POS tagging and suffix matching with lemma validation.

---

### 3. Modal + Epistemic Rate

**What it measures:**  
How often a writer expresses uncertainty, possibility, or stance.

**Modal verbs:**
- `can`, `could`, `may`, `might`, `must`, `shall`, `should`, `will`, `would`

**Epistemic markers (adverbs):**
- `probably`, `possibly`, `certainly`, `clearly`, `obviously`, `perhaps`, `apparently`, `arguably`, `presumably`, `supposedly`, `definitely`, `undoubtedly`, `likely`, `unlikely`, `maybe`

**How it is calculated:**
```
modal_epistemic_rate = (num_modal_epistemic / total_words) * 100
```

Uses lemma matching for robustness (e.g., "can", "could" both match lemma "can").

**Why it matters:**  
Humans hedge naturally and express uncertainty. AI avoids hedging because RLHF penalizes uncertainty and rewards confident declarative statements.

**Typical values:**
- Human: 1–5 per 100 words
- AI: 0–1 per 100 words (often ~0)
- Higher = more hedging/uncertainty

**Implementation:**  
Checks both token text and lemma against the modal/epistemic phrase list.

---

### 4. Clause Complexity

**What it measures:**  
The number of **dependent clauses per sentence**.

Examples of dependent clauses:
- *"I went to the store **because I needed milk**"* (adverbial clause)
- *"She said **that she would come**"* (complement clause)
- *"The book **which I read** was good"* (relative clause)

**How it is calculated:**
1. Identify spaCy dependency labels for clausal relations:
   - `advcl` - adverbial clause modifier
   - `ccomp` - clausal complement
   - `xcomp` - open clausal complement
   - `acl` - adjectival clause
   - `relcl` - relative clause modifier
2. Count them per sentence
3. Normalize:
```
clause_complexity = total_clauses / num_sentences
```

**Why it matters:**  
Human writing fluctuates in complexity - some simple sentences, some complex. AI stays in a safer, smoother mid-range complexity to avoid parsing errors.

**Typical values:**
- Human: 0.2–4.0 clauses/sentence (high variance)
- AI: 0.8–2.0 clauses/sentence (low variance)
- Higher = more syntactic embedding

**Implementation:**  
Uses spaCy dependency parsing to identify subordinate clause constructions.

---

### 5. Passive Voice Ratio

**What it measures:**  
The proportion of sentences containing passive voice constructions.

**Passive voice examples:**
- *"The ball was thrown by John"* (passive)
- *"John threw the ball"* (active)

**How it is detected:**
Look for spaCy dependency labels:
- `nsubjpass` - passive nominal subject
- `auxpass` - passive auxiliary

A sentence is passive if it contains **either** `nsubjpass` OR `auxpass` (not both required).

**Formula:**
```
passive_ratio = num_passive_sentences / total_sentences
```

**Why it matters:**  
Humans use more passive voice naturally, especially in formal/academic writing. AI often defaults to active voice to maintain clarity.

**Typical values:**
- Human: 0.10–0.30 (10–30% of sentences)
- AI: 0.05–0.20 (5–20% of sentences)
- Higher = more passive constructions

**Note:** spaCy's dependency labels `nsubjpass` and `auxpass` are from v2.x but still work in v3.x for compatibility.

**Implementation:**  
Checks each sentence for passive dependency markers.

---

### 6. Sentence-to-Sentence Cosine Similarity (S2S Coherence)

**What it measures:**  
How semantically similar **adjacent** sentences are.

Higher = smoother, more predictable discourse transitions.

**How it is calculated:**
1. Split text into sentences
2. Embed each sentence using `sentence-transformers` (all-MiniLM-L6-v2 by default)
3. Compute cosine similarity between adjacent pairs:
```
cosine(s_i, s_{i+1}) = dot(v_i, v_{i+1}) / (||v_i|| * ||v_{i+1}||)
```
4. Average all adjacent pairs:
```
s2s_cosine = mean(cosine(sentence[i], sentence[i+1]) for all i)
```

Returns NaN if:
- Document has < 2 sentences
- Embeddings not computed (--enable-embeddings not set)

**Why it matters:**  
AI maintains extremely smooth, uniform transitions between sentences. Humans jump around more, introducing topic shifts and less predictable discourse flow.

**Typical values:**
- Human: 0.25–0.55 (more variable transitions)
- AI: 0.35–0.75 (smoother, more coherent)
- Higher = smoother discourse flow

**Requirements:**
- Must use `--enable-embeddings` flag
- Requires `sentence-transformers` library
- GPU recommended for large datasets

**Implementation:**  
Uses sentence-transformers for dense embeddings, computes pairwise cosine similarities.

---

## Summary Table

| Metric | Measures | Why It Differentiates AI & Human | Human Range | AI Range |
|--------|----------|----------------------------------|-------------|----------|
| **MTLD** | Lexical diversity | AI uses inflated variety; humans repeat naturally | 40–120 | 60–150 |
| **Nominalization Density** | Abstract noun use (per 1000 words) | AI compresses ideas using nominalizations | 20–60 | 30–150 |
| **Modal/Epistemic Rate** | Hedging/uncertainty (per 100 words) | Humans hedge; AI avoids uncertainty | 1–5 | 0–1 |
| **Clause Complexity** | Dependent clauses per sentence | Human complexity is irregular; AI is smoother | 0.2–4.0 | 0.8–2.0 |
| **Passive Ratio** | Proportion of passive sentences | Humans use more passives | 0.10–0.30 | 0.05–0.20 |
| **S2S Cosine** | Adjacent sentence similarity | AI transitions are too coherent | 0.25–0.55 | 0.35–0.75 |

---

## Pipeline Quick Start

### 1. Basic Run (6 Core Metrics Only)

Process your dataset with default settings:

```bash
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

**What this does:**
- Loads CSV file
- Auto-detects text, label, and topic columns
- Parses documents with spaCy (cached for reuse)
- Extracts 6 core metrics
- Saves features to `results/baseline_run/all_features.parquet`
- ⚠️ **Does NOT run statistical tests or generate plots** (see below)

---

### 2. With Embeddings (Enables S2S Cosine)

```bash
python -m src.cli run \
  --input <PATH_TO_YOUR_INPUT_CSV> \
  --output <PATH_TO_OUTPUT_DIRECTORY> \
  --enable-embeddings \
  --shards 4 \
  --workers 2
```

**Example:**
```bash
python -m src.cli run \
  --input data/gsingh1-train/train.csv \
  --output results/with_embeddings \
  --enable-embeddings \
  --shards 4 \
  --workers 2
```

**What this adds:**
- Computes sentence embeddings using MiniLM
- Enables s2s_cosine_similarity metric
- Caches embeddings for reuse

---

### 3. With IRAL Lexical Analysis

```bash
python -m src.cli run \
  --input data/gsingh1-train/train.csv \
  --output results/with_lexical \
  --enable-irral-lexical \
  --shards 4
```

**What this adds:**
- Log-odds ratios for keyword extraction
- Bigram collocations per group
- Token frequency analysis
- Keyword visualizations

---

## Complete Pipeline Workflow

### Step 1: Extract Features

```bash
# Full feature extraction with embeddings
python -m src.cli run \
  --input <PATH_TO_YOUR_INPUT_CSV> \
  --output <PATH_TO_OUTPUT_DIRECTORY> \
  --enable-embeddings \
  --shards 8 \
  --workers 2 \
  --batch-size 64
```

**Example:**
```bash
python -m src.cli run \
  --input data/gsingh1-train/train.csv \
  --output results/full_analysis \
  --enable-embeddings \
  --shards 8 \
  --workers 2 \
  --batch-size 64
```

**Outputs:**
- `<OUTPUT_DIR>/all_features.parquet` - All extracted features
- `<OUTPUT_DIR>/all_features.csv` - Human-readable version
- `<OUTPUT_DIR>/metrics_schema.json` - Metric definitions
- `<OUTPUT_DIR>/cache/` - Cached parsed documents

---

### Step 2: Run Statistical Tests

The pipeline **does not automatically run statistical tests**. You must run them separately:

```bash
python -m src.cli analyze \
  --input <PATH_TO_OUTPUT_DIRECTORY>/all_features.parquet \
  --output <PATH_TO_OUTPUT_DIRECTORY>
```

**Example:**
```bash
python -m src.cli analyze \
  --input results/full_analysis/all_features.parquet \
  --output results/full_analysis
```

**What this does:**
- Loads pre-computed features
- Runs group comparisons (human vs AI)
- Performs statistical tests:
  - Welch's t-test (for unequal variances)
  - Mann-Whitney U test (non-parametric)
  - Cohen's d effect size
  - Levene's test (variance homogeneity)
  - Multiple testing corrections (Holm-Bonferroni, Benjamini-Hochberg FDR)

**Outputs:**
- `<OUTPUT_DIR>/tables/statistical_tests.csv` - Complete test results

---

### Step 3: Generate IRAL-Style Plots

The pipeline **does not automatically generate plots**. You must run visualization separately:

```python
# Run in Python (or save as a script and run with: python generate_plots.py)
from pathlib import Path
import pandas as pd
from src.visualize import create_all_figures
from src.metrics_core import CORE_METRIC_NAMES

# REPLACE WITH YOUR OUTPUT DIRECTORY PATH
output_dir = Path('<PATH_TO_OUTPUT_DIRECTORY>')

# Load features
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
  --enable-irral-lexical
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
  --enable-irral-lexical
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
  --enable-irral-lexical \
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
