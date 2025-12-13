# Metrics Reference & Pipeline Guide

**Complete reference for the 6 core linguistic metrics and step-by-step pipeline usage.**

----

## Table of Contents

1. [Six Core Metrics Explained](#six-core-metrics-explained)
2. [Pipeline Quick Start](#pipeline-quick-start)
3. [Complete Pipeline Workflow](#complete-pipeline-workflow)
4. [Output Files Reference](#output-files-reference)
5. [Statistical Tests Explained](#statistical-tests-explained)
6. [Troubleshooting](#troubleshooting)

----

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

----

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

----

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

----

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

----

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

----

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
-- Document has < 2 sentences
-- Embeddings not computed (--enable-embeddings not set)

**Why it matters:**  
AI maintains extremely smooth, uniform transitions between sentences. Humans jump around more, introducing topic shifts and less predictable discourse flow.

**Typical values:**
- Human: 0.25–0.55 (more variable transitions)
- AI: 0.35–0.75 (smoother, more coherent)
- Higher = smoother discourse flow

**Requirements:**
-- Must use `--enable-embeddings` flag
-- Requires `sentence-transformers` library
-- GPU recommended for large datasets

**Implementation:**  
Uses sentence-transformers for dense embeddings, computes pairwise cosine similarities.

----

## Summary Table

| Metric | Measures | Why It Differentiates AI & Human | Human Range | AI Range |
|--------|----------|----------------------------------|-------------|----------|
| **MTLD** | Lexical diversity | AI uses inflated variety; humans repeat naturally | 40–120 | 60–150 |
| **Nominalization Density** | Abstract noun use (per 1000 words) | AI compresses ideas using nominalizations | 20–60 | 30–150 |
| **Modal/Epistemic Rate** | Hedging/uncertainty (per 100 words) | Humans hedge; AI avoids uncertainty | 1–5 | 0–1 |
| **Clause Complexity** | Dependent clauses per sentence | Human complexity is irregular; AI is smoother | 0.2–4.0 | 0.8–2.0 |
| **Passive Ratio** | Proportion of passive sentences | Humans use more passives | 0.10–0.30 | 0.05–0.20 |
| **S2S Cosine** | Adjacent sentence similarity | AI transitions are too coherent | 0.25–0.55 | 0.35–0.75 |

----

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

----

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

----

### 3. With IRAL Lexical Analysis

IRAL lexical analysis is now a separate module available at `src.iral`. It is not run by default from the main CLI. To run IRAL analyses:

```python
from src.iral import compute_iral_outputs

# df: a pandas DataFrame with text and group columns
compute_iral_outputs(df, group_col='topic', text_col='text', output_dir='results/iral')
```

**What this adds:**
- Log-odds ratios for keyword extraction
- Bigram collocations per group
- Token frequency analysis
- Keyword visualizations

----

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

----

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

----

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

# Generate figures
create_all_figures(df, output_dir)
```

----

## Troubleshooting

- If `--enable-embeddings` is used but embeddings are missing, ensure `sentence-transformers` is installed and the cache directory is writable.
- If IRAL functions raise import errors, import from `src.iral` directly and ensure the workspace root is on `PYTHONPATH`.

----

Full content moved from top-level to keep the repository organized.
'''markdown
Contents moved from top-level METRICS_REFERENCE.md

See the original guide in the repository root or use this organized docs copy.
'''
