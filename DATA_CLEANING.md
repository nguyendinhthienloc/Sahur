# Data cleaning and preprocessing (summary)

This document summarizes the cleaning steps that produced the `cleaned.csv` and the shard files under `data/gsingh1-train/cleaned_shards/`.

Purpose
- Produce a single, reproducible tabular file with one text per row named `text`, with an optional `label` column (`Human` or `AI`) and auxiliary columns preserved (e.g., `prompt`, `length_category`).

Source files
- `data/gsingh1-train/train.csv` (original combined file)
- Shards under `data/gsingh1-train/cleaned_shards/` (split outputs used for testing and batching)

High-level cleaning steps
1. Consolidate model columns into a single `text` column
   - Melt or iterate over columns containing outputs (for example: `Human_story`, `gemma-2-9b`, `mistral-7B`, `qwen-2-72B`, `llama-8B`, `accounts/yi-01-ai/models/yi-large`, `GPT_4-o`) to create one row per generated text.
   - Create a `model` column with the source column name and map `model == 'Human_story'` to `label = 'Human'` else `label = 'AI'`.

2. Remove missing or empty text
   - Drop rows where `text` is `NaN` or empty after trimming.

3. Normalize whitespace and newlines
   - Replace any sequence of whitespace (including newlines and tabs) with a single space and strip leading/trailing whitespace.

4. Remove control characters and HTML artifacts
   - Unescape HTML entities if present and remove non-printable/control characters.

5. Deduplicate
   - Drop exact duplicate texts across the dataset (`drop_duplicates(subset=['text'])`).

6. Length filtering and `length_category`
   - Compute token length via simple whitespace tokenization (or a tokenizer if available) and either assign or verify `length_category` (`short`, `medium`, `long`).
   - Optionally remove extremely short (< 5 tokens) or extremely long entries that break downstream processing.

7. Sharding (optional)
   - Split the cleaned dataset into shards (for example by length category, or by random partitioning) and write CSVs like `short_shard_1.csv`, `medium_shard_1.csv`, `long_shard_1.csv`.

Result schema
- `text` (string): cleaned text ready for `run_pipeline.py` (pipeline expects a `text` column)
- `label` (optional): `Human` or `AI`
- `prompt`, `model`, `length_category`, and other metadata columns copied from source where available

Reproducible pandas example
```
import re
import html
import pandas as pd

# 1. load
df = pd.read_csv('data/gsingh1-train/train.csv')

# 2. melt model output columns into long format
model_cols = ['Human_story','gemma-2-9b','mistral-7B','qwen-2-72B','llama-8B','accounts/yi-01-ai/models/yi-large','GPT_4-o']
id_cols = [c for c in df.columns if c not in model_cols]
long = df.melt(id_vars=id_cols, value_vars=model_cols, var_name='model', value_name='text')

# 3. assign label
long['label'] = long['model'].apply(lambda m: 'Human' if m == 'Human_story' else 'AI')

# 4. basic cleaning utilities
def clean_text(s):
    if pd.isna(s):
        return s
    s = html.unescape(str(s))
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[\x00-\x1F\x7F]+', '', s)
    return s

long['text'] = long['text'].apply(clean_text)

# 5. drop empties and duplicates
long = long.dropna(subset=['text'])
long = long[long['text'].str.len() > 0]
long = long.drop_duplicates(subset=['text'])

# 6. optional length category check
long['token_count'] = long['text'].str.split().apply(len)
long.loc[long['token_count'] < 50, 'length_category'] = 'short'
long.loc[(long['token_count'] >= 50) & (long['token_count'] < 250), 'length_category'] = 'medium'
long.loc[long['token_count'] >= 250, 'length_category'] = 'long'

# 7. write cleaned file
long.to_csv('data/gsingh1-train/cleaned_shards/cleaned.csv', index=False)
```

Notes and assumptions
- The repository's pipeline (`src/run_pipeline.py`) expects a `text` column; the cleaning consolidates multiple model columns into that schema.
- If you used a different tokenization or more advanced normalization (e.g., sentence segmentation, spaCy-based tokenization, aggressive HTML stripping), record those steps here — the example above is intentionally minimal and reproducible with only `pandas` and `html`.
- If labels other than `Human`/`AI` are required (e.g., model-specific labels), preserve the `model` column and map later in analysis.

Next steps / reproducibility
- Run the example script above in a Python environment with `pandas` installed (see `requirements.txt`).
- If you want, I can (A) create a small runnable script `scripts/clean_data.py` that performs the steps and accepts CLI args, or (B) run the pipeline over the provided shard and save `results/` outputs — tell me which you prefer.
