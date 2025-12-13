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

Moved to docs/data_cleaning.md

Full content copied to `docs/data_cleaning.md` and top-level file replaced with this pointer to keep repository tidy.
4. Remove control characters and HTML artifacts
