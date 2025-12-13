"""Run IRAL lexical comparisons for Human vs model columns.

Creates full comparison outputs and 3 sampled runs of 40 rows each.
Saves outputs under `results/iral/human_vs_<model>/`.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from src.iral import compute_iral_outputs


DATA_PATH = Path("data/cleaned_by_topic/environment.csv")
OUT_BASE = Path("results/iral")


def make_long(df, human_col, model_col, human_label="human", model_label=None):
    model_label = model_label or model_col
    records = []
    for i, row in df.iterrows():
        records.append({"text": row.get(human_col, ""), "label": human_label, "doc_id": f"{i}_human"})
        records.append({"text": row.get(model_col, ""), "label": model_label, "doc_id": f"{i}_model"})
    return pd.DataFrame.from_records(records)


def run_comparison(df, human_col, model_col, model_short):
    print(f"Running full IRAL outputs for Human vs {model_short}")
    long_df = make_long(df, human_col, model_col, human_label="human", model_label=model_short)
    out_dir = OUT_BASE / f"human_vs_{model_short}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = compute_iral_outputs(long_df, group_col='label', text_col='text', output_dir=out_dir, min_count=2, top_k=100)
    print(f"Saved full outputs to {out_dir}")

    # Run 3 samples of 40 rows each (paired: each sample uses same 40 original rows)
    n_samples = 3
    sample_n = 40
    rng = np.random.default_rng(42)
    indices = np.arange(len(df))
    for s in range(n_samples):
        sample_idx = rng.choice(indices, size=sample_n, replace=False)
        sample_df = df.iloc[sample_idx].reset_index(drop=True)
        long_sample = make_long(sample_df, human_col, model_col, human_label='human', model_label=model_short)
        sample_out = out_dir / f"sample_{s+1:02d}"
        sample_out.mkdir(parents=True, exist_ok=True)
        compute_iral_outputs(long_sample, group_col='label', text_col='text', output_dir=sample_out, min_count=1, top_k=50)
        print(f"Saved sample {s+1} outputs to {sample_out}")


def main():
    df = pd.read_csv(DATA_PATH)
    # Column names observed in dataset header
    comps = [
        ("Human_story", "llama-8B", "llama"),
        ("Human_story", "GPT_4-o", "gpt4")
    ]

    for human_col, model_col, short in comps:
        if human_col not in df.columns or model_col not in df.columns:
            print(f"Missing columns: {human_col} or {model_col} not in {DATA_PATH}")
            continue
        run_comparison(df, human_col, model_col, short)


if __name__ == '__main__':
    main()
