"""
Split topic CSVs under data/cleaned_by_topic into smaller shard files.

Usage:
    python scripts/shard_by_topic.py --rows-per-shard 7000

Defaults to 7000 rows per shard but can be adjusted.
"""
import argparse
from pathlib import Path
import pandas as pd


def shard_file(infile: Path, outdir: Path, rows_per_shard: int):
    df = pd.read_csv(infile)

    # Re-validate: drop rows with missing critical columns or timeout patterns
    df = df.dropna(subset=['Human_story', 'llama-8B'])

    def has_error(text):
        if pd.isna(text):
            return True
        t = str(text).lower()
        err_patterns = ['timeout', 'error', 'failed', 'exception', 'timed out', 'connection refused', 'server error', 'unable to generate', 'generation failed', 'request failed', 'api error', 'rate limit']
        return any(p in t for p in err_patterns) or len(str(text).strip()) < 50

    mask_h = df['Human_story'].apply(has_error)
    mask_l = df['llama-8B'].apply(has_error)
    df = df[~(mask_h | mask_l)].reset_index(drop=True)

    outdir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    if total == 0:
        return []

    shards = []
    for i in range(0, total, rows_per_shard):
        shard_df = df.iloc[i:i+rows_per_shard]
        shard_path = outdir / f"shard_{i//rows_per_shard:03d}.csv"
        shard_df.to_csv(shard_path, index=False)
        shards.append(shard_path)
    return shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows-per-shard', type=int, default=7000)
    args = parser.parse_args()

    input_dir = Path('data/cleaned_by_topic')
    output_root = Path('data/shards')
    output_root.mkdir(exist_ok=True)

    created = {}
    for infile in sorted(input_dir.glob('*.csv')):
        topic = infile.stem
        outdir = output_root / topic
        shards = shard_file(infile, outdir, args.rows_per_shard)
        created[topic] = [str(p) for p in shards]
        print(f"Topic {topic}: {len(shards)} shard(s), {len(shards) and sum(pd.read_csv(p).shape[0] for p in shards) or 0} rows total")

    print('\nSharding complete. Shards stored under data/shards/<topic>/')

if __name__ == '__main__':
    main()
