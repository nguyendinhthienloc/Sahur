"""Regenerate word clouds from lexical_frequency_summary.csv for a given topic."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Add repo root to path so src can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse filtering logic (stopwords, numeric/non-word filtering)
from src.iral.iral_plots import _wordcloud_frequencies


def _resolve_csv(topic_or_path: str, results_root: Path) -> Path:
    """Resolve the CSV path from a topic name or a direct path."""
    candidate = Path(topic_or_path)

    # If user passed a directory, look for lexical/lexical_frequency_summary.csv inside it
    if candidate.is_dir():
        lexical_dir = candidate / "lexical"
        if lexical_dir.exists():
            return lexical_dir / "lexical_frequency_summary.csv"
        return candidate / "lexical_frequency_summary.csv"

    # If user passed a file directly
    if candidate.is_file():
        return candidate

    # Otherwise treat as topic under results/<topic>
    return results_root / candidate / "lexical" / "lexical_frequency_summary.csv"


def _load_frequencies(csv_path: Path, value_column: str) -> dict:
    """Load frequencies grouped by corpus from the CSV."""
    df = pd.read_csv(csv_path)
    required_cols = {"corpus", "token", value_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {', '.join(sorted(missing))}")

    frequencies = {}
    for corpus, group in df.groupby("corpus"):
        pairs = list(zip(group["token"], group[value_column]))
        freq_dict = _wordcloud_frequencies(pairs)
        if freq_dict:
            frequencies[corpus] = freq_dict
    return frequencies


def _write_wordcloud(freqs: dict, out_path: Path, title: str = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="tab20",
        prefer_horizontal=1.0,
    )
    wc.generate_from_frequencies(freqs)
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    if title:
        fig.suptitle(f'Word Cloud: {title}', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "topic",
        help="Topic name (e.g., entertainment_50x7) or path to results folder/CSV",
    )
    parser.add_argument(
        "--value-column",
        default="frequency",
        choices=["frequency", "count"],
        help="Column to use for weights (default: frequency)",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing topic result folders (default: results)",
    )

    args = parser.parse_args(argv)

    results_root = Path(args.results_root).resolve()
    csv_path = _resolve_csv(args.topic, results_root)

    if not csv_path.exists():
        sys.stderr.write(f"Cannot find lexical_frequency_summary.csv at: {csv_path}\n")
        return 1

    freqs_by_corpus = _load_frequencies(csv_path, args.value_column)
    if not freqs_by_corpus:
        sys.stderr.write("No frequencies found after filtering; nothing to plot.\n")
        return 1

    # Save in results/<topic>/lexical/figures/<corpus_name>/wordcloud.png
    figures_root = csv_path.parent / "figures"

    for corpus, freqs in freqs_by_corpus.items():
        slug = (
            str(corpus)
            .lower()
            .replace("/", "-")
            .replace(" ", "_")
            .replace(".", "-")
        )
        out_dir = figures_root / slug
        out_path = out_dir / "wordcloud.png"
        _write_wordcloud(freqs, out_path, title=corpus)
        print(f"✓ Saved word cloud for '{corpus}' -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
