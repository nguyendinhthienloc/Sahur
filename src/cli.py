"""
Command-line interface for the 6-metric linguistic baseline pipeline.

Usage:
    python -m src.cli run --input data/train.csv --output results/ [options]
    python -m src.cli analyze --input results/all_features.parquet --output results/ [options]
"""

import click
import sys
from pathlib import Path
import pandas as pd
import logging

# Prevent __pycache__ generation in src
sys.dont_write_bytecode = True

from .utils import setup_logger, write_metrics_schema
from .ingest import load_data
from .pipeline import run_pipeline
from .stats_analysis import compare_groups, export_statistical_tests
from .visualize import create_all_figures
from .iral import compute_iral_outputs
from .metrics_core import CORE_METRIC_NAMES

logger = setup_logger(__name__)


@click.group()
def cli():
    """6-Metric Linguistic Baseline Pipeline for Human vs AI Text Analysis."""
    pass


@cli.command()
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='Path to input CSV file')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory for results')
@click.option('--text-col', default=None,
              help='Text column name (auto-detected if not specified)')
@click.option('--label-col', default=None,
              help='Label column name (auto-detected if not specified)')
@click.option('--topic-col', default=None,
              help='Topic column name (auto-detected if not specified)')
@click.option('--max-rows', default=None, type=int,
              help='Maximum rows to process (for testing)')
@click.option('--shards', default=1, type=int,
              help='Number of shards for parallel processing')
@click.option('--workers', default=1, type=int,
              help='Number of spaCy workers')
@click.option('--batch-size', default=32, type=int,
              help='Batch size for parsing')
@click.option('--enable-embeddings/--no-embeddings', default=True,
              help='Enable sentence embeddings for s2s_cosine metric')
@click.option('--cache-dir', default=None, type=click.Path(),
              help='Cache directory for parsed docs (default: output_dir/cache)')
@click.option('--dry-run', is_flag=True,
              help='Show what would be done without executing')
@click.option('--debug', is_flag=True,
              help='Enable debug logging')
def run(input_path, output_dir, text_col, label_col, topic_col, max_rows,
    shards, workers, batch_size, enable_embeddings,
    cache_dir, dry_run, debug):
    """
    Run the complete pipeline: parse, extract features, analyze, and visualize.
    """
    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    
    # Validate and prepare paths
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)
    
    # Auto-create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    if cache_dir is None:
        cache_dir = output_dir / 'cache'
    else:
        cache_dir = Path(cache_dir)
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("6-METRIC LINGUISTIC BASELINE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Shards: {shards}, Workers: {workers}, Batch size: {batch_size}")
    logger.info(f"Embeddings: {enable_embeddings}")
    # Auto topic option removed — topics should be present in the dataset
    
    if dry_run:
        logger.info("DRY RUN - No changes will be made")
        return
    
    # Step 1: Load data
    logger.info("\n[1/5] Loading data...")
    df = load_data(
        path=input_path,
        text_col=text_col,
        label_col=label_col,
        topic_col=topic_col,
        max_rows=max_rows,
        clean=True,
        cache_dir=cache_dir
    )
    
    logger.info(f"Loaded {len(df)} documents")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Step 2: Run pipeline (parse + extract features)
    logger.info("\n[2/5] Running feature extraction pipeline...")
    features_df = run_pipeline(
        df=df,
        output_dir=output_dir,
        cache_dir=cache_dir,
        enable_embeddings=enable_embeddings,
        n_shards=shards,
        batch_size=batch_size,
        n_process=workers
    )
    
    logger.info(f"Extracted features for {len(features_df)} documents")
    
    # Step 3: Statistical analysis (always run if possible)
    logger.info("\n[3/5] Performing statistical analysis...")
    stats_results = None
    if 'label' in features_df.columns:
        from src.stats_analysis import compare_one_vs_many
        stats_results = compare_one_vs_many(
            df=features_df,
            metrics=CORE_METRIC_NAMES,
            group_col='label',
            reference_group='Human_story'
        )
        # Export statistics
        stats_dir = output_dir / 'tables'
        export_statistical_tests(stats_results, stats_dir / 'statistical_tests.csv')
        logger.info(f"Statistical tests saved to {stats_dir / 'statistical_tests.csv'}")
    else:
        logger.warning("No 'label' column found, skipping statistical comparison")

    # Step 4: IRAL lexical analysis (always run if possible)
    from src.iral.iral_cli import run_iral_cli
    logger.info("\n[4/5] Running IRAL lexical analysis (multi-corpus 1-vs-N)...")
    run_iral_cli(features_df, output_dir, group_col='label', text_col='text', reference_group='Human_story')
    lexical_outputs = None
    
    # Step 5: Visualization
    logger.info("\n[5/5] Creating visualizations...")
    
    if 'label' in features_df.columns:
        create_all_figures(
            df=features_df,
            metrics=CORE_METRIC_NAMES,
            group_col='label',
            output_dir=output_dir,
            lexical_outputs=lexical_outputs
        )
        logger.info(f"Figures saved to {output_dir / 'figures'}")
    else:
        logger.warning("No 'label' column, skipping visualizations")
    
    # Generate schema
    schema_path = output_dir / 'metrics_schema.json'
    write_metrics_schema(
        output_path=schema_path,
        core_metrics=CORE_METRIC_NAMES,
        optional_metrics=['heavy_metrics'] if not enable_embeddings else []
    )
    
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results directory: {output_dir}")
    logger.info(f"  - Features: {output_dir / 'all_features.parquet'}")
    if stats_results is not None:
        logger.info(f"  - Statistics: {output_dir / 'tables' / 'statistical_tests.csv'}")
    logger.info(f"  - Figures: {output_dir / 'figures'}")
    if lexical_outputs:
        logger.info(f"  - Lexical: {output_dir / 'lexical'}")


@cli.command()
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='Path to features parquet/csv file')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory for analysis results')
def analyze(input_path, output_dir):
    """
    Analyze pre-computed features (skip parsing and extraction).
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    logger.info("=" * 80)
    logger.info("ANALYZING PRE-COMPUTED FEATURES")
    logger.info("=" * 80)
    
    # Load features
    logger.info(f"Loading features from {input_path}")
    
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    logger.info(f"Loaded {len(df)} documents with {len(df.columns)} columns")
    
    # Statistical analysis
    if 'label' in df.columns:
        logger.info("Performing statistical analysis...")
        stats_results = compare_groups(
            df=df,
            metrics=CORE_METRIC_NAMES,
            group_col='label'
        )
        
        stats_dir = output_dir / 'tables'
        export_statistical_tests(stats_results, stats_dir / 'statistical_tests.csv')
        logger.info(f"Statistical tests saved to {stats_dir / 'statistical_tests.csv'}")
    
    # IRAL lexical analysis moved to separate module/package (src/iral)
    lexical_outputs = None
    
    # Visualizations
    if 'label' in df.columns:
        logger.info("Creating visualizations...")
        create_all_figures(
            df=df,
            metrics=CORE_METRIC_NAMES,
            group_col='label',
            output_dir=output_dir,
            lexical_outputs=lexical_outputs
        )
        logger.info(f"Figures saved to {output_dir / 'figures'}")
    
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


@cli.command()
def version():
    """Show version information."""
    click.echo("6-Metric Linguistic Baseline Pipeline v1.0.0")
    click.echo("Core metrics: mtld, nominalization_density, modal_epistemic_rate,")
    click.echo("              clause_complexity, passive_voice_ratio, s2s_cosine_similarity")


if __name__ == '__main__':
    cli()
