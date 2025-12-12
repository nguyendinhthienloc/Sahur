"""
6-Metric Linguistic Baseline Pipeline

Core modules for human vs AI text classification.
"""

from .metrics_core import (
    extract_core_metrics,
    compute_mtld,
    compute_nominalization_density,
    compute_modal_epistemic_rate,
    compute_clause_complexity,
    compute_passive_voice_ratio,
    compute_s2s_cosine_similarity,
    CORE_METRIC_NAMES
)

from .pipeline import run_pipeline, process_shard
from .ingest import load_data, create_shards
from .parse_and_cache import parse_documents, get_spacy_pipeline
from .stats_analysis import compare_groups, export_statistical_tests
from .visualize import create_all_figures
from .irral_lexical import compute_irral_outputs

__version__ = "1.0.0"
__all__ = [
    'extract_core_metrics',
    'run_pipeline',
    'load_data',
    'parse_documents',
    'compare_groups',
    'create_all_figures',
    'compute_irral_outputs',
    'CORE_METRIC_NAMES'
]
