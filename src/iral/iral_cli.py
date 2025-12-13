"""
Update IRAL pipeline usage in CLI to use the new orchestrator for multi-corpus 1-vs-N IRAL analysis.
"""
import pandas as pd
from pathlib import Path
from .iral_orchestrator import run_iral_pipeline

def run_iral_cli(df: pd.DataFrame, output_dir: Path, group_col: str = 'label', text_col: str = 'text', reference_group: str = 'Human_story'):
    run_iral_pipeline(df, group_col=group_col, text_col=text_col, output_dir=output_dir / 'lexical', reference_group=reference_group)
