"""
Pipeline orchestrator for shard-local feature extraction runs.

Handles:
- Processing individual shards independently
- Parsing → feature extraction → embedding (optional)
- Writing shard outputs (parquet/csv)
- Idempotent and resumable execution
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import logging

from .parse_and_cache import parse_documents
from .metrics_core import extract_core_metrics, CORE_METRIC_NAMES
from .embeddings import create_embed_function
from .utils import setup_logger

logger = setup_logger(__name__)


def process_shard(shard_df: pd.DataFrame,
                 shard_id: int,
                 output_dir: Path,
                 cache_dir: Optional[Path] = None,
                 enable_embeddings: bool = False,
                 batch_size: int = 32,
                 n_process: int = 1) -> pd.DataFrame:
    """
    Process a single shard: parse docs and extract features.
    
    Parameters
    ----------
    shard_df : pd.DataFrame
        Shard dataframe with 'text' and 'doc_id' columns
    shard_id : int
        Shard identifier
    output_dir : Path
        Output directory for shard results
    cache_dir : Path, optional
        Cache directory for parsed docs
    enable_embeddings : bool
        Whether to compute embeddings for s2s_cosine
    batch_size : int
        Batch size for parsing
    n_process : int
        Number of processes for parsing
        
    Returns
    -------
    pd.DataFrame
        Shard dataframe with extracted features
    """
    logger.info(f"Processing shard {shard_id} ({len(shard_df)} documents)")
    
    # Check if already processed
    shard_output_path = output_dir / f"shard_{shard_id}" / "features.parquet"
    if shard_output_path.exists():
        logger.info(f"Shard {shard_id} already processed, loading cached results")
        return pd.read_parquet(shard_output_path)
    
    # Ensure text and doc_id columns exist
    if 'text' not in shard_df.columns:
        raise ValueError("Shard dataframe must have 'text' column")
    
    if 'doc_id' not in shard_df.columns:
        shard_df['doc_id'] = [f"shard{shard_id}_doc{i}" for i in range(len(shard_df))]
    
    texts = shard_df['text'].tolist()
    doc_ids = shard_df['doc_id'].tolist()
    
    # Parse documents
    logger.info(f"Parsing {len(texts)} documents for shard {shard_id}")
    
    shard_cache_dir = None
    if cache_dir:
        shard_cache_dir = cache_dir / f"shard_{shard_id}"
    
    docs = parse_documents(
        texts=texts,
        doc_ids=doc_ids,
        cache_dir=shard_cache_dir,
        use_cache=cache_dir is not None,
        batch_size=batch_size,
        n_process=n_process
    )
    
    # Extract features
    logger.info(f"Extracting features for shard {shard_id}")
    
    embed_cache_dir = None
    if enable_embeddings and cache_dir:
        embed_cache_dir = cache_dir / f"shard_{shard_id}_embeds"
    
    features_list = []
    
    for doc, doc_id, text in tqdm(zip(docs, doc_ids, texts), total=len(docs), desc=f"Shard {shard_id}"):
        # Create embedding function if needed
        embed_fn = None
        if enable_embeddings:
            embed_fn = create_embed_function(
                cache_dir=embed_cache_dir,
                doc_id=doc_id
            )
        
        # Extract metrics
        metrics = extract_core_metrics(doc, embed_fn=embed_fn, doc_id=doc_id)
        metrics['doc_id'] = doc_id
        
        features_list.append(metrics)
    
    # Create features dataframe
    features_df = pd.DataFrame(features_list)
    
    # Merge with original shard data
    result_df = shard_df.merge(features_df, on='doc_id', how='left')
    
    # Save shard results
    shard_output_dir = output_dir / f"shard_{shard_id}"
    shard_output_dir.mkdir(parents=True, exist_ok=True)
    
    result_df.to_parquet(shard_output_path)
    result_df.to_csv(shard_output_dir / "features.csv", index=False)
    
    logger.info(f"Shard {shard_id} processing complete, saved to {shard_output_path}")
    
    return result_df


def run_pipeline(df: pd.DataFrame,
                output_dir: Path,
                cache_dir: Optional[Path] = None,
                enable_embeddings: bool = False,
                n_shards: int = 1,
                batch_size: int = 32,
                n_process: int = 1) -> pd.DataFrame:
    """
    Run the full pipeline on a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    output_dir : Path
        Output directory
    cache_dir : Path, optional
        Cache directory
    enable_embeddings : bool
        Whether to compute embeddings
    n_shards : int
        Number of shards (for multi-shard processing)
    batch_size : int
        Batch size for parsing
    n_process : int
        Number of processes for parsing
        
    Returns
    -------
    pd.DataFrame
        Complete dataframe with all features
    """
    logger.info(f"Starting pipeline: {len(df)} documents, {n_shards} shards")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process single shard or multiple
    if n_shards == 1:
        result_df = process_shard(
            shard_df=df,
            shard_id=0,
            output_dir=output_dir,
            cache_dir=cache_dir,
            enable_embeddings=enable_embeddings,
            batch_size=batch_size,
            n_process=n_process
        )
    else:
        # Create shards
        from .ingest import create_shards
        
        shards = create_shards(df, n_shards, balance_by='topic' if 'topic' in df.columns else None)
        
        # Process each shard
        shard_results = []
        for i, shard in enumerate(shards):
            shard_result = process_shard(
                shard_df=shard,
                shard_id=i,
                output_dir=output_dir,
                cache_dir=cache_dir,
                enable_embeddings=enable_embeddings,
                batch_size=batch_size,
                n_process=n_process
            )
            shard_results.append(shard_result)
        
        # Combine shards
        result_df = pd.concat(shard_results, ignore_index=True)
    
    # Save combined results
    combined_path = output_dir / "all_features.parquet"
    result_df.to_parquet(combined_path)
    result_df.to_csv(output_dir / "all_features.csv", index=False)
    
    logger.info(f"Pipeline complete: {len(result_df)} documents processed")
    logger.info(f"Results saved to {combined_path}")
    
    return result_df
