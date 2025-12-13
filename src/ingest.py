"""
Data ingestion module for the 6-metric baseline pipeline.

Handles:
- Loading train.csv and other CSV datasets
- Column normalization (auto-detect common names)
- Minimal text cleaning (unicode normalize, strip HTML)
- Shard creation for parallel processing
"""

import pandas as pd
import unicodedata
import re
import json
from typing import Optional, List, Tuple
from pathlib import Path
import logging
# Note: zero-shot auto-classification removed — dataset is expected to include `topic`.

from .utils import setup_logger

logger = setup_logger(__name__)



def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form and handle common issues."""
    if not isinstance(text, str):
        return str(text)
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    # Fix common HTML entities that slip through
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&quot;', '"')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    return text


# zero-shot helper removed — topic assignment now relies on existing `topic` column


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not isinstance(text, str):
        return str(text)
    # Simple regex to remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    return clean


def minimal_clean(text: str) -> str:
    """
    Apply minimal cleaning: unicode normalize and strip HTML.
    
    Does NOT perform aggressive cleaning (preserve original text characteristics).
    """
    text = normalize_unicode(text)
    text = strip_html(text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Auto-detect text, label, and topic columns.
    
    Returns
    -------
    tuple
        (text_col, label_col, topic_col) - any can be None if not found
    """
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Detect text column
    text_col = None
    for candidate in ['text', 'content', 'document', 'human_story']:
        if candidate in columns_lower:
            text_col = columns_lower[candidate]
            break
    
    # Detect label column
    label_col = None
    for candidate in ['label', 'class', 'category', 'is_ai', 'is_human']:
        if candidate in columns_lower:
            label_col = columns_lower[candidate]
            break
    
    # Detect topic column
    topic_col = None
    for candidate in ['topic', 'subject', 'domain', 'category', 'prompt']:
        if candidate in columns_lower:
            topic_col = columns_lower[candidate]
            break
    
    return text_col, label_col, topic_col


def load_data(path: Path, 
              text_col: Optional[str] = None,
              label_col: Optional[str] = None, 
              topic_col: Optional[str] = None,
              max_rows: Optional[int] = None,
              clean: bool = True,
              cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and prepare dataset.
    
    Parameters
    ----------
    path : Path
        Path to CSV file
    text_col : str, optional
        Name of text column (auto-detected if None)
    label_col : str, optional
        Name of label column (auto-detected if None)
    topic_col : str, optional
        Name of topic column (auto-detected if None)
    max_rows : int, optional
        Maximum rows to load (for testing)
    clean : bool, default=True
        Apply minimal cleaning to text
    cache_dir : Path, optional
        (Unused) Directory placeholder for compatibility
        
    Returns
    -------
    pd.DataFrame
        Loaded and cleaned dataframe with standardized columns
    """
    logger.info(f"Loading data from {path}")
    
    # Load CSV
    df = pd.read_csv(path, nrows=max_rows)
    logger.info(f"Loaded {len(df)} rows")
    
    # Auto-detect columns if not specified
    if text_col is None or label_col is None or topic_col is None:
        detected_text, detected_label, detected_topic = detect_columns(df)
        
        text_col = text_col or detected_text
        label_col = label_col or detected_label
        topic_col = topic_col or detected_topic
    
    # Validate text column
    if text_col is None:
        raise ValueError(f"Could not detect text column. Available columns: {df.columns.tolist()}")
    
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found. Available: {df.columns.tolist()}")
    
    logger.info(f"Using columns - text: {text_col}, label: {label_col}, topic: {topic_col}")
    

    # If label_col is not provided, and the dataset is wide (multiple text columns for different corpora), automatically melt to long format
    if label_col is None:
        non_text_cols = {'prompt', 'topic', 'subject', 'domain', 'category'}
        text_candidates = [col for col in df.columns if col not in non_text_cols]
        if len(text_candidates) > 1:
            logger.info(f"Detected wide-format dataset with columns: {text_candidates}. Melting to long format.")
            id_vars = [col for col in df.columns if col not in text_candidates]
            # Use a temporary value_name to avoid collision
            df_long = df.melt(id_vars=id_vars, value_vars=text_candidates, var_name='label', value_name='text_value')
            df_long = df_long.dropna(subset=['text_value'])
            df_long = df_long[df_long['text_value'].astype(str).str.strip() != '']
            df_long = df_long.reset_index(drop=True)
            if 'topic' in df_long.columns:
                df_long['topic'] = df_long['topic'].fillna('SOCIETY')
            else:
                logger.warning("No topic column found. Assigning default topic: SOCIETY for all documents")
                df_long['topic'] = 'SOCIETY'
            # Now rename text_value to text
            df_long = df_long.rename(columns={'text_value': 'text'})
            df = df_long
            label_col = 'label'
            text_col = 'text'
        else:
            # Standardize column names if not melting
            df = df.rename(columns={text_col: 'text'})
    else:
        # Standardize column names if label_col is provided
        df = df.rename(columns={text_col: 'text'})
    # Always set initial_len after all conditional logic
    initial_len = len(df)

    df = df.dropna(subset=['text'])
    # Drop empty strings
    df = df[df['text'].astype(str).str.strip() != '']
    
    # Clean text if requested
    if clean:
        logger.info("Applying minimal text cleaning")
        df['text'] = df['text'].apply(minimal_clean)
    
    # Final check: drop any rows that became empty after cleaning
    df = df[df['text'].astype(str).str.strip() != '']
    df = df.reset_index(drop=True)
    
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with empty text")
    
    # Add row IDs if not present
    if 'doc_id' not in df.columns:
        df['doc_id'] = [f"doc_{i:06d}" for i in range(len(df))]
    
    # Ensure topic column exists; if missing, fill with default
    if 'topic' not in df.columns:
        logger.warning("No topic column found. Assigning default topic: SOCIETY for all documents")
        df['topic'] = "SOCIETY"
    else:
        logger.info(f"Using existing topic column. Distribution: {df['topic'].value_counts().to_dict()}")
    
    # Validate topic column
    if df['topic'].isna().any() or (df['topic'] == '').any():
        logger.warning(f"Found {df['topic'].isna().sum()} null/empty topics. Filling with SOCIETY.")
        df['topic'] = df['topic'].fillna('SOCIETY')
        df.loc[df['topic'] == '', 'topic'] = 'SOCIETY'
    
    return df


def create_shards(df: pd.DataFrame, n_shards: int, 
                 balance_by: Optional[List[str]] = None) -> List[pd.DataFrame]:
    """
    Split dataframe into shards for parallel processing.
    
    Implements stratified splitting to ensure balanced representation
    of both labels and topics across all shards.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    n_shards : int
        Number of shards to create
    balance_by : List[str], optional
        List of column names to stratify by (e.g., ['label', 'topic'])
        If None, will use ['label', 'topic'] if both exist
        
    Returns
    -------
    List[pd.DataFrame]
        List of shard dataframes
    """
    if n_shards <= 1:
        return [df]
    
    # Auto-detect balance columns if not specified
    if balance_by is None:
        balance_by = []
        if 'label' in df.columns:
            balance_by.append('label')
        if 'topic' in df.columns:
            balance_by.append('topic')
    
    # Filter to only existing columns
    balance_by = [col for col in balance_by if col in df.columns]
    
    if len(balance_by) == 0:
        # Simple sequential splitting
        logger.info(f"Creating {n_shards} sequential shards (no stratification)")
        shard_size = len(df) // n_shards
        
        shards = []
        for i in range(n_shards):
            start_idx = i * shard_size
            if i == n_shards - 1:
                # Last shard gets remaining rows
                end_idx = len(df)
            else:
                end_idx = (i + 1) * shard_size
            
            shards.append(df.iloc[start_idx:end_idx].copy())
    
    elif len(balance_by) == 1:
        # Single-column stratification
        col = balance_by[0]
        logger.info(f"Creating {n_shards} stratified shards by '{col}'")
        
        groups = df.groupby(col)
        shards = [[] for _ in range(n_shards)]
        
        for group_name, group_df in groups:
            # Distribute this group across shards
            for i in range(n_shards):
                shard_chunk = group_df.iloc[i::n_shards].copy()
                shards[i].append(shard_chunk)
        
        # Concatenate chunks
        shards = [pd.concat(shard_list, ignore_index=True) if shard_list else pd.DataFrame() 
                 for shard_list in shards]
    
    else:
        # Multi-column stratification (label + topic)
        logger.info(f"Creating {n_shards} stratified shards by {balance_by}")
        
        # Create combined grouping key
        grouping_key = df[balance_by].apply(lambda x: '_'.join(map(str, x)), axis=1)
        groups = df.groupby(grouping_key)
        
        shards = [[] for _ in range(n_shards)]
        
        for group_name, group_df in groups:
            # Distribute this group across shards
            for i in range(n_shards):
                shard_chunk = group_df.iloc[i::n_shards].copy()
                if len(shard_chunk) > 0:
                    shards[i].append(shard_chunk)
        
        # Concatenate chunks
        shards = [pd.concat(shard_list, ignore_index=True) if shard_list else pd.DataFrame() 
                 for shard_list in shards]
    
    # Filter out empty shards
    shards = [s for s in shards if len(s) > 0]
    
    logger.info(f"Created {len(shards)} shards with sizes: {[len(s) for s in shards]}")
    
    # Log balance verification
    for i, shard in enumerate(shards):
        if 'label' in shard.columns:
            label_dist = shard['label'].value_counts().to_dict()
            logger.debug(f"Shard {i} label distribution: {label_dist}")
        if 'topic' in shard.columns:
            topic_dist = shard['topic'].value_counts().to_dict()
            logger.debug(f"Shard {i} topic distribution: {topic_dist}")
    
    return shards
