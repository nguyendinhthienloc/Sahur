"""
Data ingestion module for the 6-metric baseline pipeline.

Handles:
- Loading train.csv and other CSV datasets
- Column normalization (auto-detect common names)
- Minimal text cleaning (unicode normalize, strip HTML)
- Shard creation for parallel processing
- Auto topic classification (zero-shot)
"""

import pandas as pd
import unicodedata
import re
import json
from typing import Optional, List, Tuple
from pathlib import Path
import logging
from transformers import pipeline

from .utils import setup_logger

logger = setup_logger(__name__)

# Hardcoded topic list for zero-shot classification
TOPIC_LABELS = [
    "EDUCATION",
    "TECHNOLOGY",
    "ENVIRONMENT",
    "HEALTH",
    "SOCIETY",
    "ECONOMY",
    "POLITICS",
    "CULTURE"
]


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


def assign_topic(text: str, classifier=None) -> str:
    """
    Assign a topic to text using zero-shot classification.
    
    Parameters
    ----------
    text : str
        Text to classify
    classifier : transformers.Pipeline, optional
        Pre-loaded zero-shot classifier (to avoid reloading)
        
    Returns
    -------
    str
        One of the 8 hardcoded topics, or "SOCIETY" as fallback
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        logger.warning(f"Very short text for topic classification, using fallback: SOCIETY")
        return "SOCIETY"
    
    try:
        # Truncate long texts to first 512 characters for speed
        text_truncated = text[:512]
        
        result = classifier(text_truncated, TOPIC_LABELS)
        
        # Get highest scoring label
        if result and 'labels' in result and len(result['labels']) > 0:
            topic = result['labels'][0]
            return topic
        else:
            logger.warning("Zero-shot classification returned no labels, using fallback: SOCIETY")
            return "SOCIETY"
            
    except Exception as e:
        logger.warning(f"Topic classification failed: {e}, using fallback: SOCIETY")
        return "SOCIETY"


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
              auto_topic: bool = True,
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
    auto_topic : bool, default=True
        Automatically classify topics using zero-shot if no topic column exists
    cache_dir : Path, optional
        Directory to cache topic predictions
        
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
    
    # Standardize column names
    df = df.rename(columns={text_col: 'text'})
    if label_col and label_col in df.columns:
        df = df.rename(columns={label_col: 'label'})
    if topic_col and topic_col in df.columns:
        df = df.rename(columns={topic_col: 'topic'})
    
    # Drop rows with missing or empty text BEFORE cleaning
    initial_len = len(df)
    # Drop NaN/None values first
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
    
    # Auto topic classification if needed
    if auto_topic and 'topic' not in df.columns:
        logger.info("No topic column found. Auto-classifying topics using zero-shot model...")
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.cwd() / 'cache' / 'topics'
        else:
            cache_dir = Path(cache_dir) / 'topics'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load zero-shot classifier
            logger.info("Loading facebook/bart-large-mnli for topic classification...")
            classifier = pipeline("zero-shot-classification", 
                                model="facebook/bart-large-mnli",
                                device=-1)  # CPU
            
            topics = []
            for idx, row in df.iterrows():
                doc_id = row['doc_id']
                text = row['text']
                
                # Check cache
                cache_file = cache_dir / f"{doc_id}.json"
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                        topics.append(cached['topic'])
                else:
                    # Classify
                    topic = assign_topic(text, classifier)
                    topics.append(topic)
                    
                    # Save to cache
                    with open(cache_file, 'w') as f:
                        json.dump({'topic': topic}, f)
                
                # Log progress
                if (idx + 1) % 100 == 0:
                    logger.info(f"Classified {idx + 1}/{len(df)} documents")
            
            df['topic'] = topics
            logger.info(f"Topic classification complete. Distribution: {df['topic'].value_counts().to_dict()}")
            
        except Exception as e:
            logger.error(f"Auto topic classification failed: {e}")
            logger.warning("Falling back to default topic: SOCIETY for all documents")
            df['topic'] = "SOCIETY"
    
    elif 'topic' not in df.columns:
        logger.warning("No topic column and auto_topic=False. Adding default topic: SOCIETY")
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
