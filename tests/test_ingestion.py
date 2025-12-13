"""
Tests for ingestion module including topic classification.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.ingest import (
    load_data, 
    create_shards
)


# Zero-shot topic helpers removed; ingestion now requires topic column or fills default


def test_load_data_basic(tmp_path):
    """Test basic data loading without topic classification."""
    # Create temp CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['Sample text one', 'Sample text two'],
        'label': ['human', 'ai']
    })
    df.to_csv(csv_path, index=False)
    
    result = load_data(csv_path)
    
    assert len(result) == 2
    assert 'text' in result.columns
    assert 'label' in result.columns
    assert 'doc_id' in result.columns
    assert 'topic' in result.columns  # Should be added as default
    assert result['topic'].iloc[0] == 'SOCIETY'


def test_load_data_with_existing_topic(tmp_path):
    """Test that existing topic column is preserved."""
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['Education text', 'Health text'],
        'label': ['human', 'ai'],
        'topic': ['EDUCATION', 'HEALTH']
    })
    df.to_csv(csv_path, index=False)
    
    result = load_data(csv_path)
    
    assert len(result) == 2
    assert 'topic' in result.columns
    assert result['topic'].tolist() == ['EDUCATION', 'HEALTH']


def test_load_data_max_rows(tmp_path):
    """Test max_rows parameter."""
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': [f'Text {i}' for i in range(100)],
        'label': ['human'] * 100
    })
    df.to_csv(csv_path, index=False)
    
    result = load_data(csv_path, max_rows=10)
    
    assert len(result) == 10


def test_create_shards_single():
    """Test shard creation with n_shards=1."""
    df = pd.DataFrame({
        'text': ['Text 1', 'Text 2'],
        'label': ['human', 'ai']
    })
    
    shards = create_shards(df, n_shards=1)
    
    assert len(shards) == 1
    assert len(shards[0]) == 2


def test_create_shards_balanced():
    """Test balanced sharding by label and topic."""
    df = pd.DataFrame({
        'text': [f'Text {i}' for i in range(20)],
        'label': ['human', 'ai'] * 10,
        'topic': ['EDUCATION', 'HEALTH'] * 10
    })
    
    shards = create_shards(df, n_shards=2, balance_by=['label', 'topic'])
    
    assert len(shards) == 2
    
    # Each shard should have both labels
    for shard in shards:
        assert 'human' in shard['label'].values
        assert 'ai' in shard['label'].values
        
    # Each shard should have both topics
    for shard in shards:
        assert 'EDUCATION' in shard['topic'].values
        assert 'HEALTH' in shard['topic'].values


def test_create_shards_no_balance():
    """Test sequential sharding without balance."""
    df = pd.DataFrame({
        'text': [f'Text {i}' for i in range(10)],
    })
    
    shards = create_shards(df, n_shards=2, balance_by=[])
    
    assert len(shards) == 2
    # Sequential split should give 5 each
    assert len(shards[0]) == 5
    assert len(shards[1]) == 5


def test_load_data_empty_text_removal(tmp_path):
    """Test that empty texts are removed."""
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['Valid text', 'Another valid', 'Third text'],
        'label': ['human', 'ai', 'human']
    })
    # Add some rows with empty/whitespace text by appending
    empty_df = pd.DataFrame({
        'text': ['', '  ', None],
        'label': ['ai', 'human', 'ai']
    })
    df = pd.concat([df, empty_df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    
    result = load_data(csv_path)
    
    # Should only keep texts with actual content (3 valid ones)
    assert len(result) == 3
    # Check that valid texts are present
    assert 'Valid text' in result['text'].values
    assert 'Another valid' in result['text'].values
    assert 'Third text' in result['text'].values
    # No empty or None values
    assert not result['text'].isna().any()
    assert all(len(t.strip()) > 0 for t in result['text'])


def test_load_data_column_detection(tmp_path):
    """Test auto-detection of column names."""
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'content': ['Text 1', 'Text 2'],  # Should be detected as 'text'
        'class': ['human', 'ai']  # Should be detected as 'label'
    })
    df.to_csv(csv_path, index=False)
    
    result = load_data(csv_path)
    
    assert 'text' in result.columns
    assert 'label' in result.columns
    assert result['text'].tolist() == ['Text 1', 'Text 2']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
