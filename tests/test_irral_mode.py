"""
Unit tests for IRAL lexical explainability module.

Tests Log-Odds, collocations, and frequency extraction.
"""

import pytest
import pandas as pd
import numpy as np
from src.irral_lexical import (
    compute_log_odds,
    extract_token_counts,
    compute_group_log_odds,
    extract_bigram_collocations,
    compute_top_k_frequencies,
    compute_irral_outputs
)


@pytest.fixture
def sample_texts_a():
    """Sample texts for group A."""
    return [
        "The implementation requires careful analysis and evaluation.",
        "We need implementation of new solutions for analysis.",
        "The evaluation process shows implementation success."
    ]


@pytest.fixture
def sample_texts_b():
    """Sample texts for group B."""
    return [
        "I think this works well and seems good to me.",
        "It looks good and I believe it works fine.",
        "This seems to work and I think it is good."
    ]


@pytest.fixture
def sample_df():
    """Sample dataframe for testing."""
    return pd.DataFrame({
        'text': [
            "The implementation requires analysis.",
            "We need new solutions.",
            "I think this works well.",
            "It looks good to me."
        ],
        'label': [0, 0, 1, 1],
        'doc_id': ['doc1', 'doc2', 'doc3', 'doc4']
    })


class TestLogOdds:
    """Test Log-Odds ratio computation."""
    
    def test_log_odds_basic(self):
        """Test basic log-odds calculation."""
        # Word appears more in group A
        lor = compute_log_odds(count_a=10, total_a=100, count_b=2, total_b=100)
        
        # Should be positive (more frequent in A)
        assert lor > 0
    
    def test_log_odds_reverse(self):
        """Test log-odds with reversed counts."""
        # Word appears more in group B
        lor = compute_log_odds(count_a=2, total_a=100, count_b=10, total_b=100)
        
        # Should be negative (more frequent in B)
        assert lor < 0
    
    def test_log_odds_equal(self):
        """Test log-odds with equal frequency."""
        # Word appears equally in both groups
        lor = compute_log_odds(count_a=5, total_a=100, count_b=5, total_b=100)
        
        # Should be close to 0
        assert abs(lor) < 0.5


class TestGroupLogOdds:
    """Test group comparison log-odds."""
    
    def test_compute_group_log_odds(self, sample_texts_a, sample_texts_b):
        """Test log-odds between two groups."""
        df = compute_group_log_odds(
            group_a_texts=sample_texts_a,
            group_b_texts=sample_texts_b,
            min_count=1,
            top_k=50
        )
        
        # Should return dataframe
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check columns
        assert 'token' in df.columns
        assert 'log_odds' in df.columns


class TestComputeIrralOutputs:
    """Test full IRAL output computation."""
    
    def test_compute_full_outputs(self, sample_df, tmp_path):
        """Test computing all IRAL outputs."""
        outputs = compute_irral_outputs(
            df=sample_df,
            group_col='label',
            text_col='text',
            output_dir=tmp_path,
            min_count=1,
            top_k=10
        )
        
        # Should return dictionary of outputs
        assert isinstance(outputs, dict)
        
        # Check expected keys
        assert 'log_odds' in outputs
        
        # Check files were created
        assert (tmp_path / 'log_odds.csv').exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
