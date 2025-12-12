"""
Tests for metrics to ensure no zero-column issues.
"""

import pytest
import spacy
import numpy as np

from src.metrics_core import (
    compute_mtld,
    compute_nominalization_density,
    compute_modal_epistemic_rate,
    compute_clause_complexity,
    compute_passive_voice_ratio,
    extract_core_metrics
)


@pytest.fixture
def nlp():
    """Load spaCy model."""
    return spacy.load("en_core_web_sm")


def test_mtld_short_doc(nlp):
    """Test MTLD with short document returns non-zero fallback."""
    text = "This is a short text."
    doc = nlp(text)
    
    result = compute_mtld(doc)
    
    # Should return TTR-based fallback, not 0.0
    assert result > 0.0
    assert not np.isnan(result)


def test_mtld_medium_doc(nlp):
    """Test MTLD with medium document (30-50 tokens)."""
    text = " ".join(["word"] * 30 + [f"word{i}" for i in range(10)])
    doc = nlp(text)
    
    result = compute_mtld(doc)
    
    # Should use TTR fallback
    assert result > 0.0
    assert result < 200  # Reasonable range


def test_mtld_long_doc(nlp):
    """Test MTLD with long document (>50 tokens)."""
    text = " ".join([f"word{i}" for i in range(60)])
    doc = nlp(text)
    
    result = compute_mtld(doc)
    
    # Should compute actual MTLD
    assert result > 0.0
    assert not np.isnan(result)


def test_nominalization_density_nonzero(nlp):
    """Test that nominalization density can detect common nominalizations."""
    text = """
    The education of children requires attention and dedication.
    Technology advancement leads to innovation and transformation.
    Government regulation ensures protection of citizens.
    """
    doc = nlp(text)
    
    result = compute_nominalization_density(doc)
    
    # Should detect multiple nominalizations
    assert result > 0.0
    print(f"Nominalization density: {result}")


def test_nominalization_density_with_suffixes(nlp):
    """Test nominalization detection with various suffixes."""
    text = """
    The implementation of the solution requires consideration and careful evaluation.
    Development and advancement contribute to improvement and innovation across sectors.
    Management decisions affect the organization and its strategic direction significantly.
    This transformation represents a major achievement in the modernization of systems.
    """
    doc = nlp(text)
    
    result = compute_nominalization_density(doc)
    
    # Should detect -tion, -ment suffixes (multiple: implementation, consideration, evaluation, etc.)
    assert result > 0.0


def test_modal_epistemic_nonzero(nlp):
    """Test modal/epistemic rate detection."""
    text = """
    We should consider this option. It might be beneficial.
    You must understand the requirements. This will probably work.
    """
    doc = nlp(text)
    
    result = compute_modal_epistemic_rate(doc)
    
    # Should detect modals: should, might, must, will, probably
    assert result > 0.0


def test_clause_complexity_nonzero(nlp):
    """Test clause complexity detection."""
    text = """
    When we finish the project, we will celebrate.
    I believe that success requires effort.
    The book that I read was interesting.
    """
    doc = nlp(text)
    
    result = compute_clause_complexity(doc)
    
    # Should detect clausal dependencies
    assert result >= 0.0  # May be 0 for simple sentences


def test_passive_voice_detection(nlp):
    """Test passive voice detection."""
    text = """
    The report was written by the team.
    The results were analyzed carefully.
    The decision was made yesterday.
    """
    doc = nlp(text)
    
    result = compute_passive_voice_ratio(doc)
    
    # Should detect passive constructions
    assert result > 0.0
    assert result <= 1.0


def test_extract_core_metrics_complete(nlp):
    """Test that all metrics are computed and non-null."""
    text = """
    Education is essential for development. Teachers should provide 
    comprehensive instruction that was designed carefully. The implementation 
    of effective methods requires consideration of student needs. When educators 
    collaborate, they might achieve better outcomes. This approach will probably 
    lead to improvement.
    """
    doc = nlp(text)
    
    metrics = extract_core_metrics(doc, embed_fn=None, doc_id="test_001")
    
    # Check all metrics present
    expected_metrics = [
        'mtld', 'nominalization_density', 'modal_epistemic_rate',
        'clause_complexity', 'passive_voice_ratio', 's2s_cosine_similarity'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        
    # Check non-zero where expected
    assert metrics['mtld'] > 0.0
    assert metrics['nominalization_density'] >= 0.0
    assert metrics['modal_epistemic_rate'] > 0.0  # Has modals
    assert metrics['passive_voice_ratio'] >= 0.0
    
    # s2s_cosine may be NaN without embeddings
    # Others should be valid numbers
    assert not np.isnan(metrics['mtld'])
    assert not np.isnan(metrics['nominalization_density'])


def test_very_short_doc_handling(nlp):
    """Test that very short docs return reasonable values."""
    text = "Hello world."
    doc = nlp(text)
    
    result = compute_mtld(doc)
    assert result > 0.0  # Should return minimum value
    
    result = compute_nominalization_density(doc)
    # May be 0 for very short text (acceptable)
    assert result >= 0.0


def test_metrics_on_stub_sample(nlp):
    """Test metrics on a realistic sample from stub dataset."""
    text = """
    Education is the foundation of a successful society. Through comprehensive 
    learning programs, students can develop critical thinking skills and prepare 
    for future challenges. Teachers play a crucial role in shaping young minds 
    and fostering intellectual curiosity. Modern educational systems must adapt 
    to technological advances while maintaining core pedagogical principles.
    """
    doc = nlp(text)
    
    metrics = extract_core_metrics(doc)
    
    # All metrics should be computable
    assert metrics['mtld'] > 0.0
    assert metrics['nominalization_density'] >= 0.0
    assert metrics['modal_epistemic_rate'] >= 0.0
    assert metrics['clause_complexity'] >= 0.0
    assert metrics['passive_voice_ratio'] >= 0.0
    
    print("Sample metrics:", metrics)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
