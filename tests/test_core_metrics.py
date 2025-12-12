"""
Unit tests for core metrics module.

Tests the 6 core linguistic metrics on deterministic fixtures.
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
    compute_s2s_cosine_similarity,
    extract_core_metrics
)


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm not available")


@pytest.fixture
def simple_text():
    """Simple text fixture for testing."""
    return "This is a simple test. It contains multiple sentences. We will analyze it carefully."


@pytest.fixture
def complex_text():
    """More complex text with nominalizations, modals, clauses."""
    return (
        "The implementation of the solution requires careful consideration. "
        "We should analyze the normalization process. "
        "Although the system was designed carefully, improvements might be needed. "
        "The evaluation of results can reveal important insights."
    )


@pytest.fixture
def passive_text():
    """Text with passive voice."""
    return (
        "The experiment was conducted by researchers. "
        "Results were analyzed carefully. "
        "The findings are presented in this report."
    )


class TestMTLD:
    """Test lexical diversity (MTLD) computation."""
    
    def test_mtld_simple(self, nlp, simple_text):
        """Test MTLD on simple text."""
        doc = nlp(simple_text)
        mtld = compute_mtld(doc)
        
        # MTLD should be a number (may be 0 for short texts)
        assert isinstance(mtld, (int, float))
        assert mtld >= 0
    
    def test_mtld_short_text(self, nlp):
        """Test MTLD returns 0 for very short texts."""
        doc = nlp("Short text.")
        mtld = compute_mtld(doc)
        
        assert mtld == 0.0
    
    def test_mtld_diverse_text(self, nlp):
        """Test MTLD on text with diverse vocabulary."""
        # Create text with many unique words
        text = " ".join([f"word{i}" for i in range(100)])
        doc = nlp(text)
        mtld = compute_mtld(doc)
        
        # High diversity should yield higher MTLD
        assert mtld > 0


class TestNominalization:
    """Test nominalization density computation."""
    
    def test_nominalization_simple(self, nlp, complex_text):
        """Test nominalization detection."""
        doc = nlp(complex_text)
        density = compute_nominalization_density(doc)
        
        # Should detect nominalizations in complex_text
        assert isinstance(density, (int, float))
        assert density >= 0
    
    def test_nominalization_known_examples(self, nlp):
        """Test with known nominalizations."""
        # Text with clear nominalizations
        text = "The implementation of the solution and the evaluation of results."
        doc = nlp(text)
        density = compute_nominalization_density(doc)
        
        # Should detect implementation, solution, evaluation, results
        assert density > 0
    
    def test_nominalization_empty(self, nlp):
        """Test with text without nominalizations."""
        text = "The cat sat on the mat."
        doc = nlp(text)
        density = compute_nominalization_density(doc)
        
        # May be 0 or very low
        assert isinstance(density, (int, float))


class TestModalEpistemic:
    """Test modal and epistemic marker detection."""
    
    def test_modal_detection(self, nlp):
        """Test modal verb detection."""
        text = "We should analyze this. You might consider it. They could help."
        doc = nlp(text)
        rate = compute_modal_epistemic_rate(doc)
        
        # Should detect should, might, could
        assert rate > 0
    
    def test_epistemic_detection(self, nlp):
        """Test epistemic marker detection."""
        text = "This is probably correct. It certainly seems valid. Maybe it works."
        doc = nlp(text)
        rate = compute_modal_epistemic_rate(doc)
        
        # Should detect probably, certainly, maybe
        assert rate > 0
    
    def test_no_modals(self, nlp):
        """Test text without modals."""
        text = "The cat sat on the mat. It was comfortable."
        doc = nlp(text)
        rate = compute_modal_epistemic_rate(doc)
        
        # Should be 0
        assert rate == 0.0


class TestClauseComplexity:
    """Test clause complexity computation."""
    
    def test_clause_complexity(self, nlp, complex_text):
        """Test clause detection."""
        doc = nlp(complex_text)
        complexity = compute_clause_complexity(doc)
        
        # Should detect clauses
        assert isinstance(complexity, (int, float))
        assert complexity >= 0
    
    def test_simple_sentences(self, nlp):
        """Test simple sentences have lower complexity."""
        text = "The cat sat. The dog barked. The bird flew."
        doc = nlp(text)
        complexity = compute_clause_complexity(doc)
        
        # Simple sentences should have low/zero complexity
        assert complexity >= 0


class TestPassiveVoice:
    """Test passive voice ratio computation."""
    
    def test_passive_detection(self, nlp, passive_text):
        """Test passive voice detection."""
        doc = nlp(passive_text)
        ratio = compute_passive_voice_ratio(doc)
        
        # Should detect passive sentences
        assert 0 <= ratio <= 1
        assert ratio > 0
    
    def test_active_voice(self, nlp):
        """Test active voice text."""
        text = "The researchers conducted experiments. They analyzed results. We present findings."
        doc = nlp(text)
        ratio = compute_passive_voice_ratio(doc)
        
        # Should be 0 for active voice
        assert ratio == 0.0


class TestS2SCosine:
    """Test sentence-to-sentence cosine similarity."""
    
    def test_s2s_no_embeddings(self, nlp, simple_text):
        """Test without embedding function returns NaN."""
        doc = nlp(simple_text)
        similarity = compute_s2s_cosine_similarity(doc, embed_fn=None)
        
        assert np.isnan(similarity)
    
    def test_s2s_with_mock_embeddings(self, nlp, simple_text):
        """Test with mock embedding function."""
        doc = nlp(simple_text)
        
        # Mock embedding function
        def mock_embed_fn(sentences):
            # Return random embeddings
            n = len(sentences)
            return np.random.rand(n, 10)
        
        similarity = compute_s2s_cosine_similarity(doc, embed_fn=mock_embed_fn)
        
        # Should return a valid cosine similarity
        assert isinstance(similarity, (int, float))
        assert -1 <= similarity <= 1


class TestExtractCoreMetrics:
    """Test the main extraction function."""
    
    def test_extract_all_metrics(self, nlp, complex_text):
        """Test extracting all core metrics."""
        doc = nlp(complex_text)
        metrics = extract_core_metrics(doc)
        
        # Check all expected keys present
        expected_keys = {
            'mtld',
            'nominalization_density',
            'modal_epistemic_rate',
            'clause_complexity',
            'passive_voice_ratio',
            's2s_cosine_similarity',
            'word_count'
        }
        
        assert set(metrics.keys()) == expected_keys
        
        # Check all values are numbers
        for key, value in metrics.items():
            assert isinstance(value, (int, float, np.number))
    
    def test_extract_short_text(self, nlp):
        """Test extraction returns NaN for very short text."""
        doc = nlp("Short.")
        metrics = extract_core_metrics(doc)
        
        # Should return NaN for most metrics
        assert np.isnan(metrics['mtld'])
        assert np.isnan(metrics['nominalization_density'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
