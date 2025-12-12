import pytest
from src.embeddings import EmbeddingAnalyzer
import numpy as np

def test_topical_drift_with_dummy(monkeypatch, dummy_embed_model):
    # monkeypatch the __init__ to inject dummy model
    def mock_init(self, model_name='all-MiniLM-L6-v2'):
        self.model = dummy_embed_model
    
    monkeypatch.setattr(EmbeddingAnalyzer, "__init__", mock_init)
    analyzer = EmbeddingAnalyzer()
    # identical embeddings -> cosine distance should be 0
    drift = analyzer.compute_topical_drift("Para one.\n\nPara two.")
    assert pytest.approx(0.0, rel=1e-6) == drift

def test_centroid_distance_with_dummy(monkeypatch, dummy_embed_model):
    # monkeypatch the __init__ to inject dummy model
    def mock_init(self, model_name='all-MiniLM-L6-v2'):
        self.model = dummy_embed_model
    
    monkeypatch.setattr(EmbeddingAnalyzer, "__init__", mock_init)
    analyzer = EmbeddingAnalyzer()
    # provide a fake human centroid vector consistent with DummyEmbedModel dim (4)
    centroid = np.array([1.0, 0.0, 0.0, 0.0])
    dist = analyzer.compute_centroid_distance("Test text", human_centroid=centroid)
    assert dist >= 0.0
