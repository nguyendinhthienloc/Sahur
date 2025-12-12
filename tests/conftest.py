import pytest
import numpy as np

# Dummy embedding model that returns deterministic vectors
class DummyEmbedModel:
    def encode(self, texts, **kwargs):
        # If list, return list of vectors; else single vector
        def vec(x):
            # deterministic small vector derived from text length
            l = len(x) if isinstance(x, str) else sum(len(t) for t in x)
            return np.array([1.0, float(l % 5), 0.0, 0.0])
        if isinstance(texts, list):
            return [vec(t) for t in texts]
        return vec(texts)

# Dummy perplexity evaluator (no model load)
class DummyPerplexity:
    def compute_perplexity(self, text, model_name='distilgpt2'):
        # return stable deterministic score
        return 42.0
    def compute_perplexity_gap(self, text):
        return {
            'ppl_mean': 42.0,
            'ppl_std': 0.0,
            'ppl_gap': 0.0,
            'ppl_min': 42.0,
            'ppl_max': 42.0
        }
    def compute_surprisal_variance(self, text, window_size: int = 50):
        return 0.0

@pytest.fixture
def dummy_embed_model():
    return DummyEmbedModel()

@pytest.fixture
def dummy_perplexity():
    return DummyPerplexity()

# Helper: small sample document
@pytest.fixture
def small_doc():
    return "This is a small test document. It has two sentences."
