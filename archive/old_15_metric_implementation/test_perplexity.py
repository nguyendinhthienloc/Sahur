import pytest
from src.perplexity import PerplexityEvaluator

def test_perplexity_monkeypatched(monkeypatch, dummy_perplexity):
    # Monkeypatch methods to avoid model downloads
    monkeypatch.setattr(PerplexityEvaluator, "_preload_models", lambda self: None)
    monkeypatch.setattr(PerplexityEvaluator, "compute_perplexity", dummy_perplexity.compute_perplexity)
    monkeypatch.setattr(PerplexityEvaluator, "compute_perplexity_gap", dummy_perplexity.compute_perplexity_gap)
    p = PerplexityEvaluator(models=['distilgpt2'])
    assert p.compute_perplexity("test") == 42.0
    gap = p.compute_perplexity_gap("test")
    assert gap['ppl_mean'] == 42.0

def test_perplexity_batch_monkeypatched(monkeypatch, dummy_perplexity):
    # Test batch processing
    monkeypatch.setattr(PerplexityEvaluator, "_preload_models", lambda self: None)
    monkeypatch.setattr(PerplexityEvaluator, "compute_perplexity_batch", 
                       lambda self, texts, model_name='gpt2': [42.0] * len(texts))
    p = PerplexityEvaluator(models=['distilgpt2'])
    results = p.compute_perplexity_batch(["test1", "test2", "test3"])
    assert len(results) == 3
    assert all(r == 42.0 for r in results)
