import pytest
from src.lexical_diversity import compute_lexical_diversity

def test_mtld_basic():
    tokens = ["the", "cat", "sat", "on", "the", "mat"] * 20
    res = compute_lexical_diversity(tokens)
    assert 'mtld' in res and res['mtld'] > 0
    assert 'hdd' in res and res['hdd'] >= 0

def test_mtld_short_text():
    tokens = ["hello", "world"]
    res = compute_lexical_diversity(tokens)
    assert res['mtld'] == 0.0
    assert res['hdd'] == 0.0 or res['hdd'] >= 0.0
