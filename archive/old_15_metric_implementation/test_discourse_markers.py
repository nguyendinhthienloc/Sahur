import pytest
from src.discourse_markers import compute_discourse_distribution, compute_modal_rate

def test_discourse_distribution_basic():
    tokens = ["however", "the", "cat", "therefore", "sat"]
    res = compute_discourse_distribution([t.lower() for t in tokens], len(tokens))
    # Expect contingency ('therefore') and comparison/contrast ('however')
    assert res.get('discourse_contingency_per1k', 0) > 0
    assert res.get('discourse_comparison_per1k', 0) > 0

def test_discourse_empty():
    res = compute_discourse_distribution([], 0)
    # Should not crash and should return zeros
    for k, v in res.items():
        assert v == pytest.approx(0.0)

def test_modal_rate():
    tokens = ["may", "might", "definitely"]
    res = compute_modal_rate([t.lower() for t in tokens], len(tokens))
    assert res['modal_verbs_per1k'] > 0
    assert res['epistemic_markers_per1k'] >= 0
