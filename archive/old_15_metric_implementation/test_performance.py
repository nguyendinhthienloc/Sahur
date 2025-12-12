import pytest
from src.run_pipeline import run_pipeline
import pandas as pd
from pathlib import Path
import time

@pytest.mark.slow
def test_performance_small_batch(tmp_path):
    # Slow test: only run in slow CI or locally
    df = pd.DataFrame({'text': ["This is a test."] * 50, 'label': ['Human']*50})
    inp = tmp_path / "in.csv"
    out = tmp_path / "out"
    inp.write_text(df.to_csv(index=False))
    out.mkdir(exist_ok=True)
    start = time.time()
    # Run in fast mode for CI safety, but we keep this slow marker
    run_pipeline(str(inp), str(out), disable_heavy=True)
    elapsed = time.time() - start
    assert elapsed < 300  # generous limit
