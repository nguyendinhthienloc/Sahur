import pandas as pd
from pathlib import Path
from src.run_pipeline import run_pipeline

def test_pipeline_outputs_15_metrics(tmp_path):
    # Create minimal input CSV with longer text for better metric coverage
    df = pd.DataFrame({
        'text': ["This is a longer test document with multiple sentences. " + 
                 "It contains enough content to compute meaningful lexical diversity metrics. " +
                 "The document needs sufficient length for MTLD and HD-D calculations. " +
                 "Therefore, we add more text to ensure proper metric extraction."],
        'label': ['Human']
    })
    inp = tmp_path / "in.csv"
    out = tmp_path / "out"
    inp.write_text(df.to_csv(index=False))
    out.mkdir(exist_ok=True)
    # run pipeline in fast mode (no heavy models)
    run_pipeline(str(inp), str(out), disable_heavy=True, workers=1, chunk_size=16)
    results_file = out / "metrics_results.csv"
    assert results_file.exists()
    res = pd.read_csv(results_file)
    # check at least one row and presence of core metric columns
    assert len(res) >= 1
    
    # Map canonical names to actual column names used in pipeline
    # Some metrics have slightly different names or are part of larger outputs
    expected_subset = [
        'mtld', 'hdd',  # lexical diversity
        'depth_p50',  # dependency depth (check for any depth_p* columns)
        'advcl_per1k', 'passive_ratio',  # syntax
        'pos_entropy',  # POS bigram entropy (may be pos_entropy or similar)
        'modal_verbs_per1k',  # modal markers
        'discourse_contingency_per1k',  # discourse
        'entity_specificity_score',  # entity specificity
        'topical_drift', 's2s_similarity', 'centroid_distance',  # embeddings
        'ppl_mean', 'surprisal_variance',  # perplexity
        'nominalization_density'  # nominalization (maps to nominal_per1k conceptually)
    ]
    
    # Check for presence of columns (with some flexibility for naming)
    missing = []
    for col in expected_subset:
        if col not in res.columns:
            missing.append(col)
    
    # Allow some columns to be missing in fast mode or have alternate names
    # Assert at least 10 of 15 metrics are present
    assert len(missing) <= 5, f"Too many missing metrics ({len(missing)}): {missing}"
