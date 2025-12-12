"""
Main pipeline execution script.
"""

import logging
import os
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from .features import compute_basic_metrics
from .lexical_diversity import compute_lexical_diversity
from .syntax_features import compute_clause_density, compute_passive_ratio
from .pos_entropy import compute_pos_ngram_entropy
from .perplexity import PerplexityEvaluator
from .embeddings import EmbeddingAnalyzer
from .discourse_markers import (compute_discourse_distribution, 
                                compute_modal_rate)
from .function_words import compute_entity_specificity
from .nominalization import compute_nominalization_density
from .advanced_plots import (create_violin_boxplot, create_correlation_heatmap, 
                             create_radar_chart, create_pca_scatter, 
                             create_umap_scatter, create_paired_difference_plot)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_pipeline(input_file="data/input.csv", output_dir="results", 
                 disable_heavy=True, enable_embeddings=False, enable_perplexity=False,
                 workers=1, chunk_size=32):
    """
    Run the feature extraction pipeline with batching and parallel processing.
    
    Args:
        input_file: Path to input CSV with 'text' column
        output_dir: Output directory for results
        disable_heavy: If True, disable both embeddings and perplexity (overrides enable flags)
        enable_embeddings: Enable embedding-based metrics (topical drift, s2s similarity)
        enable_perplexity: Enable perplexity-based metrics
        workers: Number of spaCy workers for parallel processing (default: 1)
        chunk_size: Batch size for nlp.pipe processing (default: 32)
    """
    logger.info("Initializing language models...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        logger.warning("en_core_web_lg not found. Downloading...")
        from spacy.cli import download
        download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")

    # Load heavy models based on flags
    ppl_evaluator = None
    emb_analyzer = None
    if not disable_heavy:
        if enable_perplexity:
            logger.info("Loading perplexity evaluator...")
            ppl_evaluator = PerplexityEvaluator(models=['gpt2'])
        if enable_embeddings:
            logger.info("Loading embedding analyzer...")
            emb_analyzer = EmbeddingAnalyzer(model_name='all-MiniLM-L6-v2')
    
    # Mock loading for now if file doesn't exist
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
    else:
        logger.warning(f"Input file {input_file} not found. Creating dummy data.")
        df = pd.DataFrame({'text': ["This is a test sentence.", "Another test sentence."]})

    results = []
    texts = df['text'].tolist()
    
    logger.info(f"Processing {len(texts)} documents with {workers} workers, batch size {chunk_size}...")
    
    # Use nlp.pipe for batched processing
    docs = list(tqdm(
        nlp.pipe(texts, batch_size=chunk_size, n_process=workers),
        total=len(texts),
        desc="Processing"
    ))
    
    logger.info("Extracting features...")
    for idx, (doc, text) in enumerate(tqdm(zip(docs, texts), total=len(texts), desc="Features")):
        if not text or not text.strip():
            continue
        
        # 1. Basic & Lexical (MTLD, HD-D)
        metrics = compute_basic_metrics(doc) 
        
        # 2. Syntax (Dependency Depth, Clause Density, Passive Voice)
        # Note: Dependency Depth is computed inside compute_basic_metrics via pos_tools if updated, 
        # or we need to ensure it's called. Let's check features.py later.
        metrics.update(compute_clause_density(doc, metrics['word_count']))
        metrics['passive_ratio'] = compute_passive_ratio(doc)
        
        # 3. POS Entropy
        pos_seq = [token.pos_ for token in doc]
        metrics['pos_entropy'] = compute_pos_ngram_entropy(pos_seq)
        
        # 4. Perplexity (Gap, Surprisal Variance)
        if not disable_heavy and ppl_evaluator:
            ppl_result = ppl_evaluator.compute_perplexity(text)
            metrics['ppl_mean'] = ppl_result if isinstance(ppl_result, (int, float)) else ppl_result.get('ppl_mean', 0.0)
            metrics['surprisal_variance'] = ppl_evaluator.compute_surprisal_variance(text) if hasattr(ppl_evaluator, 'compute_surprisal_variance') else 0.0
        else:
            metrics['ppl_mean'] = 0.0
            metrics['surprisal_variance'] = 0.0
        
        # 5. Embeddings (Topical Drift, Centroid Distance, Sentence Similarity)
        if not disable_heavy and emb_analyzer:
            metrics['topical_drift'] = emb_analyzer.compute_topical_drift(text)
            metrics['s2s_similarity'] = emb_analyzer.compute_sentence_similarity(text) if hasattr(emb_analyzer, 'compute_sentence_similarity') else 0.0
            metrics['centroid_distance'] = 0.0  # Needs human_centroid reference
        else:
            metrics['topical_drift'] = 0.0
            metrics['s2s_similarity'] = 0.0
            metrics['centroid_distance'] = 0.0
        
        # 6. Discourse (Markers, Modals)
        tokens = [t.text for t in doc]
        metrics.update(compute_discourse_distribution(tokens, metrics['word_count']))
        metrics.update(compute_modal_rate(tokens, metrics['word_count']))
        
        # 7. Entity Specificity
        metrics.update(compute_entity_specificity(doc, metrics['word_count']))

        # 8. Nominalization
        metrics.update(compute_nominalization_density(doc, metrics['word_count']))
        
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    
    # Merge with original labels if available
    if 'label' in df.columns:
        results_df['label'] = df['label']
    elif 'source' in df.columns:
        results_df['label'] = df['source']
    
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "metrics_results.csv"), index=False)
    
    # Generate Visualizations
    logger.info("Generating visualizations...")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Correlation Heatmap (doesn't require labels)
    if len(numeric_cols) > 1:
        create_correlation_heatmap(results_df, numeric_cols, os.path.join(figures_dir, "correlation_heatmap.png"))
    
    if 'label' in results_df.columns and results_df['label'].nunique() > 1:
        label_col = 'label'
        
        # Violin Plots for top metrics
        for metric in numeric_cols[:5]: # Limit to first 5 for demo
            create_violin_boxplot(results_df, metric, label_col, os.path.join(figures_dir, f"violin_{metric}.png"))
            
        # PCA & UMAP
        create_pca_scatter(results_df, label_col, numeric_cols, os.path.join(figures_dir, "pca_scatter.png"))
        create_umap_scatter(results_df, label_col, numeric_cols, os.path.join(figures_dir, "umap_scatter.png"))
        
        # Radar Chart (Human vs AI)
        if results_df[label_col].nunique() == 2:
            means = results_df.groupby(label_col)[numeric_cols].mean()
            # Normalize means for radar chart (0-1)
            normalized_means = (means - means.min()) / (means.max() - means.min())
            # Handle NaN if max == min
            normalized_means = normalized_means.fillna(0.5)
            
            # Select a subset of metrics for readability
            radar_metrics = numeric_cols[:8] 
            create_radar_chart(normalized_means.iloc[0][radar_metrics], 
                               normalized_means.iloc[1][radar_metrics], 
                               radar_metrics, 
                               os.path.join(figures_dir, "radar_chart.png"))
                               
            create_paired_difference_plot(results_df, numeric_cols, label_col, os.path.join(figures_dir, "paired_diff.png"))

    logger.info("Pipeline completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run feature extraction pipeline with batching")
    parser.add_argument('--input', default='data/input.csv', help='Input CSV file path')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--disable-heavy', action='store_true', 
                        help='Disable all heavy models (overrides enable flags)')
    parser.add_argument('--enable-embeddings', action='store_true',
                        help='Enable embedding-based metrics (requires sentence-transformers)')
    parser.add_argument('--enable-perplexity', action='store_true',
                        help='Enable perplexity-based metrics (requires transformers + GPU recommended)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel spaCy workers (default: 1)')
    parser.add_argument('--chunk-size', type=int, default=32,
                        help='Batch size for nlp.pipe processing (default: 32)')
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output, 
                 disable_heavy=args.disable_heavy,
                 enable_embeddings=args.enable_embeddings,
                 enable_perplexity=args.enable_perplexity,
                 workers=args.workers,
                 chunk_size=args.chunk_size)
