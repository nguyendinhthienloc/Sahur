"""
Organize and clean up the codebase structure.

This script:
1. Moves old/test scripts to archive
2. Organizes results by purpose
3. Creates clear directory structure
4. Generates updated documentation
"""

import shutil
from pathlib import Path
import json

def main():
    print("=" * 80)
    print("CODEBASE ORGANIZATION")
    print("=" * 80)
    
    # Define organization structure
    archive_dir = Path("archive")
    scripts_dir = Path("scripts")
    results_dir = Path("results")
    
    # 1. Organize scripts
    print("\n1. Organizing scripts...")
    
    # Scripts to archive (old/test scripts)
    old_scripts = [
        "scripts/run_50pairs.py",  # Superseded by extract_50pairs.py
        "scripts/classify_topics_zeroshot.py",  # Failed approach, superseded by keywords
    ]
    
    archive_scripts = archive_dir / "old_scripts"
    archive_scripts.mkdir(parents=True, exist_ok=True)
    
    for script in old_scripts:
        script_path = Path(script)
        if script_path.exists():
            dest = archive_scripts / script_path.name
            print(f"   Moving: {script} -> {dest}")
            shutil.move(str(script_path), str(dest))
    
    # 2. Organize results directories
    print("\n2. Organizing results...")
    
    # Keep only relevant results, document the rest
    active_results = [
        "results/50pairs_analysis",
        "results/gpt4_100pairs_analysis",
    ]
    
    results_archive = archive_dir / "old_results"
    results_archive.mkdir(parents=True, exist_ok=True)
    
    # Archive stub analysis results (test runs)
    test_results = [
        "results/stub_analysis",
        "results/stub_improved_analysis",
        "results/stub_improved_with_embeddings",
    ]
    
    for result_dir in test_results:
        result_path = Path(result_dir)
        if result_path.exists():
            dest = results_archive / result_path.name
            if not dest.exists():
                print(f"   Archiving: {result_dir} -> {dest}")
                shutil.move(str(result_path), str(dest))
    
    # 3. Create organized directory structure documentation
    print("\n3. Creating directory structure documentation...")
    
    structure_doc = Path("DIRECTORY_STRUCTURE.md")
    with open(structure_doc, 'w') as f:
        f.write("# Directory Structure\n\n")
        f.write("## Active Directories\n\n")
        f.write("### `/scripts/`\n")
        f.write("Active data processing and analysis scripts:\n\n")
        f.write("- `extract_50pairs.py` - Extract sample pairs from training data\n")
        f.write("- `classify_topics_keywords.py` - Keyword-based topic classification\n")
        f.write("- `clean_and_organize_dataset.py` - Clean and split main dataset by topic\n")
        f.write("- `update_topics.py` - Update features with classified topics\n")
        f.write("- `analyze_by_topic.py` - Run topic-specific IRAL analysis\n\n")
        
        f.write("### `/src/`\n")
        f.write("Core library modules:\n\n")
        f.write("- `cli.py` - Command-line interface\n")
        f.write("- `pipeline.py` - Main processing pipeline\n")
        f.write("- `metrics_core.py` - Linguistic feature extraction\n")
        f.write("- `embeddings.py` - Semantic similarity (sentence embeddings)\n")
        f.write("- `iral_lexical.py` - IRAL lexical analysis (log-odds, collocations)\n")
        f.write("- `stats_analysis.py` - Statistical testing\n")
        f.write("- `visualize.py` - Plotting and visualization\n")
        f.write("- `ingest.py` - Data loading and validation\n")
        f.write("- `parse_and_cache.py` - spaCy parsing with caching\n")
        f.write("- `utils.py` - Utility functions\n\n")
        
        f.write("### `/data/`\n")
        f.write("Dataset storage:\n\n")
        f.write("- `gsingh1-train/train.csv` - Main training dataset (907k rows, 153 MB)\n")
        f.write("- `train_cleaned.csv` - Cleaned version (after removing errors/timeouts)\n")
        f.write("- `cleaned_by_topic/` - Topic-specific shards (~7k each)\n")
        f.write("- `50pairs_human_llama8b.csv` - Sample extraction (50 pairs)\n")
        f.write("- `50pairs_human_llama8b_classified.csv` - Sample with topic classification\n")
        f.write("- `gpt4_100pairs.csv` - GPT-4 comparison dataset\n\n")
        
        f.write("### `/results/`\n")
        f.write("Analysis outputs:\n\n")
        f.write("- `50pairs_analysis/` - Complete analysis of 50-pair sample\n")
        f.write("- `50pairs_by_topic/` - Topic-specific analyses\n")
        f.write("- `gpt4_100pairs_analysis/` - GPT-4 comparison results\n\n")
        f.write("Each analysis directory contains:\n")
        f.write("- `all_features.parquet` - Extracted features\n")
        f.write("- `figures/` - Visualization PNG files\n")
        f.write("- `tables/` - Statistical test results\n")
        f.write("- `lexical/` - IRAL lexical analysis (log-odds, collocations)\n")
        f.write("- `cache/` - Cached spaCy parses\n\n")
        
        f.write("### `/config/`\n")
        f.write("Configuration files:\n\n")
        f.write("- `metrics_config.yaml` - Metric computation settings\n")
        f.write("- `metrics_schema.json` - Metric definitions and descriptions\n\n")
        
        f.write("### `/tests/`\n")
        f.write("Unit and integration tests:\n\n")
        f.write("- `test_core_metrics.py` - Metric computation tests\n")
        f.write("- `test_ingestion.py` - Data loading tests\n")
        f.write("- `test_iral_mode.py` - IRAL lexical analysis tests\n")
        f.write("- `test_metrics_nonzero.py` - Validation tests\n")
        f.write("- `conftest.py` - pytest configuration\n\n")
        
        f.write("### `/notebooks/`\n")
        f.write("Jupyter notebooks for exploration and analysis\n\n")
        
        f.write("### `/iral_plots/`\n")
        f.write("IRAL-specific plotting utilities\n\n")
        
        f.write("## Archived Directories\n\n")
        f.write("### `/archive/`\n")
        f.write("Deprecated code and old implementations:\n\n")
        f.write("- `old_15_metric_implementation/` - Original 15-metric system (deprecated)\n")
        f.write("- `old_scripts/` - Superseded scripts\n")
        f.write("- `old_results/` - Test run results\n\n")
        
        f.write("## Key Files\n\n")
        f.write("- `README.md` - Project overview and quick start\n")
        f.write("- `REPRODUCIBILITY.md` - Reproducibility guide\n")
        f.write("- `METRICS_REFERENCE.md` - Metric definitions\n")
        f.write("- `DATA_CLEANING.md` - Data preprocessing documentation\n")
        f.write("- `requirements.txt` - Python dependencies\n")
        f.write("- `setup_environment.py` - Environment setup script\n")
        f.write("- `pytest.ini` - pytest configuration\n\n")
        
        f.write("## Workflow\n\n")
        f.write("### 1. Data Preparation\n")
        f.write("```bash\n")
        f.write("# Clean main dataset and split by topic\n")
        f.write("python scripts/clean_and_organize_dataset.py\n")
        f.write("```\n\n")
        
        f.write("### 2. Feature Extraction\n")
        f.write("```bash\n")
        f.write("# Run full pipeline with embeddings\n")
        f.write("python -m src.cli run --input data/cleaned_by_topic/politics.csv \\\n")
        f.write("                       --output results/politics_analysis \\\n")
        f.write("                       --enable-embeddings \\\n")
        f.write("                       --shards 4 --workers 4\n")
        f.write("```\n\n")
        
        f.write("### 3. Statistical Analysis\n")
        f.write("```bash\n")
        f.write("# Run statistical tests\n")
        f.write("python -m src.cli analyze --input results/politics_analysis/all_features.parquet \\\n")
        f.write("                          --output results/politics_analysis \\\n")
            f.write("                          --enable-iral-lexical\n")
        f.write("```\n\n")
        
        f.write("### 4. Visualization\n")
        f.write("Check `results/*/figures/` for generated plots\n\n")
    
    print(f"   ✓ Created: {structure_doc}")
    
    # 4. Create a cleaned requirements.txt (remove duplicates if any)
    print("\n4. Validating requirements.txt...")
    req_file = Path("requirements.txt")
    if req_file.exists():
        with open(req_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Remove duplicates while preserving order
        seen = set()
        clean_lines = []
        for line in lines:
            pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].lower()
            if pkg_name not in seen:
                seen.add(pkg_name)
                clean_lines.append(line)
        
        print(f"   Original packages: {len(lines)}")
        print(f"   After deduplication: {len(clean_lines)}")
    
    # 5. Create a quick reference guide
    print("\n5. Creating quick reference guide...")
    
    quickref_path = Path("QUICK_REFERENCE.md")
    with open(quickref_path, 'w') as f:
        f.write("# Quick Reference\n\n")
        f.write("## Common Commands\n\n")
        f.write("### Data Preparation\n")
        f.write("```bash\n")
        f.write("# Clean and organize main dataset\n")
        f.write("python scripts/clean_and_organize_dataset.py\n\n")
        f.write("# Extract sample pairs\n")
        f.write("python scripts/extract_50pairs.py\n\n")
        f.write("# Classify topics\n")
        f.write("python scripts/classify_topics_keywords.py\n")
        f.write("```\n\n")
        
        f.write("### Analysis Pipeline\n")
        f.write("```bash\n")
        f.write("# Full pipeline (6 core metrics + embeddings)\n")
        f.write("python -m src.cli run --input INPUT.csv --output OUTPUT_DIR --enable-embeddings\n\n")
        f.write("# With parallelization\n")
        f.write("python -m src.cli run --input INPUT.csv --output OUTPUT_DIR \\\n")
        f.write("                       --enable-embeddings --shards 4 --workers 4\n\n")
        f.write("# Statistical analysis + IRAL lexical\n")
        f.write("python -m src.cli analyze --input OUTPUT_DIR/all_features.parquet \\\n")
        f.write("                          --output OUTPUT_DIR --enable-iral-lexical\n")
        f.write("```\n\n")
        
        f.write("### Testing\n")
        f.write("```bash\n")
        f.write("# Run all tests\n")
        f.write("pytest\n\n")
        f.write("# Run specific test file\n")
        f.write("pytest tests/test_core_metrics.py\n\n")
        f.write("# Check reproducibility\n")
        f.write("python check_reproducibility.py\n")
        f.write("```\n\n")
        
        f.write("## Key Metrics (6 Core Features)\n\n")
        f.write("1. **MTLD** - Measure of Textual Lexical Diversity\n")
        f.write("2. **Nominalization Density** - Abstract noun usage rate\n")
        f.write("3. **Modal/Epistemic Rate** - Hedging and certainty markers\n")
        f.write("4. **Clause Complexity** - Mean clauses per sentence\n")
        f.write("5. **Passive Voice Ratio** - Passive construction frequency\n")
        f.write("6. **S2S Cosine Similarity** - Semantic coherence (sentence embeddings)\n\n")
        
        f.write("## Directory Quick Links\n\n")
        f.write("- **Scripts**: `scripts/` - Data processing and analysis scripts\n")
        f.write("- **Source**: `src/` - Core library modules\n")
        f.write("- **Data**: `data/cleaned_by_topic/` - Topic-specific datasets\n")
        f.write("- **Results**: `results/` - Analysis outputs\n")
        f.write("- **Tests**: `tests/` - Unit and integration tests\n")
        f.write("- **Config**: `config/` - Configuration files\n\n")
        
        f.write("## Output Structure\n\n")
        f.write("After running analysis, each output directory contains:\n\n")
        f.write("```\n")
        f.write("output_dir/\n")
        f.write("├── all_features.parquet      # Extracted features\n")
        f.write("├── all_features.csv           # CSV version\n")
        f.write("├── metrics_schema.json        # Metric definitions\n")
        f.write("├── figures/                   # Visualizations\n")
        f.write("│   ├── violin_plots.png\n")
        f.write("│   ├── radar_chart.png\n")
        f.write("│   ├── keywords_log_odds_positive.png\n")
        f.write("│   └── keywords_log_odds_negative.png\n")
        f.write("├── tables/                    # Statistical results\n")
        f.write("│   └── statistical_tests.csv\n")
        f.write("├── lexical/                   # IRAL lexical analysis\n")
        f.write("│   ├── log_odds.csv\n")
        f.write("│   ├── collocations_group_ai.csv\n")
        f.write("│   ├── collocations_group_human.csv\n")
        f.write("│   ├── top_freq_group_ai.csv\n")
        f.write("│   └── top_freq_group_human.csv\n")
        f.write("└── cache/                     # Cached spaCy parses\n")
        f.write("    └── shard_0/\n")
        f.write("```\n\n")
        
        f.write("## Environment Setup\n\n")
        f.write("```bash\n")
        f.write("# Create virtual environment\n")
        f.write("python -m venv .venv\n\n")
        f.write("# Activate (Windows)\n")
        f.write(".venv\\Scripts\\activate\n\n")
        f.write("# Install dependencies\n")
        f.write("pip install -r requirements.txt\n\n")
        f.write("# Download spaCy model\n")
        f.write("python -m spacy download en_core_web_lg\n")
        f.write("```\n\n")
        
        f.write("## Dataset Info\n\n")
        f.write("- **Main Dataset**: `data/gsingh1-train/train.csv` (~907k rows, 153 MB)\n")
        f.write("- **Cleaned Dataset**: `data/train_cleaned.csv` (after removing errors)\n")
        f.write("- **Topic Shards**: `data/cleaned_by_topic/*.csv` (~7k articles each)\n")
        f.write("- **Models**: Human_story vs llama-8B, gemma-2-9b, mistral-7B, qwen-2-72B, GPT_4-o\n\n")
    
    print(f"   ✓ Created: {quickref_path}")
    
    print("\n" + "=" * 80)
    print("ORGANIZATION COMPLETE!")
    print("=" * 80)
    print("\nCreated documentation:")
    print(f"  - {structure_doc}")
    print(f"  - {quickref_path}")
    print("\nNext steps:")
    print("  1. Run: python scripts/clean_and_organize_dataset.py")
    print("  2. Check: data/cleaned_by_topic/ for topic-specific datasets")
    print("  3. Review: DIRECTORY_STRUCTURE.md and QUICK_REFERENCE.md")
    print()

if __name__ == "__main__":
    main()
