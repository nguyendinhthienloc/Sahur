# Data Directory

This directory contains datasets for AI vs Human text classification.

## 📁 Directory Structure

```
data/
├── stub_train.csv              # Small test dataset (included in repo)
├── stub_train_improved.csv     # Improved test dataset (included)
├── gpt4_100pairs.csv           # 100 AI/Human pairs (included)
└── gsingh1-train/              # Large training dataset (NOT in repo)
    └── train.csv               # Download separately
```

## 📥 Getting the Full Dataset

The large training dataset (`gsingh1-train/train.csv`, ~153 MB) is **not included** in this repository to keep it lightweight.

### Option 1: Download from Kaggle

```bash
# If you have kaggle API configured
kaggle datasets download -d username/ai-vs-human-text
unzip ai-vs-human-text.zip -d data/gsingh1-train/
```

### Option 2: Manual Download

1. Download the dataset from [your data source]
2. Place it in `data/gsingh1-train/train.csv`

### Option 3: Use Your Own Data

Place your CSV file anywhere and reference it with `--input`:

```bash
python -m src.cli run \
  --input /path/to/your/data.csv \
  --output results/my_analysis \
  --shards 4 --workers 2
```

## 📋 Required Data Format

Your CSV must have these columns:

```csv
text,label
"This is AI-generated text...",ai
"This is human-written text...",human
```

**Required columns:**
- `text` - The text content to analyze
- `label` - Binary label (`ai`, `human`, or `0`, `1`)

**Optional columns:**
- `topic` - Topic/domain categorization
- `doc_id` - Unique document identifier
- `source` - Source of the text

The pipeline will auto-detect column names even if they differ slightly (e.g., "content", "text_content", "document").

## 🧪 Testing Without Full Dataset

Use the included test files for quick testing:

```bash
# Small test run
python -m src.cli run \
  --input data/stub_train.csv \
  --output results/test \
  --shards 1 --workers 1

# 100 pairs test
python -m src.cli run \
  --input data/gpt4_100pairs.csv \
  --output results/gpt4_test \
  --shards 2 --workers 2
```

## 🔒 Data Privacy

- Large datasets are automatically excluded from git via [.gitignore](../.gitignore)
- Only small test files are tracked in version control
- Your data stays local and is never committed

## 📚 Data Sources

The included test datasets were curated from:
- **stub_train.csv** - Manually created examples for testing
- **gpt4_100pairs.csv** - GPT-4 vs Human writing samples

For the full dataset, please refer to the original source or contact the repository maintainer.
