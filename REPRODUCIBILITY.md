# Reproducibility Summary

This document summarizes all changes made to ensure the project is reproducible across different environments (local, Google Colab, Kaggle).

## ✅ Changes Made

### 1. Enhanced `.gitignore`

Added comprehensive patterns to exclude heavy files:
- Virtual environments (`.venv/`, `venv/`, `.conda/`)
- Cache directories (`cache/`, `**/cache/`, `**/*_embeds/`)
- Model files (`*.pt`, `*.bin`, `*.h5`, `*.safetensors`)
- Results and embeddings (`results/`, `*.parquet`, `*.npy`)
- Large datasets (`data/gsingh1-train/`, large CSVs)
- Python bytecode (`*.pyc`, `__pycache__/`)
- Jupyter checkpoints (`.ipynb_checkpoints/`)
- Cloud environment folders (`/content/`, `/kaggle/`)

### 2. Cloud Setup Script (`setup_environment.py`)

Automated setup script that:
- Detects environment (local, Colab, Kaggle)
- Clones repository (for cloud environments)
- Installs all dependencies from `requirements.txt`
- Downloads required spaCy models
- Creates necessary directories
- Provides clear next steps

**Usage:**
```python
!python setup_environment.py
```

### 3. Cloud Setup Guide (`CLOUD_SETUP.md`)

Comprehensive documentation for cloud environments:
- Google Colab setup (automated & manual)
- Kaggle Notebooks setup
- Complete workflow examples
- Tips for memory management
- Troubleshooting guide
- Example notebook templates

### 4. Reproducibility Check (`check_reproducibility.py`)

Automated validation tool that checks:
- `.gitignore` has essential patterns
- No hardcoded absolute paths in code
- `requirements.txt` exists and has essential packages
- No manual `sys.path` modifications
- Essential documentation exists
- Large files are not tracked in git

**Usage:**
```bash
python check_reproducibility.py
```

### 5. Data Directory Documentation (`data/README.md`)

Clear instructions for:
- Where to get the full dataset
- How to use your own data
- Required data format
- Testing without full dataset
- Data privacy notes

### 6. Updated Main README

Added section on cloud environments with link to detailed guide.

## 🎯 Key Features for Reproducibility

### ✓ No Hardcoded Paths
All paths use relative references or are configurable via CLI arguments.

### ✓ Isolated Dependencies
Uses `requirements.txt` with pinned versions for consistent environments.

### ✓ Environment Detection
`setup_environment.py` automatically detects and configures for local/Colab/Kaggle.

### ✓ Lightweight Repository
Heavy files (models, results, large datasets) are excluded from git.

### ✓ One-Command Setup
Single command to set up the entire environment in cloud platforms.

### ✓ Clear Documentation
Multiple levels of documentation for different use cases.

## 🚀 Quick Start Guide

### Local Environment

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/AIvsHuman.git
cd AIvsHuman
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# 2. Install and verify
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python check_reproducibility.py

# 3. Run analysis
python -m src.cli run \
  --input data/stub_train.csv \
  --output results/test \
  --shards 2 --workers 2
```

### Google Colab

```python
# Single cell setup
!git clone https://github.com/YOUR_USERNAME/AIvsHuman.git
%cd AIvsHuman
!python setup_environment.py

# Run analysis
!python -m src.cli run \
  --input data/stub_train.csv \
  --output results/colab_run \
  --shards 2 --workers 2
```

### Kaggle

```python
# Single cell setup
!git clone https://github.com/YOUR_USERNAME/AIvsHuman.git
%cd /kaggle/working/AIvsHuman
!python setup_environment.py

# Run analysis with Kaggle dataset
!python -m src.cli run \
  --input /kaggle/input/your-dataset/data.csv \
  --output /kaggle/working/results \
  --shards 4 --workers 2
```

## 📋 Pre-Commit Checklist

Before committing code, run:

```bash
# 1. Check reproducibility
python check_reproducibility.py

# 2. Run tests
pytest tests/ -v

# 3. Verify gitignore
git status  # Should not show cache/, results/, .venv/

# 4. Test minimal install
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
python -m src.cli run --input data/stub_train.csv --output /tmp/test --shards 1
deactivate
rm -rf test_env
```

## 🔧 Troubleshooting

### "Large files in repository" warning

```bash
# Remove from git but keep locally
git rm --cached <file>

# Add to .gitignore
echo "<pattern>" >> .gitignore

# Commit the change
git add .gitignore
git commit -m "Untrack large files"
```

### Hardcoded paths detected

Search and replace with relative paths:
```bash
# Find hardcoded paths
grep -r "C:\\\\" src/
grep -r "/Users/" src/
grep -r "/home/" src/

# Use Path() or relative references instead
from pathlib import Path
data_file = Path("data/myfile.csv")  # ✓ Good
data_file = "C:\\Users\\me\\data.csv"  # ✗ Bad
```

### Environment-specific issues

Use environment detection:
```python
import sys

if 'google.colab' in sys.modules:
    # Colab-specific setup
elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    # Kaggle-specific setup
else:
    # Local setup
```

## 📊 Validation Results

After implementing these changes:

```
🔎 Reproducibility Check
========================

✓ .gitignore looks good
✓ No hardcoded paths found
✓ requirements.txt looks good
✓ No sys.path modifications found
✓ Essential documentation present
✓ setup_environment.py found
✓ No large data files tracked in git

✅ All checks passed (7/7)
✨ Your project is reproducible!
```

## 🎓 Best Practices Implemented

1. **Environment Isolation**: Virtual environments and containerization
2. **Dependency Management**: Pinned versions in requirements.txt
3. **Path Handling**: Relative paths and Path objects
4. **Documentation**: Multi-level docs for different audiences
5. **Validation**: Automated reproducibility checks
6. **Git Hygiene**: Proper .gitignore for ML projects
7. **Cloud Ready**: One-command setup for cloud platforms

## 📚 Additional Resources

- [Main README](README.md) - Project overview and usage
- [CLOUD_SETUP.md](CLOUD_SETUP.md) - Detailed cloud setup guide
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md) - Metric definitions
- [data/README.md](data/README.md) - Data acquisition guide

## 🤝 Contributing

When contributing, ensure:
1. Run `python check_reproducibility.py` before committing
2. No hardcoded paths in your code
3. Update documentation if adding new dependencies
4. Test in at least one cloud environment (Colab or Kaggle)
5. Add new files to `.gitignore` if they're heavy or generated

---

**Last Updated**: December 12, 2025
**Status**: ✅ Fully Reproducible
