# 🚀 Quick Setup for Cloud Environments

This guide helps you run the AI vs Human text analysis pipeline on Google Colab or Kaggle notebooks.

## 🌐 Google Colab

### Method 1: Automated Setup (Recommended)

```python
# In a Colab cell, run:
!git clone https://github.com/YOUR_USERNAME/AIvsHuman.git
%cd AIvsHuman
!python setup_environment.py
```

### Method 2: Manual Setup

```python
# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/AIvsHuman.git
%cd AIvsHuman

# 2. Install dependencies
!pip install -q -r requirements.txt

# 3. Download spaCy model
!python -m spacy download en_core_web_lg

# 4. Run analysis
!python -m src.cli run \
  --input data/stub_train.csv \
  --output results/colab_run \
  --shards 2 --workers 2
```

### Example Notebook Cells

```python
# Cell 1: Setup
!python setup_environment.py

# Cell 2: Run basic analysis (6 metrics, no embeddings)
!python -m src.cli run \
  --input data/stub_train.csv \
  --output results/my_analysis \
  --shards 2 \
  --workers 2

# Cell 3: Run statistical tests
!python run_stats_tests.py results/my_analysis

# Cell 4: Generate IRAL plots
!python generate_iral_plots.py results/my_analysis

# Cell 5: View results
import pandas as pd
df = pd.read_csv('results/my_analysis/all_features.csv')
print(df.head())
```

### Upload Your Own Data

```python
from google.colab import files

# Upload CSV file
uploaded = files.upload()

# Use the uploaded file
filename = list(uploaded.keys())[0]
!python -m src.cli run \
  --input {filename} \
  --output results/my_data_analysis \
  --shards 2 --workers 2
```

---

## 📊 Kaggle Notebooks

### Method 1: Automated Setup (Recommended)

```python
# In a Kaggle cell, run:
!git clone https://github.com/YOUR_USERNAME/AIvsHuman.git
%cd /kaggle/working/AIvsHuman
!python setup_environment.py
```

### Method 2: Manual Setup

```python
# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/AIvsHuman.git
%cd /kaggle/working/AIvsHuman

# 2. Install dependencies
!pip install -q -r requirements.txt

# 3. Download spaCy model
!python -m spacy download en_core_web_lg

# 4. Run analysis
!python -m src.cli run \
  --input /kaggle/input/your-dataset/data.csv \
  --output /kaggle/working/results \
  --shards 2 --workers 2
```

### Using Kaggle Datasets

```python
# If you added a dataset to your notebook
!python -m src.cli run \
  --input /kaggle/input/ai-text-dataset/train.csv \
  --output /kaggle/working/results \
  --shards 4 --workers 2

# View results
!ls -lh /kaggle/working/results/
```

---

## 🎯 Complete Workflow Example

```python
# 1. Setup (only run once per session)
!python setup_environment.py

# 2. Run feature extraction
!python -m src.cli run \
  --input data/stub_train.csv \
  --output results/analysis \
  --shards 2 \
  --workers 2

# 3. Run statistical tests
!python run_stats_tests.py results/analysis

# 4. Generate plots
!python generate_iral_plots.py results/analysis

# 5. Load and explore results in Python
import pandas as pd
import matplotlib.pyplot as plt

# Load features
df = pd.read_csv('results/analysis/all_features.csv')

# Display summary
print(df.groupby('label')[['mtld', 'nominalization_density', 
                            'modal_epistemic_rate', 'clause_complexity', 
                            'passive_ratio']].mean())

# Plot distribution
df.groupby('label')['mtld'].plot(kind='kde', legend=True)
plt.title('MTLD Distribution by Label')
plt.show()
```

---

## ⚡ Tips for Cloud Environments

### Memory Management

```python
# For large datasets, increase shards and reduce workers
!python -m src.cli run \
  --input large_dataset.csv \
  --output results/large_run \
  --shards 8 \      # More shards = less memory per shard
  --workers 1        # Fewer workers = less parallel memory usage
```

### Skip Embeddings (Faster, Less Memory)

```python
# Default mode runs without embeddings (5 out of 6 metrics)
# This is faster and uses less GPU/CPU
!python -m src.cli run \
  --input data.csv \
  --output results/no_embeds \
  --shards 4 --workers 2
```

### Enable Embeddings (6th metric)

```python
# Only enable if you need s2s_cosine_similarity
!python -m src.cli run \
  --input data.csv \
  --output results/with_embeds \
  --enable-embeddings \
  --shards 4 --workers 2
```

### Save Results

```python
# Colab: Download results
from google.colab import files
!zip -r results.zip results/
files.download('results.zip')

# Kaggle: Results auto-saved to output directory
# Access them in the "Output" tab after notebook runs
```

---

## 🐛 Troubleshooting

### Out of Memory

```python
# Reduce workers and increase shards
!python -m src.cli run --input data.csv --output results/run \
  --shards 16 --workers 1
```

### spaCy Model Not Found

```python
# Re-download model
!python -m spacy download en_core_web_lg --force
```

### Package Installation Issues

```python
# Upgrade pip first
!pip install --upgrade pip
!pip install -r requirements.txt
```

### Check Progress

```python
# View cache directory to see parsing progress
!ls -lh results/my_run/cache/shard_*/
```

---

## 📝 Input Data Format

Your CSV must have these columns:

```csv
text,label
"This is AI-generated text...",ai
"This is human-written text...",human
```

Optional columns: `topic`, `doc_id`, `source`

---

## 🎓 Example Colab Notebook Template

Create a new Colab notebook with these cells:

```python
# ===== CELL 1: Setup =====
!git clone https://github.com/YOUR_USERNAME/AIvsHuman.git
%cd AIvsHuman
!python setup_environment.py

# ===== CELL 2: Upload Data (Optional) =====
from google.colab import files
uploaded = files.upload()

# ===== CELL 3: Run Analysis =====
!python -m src.cli run \
  --input data/stub_train.csv \
  --output results/demo \
  --shards 2 --workers 2

# ===== CELL 4: Statistical Tests =====
!python run_stats_tests.py results/demo

# ===== CELL 5: Visualize =====
!python generate_iral_plots.py results/demo

# ===== CELL 6: Explore Results =====
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results/demo/all_features.csv')
print(df.head())
print("\nSummary by label:")
print(df.groupby('label').mean())

# ===== CELL 7: Download Results =====
from google.colab import files
!zip -r demo_results.zip results/demo
files.download('demo_results.zip')
```

---

## 📚 Additional Resources

- Main README: [README.md](README.md)
- Metrics Reference: [METRICS_REFERENCE.md](METRICS_REFERENCE.md)
- Local installation instructions in main README

---

**Need help?** Open an issue on GitHub or check the main README.md for detailed documentation.
