# 🚀 Quick Setup for Cloud Environments

This guide helps you run the AI vs Human text analysis pipeline on Google Colab or Kaggle notebooks.

## 🌐 Google Colab

### Method 1: Automated Setup (Recommended)

```python
# In a Colab cell, run:
!git clone https://github.com/nguyendinhthienloc/AIvsHuman.git
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

# Cell 4: (Optional) Generate plots
# The repository no longer includes an automatic plotting helper. Use the CSV outputs in `results/my_analysis/lexical/` to create figures in a notebook.

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

# 4. Generate plots (optional)
# Use CSVs in `results/analysis/lexical/` to create figures in a notebook.

# 5. Load and explore results in Python
import pandas as pd
import matplotlib.pyplot as plt

# Load features
Moved to docs/cloud_setup.md

Full content copied to `docs/cloud_setup.md` and top-level file replaced with this pointer to keep repository tidy.
print(df.groupby('label')[['mtld', 'nominalization_density', 
