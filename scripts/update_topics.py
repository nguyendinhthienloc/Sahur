#!/usr/bin/env python3
"""
Update the existing analysis results with classified topics.
"""

import pandas as pd
from pathlib import Path

# Load the original features (with prompts as topics)
original_features = pd.read_parquet("results/50pairs_analysis/all_features.parquet")

# Load the classified data
classified_data = pd.read_csv("data/50pairs_human_llama8b_classified.csv")

# Merge on text and label to get the correct topic classification
# First, ensure we have matching text
merged = original_features.merge(
    classified_data[['text', 'label', 'topic']], 
    on=['text', 'label'],
    how='left',
    suffixes=('_old', '_new')
)

# Replace the old topic column with the new classified topics
merged['topic'] = merged['topic_new']
merged = merged.drop(columns=['topic_old', 'topic_new'])

# Save updated features
merged.to_parquet("results/50pairs_analysis/all_features.parquet", index=False)
merged.to_csv("results/50pairs_analysis/all_features.csv", index=False)

print("✓ Updated all_features with classified topics")
print(f"\nTopic distribution:")
print(merged['topic'].value_counts())
