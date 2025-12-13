#!/usr/bin/env python3
"""
Run IRAL lexical analysis separately for each topic group.
This avoids bias from topic-specific vocabulary.
"""

import pandas as pd
from pathlib import Path
import subprocess
import sys

def analyze_by_topic(features_file: str, output_base_dir: str):
    """
    Split features by topic and run IRAL lexical analysis on each group.
    
    Args:
        features_file: Path to all_features.parquet or .csv
        output_base_dir: Base directory for topic-specific outputs
    """
    print(f"Loading features from: {features_file}")
    
    # Load features
    if features_file.endswith('.parquet'):
        df = pd.read_parquet(features_file)
    else:
        df = pd.read_csv(features_file)
    
    print(f"Total documents: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get topic distribution
    topic_counts = df['topic'].value_counts()
    print(f"\nTopic distribution:")
    for topic, count in topic_counts.items():
        human_count = len(df[(df['topic'] == topic) & (df['label'] == 'human')])
        ai_count = len(df[(df['topic'] == topic) & (df['label'] == 'ai')])
        print(f"  {topic}: {count} total ({human_count} human, {ai_count} ai)")
    
    # Filter topics with sufficient data (at least 4 documents, 2 per class)
    min_docs_per_class = 2
    valid_topics = []
    
    for topic in df['topic'].unique():
        topic_df = df[df['topic'] == topic]
        human_count = len(topic_df[topic_df['label'] == 'human'])
        ai_count = len(topic_df[topic_df['label'] == 'ai'])
        
        if human_count >= min_docs_per_class and ai_count >= min_docs_per_class:
            valid_topics.append(topic)
    
    print(f"\nTopics with sufficient data for analysis: {len(valid_topics)}")
    print(f"Topics: {', '.join(valid_topics)}")
    
    # Create base output directory
    base_dir = Path(output_base_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each topic
    python_exe = sys.executable
    
    for topic in valid_topics:
        print(f"\n{'='*60}")
        print(f"Processing topic: {topic}")
        print(f"{'='*60}")
        
        # Filter data for this topic
        topic_df = df[df['topic'] == topic]
        
        # Create topic-specific directory
        topic_dir = base_dir / f"topic_{topic.lower()}"
        topic_dir.mkdir(exist_ok=True, parents=True)
        
        # Save topic-specific features
        topic_features_file = topic_dir / "all_features.parquet"
        topic_df.to_parquet(topic_features_file, index=False)
        
        print(f"  Saved {len(topic_df)} documents to {topic_features_file}")
        
        # Run IRAL lexical analysis for this topic
        print(f"  Running IRAL lexical analysis...")
        
        cmd = [
            python_exe,
            "-m", "src.cli",
            "analyze",
            "--input", str(topic_features_file),
            "--output", str(topic_dir),
            "--enable-iral-lexical"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✓ Analysis complete for {topic}")
        else:
            print(f"  ✗ Error analyzing {topic}")
            print(f"    Error: {result.stderr}")
    
    print(f"\n{'='*60}")
    print(f"All topic analyses complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {base_dir}")
    print(f"\nTopic directories created:")
    for topic in valid_topics:
        topic_dir = base_dir / f"topic_{topic.lower()}"
        print(f"  - {topic_dir}")

if __name__ == "__main__":
    # Use the classified features
    features_file = "results/50pairs_analysis/all_features.parquet"
    output_dir = "results/50pairs_by_topic"
    
    analyze_by_topic(features_file, output_dir)
