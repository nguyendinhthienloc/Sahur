"""
Clean and organize the main training dataset into topic-based shards.

This script:
1. Loads the full training dataset
2. Removes rows with missing or timeout errors
3. Classifies articles into topic categories using keyword matching
4. Splits into topic-based shards (~7k articles per topic group)
5. Saves cleaned datasets for easier future processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter

# Topic classification keywords (same as classify_topics_keywords.py)
TOPIC_KEYWORDS = {
    'POLITICS': ['government', 'election', 'president', 'congress', 'senate', 'parliament', 
                 'vote', 'policy', 'democrat', 'republican', 'political', 'minister', 
                 'legislation', 'campaign', 'ballot', 'governance', 'administration'],
    'SPORTS': ['game', 'player', 'team', 'sport', 'match', 'championship', 'league', 
               'tournament', 'football', 'basketball', 'baseball', 'soccer', 'tennis',
               'olympics', 'athlete', 'coach', 'stadium', 'score', 'season'],
    'TECHNOLOGY': ['technology', 'software', 'computer', 'internet', 'digital', 'tech',
                   'app', 'device', 'ai', 'algorithm', 'data', 'code', 'programming',
                   'startup', 'silicon valley', 'innovation', 'cybersecurity', 'cloud'],
    'BUSINESS': ['business', 'company', 'economy', 'market', 'stock', 'finance', 'trade',
                 'investor', 'profit', 'revenue', 'corporate', 'ceo', 'industry',
                 'economic', 'commercial', 'enterprise', 'venture', 'banking'],
    'ENTERTAINMENT': ['movie', 'film', 'music', 'actor', 'celebrity', 'show', 'concert',
                      'theater', 'entertainment', 'hollywood', 'album', 'song', 'band',
                      'television', 'streaming', 'netflix', 'director', 'performance'],
    'HEALTH': ['health', 'medical', 'doctor', 'hospital', 'patient', 'disease', 'treatment',
               'medicine', 'healthcare', 'clinic', 'therapy', 'diagnosis', 'vaccine',
               'pharmaceutical', 'surgery', 'mental health', 'wellness', 'nutrition'],
    'SCIENCE': ['science', 'research', 'study', 'scientist', 'experiment', 'discovery',
                'physics', 'chemistry', 'biology', 'laboratory', 'nasa', 'space',
                'astronomy', 'genetic', 'quantum', 'scientific', 'molecule'],
    'ENVIRONMENT': ['climate', 'environment', 'pollution', 'sustainability', 'renewable',
                    'carbon', 'energy', 'conservation', 'ecosystem', 'global warming',
                    'greenhouse', 'emissions', 'wildlife', 'biodiversity', 'recycling'],
    'EDUCATION': ['education', 'school', 'student', 'teacher', 'university', 'college',
                  'learning', 'academic', 'curriculum', 'classroom', 'degree', 'campus',
                  'scholarship', 'tuition', 'professor', 'graduation', 'literacy'],
    'CRIME': ['crime', 'police', 'arrest', 'criminal', 'law enforcement', 'investigation',
              'murder', 'theft', 'robbery', 'court', 'justice', 'prison', 'trial',
              'guilty', 'verdict', 'detective', 'violent', 'fraud'],
    'TRAVEL': ['travel', 'tourism', 'tourist', 'vacation', 'hotel', 'flight', 'airport',
               'destination', 'journey', 'trip', 'airline', 'passenger', 'visa',
               'adventure', 'resort', 'cruise', 'traveler', 'sightseeing'],
    'FOOD': ['food', 'restaurant', 'chef', 'cooking', 'recipe', 'cuisine', 'meal',
             'culinary', 'dish', 'flavor', 'ingredient', 'dining', 'taste',
             'menu', 'beverage', 'wine', 'nutrition', 'kitchen'],
    'FASHION': ['fashion', 'style', 'designer', 'clothing', 'model', 'runway', 'brand',
                'wardrobe', 'trend', 'apparel', 'boutique', 'couture', 'textile',
                'accessories', 'luxury', 'garment', 'fashionable'],
    'ARTS': ['art', 'artist', 'painting', 'gallery', 'museum', 'exhibition', 'sculpture',
             'creative', 'artwork', 'culture', 'cultural', 'artistic', 'canvas',
             'masterpiece', 'installation', 'contemporary art', 'classical'],
    'WORLD_NEWS': ['international', 'global', 'world', 'foreign', 'diplomatic', 'embassy',
                   'united nations', 'treaty', 'alliance', 'geopolitical', 'sanctions',
                   'bilateral', 'summit', 'refugee', 'humanitarian'],
    'LOCAL_NEWS': ['local', 'community', 'neighborhood', 'town', 'city council', 'mayor',
                   'municipal', 'county', 'residents', 'regional', 'district', 'suburb']
}

def detect_timeout_or_error(text):
    """Check if text contains timeout errors or is invalid."""
    if pd.isna(text) or text is None:
        return True
    
    text_str = str(text).strip()
    
    # Check for empty or too short
    if len(text_str) < 50:
        return True
    
    # Check for timeout/error patterns
    error_patterns = [
        'timeout', 'error', 'failed', 'exception',
        'timed out', 'connection refused', 'server error',
        'unable to generate', 'generation failed',
        'request failed', 'api error', 'rate limit'
    ]
    
    text_lower = text_str.lower()
    for pattern in error_patterns:
        if pattern in text_lower:
            return True
    
    return False

def classify_by_keywords(text):
    """Classify text into topic category using keyword matching."""
    if pd.isna(text):
        return 'UNCATEGORIZED'
    
    text_lower = str(text).lower()
    
    # Count keyword matches for each topic
    topic_scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        if score > 0:
            topic_scores[topic] = score
    
    # Return topic with highest score
    if topic_scores:
        return max(topic_scores, key=topic_scores.get)
    
    return 'UNCATEGORIZED'

def main():
    print("=" * 80)
    print("DATASET CLEANING AND ORGANIZATION")
    print("=" * 80)
    
    # Load the main dataset
    data_path = Path("data/gsingh1-train/train.csv")
    print(f"\n1. Loading dataset: {data_path}")
    print(f"   File size: {data_path.stat().st_size / (1024*1024):.2f} MB")
    
    df = pd.read_csv(data_path)
    print(f"   Initial rows: {len(df):,}")
    print(f"   Columns: {', '.join(df.columns)}")
    
    # Check for missing values
    print("\n2. Checking data quality...")
    print(f"   Columns to check: Human_story, llama-8B")
    
    initial_count = len(df)
    
    # Remove rows with missing Human_story or llama-8B
    df = df.dropna(subset=['Human_story', 'llama-8B'])
    print(f"   After removing NaN: {len(df):,} rows ({initial_count - len(df):,} removed)")
    
    # Detect and remove timeout/error entries
    print("\n3. Detecting timeout and error entries...")
    human_errors = df['Human_story'].apply(detect_timeout_or_error)
    llama_errors = df['llama-8B'].apply(detect_timeout_or_error)
    
    error_count = (human_errors | llama_errors).sum()
    print(f"   Found {error_count:,} rows with errors/timeouts")
    
    df = df[~(human_errors | llama_errors)]
    print(f"   After cleaning: {len(df):,} rows")
    
    # Classify into topics
    print("\n4. Classifying articles into topics...")
    print("   Using prompt text for classification...")
    
    # Use both prompt and Human_story for better classification
    df['topic'] = df.apply(
        lambda row: classify_by_keywords(str(row['prompt']) + ' ' + str(row['Human_story'])[:500]),
        axis=1
    )
    
    # Show topic distribution before filtering
    print("\n5. Topic distribution (before filtering):")
    topic_counts_all = df['topic'].value_counts()
    for topic, count in topic_counts_all.items():
        print(f"   {topic:20s}: {count:>6,} articles")
    
    # Remove uncategorized articles
    uncategorized_count = (df['topic'] == 'UNCATEGORIZED').sum()
    df = df[df['topic'] != 'UNCATEGORIZED']
    print(f"\n   Removing UNCATEGORIZED: {uncategorized_count:,} rows")
    print(f"   After filtering: {len(df):,} rows")
    
    # Show final topic distribution
    print("\n6. Final topic distribution:")
    topic_counts = df['topic'].value_counts()
    for topic, count in topic_counts.items():
        print(f"   {topic:20s}: {count:>6,} articles")
    
    # Create output directory
    output_dir = Path("data/cleaned_by_topic")
    output_dir.mkdir(exist_ok=True)
    print(f"\n7. Saving cleaned datasets to: {output_dir}/")
    
    # Save overall cleaned dataset
    cleaned_full_path = Path("data/train_cleaned.csv")
    df.to_csv(cleaned_full_path, index=False)
    print(f"   ✓ Full cleaned dataset: {cleaned_full_path}")
    print(f"     Size: {cleaned_full_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Save topic-based shards
    print("\n8. Creating topic-based shards...")
    for topic in sorted(topic_counts.index):
        topic_df = df[df['topic'] == topic]
        topic_file = output_dir / f"{topic.lower()}.csv"
        topic_df.to_csv(topic_file, index=False)
        
        file_size = topic_file.stat().st_size / (1024*1024)
        print(f"   ✓ {topic:20s}: {len(topic_df):>6,} rows ({file_size:>6.2f} MB) -> {topic_file.name}")
    
    # Save summary
    summary_path = output_dir / "README.md"
    with open(summary_path, 'w') as f:
        f.write("# Cleaned Dataset by Topic\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Original dataset: `{data_path}`\n")
        f.write(f"- Original rows: {initial_count:,}\n")
        f.write(f"- Cleaned rows: {len(df):,}\n")
        f.write(f"- Removed: {initial_count - len(df):,} rows\n")
        f.write(f"  - Missing data: {initial_count - len(df.dropna(subset=['Human_story', 'llama-8B'])):,}\n")
        f.write(f"  - Timeouts/errors: {error_count:,}\n")
        f.write(f"  - Uncategorized: {uncategorized_count:,}\n\n")
        f.write("## Topic Distribution\n\n")
        f.write("| Topic | Count | File |\n")
        f.write("|-------|------:|------|\n")
        for topic in sorted(topic_counts.index):
            count = topic_counts[topic]
            f.write(f"| {topic} | {count:,} | `{topic.lower()}.csv` |\n")
        f.write("\n## Data Columns\n\n")
        f.write("Each CSV file contains:\n")
        for col in df.columns:
            f.write(f"- `{col}`\n")
        f.write("\n## Usage\n\n")
        f.write("These cleaned datasets are ready for:\n")
        f.write("- Feature extraction pipelines\n")
        f.write("- Topic-specific linguistic analysis\n")
        f.write("- Training/testing splits\n")
        f.write("- Statistical comparisons\n")
    
    print(f"\n   ✓ Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("CLEANING COMPLETE!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Original:     {initial_count:>10,} rows")
    print(f"  Cleaned:         {initial_count:>10,} rows")
    print(f"  Cleaned:         {len(df):>10,} rows")
    print(f"  Removed:         {initial_count - len(df):>10,} rows ({(initial_count - len(df))/initial_count*100:.1f}%)")
    print(f"    - Missing:     {initial_count - len(df) + error_count + uncategorized_count:>10,}")
    print(f"    - Errors:      {error_count:>10,}")
    print(f"    - Uncategorized: {uncategorized_count:>8,}")
    print(f"  Valid Topics:    {len(topic_counts):>10,}")
    print(f"\nOutput:")
    print(f"  Full dataset: data/train_cleaned.csv")
    print(f"  By topic:     data/cleaned_by_topic/*.csv")
    print()

if __name__ == "__main__":
    main()
