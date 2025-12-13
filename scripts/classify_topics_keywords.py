#!/usr/bin/env python3
"""
Classify articles into topic categories using a lightweight approach.
Uses spaCy and keyword matching for faster classification.
"""

import pandas as pd
from pathlib import Path
import spacy
from collections import Counter
import re

# Topic keywords for classification
TOPIC_KEYWORDS = {
    "POLITICS": [
        "president", "congress", "senate", "election", "vote", "government", "political",
        "republican", "democrat", "policy", "legislation", "campaign", "governor",
        "mayor", "administration", "white house", "capitol", "politician", "ballot"
    ],
    "SPORTS": [
        "game", "team", "player", "score", "season", "coach", "championship", "league",
        "tournament", "athletic", "stadium", "match", "win", "defeat", "victory"
    ],
    "TECHNOLOGY": [
        "software", "computer", "digital", "app", "tech", "data", "internet", "online",
        "technology", "innovation", "startup", "algorithm", "ai", "robot", "device"
    ],
    "BUSINESS": [
        "company", "market", "stock", "economy", "business", "financial", "investor",
        "profit", "revenue", "trade", "corporate", "industry", "economic", "commerce"
    ],
    "ENTERTAINMENT": [
        "movie", "film", "actor", "actress", "show", "theater", "music", "concert",
        "celebrity", "performance", "entertainment", "stage", "audience", "broadway"
    ],
    "HEALTH": [
        "health", "medical", "doctor", "hospital", "patient", "disease", "treatment",
        "medicine", "healthcare", "virus", "pandemic", "vaccine", "illness", "covid"
    ],
    "SCIENCE": [
        "research", "scientist", "study", "experiment", "discovery", "laboratory",
        "scientific", "theory", "analysis", "evidence", "academic", "university"
    ],
    "ENVIRONMENT": [
        "climate", "environment", "pollution", "nature", "conservation", "wildlife",
        "sustainability", "green", "ecological", "emissions", "carbon", "renewable"
    ],
    "EDUCATION": [
        "school", "student", "teacher", "education", "university", "college", "learning",
        "classroom", "academic", "curriculum", "graduation", "campus", "professor"
    ],
    "CRIME": [
        "police", "crime", "arrest", "investigation", "criminal", "law enforcement",
        "suspect", "victim", "court", "trial", "justice", "prosecutor", "detective"
    ],
    "TRAVEL": [
        "travel", "tourism", "destination", "vacation", "hotel", "flight", "airport",
        "trip", "tourist", "journey", "visit", "explore", "adventure"
    ],
    "FOOD": [
        "food", "restaurant", "chef", "cuisine", "recipe", "meal", "dish", "cooking",
        "dining", "culinary", "ingredient", "flavor", "menu", "wine"
    ],
    "WORLD_NEWS": [
        "international", "global", "foreign", "country", "nation", "world", "embassy",
        "diplomatic", "treaty", "refugee", "migration", "border", "conflict"
    ],
    "LOCAL_NEWS": [
        "local", "community", "neighborhood", "city", "town", "resident", "county",
        "municipal", "downtown", "suburb", "regional"
    ],
    "ARTS": [
        "art", "artist", "museum", "gallery", "exhibition", "painting", "sculpture",
        "creative", "artwork", "culture", "cultural", "design", "aesthetic"
    ]
}

def classify_by_keywords(text: str) -> str:
    """
    Classify text based on keyword matching.
    
    Args:
        text: Article text to classify
        
    Returns:
        Predicted topic category
    """
    text_lower = text.lower()
    
    # Count keyword matches for each topic
    topic_scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        topic_scores[topic] = score
    
    # Get topic with highest score
    if max(topic_scores.values()) == 0:
        # No keywords matched, default to WORLD_NEWS
        return "WORLD_NEWS"
    
    predicted_topic = max(topic_scores, key=topic_scores.get)
    return predicted_topic

def classify_articles(input_csv: str, output_csv: str):
    """
    Classify articles using keyword-based approach.
    Maintains pairing - both human and AI articles in a pair get the same topic.
    
    Args:
        input_csv: Path to input CSV with text column
        output_csv: Path to output CSV with classified topics
    """
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Total articles to classify: {len(df)}")
    print(f"Processing in pairs (human + AI respond to same prompt)...")
    print(f"\nClassifying articles using keyword matching...")
    
    topics = []
    
    # Process in pairs (rows 0-1, 2-3, 4-5, etc.)
    for pair_idx in range(0, len(df), 2):
        # Get both texts in the pair
        human_text = df.iloc[pair_idx]['text']
        ai_text = df.iloc[pair_idx + 1]['text']
        
        # Combine both texts for classification to ensure consistency
        combined_text = human_text + " " + ai_text
        
        # Classify based on combined text
        predicted_topic = classify_by_keywords(combined_text)
        
        # Assign same topic to both articles in the pair
        topics.append(predicted_topic)
        topics.append(predicted_topic)
        
        if (pair_idx + 2) % 10 == 0:
            print(f"  Classified {pair_idx + 2}/{len(df)} articles... (Current pair: {predicted_topic})")
    
    # Update dataframe with classified topics
    df['topic'] = topics
    
    # Save to output
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Classification complete!")
    print(f"✓ Saved to: {output_csv}")
    
    # Print summary
    print(f"\nTopic Distribution:")
    topic_counts = df['topic'].value_counts()
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count}")
    
    # Print distribution by label
    print(f"\nTopic Distribution by Label:")
    for label in ['human', 'ai']:
        label_df = df[df['label'] == label]
        print(f"\n{label.upper()}:")
        label_counts = label_df['topic'].value_counts()
        for topic, count in label_counts.items():
            print(f"  {topic}: {count}")
    
    return df

if __name__ == "__main__":
    input_file = Path("data/50pairs_human_llama8b.csv")
    output_file = Path("data/50pairs_human_llama8b_classified.csv")
    
    classify_articles(str(input_file), str(output_file))
