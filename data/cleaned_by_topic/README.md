# Cleaned Dataset by Topic

## Overview

- Original dataset: `data\gsingh1-train\train.csv`
- Original rows: 7,321
- Cleaned rows: 5,577
- Removed: 1,744 rows
  - Missing data: 1,744
  - Timeouts/errors: 1,530
  - Uncategorized: 173

## Topic Distribution

| Topic | Count | File |
|-------|------:|------|
| ARTS | 306 | `arts.csv` |
| BUSINESS | 233 | `business.csv` |
| CRIME | 100 | `crime.csv` |
| EDUCATION | 181 | `education.csv` |
| ENTERTAINMENT | 806 | `entertainment.csv` |
| ENVIRONMENT | 40 | `environment.csv` |
| FASHION | 141 | `fashion.csv` |
| FOOD | 141 | `food.csv` |
| HEALTH | 366 | `health.csv` |
| LOCAL_NEWS | 61 | `local_news.csv` |
| POLITICS | 1,187 | `politics.csv` |
| SCIENCE | 88 | `science.csv` |
| SPORTS | 329 | `sports.csv` |
| TECHNOLOGY | 1,459 | `technology.csv` |
| TRAVEL | 61 | `travel.csv` |
| WORLD_NEWS | 78 | `world_news.csv` |

## Data Columns

Each CSV file contains:
- `prompt`
- `Human_story`
- `gemma-2-9b`
- `mistral-7B`
- `qwen-2-72B`
- `llama-8B`
- `accounts/yi-01-ai/models/yi-large`
- `GPT_4-o`
- `topic`

## Usage

These cleaned datasets are ready for:
- Feature extraction pipelines
- Topic-specific linguistic analysis
- Training/testing splits
- Statistical comparisons
