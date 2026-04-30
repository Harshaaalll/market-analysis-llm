# ============================================================
# README: Market Analysis Using LLMs
# File: README-market-analysis-llm.md
# ============================================================

# Market Analysis Using Large Language Models

> Automated financial news sentiment analysis pipeline. Scrapes 500+ articles, summarises with two-pass BART, classifies sentiment with RoBERTa, and serves real-time scores via Flask API with Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/BART-facebook-FF9D00?style=flat-square)](https://huggingface.co/facebook/bart-large-cnn)
[![HuggingFace](https://img.shields.io/badge/RoBERTa-cardiffnlp-FF9D00?style=flat-square)](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

Financial traders and analysts need to process enormous volumes of news to understand market sentiment. Reading 500 articles manually is impossible. This pipeline automates the full workflow: scraping → summarising → sentiment scoring → serving results via API.

---

## Pipeline

```
News Sources (500+ articles)
         │
         ▼
BeautifulSoup Scraping
  Financial news sites
         │
         ▼
LangChain Text Chunking
  Unstructured data workflow
         │
         ▼
Two-Pass BART Summarisation
  facebook/bart-large-cnn
  4-beam search
  Pass 1: chunk summaries
  Pass 2: summary of summaries
  → 70% length reduction
         │
         ▼
RoBERTa Sentiment
  cardiffnlp/twitter-roberta-base-sentiment
  Outputs: positive / neutral / negative probs
  Compound score = softmax weighted sum
         │
         ▼
SMOTE Balancing (training only)
  Applied AFTER train-test split
  55:46 → balanced training set
         │
         ▼
Flask API + Streamlit Dashboard
  Real-time sentiment scores
  Historical trend visualisation
```

---

## Key Technical Decisions

### Why Two-Pass Summarisation?

BART (facebook/bart-large-cnn) has a **1024-token context limit**. Financial news articles frequently exceed this. Single-pass summarisation on truncated text misses information in the second half of the article.

**Two-pass approach:**
- Pass 1: Split article into 900-token chunks. Summarise each independently.
- Pass 2: Concatenate all chunk summaries. Summarise the concatenated summaries.

```python
from transformers import pipeline

summariser = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # CPU
)

def two_pass_summarise(text: str, target_length: int = 150) -> str:
    tokens = text.split()
    
    # Pass 1: chunk-level summaries
    chunk_size = 900
    chunks = [' '.join(tokens[i:i+chunk_size])
              for i in range(0, len(tokens), chunk_size)]
    
    chunk_summaries = []
    for chunk in chunks:
        summary = summariser(
            chunk,
            max_length=200,
            min_length=50,
            num_beams=4,  # Beam search for quality
            do_sample=False
        )[0]['summary_text']
        chunk_summaries.append(summary)
    
    # Pass 2: summarise the summaries
    combined = ' '.join(chunk_summaries)
    if len(combined.split()) > 900:
        final_summary = summariser(
            combined,
            max_length=target_length,
            min_length=50,
            num_beams=4,
            do_sample=False
        )[0]['summary_text']
    else:
        final_summary = combined
    
    return final_summary
```

### Why RoBERTa compound score over argmax?

A standard sentiment model returns the highest-probability class: positive, neutral, or negative. But for market analysis, you want a **continuous signal** — knowing whether sentiment shifted from 0.6 positive to 0.7 positive over a week is valuable. Argmax loses this.

**Compound score:**
```python
import torch

def compute_compound_score(probs: dict) -> float:
    """
    probs: {'negative': 0.1, 'neutral': 0.3, 'positive': 0.6}
    Returns: continuous score from -1 (most negative) to +1 (most positive)
    """
    return probs['positive'] - probs['negative']
    # Range: -1.0 to +1.0
    # Negative: -1.0 to -0.1
    # Neutral: -0.1 to +0.1
    # Positive: +0.1 to +1.0
```

### Why SMOTE only on training data?

This is the most important data science rule in this project. A common mistake:
```python
# WRONG — leaks synthetic samples into validation
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
```

The correct approach:
```python
# CORRECT — split first, then balance only training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
# X_test remains original, uncontaminated
```

Applying SMOTE before splitting creates synthetic validation samples derived from training data, inflating accuracy metrics. The model appears to generalise better than it actually does.

---

## Project Structure

```
market-analysis-llm/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml          # Sources, model paths, thresholds
├── src/
│   ├── scraper/
│   │   └── news_scraper.py  # BeautifulSoup multi-source scraping
│   ├── summariser/
│   │   └── bart_summariser.py  # Two-pass BART pipeline
│   ├── sentiment/
│   │   ├── roberta_sentiment.py
│   │   └── compound_scorer.py
│   ├── training/
│   │   └── train_classifier.py  # SMOTE + classifier training
│   └── api/
│       ├── flask_api.py         # REST API
│       └── streamlit_app.py     # Dashboard
├── tests/
│   ├── test_scraper.py
│   ├── test_summariser.py
│   └── test_sentiment.py
├── notebooks/
│   ├── 01_scraping_eda.ipynb
│   ├── 02_summarisation_quality.ipynb
│   └── 03_sentiment_analysis.ipynb
└── Dockerfile
```

---

## Requirements

```
transformers==4.26.0
torch==1.13.1
beautifulsoup4==4.12.0
requests==2.28.2
langchain==0.0.150
imbalanced-learn==0.10.1
scikit-learn==1.2.1
pandas==1.5.3
numpy==1.24.2
flask==2.2.3
streamlit==1.20.0
plotly==5.13.1
```

---

## Usage

```bash
# Run the scraping + analysis pipeline
python src/api/flask_api.py

# Access API
curl http://localhost:5000/sentiment?ticker=RELIANCE

# Launch Streamlit dashboard
streamlit run src/api/streamlit_app.py
```

---

*Independent Project · Jun–Jul 2024*
*Harshal Bhambhani · BITS Hyderabad · 2026*