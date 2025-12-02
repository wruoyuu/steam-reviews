# steam-reviews
Repository for UIUC's CS 410 class's project.

# Steam Review Search System with Aspect-Based Sentiment Analysis

A comprehensive information retrieval and text mining system for Steam game reviews that compares multiple search algorithms (BM25, TF-IDF, FAISS (removed)) and performs aspect-based sentiment analysis.

## Project Overview

This system helps gamers find relevant information within Steam reviews by:
- Implementing and comparing three retrieval algorithms
- Extracting sentiment toward specific game aspects (graphics, performance, gameplay, etc.)
- Tracking sentiment evolution over time
- Providing an intuitive web interface for search and analysis

## Features

### 1. Multi-Algorithm Search
- **TF-IDF**: Term frequency-inverse document frequency ranking
- **BM25**: Probabilistic ranking with parameter tuning (k1=1.5, b=0.75)
- **FAISS (retracted)**: Semantic search using Sentence-BERT embeddings

### 2. Aspect-Based Sentiment Analysis
Automatically extracts sentiment for:
- Graphics & Visuals
- Performance & Optimization
- Gameplay & Mechanics
- Story & Narrative
- Multiplayer Features
- Price & Value

### 3. Temporal Analysis
- Track sentiment changes over time
- Compare launch reviews vs. post-patch feedback
- Analyze sentiment by player experience level (playtime)

## Installation
```bash
# Clone repository
git clone https://github.com/wruoyuu/steam-reviews.git
cd steam-reviews

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Collection
```bash
python data_collection.py
```

### 2. Preprocessing
```bash
python preprocessing.py
```

### 3. Run Web Application
```bash
python app.py
```
Then open http://[localhost:5000](http://127.0.0.1:5000/) in your browser.

### 4. Run Evaluation
```bash
python evaluation.py
```

### 5. Generate Visualizations
```bash
python visualization.py
```

## Project Structure
```
steam-review-system/
├── data/
│   ├── raw/                    # Raw review data
│   ├── processed/              # Preprocessed data
│   ├── evaluation/             # Evaluation results
│   └── visualizations/         # Generated plots
├── templates/
│   └── index.html             # Web interface
├── data_collection.py         # Steam API data collection
├── preprocessing.py           # Text preprocessing
├── retrieval.py               # BM25, TF-IDF, FAISS implementation
├── sentiment_analysis.py      # Aspect sentiment extraction
├── app.py                     # Flask web application
├── evaluation.py              # Evaluation metrics
├── visualization.py           # Result visualization
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Evaluation Results

### Retrieval Performance
- MAP scores, P@10, NDCG@10 for each algorithm
- See `data/evaluation/retrieval_results.json`

### Sentiment Analysis
- Aspect extraction: Precision, Recall, F1 scores
- Sentiment classification accuracy
- See `data/evaluation/sentiment_results.json`

## Example Queries

- "performance issues RTX 4080"
- "great story campaign single player"
- "multiplayer lag server problems"
- "beautiful graphics art style"
- "worth the price"

## Implementation Details

### BM25 Parameters
- k1: 1.5 (term saturation parameter)
- b: 0.75 (length normalization)

### Preprocessing Pipeline
1. Lowercase conversion
2. URL removal
3. Tokenization
4. Stopword removal (custom list)
5. Porter stemming

### Semantic Search
- Model: all-MiniLM-L6-v2 (Sentence-BERT)
- Embedding dimension: 384
- Similarity: Cosine (via FAISS Inner Product)

## Technologies Used

- **Python 3.8+**
- **Flask**: Web framework
- **FAISS (not used anymore)**: Fast similarity search
- **Sentence-Transformers**: Semantic embeddings
- **Scikit-learn**: TF-IDF implementation
- **NLTK**: Text preprocessing
- **TextBlob**: Sentiment analysis
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

## Future Improvements

- Add more sophisticated aspect extraction (dependency parsing)
- Implement query expansion
- Add more games to dataset
- Support for multilingual reviews
- Real-time updates via Steam API

## Author

Lillian Wang
CS410: Text Information Systems
University of Illinois Urbana-Champaign
Fall 2025