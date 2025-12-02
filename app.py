from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from retrieval import TFIDFRetriever, BM25Retriever
from sentiment_analysis import AspectSentimentAnalyzer, TemporalAnalyzer
import json

app = Flask(__name__)

# Global variables for loaded data and models
df = None
tfidf_retriever = None
bm25_retriever = None
sentiment_analyzer = None

def load_data_and_models():
    """Load data and initialize retrievers"""
    global df, tfidf_retriever, bm25_retriever, sentiment_analyzer
    
    print("Loading data and models...")
    
    # Load data
    df = pd.read_csv('data/processed/reviews_with_sentiment.csv')
    documents = df['processed_text'].tolist()
    
    # Initialize retrievers
    print("Building TF-IDF index...")
    tfidf_retriever = TFIDFRetriever()
    tfidf_retriever.fit(documents)
    
    print("Building BM25 index...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(documents)
    
    # Initialize sentiment analyzer
    sentiment_analyzer = AspectSentimentAnalyzer()
    
    print("All models loaded!")

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint"""
    data = request.get_json()
    query = data.get('query', '')
    algorithm = data.get('algorithm', 'all')
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    results = {}
    
    # Get results from selected algorithm(s)
    if algorithm in ['tfidf', 'all']:
        tfidf_results = tfidf_retriever.search(query, top_k=top_k)
        results['tfidf'] = format_results(tfidf_results, 'tfidf')
    
    if algorithm in ['bm25', 'all']:
        bm25_results = bm25_retriever.search(query, top_k=top_k)
        results['bm25'] = format_results(bm25_results, 'bm25')
    
    return jsonify(results)

def format_results(results, algorithm):
    """Format search results with review details"""
    formatted = []
    
    for doc_id, score in results:
        review = df.iloc[doc_id]
        
        formatted.append({
            'doc_id': int(doc_id),
            'score': float(score),
            'game_name': review['game_name'],
            'review_text': review['review_text'],
            'voted_up': bool(review['voted_up']),
            'playtime_forever': int(review['playtime_forever']),
            'overall_sentiment': float(review['overall_sentiment']) if pd.notna(review['overall_sentiment']) else None,
            'timestamp_created': int(review['timestamp_created'])
        })
    
    return formatted

@app.route('/games', methods=['GET'])
def get_games():
    """Get list of all games"""
    games = df['game_name'].unique().tolist()
    return jsonify({'games': games})

@app.route('/game_stats/<game_name>', methods=['GET'])
def game_stats(game_name):
    """Get statistics for a specific game"""
    game_df = df[df['game_name'] == game_name]
    
    if len(game_df) == 0:
        return jsonify({'error': 'Game not found'}), 404
    
    # Get aspect statistics
    stats = sentiment_analyzer.get_aspect_statistics(game_df)
    
    # Get temporal trends
    temporal_analyzer = TemporalAnalyzer()
    temporal_trends = temporal_analyzer.analyze_temporal_trends(game_df)
    
    # Convert to JSON-serializable format
    temporal_data = []
    for period, row in temporal_trends.iterrows():
        temporal_data.append({
            'period': str(period),
            'sentiment': float(row['overall_sentiment']),
            'count': int(row['review_count'])
        })
    
    # Playtime analysis
    playtime_sentiment = temporal_analyzer.analyze_by_playtime(game_df)
    playtime_data = []
    for category, row in playtime_sentiment.iterrows():
        playtime_data.append({
            'category': str(category),
            'sentiment': float(row['overall_sentiment']),
            'count': int(row['review_count'])
        })
    
    return jsonify({
        'game_name': game_name,
        'total_reviews': len(game_df),
        'average_sentiment': float(game_df['overall_sentiment'].mean()),
        'positive_percentage': float((game_df['voted_up'].sum() / len(game_df)) * 100),
        'aspect_stats': stats,
        'temporal_trends': temporal_data,
        'playtime_sentiment': playtime_data
    })

@app.route('/compare', methods=['POST'])
def compare_algorithms():
    """Compare different algorithms on the same query"""
    data = request.get_json()
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Get results from all algorithms
    tfidf_results = tfidf_retriever.search(query, top_k=top_k)
    bm25_results = bm25_retriever.search(query, top_k=top_k)
    
    # Calculate overlap
    tfidf_docs = set([doc_id for doc_id, _ in tfidf_results])
    bm25_docs = set([doc_id for doc_id, _ in bm25_results])
    
    overlap_tfidf_bm25 = tfidf_docs & bm25_docs
    
    return jsonify({
        'tfidf': format_results(tfidf_results, 'tfidf'),
        'bm25': format_results(bm25_results, 'bm25'),
        'overlap_stats': {
            'tfidf_bm25': len(overlap_tfidf_bm25)
        }
    })

if __name__ == '__main__':
    load_data_and_models()
    app.run(debug=True, host='0.0.0.0', port=5000)