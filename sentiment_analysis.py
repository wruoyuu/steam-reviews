import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from collections import defaultdict
import nltk
from nltk import pos_tag, word_tokenize

nltk.download('averaged_perceptron_tagger')

class AspectSentimentAnalyzer:
    def __init__(self):
        # Define aspect keywords
        self.aspect_keywords = {
            'graphics': ['graphic', 'visual', 'art', 'style', 'animation', 'render', 'texture', 
                        'fps', 'resolution', 'pixel', 'screen', 'display'],
            'performance': ['performance', 'fps', 'lag', 'stutter', 'crash', 'freeze', 'bug', 
                           'optimize', 'run', 'smooth', 'frame', 'load', 'loading'],
            'gameplay': ['gameplay', 'mechanic', 'control', 'play', 'combat', 'difficulty', 
                        'fun', 'boring', 'repetitive', 'engage', 'addict'],
            'story': ['story', 'plot', 'narrative', 'character', 'dialogue', 'quest', 
                     'mission', 'campaign', 'lore', 'writing'],
            'multiplayer': ['multiplayer', 'online', 'coop', 'co-op', 'pvp', 'server', 
                           'matchmaking', 'team', 'player', 'community'],
            'price': ['price', 'worth', 'money', 'value', 'cost', 'expensive', 'cheap', 
                     'sale', 'dlc', 'microtransaction']
        }
        
    def get_sentiment(self, text):
        """Get overall sentiment using TextBlob"""
        if not text or pd.isna(text):
            return 0.0
        
        blob = TextBlob(str(text))
        return blob.sentiment.polarity  # -1 (negative) to 1 (positive)
    
    def extract_sentences_with_aspect(self, text, aspect):
        """Extract sentences mentioning a specific aspect"""
        if not text or pd.isna(text):
            return []
        
        sentences = re.split(r'[.!?]+', str(text))
        keywords = self.aspect_keywords.get(aspect, [])
        
        matching_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                matching_sentences.append(sentence.strip())
        
        return matching_sentences
    
    def get_aspect_sentiment(self, text, aspect):
        """Get sentiment for a specific aspect"""
        sentences = self.extract_sentences_with_aspect(text, aspect)
        
        if not sentences:
            return None  # Aspect not mentioned
        
        # Average sentiment of all sentences mentioning this aspect
        sentiments = [self.get_sentiment(sent) for sent in sentences]
        return np.mean(sentiments) if sentiments else None
    
    def analyze_review(self, text):
        """Analyze all aspects in a review"""
        results = {
            'overall_sentiment': self.get_sentiment(text)
        }
        
        for aspect in self.aspect_keywords.keys():
            aspect_sentiment = self.get_aspect_sentiment(text, aspect)
            results[f'{aspect}_sentiment'] = aspect_sentiment
            results[f'{aspect}_mentioned'] = aspect_sentiment is not None
        
        return results
    
    def analyze_dataframe(self, df, text_column='review_text'):
        """Analyze entire dataframe"""
        print("Analyzing sentiment...")
        
        # Analyze each review
        sentiment_data = []
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(df)} reviews")
            
            analysis = self.analyze_review(row[text_column])
            sentiment_data.append(analysis)
        
        # Create sentiment dataframe
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Combine with original dataframe
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        print("Sentiment analysis complete!")
        return result_df
    
    def get_aspect_statistics(self, df):
        """Get statistics about aspects"""
        stats = {}
        
        for aspect in self.aspect_keywords.keys():
            mentioned_col = f'{aspect}_mentioned'
            sentiment_col = f'{aspect}_sentiment'
            
            if mentioned_col in df.columns and sentiment_col in df.columns:
                mentioned_count = df[mentioned_col].sum()
                avg_sentiment = df[df[mentioned_col]][sentiment_col].mean()
                
                stats[aspect] = {
                    'mention_count': int(mentioned_count),
                    'mention_percentage': float(mentioned_count / len(df) * 100),
                    'avg_sentiment': float(avg_sentiment) if not pd.isna(avg_sentiment) else None
                }
        
        return stats


class TemporalAnalyzer:
    """Analyze how sentiment changes over time"""
    
    def __init__(self):
        pass
    
    def analyze_temporal_trends(self, df, game_name=None):
        """Analyze sentiment trends over time"""
        # Filter by game if specified
        if game_name:
            df = df[df['game_name'] == game_name].copy()
        
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp_created'], unit='s')
        
        # Group by month
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Calculate monthly average sentiment
        monthly_sentiment = df.groupby('year_month').agg({
            'overall_sentiment': 'mean',
            'review_id': 'count'
        }).rename(columns={'review_id': 'review_count'})
        
        return monthly_sentiment
    
    def analyze_by_playtime(self, df):
        """Analyze sentiment by player experience (playtime)"""
        # Create playtime buckets
        df['playtime_category'] = pd.cut(
            df['playtime_forever'],
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['<10h', '10-50h', '50-100h', '100-500h', '500+h']
        )
        
        # Calculate sentiment by playtime
        playtime_sentiment = df.groupby('playtime_category').agg({
            'overall_sentiment': 'mean',
            'review_id': 'count'
        }).rename(columns={'review_id': 'review_count'})
        
        return playtime_sentiment


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/reviews_processed.csv')
    print(f"Loaded {len(df)} reviews")
    
    # Analyze sentiment
    analyzer = AspectSentimentAnalyzer()
    df_with_sentiment = analyzer.analyze_dataframe(df)
    
    # Save
    df_with_sentiment.to_csv('data/processed/reviews_with_sentiment.csv', index=False)
    print("Saved sentiment analysis results")
    
    # Print statistics
    print("\n=== Aspect Statistics ===")
    stats = analyzer.get_aspect_statistics(df_with_sentiment)
    for aspect, data in stats.items():
        print(f"\n{aspect.upper()}:")
        print(f"  Mentioned in: {data['mention_percentage']:.1f}% of reviews")
        if data['avg_sentiment'] is not None:
            sentiment_label = "Positive" if data['avg_sentiment'] > 0 else "Negative"
            print(f"  Average sentiment: {data['avg_sentiment']:.3f} ({sentiment_label})")
    
    # Temporal analysis example
    temporal_analyzer = TemporalAnalyzer()
    
    # Get unique games
    games = df_with_sentiment['game_name'].unique()
    if len(games) > 0:
        example_game = games[0]
        print(f"\n=== Temporal Analysis for {example_game} ===")
        temporal_trends = temporal_analyzer.analyze_temporal_trends(
            df_with_sentiment, 
            game_name=example_game
        )
        print(temporal_trends.head(10))
    
    # Playtime analysis
    print("\n=== Sentiment by Playtime ===")
    playtime_sentiment = temporal_analyzer.analyze_by_playtime(df_with_sentiment)
    print(playtime_sentiment)