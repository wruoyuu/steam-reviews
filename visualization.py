import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ResultsVisualizer:
    def __init__(self, output_dir='data/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_retrieval_comparison(self, results_file):
        """Plot retrieval algorithm comparison"""
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Prepare data
        algorithms = list(results.keys())
        metrics = ['precision', 'recall', 'ndcg']
        k_values = [5, 10, 20]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for algo in algorithms:
                values = [results[algo][metric][k] for k in k_values]
                ax.plot(k_values, values, marker='o', label=algo, linewidth=2)
            
            ax.set_xlabel('k', fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} Comparison', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'retrieval_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'retrieval_comparison.png'}")
        plt.close()
        
        # Plot MAP
        fig, ax = plt.subplots(figsize=(8, 6))
        map_scores = [results[algo]['map'] for algo in algorithms]
        bars = ax.bar(algorithms, map_scores, color=['#667eea', '#764ba2', '#f093fb'])
        
        ax.set_ylabel('MAP Score', fontsize=12)
        ax.set_title('Mean Average Precision Comparison', fontsize=14)
        ax.set_ylim(0, max(map_scores) * 1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'map_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'map_comparison.png'}")
        plt.close()
    
    def plot_aspect_sentiment(self, df):
        """Plot aspect-based sentiment analysis"""
        aspects = ['graphics', 'performance', 'gameplay', 'story', 'multiplayer', 'price']
        
        # Calculate statistics
        mention_rates = []
        avg_sentiments = []
        
        for aspect in aspects:
            mentioned_col = f'{aspect}_mentioned'
            sentiment_col = f'{aspect}_sentiment'
            
            mention_rate = df[mentioned_col].sum() / len(df) * 100
            avg_sentiment = df[df[mentioned_col]][sentiment_col].mean()
            
            mention_rates.append(mention_rate)
            avg_sentiments.append(avg_sentiment)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mention rates
        bars1 = ax1.barh(aspects, mention_rates, color='#667eea')
        ax1.set_xlabel('Mention Rate (%)', fontsize=12)
        ax1.set_title('Aspect Mention Frequency', fontsize=14)
        
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center', fontsize=10)
        
        # Sentiment scores
        colors = ['#10b981' if s > 0 else '#ef4444' for s in avg_sentiments]
        bars2 = ax2.barh(aspects, avg_sentiments, color=colors)
        ax2.set_xlabel('Average Sentiment', fontsize=12)
        ax2.set_title('Aspect Sentiment Scores', fontsize=14)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left' if width > 0 else 'right',
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'aspect_sentiment.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'aspect_sentiment.png'}")
        plt.close()
    
    def plot_temporal_trends(self, df, game_name):
        """Plot sentiment trends over time for a game"""
        game_df = df[df['game_name'] == game_name].copy()
        game_df['date'] = pd.to_datetime(game_df['timestamp_created'], unit='s')
        game_df['year_month'] = game_df['date'].dt.to_period('M')
        
        monthly = game_df.groupby('year_month').agg({
            'overall_sentiment': 'mean',
            'review_id': 'count'
        })
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Sentiment trend
        x = range(len(monthly))
        ax1.plot(x, monthly['overall_sentiment'], marker='o', linewidth=2, color='#667eea')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Average Sentiment', fontsize=12)
        ax1.set_title(f'Sentiment Trend for {game_name}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(x, 0, monthly['overall_sentiment'],
                         where=(monthly['overall_sentiment'] > 0),
                         color='#10b981', alpha=0.3)
        ax1.fill_between(x, 0, monthly['overall_sentiment'],
                         where=(monthly['overall_sentiment'] < 0),
                         color='#ef4444', alpha=0.3)
        
        # Review volume
        ax2.bar(x, monthly['review_id'], color='#764ba2', alpha=0.7)
        ax2.set_xlabel('Time Period', fontsize=12)
        ax2.set_ylabel('Number of Reviews', fontsize=12)
        ax2.set_title('Review Volume', fontsize=14)
        
        # Set x-axis labels
        labels = [str(period) for period in monthly.index]
        ax2.set_xticks(x[::max(1, len(x)//10)])  # Show ~10 labels
        ax2.set_xticklabels(labels[::max(1, len(labels)//10)], rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'temporal_trend_{game_name.replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        print(f"Saved temporal trend for {game_name}")
        plt.close()
    
    def plot_playtime_sentiment(self, df):
        """Plot sentiment vs playtime"""
        df['playtime_category'] = pd.cut(
            df['playtime_forever'],
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['<10h', '10-50h', '50-100h', '100-500h', '500+h']
        )
        
        playtime_stats = df.groupby('playtime_category').agg({
            'overall_sentiment': 'mean',
            'review_id': 'count'
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(playtime_stats))
        bars = ax.bar(x, playtime_stats['overall_sentiment'],
                     color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe'])
        
        ax.set_xticks(x)
        ax.set_xticklabels(playtime_stats.index)
        ax.set_xlabel('Playtime Category', fontsize=12)
        ax.set_ylabel('Average Sentiment', fontsize=12)
        ax.set_title('Sentiment by Player Experience Level', fontsize=14)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Add count labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = playtime_stats.iloc[i]['review_id']
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}\n(n={count})',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'playtime_sentiment.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'playtime_sentiment.png'}")
        plt.close()


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/reviews_with_sentiment.csv')
    
    visualizer = ResultsVisualizer()
    
    print("Generating visualizations...")
    
    # 1. Retrieval comparison (if evaluation results exist)
    if Path('data/evaluation/retrieval_results.json').exists():
        visualizer.plot_retrieval_comparison('data/evaluation/retrieval_results.json')
    
    # 2. Aspect sentiment
    visualizer.plot_aspect_sentiment(df)
    
    # 3. Temporal trends (for top games)
    games = df['game_name'].value_counts().head(5).index
    for game in games:
        visualizer.plot_temporal_trends(df, game)
    
    # 4. Playtime vs sentiment
    visualizer.plot_playtime_sentiment(df)
    
    print("\nAll visualizations complete!")
    print(f"Saved to: {visualizer.output_dir}")