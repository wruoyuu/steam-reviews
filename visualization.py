import json
import matplotlib.pyplot as plt
from pathlib import Path

class Visualization:
    def __init__(self, output_dir="data/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_retrieval_comparison(self, results_file):
        """Plot retrieval algorithm comparison"""
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        algorithms = list(results.keys())
        metrics = ['precision', 'recall', 'ndcg']

        # k values (convert string keys to int)
        first_algo = algorithms[0]
        k_values = sorted([int(k) for k in results[first_algo]['precision'].keys()])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for algo in algorithms:
                values = [results[algo][metric][str(k)] for k in k_values]
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

        # MAP plot
        fig, ax = plt.subplots(figsize=(8, 6))
        map_scores = [results[algo]['map'] for algo in algorithms]

        colors = ['#667eea', '#764ba2']
        bars = ax.bar(algorithms, map_scores, color=colors)

        ax.set_ylabel('MAP Score', fontsize=12)
        ax.set_title('Mean Average Precision Comparison', fontsize=14)
        ax.set_ylim(0, max(map_scores) * 1.2)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'map_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'map_comparison.png'}")
        plt.close()


if __name__ == "__main__":
    viz = Visualization("data/visualizations")
    viz.plot_retrieval_comparison("data/evaluation/retrieval_results.json")