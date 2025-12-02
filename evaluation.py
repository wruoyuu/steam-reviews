import pandas as pd
import numpy as np
from retrieval import TFIDFRetriever, BM25Retriever
import json
from collections import defaultdict

class RetrievalEvaluator:
    def __init__(self, df, retrievers):
        self.df = df
        self.retrievers = retrievers
        
    def create_test_queries(self):
        """Create test queries with manual relevance judgments"""
        # Manually create these based on my dataset
        test_queries = [
            {
                'query': 'performance issues crashes',
                'relevant_docs': [23155,23150]  # Add manually after inspection
            },
            {
                'query': 'great story campaign',
                'relevant_docs': [62192,62300,35190,62145,61720,62030,61684,61898,11010]
            },
            {
                'query': 'multiplayer bugs server',
                'relevant_docs': [62101,29459,29685,21717,21292,19777,31498,56902,40209,20295,29660,20429,39966]
            },
            {
                'query': 'beautiful graphics art style',
                'revelant_docs': [9920,7546,14002,44559,44389,14656,18615,39468,65305,58202,39677,33939,4741,69049]
            },
            {
                'query': 'difficult challenging gameplay',
                'relevant_docs': [62296,16289,5340,40157,34121,4550,65151,32131]
            },
            {
                'query': 'worth the price',
                'relevant_docs': [69066,56866,52158,57586,48064,12009,69083,38225,12261,48301,37301,58988,59232,47841,59428]
            },
            {
                'query': 'bugs glitches broken',
                'relevant_docs': [66754,13352,5453,25819,5161,67311,55539,19445,21134,53634,9567,9486,41886,2690052682,33478,53481]
            },
            {
                'query': 'addictive fun gameplay loop',
                'relevant_docs': [65066,15849,48233,20586,46638,4485,17748,25646,22727,31739,64915,58085,60895,15853,68893,51090,46187]
            },
            {
                'query': 'boring repetitive missions',
                'relevant_docs': [23151]
            },
            {
                'query': 'optimization frame rate',
                'relevant_docs': [52397,22437,24640,51479,55562,44903,63200,23991,14325,22490,49253,21965,45594,21934,21905,62572,55368,22129]
            }
        ]
        return test_queries
    
    def precision_at_k(self, retrieved, relevant, k):
        """Calculate Precision@k"""
        if k == 0 or len(relevant) == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant))
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved, relevant, k):
        """Calculate Recall@k"""
        if len(relevant) == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant))
        return relevant_retrieved / len(relevant)
    
    def average_precision(self, retrieved, relevant):
        """Calculate Average Precision"""
        if len(relevant) == 0:
            return 0.0
        
        precisions = []
        num_relevant = 0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_relevant += 1
                precisions.append(num_relevant / i)
        
        if len(precisions) == 0:
            return 0.0
        
        return sum(precisions) / len(relevant)
    
    def ndcg_at_k(self, retrieved, relevant, k):
        """Calculate NDCG@k"""
        if len(relevant) == 0:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        # IDCG (ideal DCG)
        idcg = 0.0
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retriever(self, retriever, test_queries, k_values=[5, 10, 20]):
        """Evaluate a single retriever"""
        results = {
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'map': []
        }
        
        for query_data in test_queries:
            query = query_data['query']
            relevant = set(query_data['relevant_docs'])
            
            if len(relevant) == 0:
                continue
            
            # Get retrieval results
            search_results = retriever.search(query, top_k=max(k_values))
            retrieved = [doc_id for doc_id, _ in search_results]
            
            # Calculate metrics
            for k in k_values:
                results['precision'][k].append(self.precision_at_k(retrieved, relevant, k))
                results['recall'][k].append(self.recall_at_k(retrieved, relevant, k))
                results['ndcg'][k].append(self.ndcg_at_k(retrieved, relevant, k))
            
            results['map'].append(self.average_precision(retrieved, relevant))
        
        # Average results
        averaged_results = {
            'precision': {k: np.mean(v) for k, v in results['precision'].items()},
            'recall': {k: np.mean(v) for k, v in results['recall'].items()},
            'ndcg': {k: np.mean(v) for k, v in results['ndcg'].items()},
            'map': np.mean(results['map'])
        }
        
        return averaged_results
    
    def compare_retrievers(self, test_queries):
        """Compare all retrievers"""
        comparison = {}
        
        for name, retriever in self.retrievers.items():
            print(f"\nEvaluating {name}...")
            results = self.evaluate_retriever(retriever, test_queries)
            comparison[name] = results
            
            print(f"  MAP: {results['map']:.4f}")
            print(f"  P@10: {results['precision'][10]:.4f}")
            print(f"  NDCG@10: {results['ndcg'][10]:.4f}")
        
        return comparison


class SentimentEvaluator:
    """Evaluate sentiment analysis quality"""
    
    def evaluate_aspect_extraction(self, df, sample_size=500):
        """
        Manually evaluate aspect extraction accuracy
        Returns a template for manual labeling
        """
        # Sample reviews
        sample = df.sample(n=min(sample_size, len(df)))
        
        evaluation_template = []
        
        for idx, row in sample.iterrows():
            item = {
                'review_id': row['review_id'],
                'review_text': row['review_text'],
                'extracted_aspects': {},
                'manual_labels': {}  # To be filled manually
            }
            
            # Add extracted aspects
            aspects = ['graphics', 'performance', 'gameplay', 'story', 'multiplayer', 'price']
            for aspect in aspects:
                mentioned_col = f'{aspect}_mentioned'
                sentiment_col = f'{aspect}_sentiment'
                
                if row[mentioned_col]:
                    item['extracted_aspects'][aspect] = {
                        'sentiment': float(row[sentiment_col])
                    }
            
            evaluation_template.append(item)
        
        # Save template for manual labeling
        with open('data/evaluation/sentiment_evaluation_template.json', 'w') as f:
            json.dump(evaluation_template, f, indent=2)
        
        print(f"Created evaluation template with {len(evaluation_template)} reviews")
        print("Please manually label the aspects in: data/evaluation/sentiment_evaluation_template.json")
        
        return evaluation_template
    
    def calculate_sentiment_accuracy(self, labeled_file):
        """
        Calculate accuracy after manual labeling
        labeled_file: JSON file with manual labels filled in
        """
        with open(labeled_file, 'r') as f:
            labeled_data = json.load(f)
        
        results = {
            'aspect_detection': {},
            'sentiment_classification': {}
        }
        
        aspects = ['graphics', 'performance', 'gameplay', 'story', 'multiplayer', 'price']
        
        for aspect in aspects:
            tp = fp = fn = tn = 0
            sentiment_correct = 0
            sentiment_total = 0
            
            for item in labeled_data:
                extracted = aspect in item['extracted_aspects']
                actual = aspect in item['manual_labels']
                
                if extracted and actual:
                    tp += 1
                    # Check sentiment accuracy
                    extracted_sent = item['extracted_aspects'][aspect]['sentiment']
                    actual_sent = item['manual_labels'][aspect]['sentiment']
                    
                    # Consider correct if both positive, both negative, or both neutral
                    if (extracted_sent > 0.1 and actual_sent > 0.1) or \
                       (extracted_sent < -0.1 and actual_sent < -0.1) or \
                       (abs(extracted_sent) <= 0.1 and abs(actual_sent) <= 0.1):
                        sentiment_correct += 1
                    sentiment_total += 1
                    
                elif extracted and not actual:
                    fp += 1
                elif not extracted and actual:
                    fn += 1
                else:
                    tn += 1
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['aspect_detection'][aspect] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            results['sentiment_classification'][aspect] = {
                'accuracy': sentiment_correct / sentiment_total if sentiment_total > 0 else 0,
                'total_samples': sentiment_total
            }
        
        return results


class UserStudyHelper:
    """Helper for conducting user studies"""
    
    def generate_user_study_tasks(self, df, num_queries=10):
        """Generate tasks for user study"""
        tasks = []
        
        # Sample diverse queries
        sample_queries = [
            "performance issues crashes",
            "great story single player",
            "multiplayer lag server problems",
            "beautiful graphics art style",
            "difficult challenging gameplay",
            "worth the price",
            "bugs glitches broken",
            "addictive fun gameplay loop",
            "boring repetitive missions",
            "optimization frame rate"
        ]
        
        for i, query in enumerate(sample_queries[:num_queries], 1):
            task = {
                'task_id': i,
                'query': query,
                'instructions': f"Find reviews discussing: '{query}'",
                'time_traditional': None,  # User fills in
                'time_with_tool': None,    # User fills in
                'satisfaction_traditional': None,  # 1-5 scale
                'satisfaction_with_tool': None,    # 1-5 scale
                'comments': ""
            }
            tasks.append(task)
        
        with open('data/evaluation/user_study_tasks.json', 'w') as f:
            json.dump(tasks, f, indent=2)
        
        print(f"Created user study tasks: data/evaluation/user_study_tasks.json")
        return tasks
    
    def analyze_user_study_results(self, results_file):
        """Analyze completed user study results"""
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        time_saved = []
        satisfaction_improvement = []
        
        for task in results:
            if task['time_traditional'] and task['time_with_tool']:
                time_saved.append(task['time_traditional'] - task['time_with_tool'])
            
            if task['satisfaction_traditional'] and task['satisfaction_with_tool']:
                satisfaction_improvement.append(
                    task['satisfaction_with_tool'] - task['satisfaction_traditional']
                )
        
        analysis = {
            'avg_time_saved_seconds': np.mean(time_saved) if time_saved else 0,
            'avg_satisfaction_improvement': np.mean(satisfaction_improvement) if satisfaction_improvement else 0,
            'total_participants': len(results),
            'time_saved_data': time_saved,
            'satisfaction_data': satisfaction_improvement
        }
        
        return analysis


def create_test_queries_interactive(df, retriever):
    """
    Interactive tool to help create test queries with relevance judgments
    """
    test_queries = []
    
    print("=== Test Query Creation Tool ===")
    print("This will help you create test queries with relevance judgments\n")
    
    while True:
        query = input("\nEnter a test query (or 'done' to finish): ").strip()
        
        if query.lower() == 'done':
            break
        
        if not query:
            continue
        
        # Search and show results
        print(f"\nSearching for: '{query}'")
        results = retriever.search(query, top_k=20)
        
        print("\nTop 20 results:")
        for i, (doc_id, score) in enumerate(results, 1):
            review = df.iloc[doc_id]
            print(f"\n{i}. [Doc ID: {doc_id}] Score: {score:.4f}")
            print(f"   Game: {review['game_name']}")
            print(f"   Review: {review['review_text'][:200]}...")
        
        # Get relevance judgments
        print("\nEnter relevant document IDs (comma-separated, e.g., '0,3,7'):")
        relevant_input = input("Relevant docs: ").strip()
        
        if relevant_input:
            relevant_docs = [int(x.strip()) for x in relevant_input.split(',')]
        else:
            relevant_docs = []
        
        test_queries.append({
            'query': query,
            'relevant_docs': relevant_docs
        })
        
        print(f"Added query with {len(relevant_docs)} relevant documents")
    
    # Save test queries
    with open('data/evaluation/test_queries.json', 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    print(f"\nSaved {len(test_queries)} test queries to data/evaluation/test_queries.json")
    return test_queries


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/reviews_with_sentiment.csv')
    documents = df['processed_text'].tolist()
    
    # Initialize retrievers
    print("Loading retrievers...")
    tfidf_retriever = TFIDFRetriever()
    tfidf_retriever.fit(documents)
    
    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(documents)
    
    retrievers = {
        'TF-IDF': tfidf_retriever,
        'BM25': bm25_retriever
    }
    
    # Create evaluation directory
    import os
    os.makedirs('data/evaluation', exist_ok=True)
    
    print("\n" + "="*50)
    print("EVALUATION OPTIONS")
    print("="*50)
    print("1. Create test queries interactively")
    print("2. Evaluate retrieval performance")
    print("3. Generate sentiment evaluation template")
    print("4. Generate user study tasks")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        # Interactive query creation
        test_queries = create_test_queries_interactive(df, bm25_retriever)
        
    elif choice == '2':
        # Load existing test queries
        try:
            with open('data/evaluation/test_queries.json', 'r') as f:
                test_queries = json.load(f)
            
            # Evaluate
            evaluator = RetrievalEvaluator(df, retrievers)
            comparison = evaluator.compare_retrievers(test_queries)
            
            # Save results
            with open('data/evaluation/retrieval_results.json', 'w') as f:
                json.dump(comparison, f, indent=2)
            
            print("\nResults saved to: data/evaluation/retrieval_results.json")
            
        except FileNotFoundError:
            print("Error: test_queries.json not found. Please create test queries first (option 1)")
    
    elif choice == '3':
        # Generate sentiment evaluation template
        sentiment_evaluator = SentimentEvaluator()
        sentiment_evaluator.evaluate_aspect_extraction(df, sample_size=500)
        
    elif choice == '4':
        # Generate user study tasks
        user_study = UserStudyHelper()
        user_study.generate_user_study_tasks(df)