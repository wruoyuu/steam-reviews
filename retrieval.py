# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sentence_transformers import SentenceTransformer
# import faiss
# from collections import defaultdict
# import math

# class TFIDFRetriever:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer(
#             max_features=10000,
#             ngram_range=(1, 2),  # unigrams and bigrams
#             min_df=2,
#             max_df=0.8
#         )
#         self.doc_vectors = None
#         self.documents = None
        
#     def fit(self, documents):
#         """Build TF-IDF index"""
#         print("Building TF-IDF index...")
#         self.documents = documents
#         self.doc_vectors = self.vectorizer.fit_transform(documents)
#         print(f"TF-IDF index built: {self.doc_vectors.shape}")
        
#     def search(self, query, top_k=10):
#         """Search using TF-IDF cosine similarity"""
#         query_vector = self.vectorizer.transform([query])
#         scores = (self.doc_vectors @ query_vector.T).toarray().flatten()
#         top_indices = np.argsort(scores)[::-1][:top_k]
#         return [(idx, scores[idx]) for idx in top_indices]


# class BM25Retriever:
#     def __init__(self, k1=1.5, b=0.75):
#         self.k1 = k1
#         self.b = b
#         self.doc_lengths = []
#         self.avgdl = 0
#         self.doc_freqs = defaultdict(int)
#         self.idf = {}
#         self.doc_term_freqs = []
#         self.documents = None
        
#     def fit(self, documents):
#         """Build BM25 index"""
#         print("Building BM25 index...")
#         self.documents = documents
        
#         # Calculate document frequencies and term frequencies
#         for doc in documents:
#             tokens = doc.split()
#             self.doc_lengths.append(len(tokens))
            
#             term_freq = defaultdict(int)
#             for token in tokens:
#                 term_freq[token] += 1
                
#             self.doc_term_freqs.append(dict(term_freq))
            
#             # Document frequency (how many docs contain this term)
#             for token in set(tokens):
#                 self.doc_freqs[token] += 1
        
#         # Calculate average document length
#         self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        
#         # Calculate IDF for each term
#         N = len(documents)
#         for term, freq in self.doc_freqs.items():
#             self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
        
#         print(f"BM25 index built: {len(documents)} documents, {len(self.idf)} unique terms")
        
#     def search(self, query, top_k=10):
#         """Search using BM25 scoring"""
#         query_tokens = query.split()
#         scores = []
        
#         for doc_id, (doc_tf, doc_len) in enumerate(zip(self.doc_term_freqs, self.doc_lengths)):
#             score = 0.0
            
#             for token in query_tokens:
#                 if token not in doc_tf:
#                     continue
                
#                 tf = doc_tf[token]
#                 idf = self.idf.get(token, 0)
                
#                 # BM25 formula
#                 numerator = tf * (self.k1 + 1)
#                 denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
#                 score += idf * (numerator / denominator)
            
#             scores.append(score)
        
#         # Get top-k
#         scores = np.array(scores)
#         top_indices = np.argsort(scores)[::-1][:top_k]
#         return [(idx, scores[idx]) for idx in top_indices]


# class FAISSRetriever:
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         """
#         Initialize FAISS retriever with sentence transformer
#         all-MiniLM-L6-v2 is fast and good quality
#         """
#         print(f"Loading sentence transformer model: {model_name}")
#         self.model = SentenceTransformer(model_name)
#         self.index = None
#         self.documents = None
#         self.embeddings = None
        
#     def fit(self, documents):
#         """Build FAISS index"""
#         print("Building FAISS index (this may take a few minutes)...")
#         self.documents = documents
        
#         # Generate embeddings
#         self.embeddings = self.model.encode(
#             documents,
#             show_progress_bar=True,
#             batch_size=32
#         )
        
#         # Create FAISS index
#         dimension = self.embeddings.shape[1]
#         self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
#         # Normalize embeddings for cosine similarity
#         faiss.normalize_L2(self.embeddings)
        
#         # Add to index
#         self.index.add(self.embeddings)
        
#         print(f"FAISS index built: {len(documents)} documents, {dimension} dimensions")
        
#     def search(self, query, top_k=10):
#         """Search using semantic similarity"""
#         # Encode query
#         query_embedding = self.model.encode([query])
#         faiss.normalize_L2(query_embedding)
        
#         # Search
#         scores, indices = self.index.search(query_embedding, top_k)
        
#         return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]


# class HybridRetriever:
#     """Combine multiple retrievers"""
#     def __init__(self, retrievers, weights=None):
#         self.retrievers = retrievers
#         self.weights = weights or [1.0] * len(retrievers)
        
#     def search(self, query, top_k=10):
#         """Combine scores from multiple retrievers"""
#         all_scores = defaultdict(float)
        
#         for retriever, weight in zip(self.retrievers, self.weights):
#             results = retriever.search(query, top_k=top_k*2)  # Get more for fusion
            
#             # Normalize scores to [0, 1]
#             max_score = max([score for _, score in results]) if results else 1.0
            
#             for doc_id, score in results:
#                 normalized_score = score / max_score if max_score > 0 else 0
#                 all_scores[doc_id] += weight * normalized_score
        
#         # Sort by combined score
#         sorted_docs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
#         return sorted_docs[:top_k]


# if __name__ == "__main__":
#     # Load processed data
#     df = pd.read_csv('data/processed/reviews_processed.csv')
#     documents = df['processed_text'].tolist()
    
#     # Test each retriever
#     query = "performance issues crashes"
    
#     print("\n=== Testing TF-IDF ===")
#     tfidf = TFIDFRetriever()
#     tfidf.fit(documents)
#     results = tfidf.search(query, top_k=5)
#     print(f"Top results for '{query}':")
#     for idx, score in results:
#         print(f"  Score: {score:.4f} - {df.iloc[idx]['review_text'][:100]}...")
    
#     print("\n=== Testing BM25 ===")
#     bm25 = BM25Retriever()
#     bm25.fit(documents)
#     results = bm25.search(query, top_k=5)
#     print(f"Top results for '{query}':")
#     for idx, score in results:
#         print(f"  Score: {score:.4f} - {df.iloc[idx]['review_text'][:100]}...")
    
#     print("\n=== Testing FAISS ===")
#     faiss_retriever = FAISSRetriever()
#     faiss_retriever.fit(documents)
#     results = faiss_retriever.search(query, top_k=5)
#     print(f"Top results for '{query}':")
#     for idx, score in results:
#         print(f"  Score: {score:.4f} - {df.iloc[idx]['review_text'][:100]}...")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import math

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            max_df=0.8
        )
        self.doc_vectors = None
        self.documents = None
        
    def fit(self, documents):
        """Build TF-IDF index"""
        print("Building TF-IDF index...")
        self.documents = documents
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        print(f"TF-IDF index built: {self.doc_vectors.shape}")
        
    def search(self, query, top_k=10):
        """Search using TF-IDF cosine similarity"""
        query_vector = self.vectorizer.transform([query])
        scores = (self.doc_vectors @ query_vector.T).toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.doc_term_freqs = []
        self.documents = None
        
    def fit(self, documents):
        """Build BM25 index"""
        print("Building BM25 index...")
        self.documents = documents
        
        # Calculate document frequencies and term frequencies
        for doc in documents:
            tokens = doc.split()
            self.doc_lengths.append(len(tokens))
            
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
                
            self.doc_term_freqs.append(dict(term_freq))
            
            # Document frequency (how many docs contain this term)
            for token in set(tokens):
                self.doc_freqs[token] += 1
        
        # Calculate average document length
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # Calculate IDF for each term
        N = len(documents)
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
        
        print(f"BM25 index built: {len(documents)} documents, {len(self.idf)} unique terms")
        
    def search(self, query, top_k=10):
        """Search using BM25 scoring"""
        query_tokens = query.split()
        scores = []
        
        for doc_id, (doc_tf, doc_len) in enumerate(zip(self.doc_term_freqs, self.doc_lengths)):
            score = 0.0
            
            for token in query_tokens:
                if token not in doc_tf:
                    continue
                
                tf = doc_tf[token]
                idf = self.idf.get(token, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)
            
            scores.append(score)
        
        # Get top-k
        scores = np.array(scores)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]


class HybridRetriever:
    """Combine multiple retrievers"""
    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        
    def search(self, query, top_k=10):
        """Combine scores from multiple retrievers"""
        all_scores = defaultdict(float)
        
        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.search(query, top_k=top_k*2)  # Get more for fusion
            
            # Normalize scores to [0, 1]
            max_score = max([score for _, score in results]) if results else 1.0
            
            for doc_id, score in results:
                normalized_score = score / max_score if max_score > 0 else 0
                all_scores[doc_id] += weight * normalized_score
        
        # Sort by combined score
        sorted_docs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/processed/reviews_processed.csv')
    documents = df['processed_text'].tolist()
    
    # Test each retriever
    query = "performance issues crashes"
    
    print("\n=== Testing TF-IDF ===")
    tfidf = TFIDFRetriever()
    tfidf.fit(documents)
    results = tfidf.search(query, top_k=5)
    print(f"Top results for '{query}':")
    for idx, score in results:
        print(f"  Score: {score:.4f} - {df.iloc[idx]['review_text'][:100]}...")
    
    print("\n=== Testing BM25 ===")
    bm25 = BM25Retriever()
    bm25.fit(documents)
    results = bm25.search(query, top_k=5)
    print(f"Top results for '{query}':")
    for idx, score in results:
        print(f"  Score: {score:.4f} - {df.iloc[idx]['review_text'][:100]}...")
    
    print("\n✓ Retrieval testing complete (TF-IDF and BM25 working)")
    print("Note: FAISS semantic search disabled due to PyTorch issues")