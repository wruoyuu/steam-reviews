import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class ReviewPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Keep some game-specific words that might be in stopwords
        self.stop_words -= {'not', 'no', 'very', 'too', 'can', 'will'}
    
    def clean_text(self, text):
        """Basic cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords"""
        return [token for token in tokens if token not in self.stop_words]
    
    def remove_punctuation(self, tokens):
        """Remove punctuation"""
        return [token for token in tokens if token not in string.punctuation]
    
    def stem_tokens(self, tokens):
        """Apply stemming"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text, stem=True):
        """Full preprocessing pipeline"""
        # Clean
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove punctuation
        tokens = self.remove_punctuation(tokens)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stem (optional)
        if stem:
            tokens = self.stem_tokens(tokens)
        
        return tokens
    
    def preprocess_dataframe(self, df, text_column='review_text'):
        """Preprocess entire dataframe"""
        print("Preprocessing reviews...")
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Tokenize with stemming
        df['tokens_stemmed'] = df['cleaned_text'].apply(
            lambda x: self.preprocess(x, stem=True)
        )
        
        # Tokenize without stemming (for display)
        df['tokens'] = df['cleaned_text'].apply(
            lambda x: self.preprocess(x, stem=False)
        )
        
        # Create searchable text from stemmed tokens
        df['processed_text'] = df['tokens_stemmed'].apply(lambda x: ' '.join(x))
        
        # Filter out very short reviews (less than 5 words)
        df = df[df['tokens_stemmed'].apply(len) >= 5]
        
        print(f"Preprocessing complete. {len(df)} reviews remaining.")
        return df

if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv('data/raw/all_reviews.csv')
    print(f"Loaded {len(df)} reviews")
    
    # Preprocess
    preprocessor = ReviewPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # Save
    df_processed.to_csv('data/processed/reviews_processed.csv', index=False)
    print("Saved processed reviews")
    
    # Show example
    print("\nExample processed review:")
    print(f"Original: {df_processed.iloc[0]['review_text'][:200]}...")
    print(f"Processed: {df_processed.iloc[0]['processed_text'][:200]}...")