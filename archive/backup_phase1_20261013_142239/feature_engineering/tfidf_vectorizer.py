import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFEngine:
    def __init__(self, max_features: int = 1000):
        """
        Initializes the TF-IDF Vectorizer.
        max_features=1000 limits the vocabulary to the top 1000 most frequent words.
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def fit_transform(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Learns the vocabulary (fit) and transforms the text into a TF-IDF feature matrix.
        Input: A Pandas Series of PREPROCESSED text.
        Output: A Pandas DataFrame representing the Sparse Feature Matrix.
        """
        # Learn vocabulary and generate the sparse matrix
        tfidf_matrix = self.vectorizer.fit_transform(text_series)
        
        # Extract the learned vocabulary (the exact words)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Convert the mathematical sparse matrix into a readable Pandas DataFrame
        # We append 'tfidf_' to the column names so we don't confuse them with other features
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f"tfidf_{word}" for word in feature_names]
        )
        
        return tfidf_df

    def transform_new_text(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Transforms new unseen data (like user testing input) using the ALREADY LEARNED vocabulary.
        Notice we only call .transform(), not .fit_transform()!
        """
        tfidf_matrix = self.vectorizer.transform(text_series)
        feature_names = self.vectorizer.get_feature_names_out()
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f"tfidf_{word}" for word in feature_names]
        )
        return tfidf_df

# Simple test if run directly
if __name__ == "__main__":
    # Mocking preprocessed requirements
    mock_preprocessed_reqs = pd.Series([
        "system authenticate user",
        "system process transaction quick",
        "user easy interface system"
    ])
    
    engine = TFIDFEngine(max_features=10)
    feature_matrix = engine.fit_transform(mock_preprocessed_reqs)
    
    print("--- LEARNED VOCABULARY ---")
    print(engine.vectorizer.get_feature_names_out())
    
    print("\n--- TF-IDF FEATURE MATRIX ---")
    print(feature_matrix)
