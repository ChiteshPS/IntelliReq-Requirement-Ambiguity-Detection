import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

class SBERTExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the SBERT Feature Extractor.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def fit(self, texts):
        # SBERT is pre-trained, no fit needed. Included for pipeline compatibility.
        return self

    def transform(self, texts) -> pd.DataFrame:
        """
        Generates SBERT embeddings for a list of texts.
        """
        # Convert pandas series to list if necessary
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Create a dataframe with column names sbert_0 to sbert_383
        df_embeddings = pd.DataFrame(
            embeddings, 
            columns=[f"sbert_{i}" for i in range(embeddings.shape[1])]
        )
        return df_embeddings

    def fit_transform(self, texts) -> pd.DataFrame:
        """
        Generates SBERT embeddings.
        """
        return self.transform(texts)

if __name__ == "__main__":
    extractor = SBERTExtractor()
    sample_text = ["The system shall respond quickly.", "The system shall return a response within 2 seconds."]
    embeddings_df = extractor.transform(sample_text)
    print(f"Model: {extractor.model_name}")
    print(f"Number of samples: {embeddings_df.shape[0]}")
    print(f"Embedding dimension: {embeddings_df.shape[1]}")
    print(f"Example embedding (first 5 dims): {embeddings_df.iloc[0, :5].values}")
