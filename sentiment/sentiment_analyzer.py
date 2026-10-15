import pandas as pd
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        """
        Initializes the Sentiment Analyzer. 
        TextBlob requires no complex setup.
        """
        pass

    def analyze_text(self, text: str) -> dict:
        """
        Analyzes the sentiment of a single requirement using TextBlob.
        Generates Polarity, Subjectivity, and a categorical Sentiment Label.
        """
        # Create a TextBlob object
        blob = TextBlob(text)
        
        # Extract Polarity (How positive or negative the text is)
        polarity = blob.sentiment.polarity
        
        # Extract Subjectivity (How opinionated vs factual the text is)
        subjectivity = blob.sentiment.subjectivity
        
        # Generate a Sentiment Label based on the polarity score
        if polarity > 0.05:
            sentiment_label = "Positive"
        elif polarity < -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
            
        return {
            'Polarity': round(polarity, 4),
            'Subjectivity': round(subjectivity, 4),
            'Sentiment_Label': sentiment_label
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Applies sentiment analysis to an entire DataFrame.
        """
        sentiment_results = df[text_column].apply(self.analyze_text)
        sentiment_df = pd.DataFrame(sentiment_results.tolist())
        return pd.concat([df, sentiment_df], axis=1)

# Simple test if run directly
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # A clearly objective, neutral requirement
    req1 = "The system shall store passwords in the database."
    # A subjective, slightly positive requirement
    req2 = "The system should have an amazingly fast and great user interface."
    
    print(f"Req 1: {req1}")
    print(analyzer.analyze_text(req1))
    
    print(f"\nReq 2: {req2}")
    print(analyzer.analyze_text(req2))
