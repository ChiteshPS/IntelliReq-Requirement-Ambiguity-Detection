import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# In a real environment, you might need to run these once:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        # Initialize the lemmatizer and define English stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """
        Applies standard text cleaning steps.
        Input: Raw string.
        Output: Lowercased, punctuation-free string without extra spaces.
        """
        # 1. Lowercase conversion
        text = text.lower()
        
        # 2. Remove punctuation and special characters
        # The regex [^\w\s] matches anything that is NOT a word character (\w) or whitespace (\s)
        text = re.sub(r'[^\w\s]', '', text)
        
        # 3. Remove extra spaces
        # \s+ matches one or more whitespace characters. We replace them with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize_and_lemmatize(self, text: str) -> list:
        """
        Tokenizes the text, removes stopwords, and lemmatizes each word.
        Input: Cleaned string.
        Output: List of lemmatized tokens.
        """
        # 4. Tokenization
        tokens = word_tokenize(text)
        
        # 5 & 6. Stopword removal and Lemmatization
        processed_tokens = []
        for word in tokens:
            if word not in self.stop_words:
                lemmatized_word = self.lemmatizer.lemmatize(word)
                processed_tokens.append(lemmatized_word)
                
        return processed_tokens

    def preprocess(self, text: str) -> str:
        """
        The main pipeline method that chains all steps together.
        Input: Raw string.
        Output: Fully preprocessed string ready for feature engineering.
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        # Join the tokens back into a single string for ML models
        return ' '.join(tokens)

# Simple test if run directly
if __name__ == "__main__":
    sample_req = "The system MIGHT need to, integrate with external payment gateways usually!!!"
    print(f"Original: {sample_req}")
    preprocessor = TextPreprocessor()
    print(f"Processed: {preprocessor.preprocess(sample_req)}")
