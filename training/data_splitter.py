import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataSplitter:
    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        """
        Initializes the splitter.
        test_size=0.20 means 80% Training, 20% Testing.
        random_state ensures reproducibility across different runs.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into X_train, X_test, y_train, y_test.
        X represents the features.
        y represents the target labels.
        """
        # Separate the Target (Labels generated in Step 4) from the Features
        y = df[target_column]
        
        # X is everything EXCEPT the target column and identifiers
        # We also drop 'Requirement_Text' because the ML model only understands the numerical features
        cols_to_drop = [target_column]
        if 'Requirement_Text' in df.columns:
            cols_to_drop.append('Requirement_Text')
        if 'Requirement_ID' in df.columns:
            cols_to_drop.append('Requirement_ID')
        # Drop the Matched_Ambiguous_Words list as it's not a numeric feature
        if 'Matched_Ambiguous_Words' in df.columns:
            cols_to_drop.append('Matched_Ambiguous_Words')
            
        X = df.drop(columns=cols_to_drop)
        
        # Perform the 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y # Ensures the 80/20 ratio applies proportionally to both Ambiguous and Not Ambiguous classes
        )
        
        return X_train, X_test, y_train, y_test

# Simple test if run directly
if __name__ == "__main__":
    # Mocking a full feature dataset
    mock_data = {
        'Requirement_ID': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10'],
        'Requirement_Text': ["req text"] * 10,
        'Word_Count': [5, 10, 5, 10, 5, 10, 5, 10, 5, 10],
        'tfidf_system': [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
        'Rule_Label': ['Ambiguous', 'Not Ambiguous', 'Ambiguous', 'Not Ambiguous', 'Ambiguous', 
                       'Not Ambiguous', 'Ambiguous', 'Not Ambiguous', 'Ambiguous', 'Not Ambiguous']
    }
    df_mock = pd.DataFrame(mock_data)
    
    splitter = DataSplitter(test_size=0.20)
    X_train, X_test, y_train, y_test = splitter.split_data(df_mock, target_column='Rule_Label')
    
    print(f"Total dataset size: {len(df_mock)} rows")
    print(f"Training set (X_train) size: {len(X_train)} rows")
    print(f"Testing set (X_test) size: {len(X_test)} rows")
