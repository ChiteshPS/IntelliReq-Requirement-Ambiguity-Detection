import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initializes the Random Forest model with specified hyperparameters.
        n_estimators: The number of trees in the forest.
        random_state: Ensures reproducibility.
        """
        # We define the hyperparameter configuration here.
        # This is exactly the model requested by the paper's methodology.
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1 # Uses all available CPU cores for faster training
        )
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the Random Forest model using the 80% training data.
        X_train: The numerical feature matrix.
        y_train: The target labels.
        """
        print("Training the Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete.")

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Extracts the internal weights the model assigned to each feature.
        This answers the question: "Which feature was most important for detecting ambiguity?"
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting feature importance.")
            
        importances = self.model.feature_importances_
        
        # Create a readable dataframe mapping feature names to their importance score
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance (highest first)
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        return importance_df

# Simple test if run directly
if __name__ == "__main__":
    # Mock data to test the class
    mock_X_train = pd.DataFrame({
        'Word_Count': [10, 5, 20, 4],
        'tfidf_usually': [0.8, 0.0, 0.9, 0.0]
    })
    mock_y_train = pd.Series(['Ambiguous', 'Not Ambiguous', 'Ambiguous', 'Not Ambiguous'])
    
    rf = RandomForestModel(n_estimators=10)
    rf.train(mock_X_train, mock_y_train)
    
    print("\nFeature Importances:")
    print(rf.get_feature_importance(mock_X_train.columns.tolist()))
