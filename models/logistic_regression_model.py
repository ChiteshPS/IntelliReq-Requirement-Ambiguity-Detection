import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

class LogisticRegressionModel:
    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        """
        Initializes the Logistic Regression model.
        Better suited for dense semantic embeddings (SBERT).
        """
        self.base_model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight='balanced',
            penalty='l1',
            solver='liblinear',
            C=0.1
        )
        # Use calibration to get realistic confidence scores
        self.model = CalibratedClassifierCV(self.base_model, method='sigmoid', cv=5)
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the calibrated Logistic Regression model.
        """
        print("Training the Logistic Regression model (with Calibration)...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # We need to fit the base model separately on all data just to extract coefficients 
        # because CalibratedClassifierCV wraps it and makes coefficients hard to extract.
        self.base_model.fit(X_train, y_train)
        print("Training complete.")

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Extracts the coefficients as feature importance.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting feature importance.")
            
        importances = abs(self.base_model.coef_[0])
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        return importance_df
