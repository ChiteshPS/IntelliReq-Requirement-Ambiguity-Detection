import pandas as pd
import numpy as np

class ModelPredictor:
    def __init__(self, trained_model, upper_threshold: float = 0.80, lower_threshold: float = 0.20):
        """
        Initializes the predictor with a calibrated model and review thresholds.
        """
        self.trained_model = trained_model
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def predict(self, X_test_scaled: pd.DataFrame, original_features: pd.DataFrame = None) -> tuple[pd.Series, pd.DataFrame]:
        """
        Generates predictions based strictly on calibrated probabilities.
        """
        probabilities = self.trained_model.predict_proba(X_test_scaled)
        class_labels = self.trained_model.classes_
        
        prob_df = pd.DataFrame(
            probabilities,
            columns=[f"Probability_{label}" for label in class_labels]
        )
        
        if "Ambiguous" in class_labels:
            ambiguous_idx = np.where(class_labels == "Ambiguous")[0][0]
        else:
            ambiguous_idx = 1
            
        ambiguous_probs = probabilities[:, ambiguous_idx]
        
        preds = []
        for i, prob in enumerate(ambiguous_probs):
            if prob >= self.upper_threshold:
                final_pred = "Ambiguous"
            elif prob <= self.lower_threshold:
                final_pred = "Not Ambiguous"
            else:
                final_pred = "NEEDS REVIEW"
                    
            preds.append(final_pred)

        predictions = pd.Series(preds, name="Predicted_Label")
        prob_df = prob_df * 100.0
        return predictions, prob_df
