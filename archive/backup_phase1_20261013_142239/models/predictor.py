import pandas as pd
import numpy as np


class ModelPredictor:
    """
    Applies a calibrated classifier's probability output to produce
    AMBIGUOUS / NOT AMBIGUOUS / NEEDS REVIEW decisions.

    Thresholds (from saved metadata, default 0.75 / 0.25):
        ambiguity_prob >= upper_threshold  -> "Ambiguous"
        ambiguity_prob <= lower_threshold  -> "Not Ambiguous"
        otherwise                          -> "NEEDS REVIEW"

    The predictor NEVER overrides the classifier's probability.
    The explanation engine is kept strictly separate.
    """

    def __init__(
        self,
        trained_model,
        upper_threshold: float = 0.75,
        lower_threshold: float = 0.25,
    ):
        self.trained_model = trained_model
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        if not (0.0 < lower_threshold < upper_threshold < 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < lower_threshold ({lower_threshold}) "
                f"< upper_threshold ({upper_threshold}) < 1"
            )

    def predict(
        self,
        X_scaled: pd.DataFrame,
        original_features: pd.DataFrame = None,  # kept for API compatibility; not used
    ) -> tuple:
        """
        Returns:
            predictions : pd.Series  with values 'Ambiguous', 'Not Ambiguous', 'NEEDS REVIEW'
            prob_df     : pd.DataFrame  columns 'Probability_Ambiguous', 'Probability_Not Ambiguous'
                          values are percentages (0-100)
        """
        raw_probs = self.trained_model.predict_proba(X_scaled)
        class_labels = self.trained_model.classes_

        prob_df = pd.DataFrame(
            raw_probs,
            columns=[f"Probability_{label}" for label in class_labels],
        )

        if "Ambiguous" in class_labels:
            amb_idx = int(np.where(class_labels == "Ambiguous")[0][0])
        else:
            amb_idx = 1

        amb_probs = raw_probs[:, amb_idx]

        preds = []
        for prob in amb_probs:
            if prob >= self.upper_threshold:
                preds.append("Ambiguous")
            elif prob <= self.lower_threshold:
                preds.append("Not Ambiguous")
            else:
                preds.append("NEEDS REVIEW")

        predictions = pd.Series(preds, name="Predicted_Label")
        prob_df = prob_df * 100.0  # convert to percentage for display
        return predictions, prob_df
