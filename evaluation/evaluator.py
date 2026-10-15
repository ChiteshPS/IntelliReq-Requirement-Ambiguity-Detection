import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import json

class ModelEvaluator:
    def __init__(self):
        """
        Initializes the Model Evaluator.
        It uses scikit-learn's powerful metrics module.
        """
        pass

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Calculates all required evaluation metrics by comparing the actual answers (y_true)
        to the model's predictions (y_pred).
        """
        # Because our classes are text ("Ambiguous", "Not Ambiguous"), we need to specify 
        # which one is the "positive" class for precision and recall calculations.
        # In our case, detecting "Ambiguous" is the main goal, so it is the positive class (pos_label).
        pos_class = "Ambiguous"
        
        # 1. Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # 2. Precision
        prec = precision_score(y_true, y_pred, pos_label=pos_class, zero_division=0)
        
        # 3. Recall
        rec = recall_score(y_true, y_pred, pos_label=pos_class, zero_division=0)
        
        # 4. F1 Score
        f1 = f1_score(y_true, y_pred, pos_label=pos_class, zero_division=0)
        
        # 5. Confusion Matrix
        # Returns a 2x2 array: [[True Negatives, False Positives], [False Negatives, True Positives]]
        cm = confusion_matrix(y_true, y_pred)
        
        # 6. Classification Report (String format)
        report = classification_report(y_true, y_pred, zero_division=0)
        
        metrics = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'Confusion_Matrix': cm.tolist(), # Convert numpy array to list for easy JSON saving
            'Classification_Report': report
        }
        
        return metrics

    def display_metrics(self, metrics: dict):
        """
        Helper function to nicely print the metrics to the console.
        """
        print("\n--- MODEL EVALUATION RESULTS ---")
        print(f"Accuracy:  {metrics['Accuracy'] * 100:.2f}%")
        print(f"Precision: {metrics['Precision'] * 100:.2f}%")
        print(f"Recall:    {metrics['Recall'] * 100:.2f}%")
        print(f"F1 Score:  {metrics['F1_Score'] * 100:.2f}%")
        
        print("\n--- CONFUSION MATRIX ---")
        cm = metrics['Confusion_Matrix']
        print(f"True Negatives (TN): {cm[0][0]} | False Positives (FP): {cm[0][1]}")
        print(f"False Negatives (FN): {cm[1][0]} | True Positives (TP): {cm[1][1]}")
        
        print("\n--- CLASSIFICATION REPORT ---")
        print(metrics['Classification_Report'])

# Simple test if run directly
if __name__ == "__main__":
    # Mock 'True' answers vs Mock 'Predicted' answers
    mock_y_true = pd.Series(['Ambiguous', 'Not Ambiguous', 'Ambiguous', 'Ambiguous', 'Not Ambiguous'])
    mock_y_pred = pd.Series(['Ambiguous', 'Not Ambiguous', 'Not Ambiguous', 'Ambiguous', 'Ambiguous'])
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(mock_y_true, mock_y_pred)
    evaluator.display_metrics(results)
