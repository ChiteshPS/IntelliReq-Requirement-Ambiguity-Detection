import joblib
import os
import json
from datetime import datetime

class ModelSaver:
    def __init__(self):
        """
        Initializes the ModelSaver.
        Ensures the 'saved_models' directory exists.
        """
        self.save_dir = "saved_models"
        os.makedirs(self.save_dir, exist_ok=True)

    def save_pipeline(self, primary_classifier, scaler=None):
        """
        Saves all necessary components to disk (Phase 22).
        """
        print("Saving models to disk...")
        
        # Save the Primary Classifier (Hybrid Model)
        clf_path = os.path.join(self.save_dir, "primary_classifier.pkl")
        joblib.dump(primary_classifier, clf_path)
        print(f"Saved: {clf_path}")
        
        if scaler is not None:
            scaler_path = os.path.join(self.save_dir, "feature_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            print(f"Saved: {scaler_path}")
            
        # Save metadata (Phase 23)
        metadata = {
            "dataset_version": "v2",
            "training_date": datetime.now().isoformat(),
            "feature_version": "hybrid_sbert_ling_spec",
            "sbert_model": "all-MiniLM-L6-v2",
            "classifier_type": "Calibrated Logistic Regression (L1)",
            "embedding_dimension": 384
        }
        meta_path = os.path.join(self.save_dir, "model_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved: {meta_path}")
            
        print("All models successfully saved!")

    def load_pipeline(self) -> dict:
        """
        Loads all saved components from disk.
        """
        components = {}
        
        clf_path = os.path.join(self.save_dir, "primary_classifier.pkl")
        if os.path.exists(clf_path):
            components['primary_classifier'] = joblib.load(clf_path)
        else:
            raise FileNotFoundError("primary_classifier.pkl not found! Train the model first.")
            
        scaler_path = os.path.join(self.save_dir, "feature_scaler.pkl")
        if os.path.exists(scaler_path):
            components['scaler'] = joblib.load(scaler_path)
            
        meta_path = os.path.join(self.save_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                components['metadata'] = json.load(f)
            
        return components

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    dummy_clf = LogisticRegression()
    saver = ModelSaver()
    saver.save_pipeline(primary_classifier=dummy_clf)
    loaded = saver.load_pipeline()
    print("Loaded components:", loaded.keys())
