import os
import joblib
import json

def print_header(title):
    print("\n" + "="*80)
    print(f" {title.upper()} ".center(80, "="))
    print("="*80 + "\n")

def inspect_model():
    print_header("MODEL INSPECTION (PHASE 4)")
    
    if not os.path.exists('saved_models/model.joblib'):
        print("Model file not found.")
        return
        
    try:
        model = joblib.load('saved_models/model.joblib')
        print(f"Model Type: {type(model)}")
        if hasattr(model, 'classes_'):
            print(f"Classes: {model.classes_}")
            
        with open('saved_models/metadata.json', 'r') as f:
            meta = json.load(f)
            
        print("\nMetadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
            
    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    inspect_model()
