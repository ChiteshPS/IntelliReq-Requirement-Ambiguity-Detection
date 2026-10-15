import pandas as pd
import warnings
import sys
import os
import joblib
import json
import subprocess
warnings.filterwarnings('ignore')

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from rule_engine.explanation_engine import ExplanationEngine
from models.predictor import ModelPredictor
from preprocessing.document_parser import DocumentParser

def is_valid_requirement(text: str) -> bool:
    if not isinstance(text, str): return False
    text = text.strip()
    if not text or len(text.split()) < 3: return False
    return True

class IntelliReqCLI:
    def __init__(self):
        try:
            self.model = joblib.load('saved_models/model.joblib')
            self.scaler = joblib.load('saved_models/scaler.joblib')
            with open('saved_models/metadata.json', 'r') as f:
                self.metadata = json.load(f)
        except (KeyError, FileNotFoundError):
            print("\nERROR: Saved models not found! Run python train.py first.")
            sys.exit(1)

        self.feat_extractor = FeatureExtractor()
        self.sbert_extractor = SBERTExtractor()
        self.explanation_engine = ExplanationEngine()
        self.predictor = ModelPredictor(
            trained_model=self.model,
            upper_threshold=self.metadata.get('upper_threshold', 0.75),
            lower_threshold=self.metadata.get('lower_threshold', 0.25)
        )
        self.doc_parser = DocumentParser()

    def print_header(self):
        print("\n" + "="*60)
        print("INTELLIREQ — SOFTWARE REQUIREMENT ANALYZER")
        print("="*60)

    def main_menu(self):
        while True:
            self.print_header()
            print("1. Analyze single requirement")
            print("2. Analyze TXT/PDF/DOCX")
            print("3. Analyze CSV")
            print("4. Analyze Excel")
            print("5. Inspect model")
            print("6. Run evaluation")
            print("7. Exit")
            
            choice = input("\nSelect an option (1-7): ").strip()
            
            if choice == '1': self.analyze_single()
            elif choice == '2': self.analyze_document()
            elif choice == '3': self.analyze_batch(mode='csv')
            elif choice == '4': self.analyze_batch(mode='excel')
            elif choice == '5': self.inspect_model()
            elif choice == '6': self.run_eval()
            elif choice == '7': 
                print("Exiting. Goodbye!")
                break
            else: print("Invalid choice.")

    def process_text(self, text):
        ling_feats = self.feat_extractor.extract_features(text)
        X_ling = pd.DataFrame([ling_feats])
        X_sbert = self.sbert_extractor.transform([text])
        
        X_final = pd.concat([X_sbert.reset_index(drop=True), X_ling.reset_index(drop=True)], axis=1)
        X_scaled = pd.DataFrame(self.scaler.transform(X_final), columns=X_final.columns)

        preds, probs = self.predictor.predict(X_scaled, original_features=X_ling)
        predicted_class = preds.iloc[0]
        
        amb_col = [c for c in probs.columns if 'Ambiguous' in c and 'Not' not in c][0]
        confidence = probs[amb_col].iloc[0]
        
        explanation = self.explanation_engine.generate_explanation(text, predicted_class)
        return predicted_class, confidence, explanation, ling_feats

    def analyze_single(self):
        print("\n--- SINGLE REQUIREMENT ---")
        while True:
            text = input("\nEnter requirement (or 'q'):\n> ")
            if text.lower() in ['q', 'quit', 'back']: break
            if not is_valid_requirement(text):
                print("Invalid input.")
                continue

            pred, conf, exp, feats = self.process_text(text)

            print("\n" + "-"*60)
            print(f"Requirement: {text}\n")
            print(f"Prediction: {pred.upper()}")
            print(f"Probability of Ambiguous: {conf:.2f}%")
            if pred == "NEEDS REVIEW":
                print(f"Decision:   NEEDS REVIEW")
            else:
                print(f"Decision:   {pred.upper()}")
                
            print(f"Reason:     {exp['Reason']}")
            if exp['Missing_Specification']:
                print(f"Missing specification: {', '.join(exp['Missing_Specification'])}")
            if exp['Suggested_Rewrite']:
                print(f"Suggested rewrite: {exp['Suggested_Rewrite']}")
            print("-" * 60)

    def analyze_document(self):
        print("\n--- DOCUMENT ANALYSIS ---")
        path = input("Enter path to TXT/PDF/DOCX:\n> ").strip().strip('"').strip("'")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
            
        try:
            reqs = self.doc_parser.extract_requirements(path)
            print(f"Extracted {len(reqs)} candidate requirements.")
            df = pd.DataFrame({"Requirement": reqs})
            self._run_batch_dataframe(df, "document_results.xlsx")
        except Exception as e:
            print(f"Error parsing document: {e}")

    def analyze_batch(self, mode='csv'):
        print(f"\n--- BATCH {mode.upper()} ANALYSIS ---")
        path = input(f"Enter path to {mode.upper()}:\n> ").strip().strip('"').strip("'")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
            
        try:
            if mode == 'csv': df = pd.read_csv(path)
            else: df = pd.read_excel(path)
        except Exception as e:
            print(f"Error reading file: {e}")
            return
            
        # Interactive column selection
        print(f"Available columns: {list(df.columns)}")
        req_col = input("Enter the name of the column containing requirements:\n> ").strip()
        
        if req_col not in df.columns:
            print("Column not found.")
            return
            
        df = df.rename(columns={req_col: "Requirement"})
        out = "batch_results.xlsx"
        self._run_batch_dataframe(df, out)
        
    def _run_batch_dataframe(self, df, output_filename):
        results = []
        for idx, row in df.iterrows():
            text = str(row.get("Requirement", ""))
            try:
                if not is_valid_requirement(text):
                    results.append({"Requirement": text, "Prediction": "INVALID", "Status": "Error"})
                    continue
                    
                pred, conf, exp, feats = self.process_text(text)
                results.append({
                    "Requirement": text,
                    "Prediction": pred,
                    "Probability": round(conf, 2),
                    "Decision": "Review" if pred == "NEEDS REVIEW" else "Clear",
                    "Reason": exp['Reason'],
                    "Missing_Specification": "; ".join(exp['Missing_Specification']) if exp['Missing_Specification'] else "",
                    "Suggested_Rewrite": exp['Suggested_Rewrite'] if exp['Suggested_Rewrite'] else ""
                })
            except Exception as e:
                results.append({"Requirement": text, "Prediction": "ERROR", "Status": str(e)})
                
        out_df = pd.DataFrame(results)
        
        # Save to excel
        try:
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                out_df.to_excel(writer, sheet_name='Analysis Results', index=False)
                
                # Summary Sheet
                total = len(out_df)
                amb = len(out_df[out_df['Prediction'] == 'Ambiguous'])
                namb = len(out_df[out_df['Prediction'] == 'Not Ambiguous'])
                rev = len(out_df[out_df['Prediction'] == 'NEEDS REVIEW'])
                
                summary = pd.DataFrame({
                    "Metric": ["Total Processed", "Ambiguous", "Not Ambiguous", "Needs Review"],
                    "Value": [total, amb, namb, rev]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
            print(f"\nBatch processing complete. Saved to {output_filename}")
        except Exception as e:
            print(f"Error saving Excel (make sure openpyxl is installed): {e}")
            out_df.to_csv(output_filename.replace('.xlsx', '.csv'), index=False)
            print(f"Fell back to saving CSV: {output_filename.replace('.xlsx', '.csv')}")

    def inspect_model(self):
        print("\n--- MODEL INSPECTION ---")
        for k, v in self.metadata.items():
            print(f"{k}: {v}")
            
    def run_eval(self):
        print("\n--- RUNNING EVALUATION ---")
        subprocess.run([sys.executable, "final_evaluation.py"])

if __name__ == "__main__":
    cli = IntelliReqCLI()
    cli.main_menu()
