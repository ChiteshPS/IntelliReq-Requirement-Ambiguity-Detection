import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from utils.model_saver import ModelSaver
from models.predictor import ModelPredictor

def run_diagnostic():
    print("="*60)
    print("  FINAL HELD-OUT CHALLENGE EVALUATION  ")
    print("="*60)

    saver = ModelSaver()
    try:
        models = saver.load_pipeline()
        primary_classifier = models['primary_classifier']
        scaler = models.get('scaler')
    except (KeyError, FileNotFoundError):
        print("ERROR: Run train_pipeline.py first.")
        return

    print("\n[1/4] Loading Challenge Set...")
    try:
        df_challenge = pd.read_csv("data/challenge_test_set.csv")
    except FileNotFoundError:
        print("ERROR: data/challenge_test_set.csv not found.")
        return

    feat_extractor = FeatureExtractor()
    sbert_extractor = SBERTExtractor()
    predictor = ModelPredictor(trained_model=primary_classifier, threshold=0.5)

    print("\n[2/4] Extracting Features...")
    X_ling = feat_extractor.process_dataframe(df_challenge, 'Requirement_Text').drop(columns=['Requirement_ID', 'Requirement_Text', 'Ambiguity_Label']).reset_index(drop=True)
    X_sbert = sbert_extractor.transform(df_challenge['Requirement_Text']).reset_index(drop=True)
    
    X_final = pd.concat([X_sbert, X_ling], axis=1)
    if scaler:
        X_scaled = pd.DataFrame(scaler.transform(X_final), columns=X_final.columns)
    else:
        X_scaled = X_final

    print("\n[3/4] Predicting...")
    preds, probs = predictor.predict(X_scaled, original_features=X_ling)
    
    df_challenge['Predicted_Label'] = preds
    df_challenge['Confidence'] = probs.max(axis=1)

    print("\n[4/4] Evaluation Results...")
    
    correct = 0
    total = len(df_challenge)
    misclassified = []
    
    # Phase 25: Error Analysis
    for idx, row in df_challenge.iterrows():
        expected = row['Ambiguity_Label']
        predicted = row['Predicted_Label']
        
        # We handle Borderline differently - predicting "NEEDS REVIEW" is correct for Borderline
        if expected == "Borderline" and predicted == "NEEDS REVIEW":
            correct += 1
        elif expected == predicted:
            correct += 1
        else:
            misclassified.append(row)
            
    print(f"\nChallenge Set Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")
    
    print("\n--- FAILING EXAMPLES AUDIT (BEFORE VS AFTER) ---")
    failing_examples = [
        "The system shall respond to user requests quickly.",
        "The application shall display the most relevant results to the user.",
        "The system shall store user data securely.",
        "The application shall provide an easy-to-use interface.",
        "The system shall periodically back up the database.",
        "The system shall provide adequate system availability."
    ]
    
    for fail_text in failing_examples:
        match = df_challenge[df_challenge['Requirement_Text'] == fail_text]
        if not match.empty:
            print(f"Req: {fail_text}")
            print(f"Expected: Ambiguous | Old Prediction: Not Ambiguous | New Prediction: {match.iloc[0]['Predicted_Label']} ({match.iloc[0]['Confidence']:.1f}%)\n")

    if misclassified:
        print("\n--- MISCLASSIFIED EXAMPLES (Phase 25) ---")
        for idx, row in enumerate(misclassified[:10]):  # Print up to 10 errors
            print(f"[{idx+1}] Requirement: {row['Requirement_Text']}")
            print(f"    True Label: {row['Ambiguity_Label']} | Predicted: {row['Predicted_Label']}")
            print(f"    Likely Reason: Look at SBERT dominance vs Specificity constraints.\n")
            
        # Write full error analysis to CSV
        import os
        os.makedirs('evaluation', exist_ok=True)
        error_df = pd.DataFrame(misclassified)
        
        # Helper to classify error type
        def get_error_type(r):
            if r['Ambiguity_Label'] == 'Borderline':
                return 'Borderline Miss'
            elif r['Ambiguity_Label'] == 'Not Ambiguous' and r['Predicted_Label'] in ['Ambiguous', 'NEEDS REVIEW']:
                return 'False Positive'
            elif r['Ambiguity_Label'] == 'Ambiguous' and r['Predicted_Label'] in ['Not Ambiguous', 'NEEDS REVIEW']:
                return 'False Negative'
            return 'Other'
            
        error_df['Error_Type'] = error_df.apply(get_error_type, axis=1)
        error_df.to_csv('evaluation/error_analysis.csv', index=False)
        print("Full error analysis saved to evaluation/error_analysis.csv")

if __name__ == "__main__":
    run_diagnostic()
