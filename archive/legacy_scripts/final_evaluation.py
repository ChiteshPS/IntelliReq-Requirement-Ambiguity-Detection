import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from models.predictor import ModelPredictor

def load_data(path, text_col, label_col):
    df = pd.read_csv(path)
    return df[text_col].tolist(), df[label_col].tolist()

def extract_features(texts):
    df = pd.DataFrame({'requirement': texts})
    ling_extractor = FeatureExtractor()
    sbert_extractor = SBERTExtractor()
    
    X_ling = ling_extractor.process_dataframe(df, 'requirement').drop(columns=['requirement'])
    X_sbert = sbert_extractor.transform(df['requirement'])
    
    return pd.concat([X_sbert, X_ling], axis=1)

def evaluate_set(name, X_scaled, y_true, predictor, amb_val="Ambiguous", clr_val="Not Ambiguous"):
    if not y_true: return
    
    preds, prob_df = predictor.predict(X_scaled)
    
    # Needs Review counts
    review_mask = preds == "NEEDS REVIEW"
    review_rate = review_mask.mean() * 100
    
    # Selective predictions
    sel_y_true = np.array(y_true)[~review_mask]
    sel_preds = preds[~review_mask]
    
    # If borderline is the ground truth, we want to measure how many were pushed to NEEDS REVIEW.
    if name.endswith("Borderline"):
        print(f"\n--- {name} ---")
        print(f"Total rows: {len(y_true)}")
        print(f"Correctly routed to NEEDS REVIEW: {review_rate:.2f}%")
        return
        
    print(f"\n--- {name} ---")
    print(f"Total rows: {len(y_true)}")
    print(f"Review Rate (Pushed to Human): {review_rate:.2f}%")
    
    if len(sel_y_true) > 0:
        sel_acc = accuracy_score(sel_y_true, sel_preds)
        print(f"Selective Accuracy (On Confident Preds): {sel_acc*100:.2f}%")
    else:
        print("Selective Accuracy: N/A (100% review rate)")
        
    # Strict Accuracy (treating NEEDS REVIEW as incorrect)
    strict_preds = [p if p in [amb_val, clr_val] else "Incorrect" for p in preds]
    strict_acc = accuracy_score(y_true, strict_preds)
    print(f"Strict Binary Accuracy: {strict_acc*100:.2f}%")

def run_evaluation():
    if not os.path.exists('saved_models/model.joblib'):
        print("Model not trained yet.")
        return
        
    model = joblib.load('saved_models/model.joblib')
    scaler = joblib.load('saved_models/scaler.joblib')
    with open('saved_models/metadata.json', 'r') as f:
        meta = json.load(f)
        
    predictor = ModelPredictor(model, upper_threshold=meta['upper_threshold'], lower_threshold=meta['lower_threshold'])
    
    # Load Challenge
    ch_df = pd.read_csv('data/challenge_test_set.csv')
    ch_binary = ch_df[ch_df['Ambiguity_Label'] != "Borderline"]
    ch_border = ch_df[ch_df['Ambiguity_Label'] == "Borderline"]
    
    adv_df = pd.read_csv('data/adversarial_test_set.csv')
    
    print("Extracting Challenge Binary...")
    ch_X = extract_features(ch_binary['Requirement_Text'].tolist())
    ch_X_scaled = pd.DataFrame(scaler.transform(ch_X), columns=ch_X.columns)
    
    print("Extracting Challenge Borderline...")
    cb_X = extract_features(ch_border['Requirement_Text'].tolist())
    cb_X_scaled = pd.DataFrame(scaler.transform(cb_X), columns=cb_X.columns)
    
    print("Extracting Adversarial...")
    ad_X = extract_features(adv_df['requirement'].tolist())
    ad_X_scaled = pd.DataFrame(scaler.transform(ad_X), columns=ad_X.columns)
    
    evaluate_set("Challenge Binary", ch_X_scaled, ch_binary['Ambiguity_Label'].tolist(), predictor)
    evaluate_set("Adversarial Binary", ad_X_scaled, adv_df['expected_label'].tolist(), predictor)
    evaluate_set("Challenge Borderline", cb_X_scaled, ch_border['Ambiguity_Label'].tolist(), predictor)
    
if __name__ == "__main__":
    run_evaluation()
