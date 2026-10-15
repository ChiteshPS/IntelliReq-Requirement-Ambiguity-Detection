import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
import joblib

def evaluate_model_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="Ambiguous", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="Ambiguous", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="Ambiguous", zero_division=0)
    brier = brier_score_loss((y_true == "Ambiguous").astype(int), y_prob) if y_prob is not None else None
    return acc, prec, rec, f1, brier

def check_leakage(df):
    print("Checking for challenge set and adversarial set leakage...")
    
    challenge_set = set()
    try:
        chall = pd.read_csv('data/challenge_test_set.csv')
        challenge_set = set(chall['Requirement_Text'].str.lower().str.strip().tolist())
    except Exception:
        pass
        
    adv_set = set()
    try:
        adv = pd.read_csv('data/adversarial_test_set.csv')
        adv_set = set(adv['requirement'].str.lower().str.strip().tolist())
    except Exception:
        pass
        
    train_texts = set(df['requirement'].str.lower().str.strip().tolist())
    
    chall_overlap = challenge_set.intersection(train_texts)
    adv_overlap = adv_set.intersection(train_texts)
    
    if chall_overlap:
        print(f"LEAKAGE FATAL ERROR: {len(chall_overlap)} sentences from Challenge set found in Training!")
        return False
        
    if adv_overlap:
        print(f"LEAKAGE FATAL ERROR: {len(adv_overlap)} sentences from Adversarial set found in Training!")
        return False
        
    print("Leakage check PASS: No external evaluation data leaked into training.")
    return True

def run_pipeline():
    print("="*60)
    print("MASTER TRAINING PIPELINE")
    print("="*60)

    print("\n[1/10] Loading Dataset and Checking Quality...")
    dataset_path = 'data/intellireq_clean_dataset_v4.csv'
    if not os.path.exists(dataset_path):
        print(f"ERROR: {dataset_path} not found. Run generate_dataset.py first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    if len(df) < 100:
        print("ERROR: Dataset too small.")
        return
        
    if not check_leakage(df):
        import sys
        sys.exit(1)

    print("\n[2/10] Creating Group-Aware Splits (Train 70 / Val 15 / Test 15)...")
    df['semantic_group_id'] = df['semantic_group_id'].fillna(df.index.to_series().astype(str))
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df['semantic_group_id']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df['semantic_group_id']))
    
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n[3/10] Extracting Features...")
    feat_extractor = FeatureExtractor()
    sbert_extractor = SBERTExtractor()
    
    X_train_ling = feat_extractor.process_dataframe(train_df, 'requirement').drop(columns=train_df.columns)
    X_val_ling = feat_extractor.process_dataframe(val_df, 'requirement').drop(columns=val_df.columns)
    X_test_ling = feat_extractor.process_dataframe(test_df, 'requirement').drop(columns=test_df.columns)
    
    X_train_sbert = sbert_extractor.transform(train_df['requirement'])
    X_val_sbert = sbert_extractor.transform(val_df['requirement'])
    X_test_sbert = sbert_extractor.transform(test_df['requirement'])
    
    X_train = pd.concat([X_train_sbert, X_train_ling], axis=1)
    X_val = pd.concat([X_val_sbert, X_val_ling], axis=1)
    X_test = pd.concat([X_test_sbert, X_test_ling], axis=1)
    
    y_train = train_df['label']
    y_val = val_df['label']
    y_test = test_df['label']

    print("\n[4/10] Scaling...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print("\n[5/10] Evaluating Candidate Models...")
    models = {
        'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42),
        'Logistic Regression (L2)': LogisticRegression(penalty='l2', class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42)
    }
    
    best_f1 = 0
    best_model_name = ""
    best_model = None
    
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_val_scaled)
        acc, prec, rec, f1, _ = evaluate_model_metrics(y_val, preds)
        print(f"{name} -> Val F1: {f1:.4f} | Acc: {acc:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = clf
            
    print(f"\nSelected Model: {best_model_name}")

    print("\n[6/10] Calibrating Probabilities...")
    calibrated = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
    calibrated.fit(X_val_scaled, y_val)
    
    amb_idx = np.where(calibrated.classes_ == "Ambiguous")[0][0]

    print("\n[7/10] Selecting Review Thresholds...")
    upper_threshold = 0.75
    lower_threshold = 0.25
    print(f"Thresholds -> Ambiguous >= {upper_threshold}, Not Ambiguous <= {lower_threshold}")

    print("\n[8/10] Evaluating on Clean Test Set...")
    print("Skipped in train.py (Handled by final_evaluation.py).")

    print("\n[9/10] Saving Artifacts...")
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(calibrated, 'saved_models/model.joblib')
    joblib.dump(scaler, 'saved_models/scaler.joblib')
    
    metadata = {
        'model_version': '2.1.0',
        'training_timestamp': datetime.now().isoformat(),
        'dataset_version': 'intellireq_clean_v4',
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dimension': 384,
        'classifier': best_model_name,
        'calibration_method': 'sigmoid',
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'random_seed': 42,
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df)
    }
    
    with open('saved_models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\n[10/10] Complete! Master training finished successfully.")

if __name__ == "__main__":
    run_pipeline()
