import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from utils.model_saver import ModelSaver

def calculate_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="Ambiguous", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="Ambiguous", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="Ambiguous", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["Ambiguous", "Not Ambiguous"])
    return acc, prec, rec, f1, cm

def train_system():
    print("="*60)
    print("  INITIATING TRAINING PIPELINE (Full Phase Audit)  ")
    print("="*60)

    # 1. Load Dataset
    print("\n[1/10] Loading Dataset...")
    df = pd.read_csv("data/ambiguity_dataset_v2.csv")
    print(f"Loaded {len(df)} requirements.")
    
    # 2. Train/Val/Test Split (70/15/15) - Stratified
    print("\n[2/10] Splitting Dataset (70/15/15)...")
    # To prevent leakage, we use Source + Label for stratification
    stratify_col = df['Source'] + "_" + df['Ambiguity_Label']
    
    # Rare classes might crash stratification, so we fall back to just label if needed
    try:
        X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
            df['Requirement_Text'], df['Ambiguity_Label'], test_size=0.30, stratify=stratify_col, random_state=42
        )
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_temp_raw, y_temp, test_size=0.50, stratify=df.loc[y_temp.index, 'Source'] + "_" + y_temp, random_state=42
        )
    except:
        X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
            df['Requirement_Text'], df['Ambiguity_Label'], test_size=0.30, stratify=df['Ambiguity_Label'], random_state=42
        )
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

    print(f"Train: {len(X_train_raw)}, Validation: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    # 3. Extract Features
    print("\n[3/10] Extracting Linguistic & Specificity Features...")
    feat_extractor = FeatureExtractor()
    
    X_train_ling = feat_extractor.process_dataframe(pd.DataFrame(X_train_raw, columns=['Requirement_Text']), 'Requirement_Text').drop(columns=['Requirement_Text']).reset_index(drop=True)
    X_val_ling = feat_extractor.process_dataframe(pd.DataFrame(X_val_raw, columns=['Requirement_Text']), 'Requirement_Text').drop(columns=['Requirement_Text']).reset_index(drop=True)
    X_test_ling = feat_extractor.process_dataframe(pd.DataFrame(X_test_raw, columns=['Requirement_Text']), 'Requirement_Text').drop(columns=['Requirement_Text']).reset_index(drop=True)

    print("\n[4/10] Extracting SBERT Semantic Embeddings...")
    sbert_extractor = SBERTExtractor()
    X_train_sbert = sbert_extractor.transform(X_train_raw).reset_index(drop=True)
    X_val_sbert = sbert_extractor.transform(X_val_raw).reset_index(drop=True)
    X_test_sbert = sbert_extractor.transform(X_test_raw).reset_index(drop=True)

    print("\n[5/10] Extracting TF-IDF Features (For Baseline)...")
    tfidf = TfidfVectorizer(max_features=500)
    X_train_tfidf = pd.DataFrame(tfidf.fit_transform(X_train_raw).toarray())
    X_val_tfidf = pd.DataFrame(tfidf.transform(X_val_raw).toarray())
    X_test_tfidf = pd.DataFrame(tfidf.transform(X_test_raw).toarray())

    # Create hybrid datasets
    # Only Linguistic
    ling_cols = ['Word_Count', 'Sentence_Length', 'Average_Word_Length', 'Complex_Word_Percentage', 'Vague_Term_Count', 'Subjective_Term_Count', 'Approximate_Quantity_Count', 'Undefined_Reference_Count', 'Modal_Verb_Count', 'Linguistic_Ambiguity_Score']
    # Specificity
    spec_cols = ['Has_Numeric_Value', 'Has_Percentage', 'Has_Time_Unit', 'Has_Size_Unit', 'Has_Capacity_Value', 'Has_Threshold', 'Has_Range', 'Has_Format', 'Has_Explicit_Condition', 'Has_Testable_Constraint', 'Specificity_Score']
    
    X_train_sbert_ling = pd.concat([X_train_sbert, X_train_ling[ling_cols]], axis=1)
    X_test_sbert_ling = pd.concat([X_test_sbert, X_test_ling[ling_cols]], axis=1)

    X_train_sbert_all = pd.concat([X_train_sbert, X_train_ling], axis=1)
    X_val_sbert_all = pd.concat([X_val_sbert, X_val_ling], axis=1)
    X_test_sbert_all = pd.concat([X_test_sbert, X_test_ling], axis=1)

    print("\n[6/10] Scaling Features...")
    scaler = StandardScaler()
    X_train_sbert_all_scaled = pd.DataFrame(scaler.fit_transform(X_train_sbert_all), columns=X_train_sbert_all.columns)
    X_val_sbert_all_scaled = pd.DataFrame(scaler.transform(X_val_sbert_all), columns=X_val_sbert_all.columns)
    X_test_sbert_all_scaled = pd.DataFrame(scaler.transform(X_test_sbert_all), columns=X_test_sbert_all.columns)

    # 7. Ablation Study
    print("\n[7/10] Conducting Ablation Study (Phase 12, 26)...")
    results = []

    # A. TF-IDF + LR
    lr_a = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_a.fit(X_train_tfidf, y_train)
    preds_a = lr_a.predict(X_test_tfidf)
    acc, prec, rec, f1, cm = calculate_metrics(y_test, preds_a)
    results.append({'Model': 'TF-IDF + LR', 'Accuracy': acc, 'Precision (Amb)': prec, 'Recall (Amb)': rec, 'F1 (Amb)': f1})

    # B. SBERT + LR
    lr_b = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=1000)
    lr_b.fit(X_train_sbert, y_train)
    preds_b = lr_b.predict(X_test_sbert)
    acc, prec, rec, f1, cm = calculate_metrics(y_test, preds_b)
    results.append({'Model': 'SBERT + LR', 'Accuracy': acc, 'Precision (Amb)': prec, 'Recall (Amb)': rec, 'F1 (Amb)': f1})

    # C. SBERT + RF
    rf_c = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_c.fit(X_train_sbert, y_train)
    preds_c = rf_c.predict(X_test_sbert)
    acc, prec, rec, f1, cm = calculate_metrics(y_test, preds_c)
    results.append({'Model': 'SBERT + RF', 'Accuracy': acc, 'Precision (Amb)': prec, 'Recall (Amb)': rec, 'F1 (Amb)': f1})

    # D. SBERT + Linguistic + LR
    lr_d = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=1000)
    scaler_d = StandardScaler()
    X_tr_d = scaler_d.fit_transform(X_train_sbert_ling)
    X_te_d = scaler_d.transform(X_test_sbert_ling)
    lr_d.fit(X_tr_d, y_train)
    preds_d = lr_d.predict(X_te_d)
    acc, prec, rec, f1, cm = calculate_metrics(y_test, preds_d)
    results.append({'Model': 'SBERT + Ling + LR', 'Accuracy': acc, 'Precision (Amb)': prec, 'Recall (Amb)': rec, 'F1 (Amb)': f1})

    # E. SBERT + Linguistic + Specificity + LR
    lr_e = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=1000)
    
    # We calibrate the final model to output realistic probabilities
    calibrated_lr = CalibratedClassifierCV(lr_e, method='sigmoid', cv=5)
    calibrated_lr.fit(X_train_sbert_all_scaled, y_train)
    
    preds_e = calibrated_lr.predict(X_test_sbert_all_scaled)
    acc, prec, rec, f1, cm_e = calculate_metrics(y_test, preds_e)
    results.append({'Model': 'SBERT + Ling + Spec + LR (Hybrid)', 'Accuracy': acc, 'Precision (Amb)': prec, 'Recall (Amb)': rec, 'F1 (Amb)': f1})

    res_df = pd.DataFrame(results)
    print("\n--- ABLATION RESULTS ---")
    print(res_df.to_string(index=False))
    
    os.makedirs('evaluation', exist_ok=True)
    res_df.to_csv("evaluation/ablation_results.csv", index=False)
    res_df.to_csv("evaluation/model_comparison.csv", index=False)
    print("\nAblation results saved to evaluation/model_comparison.csv")
    
    # Generate Confusion Matrix PNG
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_e, annot=True, fmt='d', cmap='Blues', xticklabels=["Ambiguous", "Not Ambiguous"], yticklabels=["Ambiguous", "Not Ambiguous"])
    plt.title('Hybrid Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation/confusion_matrix.png')
    plt.close()
    print("Saved confusion matrix to evaluation/confusion_matrix.png")
    
    # Generate Classification Report CSV
    report_dict = classification_report(y_test, preds_e, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv('evaluation/classification_report.csv')
    print("Saved classification report to evaluation/classification_report.csv")

    # 10. Save Pipeline
    print("\n[10/10] Saving New Pipeline (Phase 22)...")
    
    # Remove old stale models
    for file in os.listdir("saved_models"):
        if file.endswith(".pkl"):
            os.remove(os.path.join("saved_models", file))
            print(f"Removed stale model: {file}")

    saver = ModelSaver()
    # Save the base model pipeline - explicitly as primary_classifier
    saver.save_pipeline(
        primary_classifier=calibrated_lr,
        scaler=scaler
    )
    
    print("\nTRAINING PIPELINE COMPLETE! You can now run diagnostic.py or main.py.")
    print("="*60)

if __name__ == "__main__":
    train_system()
