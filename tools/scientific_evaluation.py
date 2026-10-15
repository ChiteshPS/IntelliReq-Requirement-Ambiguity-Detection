"""
tools/scientific_evaluation.py — Uncompromising Scientific Metrics Reporter

Computes all mandatory metrics across all evaluation sets:
- Total examples
- Correct & Incorrect
- Raw binary accuracy
- Class-level Precision, Recall, F1 (Ambiguous and Not Ambiguous)
- Macro-F1
- Confusion Matrix
- Brier Score (calibration quality)
- Deferred count & Review rate
- Coverage (1 - review_rate)
- Selective Accuracy (accuracy on non-deferred decisions)
- Strict 3-way Accuracy (where deferred predictions on binary ground truth are penalized)
"""
import os
import sys
import json
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from models.predictor import ModelPredictor

def evaluate_split(y_true, probs_amb, upper_thresh=0.75, lower_thresh=0.25, name=""):
    total = len(y_true)
    
    # 1. Raw Binary argmax
    raw_preds = ["Ambiguous" if p >= 0.50 else "Not Ambiguous" for p in probs_amb]
    raw_acc = accuracy_score(y_true, raw_preds)
    raw_correct = sum(1 for yt, yp in zip(y_true, raw_preds) if yt == yp)
    raw_incorrect = total - raw_correct
    
    # Precision, Recall, F1
    p_amb = precision_score(y_true, raw_preds, pos_label="Ambiguous", zero_division=0)
    r_amb = recall_score(y_true, raw_preds, pos_label="Ambiguous", zero_division=0)
    f1_amb = f1_score(y_true, raw_preds, pos_label="Ambiguous", zero_division=0)

    p_not = precision_score(y_true, raw_preds, pos_label="Not Ambiguous", zero_division=0)
    r_not = recall_score(y_true, raw_preds, pos_label="Not Ambiguous", zero_division=0)
    f1_not = f1_score(y_true, raw_preds, pos_label="Not Ambiguous", zero_division=0)
    
    macro_f1 = (f1_amb + f1_not) / 2.0
    
    cm = confusion_matrix(y_true, raw_preds, labels=["Ambiguous", "Not Ambiguous"])
    # cm format: [[TP, FN], [FP, TN]] where pos is Ambiguous
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    
    # Brier score
    y_bin = (y_true == "Ambiguous").astype(int)
    brier = brier_score_loss(y_bin, probs_amb)
    
    # 2. Three-way Calibrated Decision
    three_way_preds = []
    for p in probs_amb:
        if p >= upper_thresh:
            three_way_preds.append("Ambiguous")
        elif p <= lower_thresh:
            three_way_preds.append("Not Ambiguous")
        else:
            three_way_preds.append("NEEDS REVIEW")
            
    review_count = three_way_preds.count("NEEDS REVIEW")
    review_rate = review_count / total
    coverage = 1.0 - review_rate
    
    # Selective accuracy: among predictions that were NOT deferred
    non_deferred_indices = [i for i, p in enumerate(three_way_preds) if p != "NEEDS REVIEW"]
    if non_deferred_indices:
        sel_true = [y_true.iloc[i] for i in non_deferred_indices]
        sel_pred = [three_way_preds[i] for i in non_deferred_indices]
        selective_acc = accuracy_score(sel_true, sel_pred)
    else:
        selective_acc = 0.0
        
    # Strict 3-way accuracy: deferred predictions are counted as incorrect
    strict_preds = [p if p in ("Ambiguous", "Not Ambiguous") else "Incorrect" for p in three_way_preds]
    strict_acc = accuracy_score(y_true, strict_preds)
    
    return {
        "Dataset": name,
        "Total": total,
        "Correct": raw_correct,
        "Incorrect": raw_incorrect,
        "Raw Binary Acc": raw_acc,
        "Prec (Amb)": p_amb,
        "Rec (Amb)": r_amb,
        "F1 (Amb)": f1_amb,
        "Prec (Not)": p_not,
        "Rec (Not)": r_not,
        "F1 (Not)": f1_not,
        "Macro-F1": macro_f1,
        "CM": f"TP={tp}, FN={fn}, FP={fp}, TN={tn}",
        "Brier": brier,
        "Review Count": review_count,
        "Review Rate": review_rate,
        "Coverage": coverage,
        "Selective Acc": selective_acc,
        "Strict 3-Way Acc": strict_acc
    }

def main():
    print("=" * 80)
    print("         INTELLIREQ — COMPREHENSIVE SCIENTIFIC METRIC AUDIT")
    print("=" * 80)
    
    model = joblib.load("saved_models/model.joblib")
    scaler = joblib.load("saved_models/scaler.joblib")
    with open("saved_models/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    fe = FeatureExtractor()
    sbert = SBERTExtractor()
    
    def extract_probs(df, text_col):
        X_ling = fe.process_dataframe(df, text_col).drop(columns=df.columns)
        X_sbert = sbert.transform(df[text_col])
        X = pd.concat([X_sbert, X_ling], axis=1)
        X_sc = pd.DataFrame(scaler.transform(X), columns=X.columns)
        probs_all = model.predict_proba(X_sc)
        amb_idx = list(model.classes_).index("Ambiguous")
        return probs_all[:, amb_idx]
        
    # 1. Clean Held-out Test Split
    df_clean = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    df_clean['semantic_group_id'] = df_clean['semantic_group_id'].fillna(df_clean.index.to_series().astype(str))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df_clean, groups=df_clean['semantic_group_id']))
    temp_df = df_clean.iloc[temp_idx].reset_index(drop=True)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df['semantic_group_id']))
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    probs_clean = extract_probs(test_df, "requirement")
    m_clean = evaluate_split(test_df['label'], probs_clean, name="Held-out Test (N=108)")
    
    # 2. Challenge Set (Binary)
    df_chall = pd.read_csv("data/challenge_test_set.csv")
    df_chall_bin = df_chall[df_chall["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].reset_index(drop=True)
    probs_chall = extract_probs(df_chall_bin, "Requirement_Text")
    m_chall = evaluate_split(df_chall_bin["Ambiguity_Label"], probs_chall, name="Challenge Binary (N=80)")
    
    # 3. Adversarial Set
    df_adv = pd.read_csv("data/adversarial_test_set.csv")
    probs_adv = extract_probs(df_adv, "requirement")
    m_adv = evaluate_split(df_adv["expected_label"], probs_adv, name="Adversarial (N=16)")
    
    # Display full report
    metrics_list = [m_clean, m_chall, m_adv]
    for m in metrics_list:
        print(f"\n--- {m['Dataset']} ---")
        print(f"  Total Examples           : {m['Total']} (Correct: {m['Correct']}, Incorrect: {m['Incorrect']})")
        print(f"  Raw Binary Accuracy      : {m['Raw Binary Acc']*100:.2f}%")
        print(f"  Precision (Ambiguous)    : {m['Prec (Amb)']*100:.2f}% | Recall: {m['Rec (Amb)']*100:.2f}% | F1: {m['F1 (Amb)']*100:.2f}%")
        print(f"  Precision (Not Ambiguous): {m['Prec (Not)']*100:.2f}% | Recall: {m['Rec (Not)']*100:.2f}% | F1: {m['F1 (Not)']*100:.2f}%")
        print(f"  Macro-F1                 : {m['Macro-F1']*100:.2f}%")
        print(f"  Confusion Matrix         : {m['CM']}")
        print(f"  Brier Score Loss         : {m['Brier']:.4f}")
        print(f"  Three-Way Deferral Rate  : {m['Review Rate']*100:.2f}% ({m['Review Count']}/{m['Total']} deferred to NEEDS REVIEW)")
        print(f"  Coverage                 : {m['Coverage']*100:.2f}%")
        print(f"  Selective Accuracy       : {m['Selective Acc']*100:.2f}% (on non-deferred decisions)")
        print(f"  Strict 3-Way Accuracy    : {m['Strict 3-Way Acc']*100:.2f}% (penalizing deferrals)")

    print("\n" + "=" * 80)
    print("                         SUMMARY COMPARISON TABLE")
    print("=" * 80)
    cols = ["Dataset", "Raw Binary Acc", "Macro-F1", "Brier", "Coverage", "Selective Acc", "Strict 3-Way Acc"]
    rows = []
    for m in metrics_list:
        rows.append({
            "Dataset": m["Dataset"],
            "Raw Binary Acc": f"{m['Raw Binary Acc']*100:.2f}%",
            "Macro-F1": f"{m['Macro-F1']*100:.2f}%",
            "Brier": f"{m['Brier']:.4f}",
            "Coverage": f"{m['Coverage']*100:.2f}%",
            "Selective Acc": f"{m['Selective Acc']*100:.2f}%",
            "Strict 3-Way Acc": f"{m['Strict 3-Way Acc']*100:.2f}%"
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main()
