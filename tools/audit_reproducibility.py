"""
tools/audit_reproducibility.py — Deep Forensic Audit of Models, Partitions, and Metrics
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    brier_score_loss
)

def main():
    print("=" * 80)
    print(" INTELLIREQ — REPRODUCIBILITY & EXACT METRIC VERIFICATION AUDIT")
    print("=" * 80)

    # 1. Inspect saved production model and scaler
    model_path = "saved_models/model.joblib"
    scaler_path = "saved_models/scaler.joblib"
    meta_path = "saved_models/metadata.json"

    mtime = datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
    sm_size = os.path.getsize(model_path)
    scaler_size = os.path.getsize(scaler_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("\n--- 1. EXACT PRODUCTION ARTIFACT SPECIFICATION ---")
    print(f"Model Path               : {model_path}")
    print(f"File Size                : {sm_size} bytes")
    print(f"File Last Modified       : {mtime}")
    print(f"Metadata Version         : {meta.get('model_version')}")
    print(f"Metadata Timestamp       : {meta.get('training_timestamp')}")
    print(f"Metadata Dataset         : {meta.get('dataset_version')}")
    print(f"Model Class              : {type(model).__name__}")
    print(f"Base Estimator           : {type(getattr(model, 'base_estimator', getattr(model, 'estimator', None))).__name__}")
    print(f"Calibration Method       : {model.method}")
    print(f"Number of Calibrators    : {len(model.calibrated_classifiers_)}")
    print(f"Thresholds in Metadata   : Upper={meta.get('upper_threshold')}, Lower={meta.get('lower_threshold')}")
    print(f"Scaler Features Count    : {scaler.n_features_in_}")
    print(f"Classes Defined          : {model.classes_}")

    # Inspect the base estimator parameters
    base_est = model.calibrated_classifiers_[0].estimator
    print("\nBase Estimator Hyperparameters:")
    print(f"  Penalty                : {base_est.penalty}")
    print(f"  C (Regularization)     : {base_est.C}")
    print(f"  Solver                 : {base_est.solver}")
    print(f"  Class Weight           : {base_est.class_weight}")
    print(f"  Random State           : {base_est.random_state}")

    # 2. Partitioning replication
    print("\n--- 2. PARTITIONING & DATA REPLICATION ---")
    df_v4 = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    df_v4['semantic_group_id'] = df_v4['semantic_group_id'].fillna(df_v4.index.to_series().astype(str))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df_v4, groups=df_v4['semantic_group_id']))
    train_df = df_v4.iloc[train_idx].reset_index(drop=True)
    temp_df = df_v4.iloc[temp_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df['semantic_group_id']))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    print(f"Total Rows: {len(df_v4)} | Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Check leakage between splits
    train_ids = set(train_df['semantic_group_id'])
    val_ids = set(val_df['semantic_group_id'])
    test_ids = set(test_df['semantic_group_id'])

    print(f"Group Overlap Train-Val : {len(train_ids.intersection(val_ids))}")
    print(f"Group Overlap Train-Test: {len(train_ids.intersection(test_ids))}")
    print(f"Group Overlap Val-Test  : {len(val_ids.intersection(test_ids))}")

    # Load cached features
    CACHE_DIR = "saved_models/benchmark_cache"
    X_test_sbert = joblib.load(os.path.join(CACHE_DIR, "test_v4_sbert.joblib"))
    X_test_ling = joblib.load(os.path.join(CACHE_DIR, "test_v4_ling.joblib"))
    X_test = pd.concat([X_test_sbert, X_test_ling], axis=1)
    X_test_scaled = scaler.transform(X_test)

    y_test_str = test_df['label'].values
    y_test_bin = (y_test_str == "Ambiguous").astype(int)

    # 3. Exact predictions with saved production model
    print("\n--- 3. EVALUATION OF PRODUCTION MODEL (saved_models/model.joblib) ---")
    amb_idx = np.where(model.classes_ == "Ambiguous")[0][0]
    probs = model.predict_proba(X_test_scaled)[:, amb_idx]
    preds_bin = (probs >= 0.50).astype(int)
    preds_str = ["Ambiguous" if p == 1 else "Not Ambiguous" for p in preds_bin]

    acc = accuracy_score(y_test_bin, preds_bin)
    bal_acc = balanced_accuracy_score(y_test_bin, preds_bin)
    p_amb = precision_score(y_test_bin, preds_bin, pos_label=1, zero_division=0)
    r_amb = recall_score(y_test_bin, preds_bin, pos_label=1, zero_division=0)
    f1_amb = f1_score(y_test_bin, preds_bin, pos_label=1, zero_division=0)
    p_clr = precision_score(y_test_bin, preds_bin, pos_label=0, zero_division=0)
    r_clr = recall_score(y_test_bin, preds_bin, pos_label=0, zero_division=0)
    f1_clr = f1_score(y_test_bin, preds_bin, pos_label=0, zero_division=0)
    macro_f1 = f1_score(y_test_bin, preds_bin, average="macro")
    cm = confusion_matrix(y_test_bin, preds_bin, labels=[1, 0])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    brier = brier_score_loss(y_test_bin, probs)
    mcc = matthews_corrcoef(y_test_bin, preds_bin)

    print(f"Total Test Samples        : {len(test_df)}")
    print(f"Correct Predictions       : {(preds_bin == y_test_bin).sum()} / {len(test_df)}")
    print(f"Raw Accuracy              : {acc * 100:.4f}% ({acc * 100:.2f}%)")
    print(f"Balanced Accuracy         : {bal_acc * 100:.4f}% ({bal_acc * 100:.2f}%)")
    print(f"Ambiguous Precision       : {p_amb * 100:.4f}% ({p_amb * 100:.2f}%)")
    print(f"Ambiguous Recall          : {r_amb * 100:.4f}% ({r_amb * 100:.2f}%)")
    print(f"Ambiguous F1-Score        : {f1_amb * 100:.4f}% ({f1_amb * 100:.2f}%)")
    print(f"Clear Precision           : {p_clr * 100:.4f}% ({p_clr * 100:.2f}%)")
    print(f"Clear Recall              : {r_clr * 100:.4f}% ({r_clr * 100:.2f}%)")
    print(f"Clear F1-Score            : {f1_clr * 100:.4f}% ({f1_clr * 100:.2f}%)")
    print(f"Macro-F1                  : {macro_f1 * 100:.4f}% ({macro_f1 * 100:.2f}%)")
    print(f"Matthews Corr (MCC)       : {mcc:.4f}")
    print(f"Brier Score Loss          : {brier:.4f}")
    print(f"Confusion Matrix (Amb=1, Clr=0):")
    print(f"  TP (True Ambiguous)     : {tp}")
    print(f"  FN (Missed Ambiguous)   : {fn}")
    print(f"  FP (False Alarm)        : {fp}")
    print(f"  TN (True Clear)         : {tn}")

    # Inspect the exact errors
    errors = test_df[preds_bin != y_test_bin].copy()
    errors['predicted'] = [preds_str[i] for i in errors.index]
    errors['probability_amb'] = [probs[i] for i in errors.index]
    print(f"\nExact Errors Made by Production Model ({len(errors)} items):")
    for idx, row in errors.iterrows():
        print(f"  Row {idx}: Ground Truth='{row['label']}', Predicted='{row['predicted']}', ProbAmb={row['probability_amb']:.4f}")
        print(f"    Text: \"{row['requirement']}\"")

    # 4. Check three-tier policy
    print("\n--- 4. THREE-TIER POLICY METRICS ---")
    u_th = meta.get("upper_threshold", 0.75)
    l_th = meta.get("lower_threshold", 0.25)
    
    y_3way = []
    for p in probs:
        if p >= u_th:
            y_3way.append("Ambiguous")
        elif p <= l_th:
            y_3way.append("Not Ambiguous")
        else:
            y_3way.append("NEEDS REVIEW")

    strict_corr = sum(1 for t, p in zip(y_test_str, y_3way) if t == p)
    strict_acc = strict_corr / len(y_test_str)
    deferred = sum(1 for p in y_3way if p == "NEEDS REVIEW")
    deferral_rate = deferred / len(y_test_str)
    non_deferred = len(y_test_str) - deferred
    sel_corr = sum(1 for t, p in zip(y_test_str, y_3way) if p != "NEEDS REVIEW" and t == p)
    sel_acc = sel_corr / non_deferred if non_deferred > 0 else 0.0

    print(f"Strict 3-Way Accuracy     : {strict_acc * 100:.2f}% ({strict_corr}/{len(y_test_str)})")
    print(f"Deferred to Review        : {deferred} ({deferral_rate * 100:.2f}%)")
    print(f"Coverage                  : {(1 - deferral_rate) * 100:.2f}% ({non_deferred}/{len(y_test_str)})")
    print(f"Selective Accuracy        : {sel_acc * 100:.2f}% ({sel_corr}/{non_deferred})")

    # 5. Challenge Set Evaluation
    print("\n--- 5. EVALUATION ON DIAGNOSTIC CHALLENGE SET ---")
    df_chall = pd.read_csv("data/challenge_test_set.csv")
    df_chall_bin = df_chall[df_chall["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].reset_index(drop=True)
    y_chall_bin = (df_chall_bin["Ambiguity_Label"] == "Ambiguous").astype(int).values

    X_chall_sbert = joblib.load(os.path.join(CACHE_DIR, "chall_bin_sbert.joblib"))
    X_chall_ling = joblib.load(os.path.join(CACHE_DIR, "chall_bin_ling.joblib"))
    X_chall = pd.concat([X_chall_sbert, X_chall_ling], axis=1)
    X_chall_sc = scaler.transform(X_chall)

    chall_probs = model.predict_proba(X_chall_sc)[:, amb_idx]
    chall_preds = (chall_probs >= 0.50).astype(int)
    chall_acc = accuracy_score(y_chall_bin, chall_preds)
    chall_f1 = f1_score(y_chall_bin, chall_preds, average="macro")
    chall_cm = confusion_matrix(y_chall_bin, chall_preds, labels=[1, 0])

    print(f"Challenge Set Total Binary Rows: {len(df_chall_bin)}")
    print(f"Challenge Raw Accuracy         : {chall_acc * 100:.2f}% ({(chall_preds == y_chall_bin).sum()}/{len(y_chall_bin)})")
    print(f"Challenge Macro-F1             : {chall_f1 * 100:.2f}%")
    print(f"Challenge Confusion Matrix (TP={chall_cm[0,0]}, FN={chall_cm[0,1]}, FP={chall_cm[1,0]}, TN={chall_cm[1,1]})")

    # Selective accuracy on challenge set
    chall_3way = []
    for p in chall_probs:
        if p >= u_th:
            chall_3way.append("Ambiguous")
        elif p <= l_th:
            chall_3way.append("Not Ambiguous")
        else:
            chall_3way.append("NEEDS REVIEW")
    ch_deferred = sum(1 for p in chall_3way if p == "NEEDS REVIEW")
    ch_non_def = len(df_chall_bin) - ch_deferred
    ch_sel_corr = sum(1 for t, p in zip(df_chall_bin["Ambiguity_Label"], chall_3way) if p != "NEEDS REVIEW" and t == p)
    ch_sel_acc = ch_sel_corr / ch_non_def if ch_non_def > 0 else 0.0
    print(f"Challenge Selective Accuracy   : {ch_sel_acc * 100:.2f}% (Coverage: {ch_non_def/len(df_chall_bin)*100:.1f}%)")

    # 6. Comparison of 97.22% vs 98.15% vs 99.07%
    print("\n--- 6. DISCREPANCY RECONCILIATION: 97.22% vs 98.15% vs 99.07% ---")
    print("Investigating where 97.22%, 98.15%, and 99.07% come from across project files...")

if __name__ == "__main__":
    main()
