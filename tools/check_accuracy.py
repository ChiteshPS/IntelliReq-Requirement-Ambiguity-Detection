"""
tools/check_accuracy.py — Standalone Model Accuracy Reporter

Computes and prints exact accuracy metrics across:
  1. Internal Held-out Test Split (data/intellireq_clean_dataset_v4.csv)
  2. Independent Challenge Test Set (data/challenge_test_set.csv)
  3. Adversarial Test Set (data/adversarial_test_set.csv)
"""
import os
import sys
import json
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings("ignore")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from models.predictor import ModelPredictor

def main():
    print("=" * 72)
    print("        INTELLIREQ — COMPREHENSIVE MODEL ACCURACY REPORT")
    print("=" * 72)

    # 1. Load artifacts
    model = joblib.load("saved_models/model.joblib")
    scaler = joblib.load("saved_models/scaler.joblib")
    with open("saved_models/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    predictor = ModelPredictor(
        trained_model=model,
        upper_threshold=meta.get("upper_threshold", 0.75),
        lower_threshold=meta.get("lower_threshold", 0.25)
    )
    fe = FeatureExtractor()
    sbert = SBERTExtractor()

    def extract_X(df, text_col):
        X_ling = fe.process_dataframe(df, text_col).drop(columns=df.columns)
        X_sbert = sbert.transform(df[text_col])
        X = pd.concat([X_sbert, X_ling], axis=1)
        return pd.DataFrame(scaler.transform(X), columns=X.columns), X_ling

    # -------------------------------------------------------------------------
    # 1. HELD-OUT TEST SPLIT (v4 Clean Dataset)
    # -------------------------------------------------------------------------
    df_clean = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    df_clean["semantic_group_id"] = df_clean["semantic_group_id"].fillna(
        df_clean.index.to_series().astype(str)
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df_clean, groups=df_clean["semantic_group_id"]))
    temp_df = df_clean.iloc[temp_idx].reset_index(drop=True)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["semantic_group_id"]))
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    X_test_scaled, X_test_ling = extract_X(test_df, "requirement")
    y_test_true = test_df["label"]

    raw_preds_test = model.predict(X_test_scaled)
    raw_acc_test = accuracy_score(y_test_true, raw_preds_test)

    pred_3tier_test, _ = predictor.predict(X_test_scaled)
    # Strict accuracy: count only exact matches (NEEDS REVIEW counts as not classified)
    strict_acc_test = accuracy_score(
        y_test_true,
        [p if p in ("Ambiguous", "Not Ambiguous") else "Incorrect" for p in pred_3tier_test]
    )

    print("\n[1] HELD-OUT CLEAN TEST SET (data/intellireq_clean_dataset_v4.csv)")
    print(f"    Total Test Requirements : {len(test_df)}")
    print(f"    Raw Binary Model Accuracy : {raw_acc_test * 100:.2f}%")
    print(f"    Calibrated Strict Accuracy: {strict_acc_test * 100:.2f}%")

    # -------------------------------------------------------------------------
    # 2. INDEPENDENT CHALLENGE TEST SET
    # -------------------------------------------------------------------------
    df_chall = pd.read_csv("data/challenge_test_set.csv")
    df_chall_bin = df_chall[df_chall["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].copy()
    X_chall_scaled, X_chall_ling = extract_X(df_chall_bin, "Requirement_Text")
    y_chall_true = df_chall_bin["Ambiguity_Label"]

    raw_preds_chall = model.predict(X_chall_scaled)
    raw_acc_chall = accuracy_score(y_chall_true, raw_preds_chall)

    pred_3tier_chall, _ = predictor.predict(X_chall_scaled)
    strict_acc_chall = accuracy_score(
        y_chall_true,
        [p if p in ("Ambiguous", "Not Ambiguous") else "Incorrect" for p in pred_3tier_chall]
    )

    print("\n[2] INDEPENDENT CHALLENGE TEST SET (data/challenge_test_set.csv)")
    print(f"    Binary Instances Evaluated: {len(df_chall_bin)} (40 Ambiguous, 40 Not Ambiguous)")
    print(f"    Raw Binary Model Accuracy : {raw_acc_chall * 100:.2f}%")
    print(f"    Calibrated Strict Accuracy: {strict_acc_chall * 100:.2f}%")

    # -------------------------------------------------------------------------
    # 3. ADVERSARIAL TEST SET
    # -------------------------------------------------------------------------
    df_adv = pd.read_csv("data/adversarial_test_set.csv")
    label_col = "expected_label" if "expected_label" in df_adv.columns else "Ambiguity_Label"
    req_col = "requirement" if "requirement" in df_adv.columns else "Requirement_Text"
    df_adv_bin = df_adv[df_adv[label_col].isin(["Ambiguous", "Not Ambiguous"])].copy()
    X_adv_scaled, X_adv_ling = extract_X(df_adv_bin, req_col)
    y_adv_true = df_adv_bin[label_col]

    raw_preds_adv = model.predict(X_adv_scaled)
    raw_acc_adv = accuracy_score(y_adv_true, raw_preds_adv)

    pred_3tier_adv, _ = predictor.predict(X_adv_scaled)
    strict_acc_adv = accuracy_score(
        y_adv_true,
        [p if p in ("Ambiguous", "Not Ambiguous") else "Incorrect" for p in pred_3tier_adv]
    )

    print("\n[3] ADVERSARIAL TEST SET (data/adversarial_test_set.csv)")
    print(f"    Adversarial Instances     : {len(df_adv_bin)} (8 Ambiguous, 8 Not Ambiguous)")
    print(f"    Raw Binary Model Accuracy : {raw_acc_adv * 100:.2f}%")
    print(f"    Calibrated Strict Accuracy: {strict_acc_adv * 100:.2f}%")

    print("\n" + "=" * 72)
    print("                          SUMMARY TABLE")
    print("=" * 72)
    summary_data = [
        {"Dataset": "Held-out Clean Test Split (v4)", "Size": len(test_df), "Raw Binary Acc": f"{raw_acc_test*100:.2f}%", "Strict Calibrated Acc": f"{strict_acc_test*100:.2f}%"},
        {"Dataset": "Independent Challenge Test Set", "Size": len(df_chall_bin), "Raw Binary Acc": f"{raw_acc_chall*100:.2f}%", "Strict Calibrated Acc": f"{strict_acc_chall*100:.2f}%"},
        {"Dataset": "Adversarial Stress Test Set", "Size": len(df_adv_bin), "Raw Binary Acc": f"{raw_acc_adv*100:.2f}%", "Strict Calibrated Acc": f"{strict_acc_adv*100:.2f}%"},
    ]
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 72)

if __name__ == "__main__":
    main()
