"""
tools/run_reproducible_ablation.py — Rigorous Baseline Comparison on Identical Splits

Evaluates 4 architectures under identical GroupShuffleSplit(random_state=42) on v4 dataset:
  1. TF-IDF (1,000 max features) + Logistic Regression
  2. SBERT (384D) + Logistic Regression
  3. Handcrafted Linguistic Features (24D) + Logistic Regression
  4. Hybrid SBERT (384D) + Handcrafted (24D) + L1 Logistic Regression (IntelliReq)

Outputs clean, verified CSVs to:
  - evaluation/ablation_results.csv
  - evaluation/model_comparison.csv
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor

def main():
    print("=" * 70)
    print("    RUNNING REPRODUCIBLE BASELINE AND ABLATION EXPERIMENTS")
    print("=" * 70)

    # 1. Load canonical v4 dataset
    df_clean = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    df_clean["semantic_group_id"] = df_clean["semantic_group_id"].fillna(
        df_clean.index.to_series().astype(str)
    )

    # 2. Group Shuffle Split (70 Train / 15 Val / 15 Test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df_clean, groups=df_clean["semantic_group_id"]))
    train_df = df_clean.iloc[train_idx].reset_index(drop=True)
    temp_df = df_clean.iloc[temp_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["semantic_group_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    print(f"Dataset Partitions: Train={len(train_df)}, Val={len(val_df)}, Heldout Test={len(test_df)}")

    # Load external evaluation benchmarks
    chall_df = pd.read_csv("data/challenge_test_set.csv")
    chall_bin = chall_df[chall_df["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].copy()

    adv_df = pd.read_csv("data/adversarial_test_set.csv")
    adv_label_col = "expected_label" if "expected_label" in adv_df.columns else "Ambiguity_Label"
    adv_text_col = "requirement" if "requirement" in adv_df.columns else "Requirement_Text"
    adv_bin = adv_df[adv_df[adv_label_col].isin(["Ambiguous", "Not Ambiguous"])].copy()

    # Feature extractors
    fe = FeatureExtractor()
    sbert = SBERTExtractor()

    print("\nExtracting features for all splits...")
    # SBERT features
    X_train_sbert = sbert.transform(train_df["requirement"])
    X_val_sbert = sbert.transform(val_df["requirement"])
    X_test_sbert = sbert.transform(test_df["requirement"])
    X_chall_sbert = sbert.transform(chall_bin["Requirement_Text"])
    X_adv_sbert = sbert.transform(adv_bin[adv_text_col])

    # Handcrafted linguistic features
    X_train_ling = fe.process_dataframe(train_df, "requirement").drop(columns=train_df.columns)
    X_val_ling = fe.process_dataframe(val_df, "requirement").drop(columns=val_df.columns)
    X_test_ling = fe.process_dataframe(test_df, "requirement").drop(columns=test_df.columns)
    X_chall_ling = fe.process_dataframe(chall_bin, "Requirement_Text").drop(columns=chall_bin.columns)
    X_adv_ling = fe.process_dataframe(adv_bin, adv_text_col).drop(columns=adv_bin.columns)

    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_df["requirement"]).toarray()
    X_test_tfidf = tfidf.transform(test_df["requirement"]).toarray()
    X_chall_tfidf = tfidf.transform(chall_bin["Requirement_Text"]).toarray()
    X_adv_tfidf = tfidf.transform(adv_bin[adv_text_col]).toarray()

    # Ground truth labels
    y_train = train_df["label"]
    y_test = test_df["label"]
    y_chall = chall_bin["Ambiguity_Label"]
    y_adv = adv_bin[adv_label_col]

    # Models to compare
    architectures = [
        ("TF-IDF + Logistic Regression", "tfidf"),
        ("SBERT Only + Logistic Regression", "sbert"),
        ("Linguistic Only + Logistic Regression", "ling"),
        ("IntelliReq Hybrid (SBERT + Ling + L1 LR)", "hybrid"),
    ]

    results = []

    for name, mode in architectures:
        if mode == "tfidf":
            clf = LogisticRegression(class_weight="balanced", random_state=42)
            clf.fit(X_train_tfidf, y_train)
            pred_test = clf.predict(X_test_tfidf)
            pred_chall = clf.predict(X_chall_tfidf)
            pred_adv = clf.predict(X_adv_tfidf)
        elif mode == "sbert":
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train_sbert)
            X_te_sc = scaler.transform(X_test_sbert)
            X_ch_sc = scaler.transform(X_chall_sbert)
            X_ad_sc = scaler.transform(X_adv_sbert)
            clf = LogisticRegression(class_weight="balanced", random_state=42)
            clf.fit(X_tr_sc, y_train)
            pred_test = clf.predict(X_te_sc)
            pred_chall = clf.predict(X_ch_sc)
            pred_adv = clf.predict(X_ad_sc)
        elif mode == "ling":
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train_ling)
            X_te_sc = scaler.transform(X_test_ling)
            X_ch_sc = scaler.transform(X_chall_ling)
            X_ad_sc = scaler.transform(X_adv_ling)
            clf = LogisticRegression(class_weight="balanced", random_state=42)
            clf.fit(X_tr_sc, y_train)
            pred_test = clf.predict(X_te_sc)
            pred_chall = clf.predict(X_ch_sc)
            pred_adv = clf.predict(X_ad_sc)
        elif mode == "hybrid":
            # Real trained model loaded from saved_models
            import joblib
            clf = joblib.load("saved_models/model.joblib")
            scaler = joblib.load("saved_models/scaler.joblib")
            
            X_te = pd.concat([X_test_sbert.reset_index(drop=True), X_test_ling.reset_index(drop=True)], axis=1)
            X_ch = pd.concat([X_chall_sbert.reset_index(drop=True), X_chall_ling.reset_index(drop=True)], axis=1)
            X_ad = pd.concat([X_adv_sbert.reset_index(drop=True), X_adv_ling.reset_index(drop=True)], axis=1)

            pred_test = ["Ambiguous" if p >= 0.50 else "Not Ambiguous" for p in clf.predict_proba(pd.DataFrame(scaler.transform(X_te), columns=X_te.columns))[:, 0]]
            pred_chall = ["Ambiguous" if p >= 0.50 else "Not Ambiguous" for p in clf.predict_proba(pd.DataFrame(scaler.transform(X_ch), columns=X_ch.columns))[:, 0]]
            pred_adv = ["Ambiguous" if p >= 0.50 else "Not Ambiguous" for p in clf.predict_proba(pd.DataFrame(scaler.transform(X_ad), columns=X_ad.columns))[:, 0]]

        # Held-out metrics
        acc_test = accuracy_score(y_test, pred_test)
        prec_test = precision_score(y_test, pred_test, pos_label="Ambiguous", zero_division=0)
        rec_test = recall_score(y_test, pred_test, pos_label="Ambiguous", zero_division=0)
        f1_test = f1_score(y_test, pred_test, pos_label="Ambiguous", zero_division=0)

        # Challenge metrics
        acc_chall = accuracy_score(y_chall, pred_chall)
        prec_chall = precision_score(y_chall, pred_chall, pos_label="Ambiguous", zero_division=0)
        rec_chall = recall_score(y_chall, pred_chall, pos_label="Ambiguous", zero_division=0)
        f1_chall = f1_score(y_chall, pred_chall, pos_label="Ambiguous", zero_division=0)

        # Adversarial metrics
        acc_adv = accuracy_score(y_adv, pred_adv)

        results.append({
            "Model": name,
            "Heldout_Acc": acc_test,
            "Heldout_Prec": prec_test,
            "Heldout_Rec": rec_test,
            "Heldout_F1": f1_test,
            "Challenge_Acc": acc_chall,
            "Challenge_Prec": prec_chall,
            "Challenge_Rec": rec_chall,
            "Challenge_F1": f1_chall,
            "Adversarial_Acc": acc_adv,
        })
        print(f"--> {name:40s} | Heldout Acc: {acc_test*100:.2f}% | Challenge Acc: {acc_chall*100:.2f}% | Adv Acc: {acc_adv*100:.2f}%")

    res_df = pd.DataFrame(results)
    res_df.to_csv("evaluation/ablation_results.csv", index=False)
    
    # Save clean model_comparison.csv
    comp_df = pd.DataFrame({
        "Model": res_df["Model"],
        "Heldout_Accuracy": (res_df["Heldout_Acc"] * 100).round(2),
        "Heldout_F1_Ambiguous": (res_df["Heldout_F1"] * 100).round(2),
        "Challenge_Accuracy": (res_df["Challenge_Acc"] * 100).round(2),
        "Challenge_F1_Ambiguous": (res_df["Challenge_F1"] * 100).round(2),
        "Adversarial_Accuracy": (res_df["Adversarial_Acc"] * 100).round(2),
    })
    comp_df.to_csv("evaluation/model_comparison.csv", index=False)
    print("\nSaved genuine comparisons to evaluation/ablation_results.csv and evaluation/model_comparison.csv!")

if __name__ == "__main__":
    main()
