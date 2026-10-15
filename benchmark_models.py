"""
benchmark_models.py — Trustworthy, Rigorous ML Benchmark Suite for IntelliReq

Implements Phase 2 benchmarking:
- Model A: Hybrid SBERT + Ling/Spec + Calibrated Logistic Regression (L1)
- Model A2: Hybrid SBERT + Ling/Spec + Calibrated Logistic Regression (L2)
- Model B: Hybrid SBERT + Ling/Spec + Calibrated Linear SVM (LinearSVC + Sigmoid)
- Model C: Classical TF-IDF (word 1-2 & char 3-5 n-grams) + Calibrated Logistic Regression
- Model C2: Classical TF-IDF + Calibrated Linear SVM
- Model D: Hybrid SBERT + Ling/Spec + Calibrated Random Forest

Evaluation metrics reported:
- Accuracy, Balanced Accuracy
- Precision, Recall, F1 for Ambiguous and Not Ambiguous
- Macro-F1
- Matthews Correlation Coefficient (MCC)
- Precision-Recall AUC (PR-AUC)
- Brier Score Loss
- Full confusion matrices (TP, FN, FP, TN)
- Calibrated 3-way decision metrics (Strict Accuracy, Deferral Rate, Selective Accuracy)

Evaluated across:
1. Held-out Clean Test Set (data/intellireq_clean_dataset_v4.csv)
2. Independent Out-of-Domain Challenge Benchmark (data/challenge_test_set.csv)
3. Adversarial Distractor Trap Set (data/adversarial_test_set.csv)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor

CACHE_DIR = "saved_models/benchmark_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cached_embeddings(texts: list[str], sbert_extractor: SBERTExtractor, cache_key: str) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}_sbert.joblib")
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    df_emb = sbert_extractor.transform(texts)
    joblib.dump(df_emb, cache_path)
    return df_emb


def get_cached_ling_features(df: pd.DataFrame, text_col: str, feat_extractor: FeatureExtractor, cache_key: str) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}_ling.joblib")
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    X_ling = feat_extractor.process_dataframe(df, text_col).drop(columns=df.columns)
    joblib.dump(X_ling, cache_path)
    return X_ling


def compute_all_metrics(y_true_binary, probs_amb, y_3way_pred=None, upper_th=0.75, lower_th=0.25):
    """
    Computes full classification, calibration, and 3-way decision metrics.
    y_true_binary: array of 0 (Not Ambiguous) and 1 (Ambiguous)
    probs_amb: probability of Ambiguous class (0.0 to 1.0)
    """
    raw_preds = (probs_amb >= 0.50).astype(int)
    acc = accuracy_score(y_true_binary, raw_preds)
    bal_acc = balanced_accuracy_score(y_true_binary, raw_preds)
    prec_amb = precision_score(y_true_binary, raw_preds, pos_label=1, zero_division=0)
    rec_amb = recall_score(y_true_binary, raw_preds, pos_label=1, zero_division=0)
    f1_amb = f1_score(y_true_binary, raw_preds, pos_label=1, zero_division=0)

    prec_clear = precision_score(y_true_binary, raw_preds, pos_label=0, zero_division=0)
    rec_clear = recall_score(y_true_binary, raw_preds, pos_label=0, zero_division=0)
    f1_clear = f1_score(y_true_binary, raw_preds, pos_label=0, zero_division=0)

    macro_f1 = f1_score(y_true_binary, raw_preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true_binary, raw_preds)

    pr_auc = average_precision_score(y_true_binary, probs_amb)
    brier = brier_score_loss(y_true_binary, probs_amb)

    cm = confusion_matrix(y_true_binary, raw_preds, labels=[1, 0])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]

    # 3-way decision policy metrics
    if y_3way_pred is None:
        y_3way_pred = []
        for p in probs_amb:
            if p >= upper_th:
                y_3way_pred.append("Ambiguous")
            elif p <= lower_th:
                y_3way_pred.append("Not Ambiguous")
            else:
                y_3way_pred.append("NEEDS REVIEW")

    true_str = ["Ambiguous" if y == 1 else "Not Ambiguous" for y in y_true_binary]
    strict_correct = sum(1 for t, p in zip(true_str, y_3way_pred) if t == p)
    strict_acc = strict_correct / len(y_true_binary)

    deferred = sum(1 for p in y_3way_pred if p == "NEEDS REVIEW")
    deferral_rate = deferred / len(y_true_binary)

    # Selective accuracy: accuracy on non-deferred decisions
    non_deferred_total = len(y_true_binary) - deferred
    if non_deferred_total > 0:
        sel_correct = sum(1 for t, p in zip(true_str, y_3way_pred) if p != "NEEDS REVIEW" and t == p)
        selective_acc = sel_correct / non_deferred_total
    else:
        selective_acc = 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_amb": prec_amb,
        "recall_amb": rec_amb,
        "f1_amb": f1_amb,
        "precision_clear": prec_clear,
        "recall_clear": rec_clear,
        "f1_clear": f1_clear,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "strict_acc": strict_acc,
        "deferral_rate": deferral_rate,
        "selective_acc": selective_acc,
    }


def run_benchmark():
    print("=" * 80)
    print("INTELLIREQ — RIGOROUS MACHINE LEARNING BENCHMARK (PHASE 2)")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/6] Loading datasets and verifying partitions...")
    df_clean = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    df_clean["semantic_group_id"] = df_clean["semantic_group_id"].fillna(
        df_clean.index.to_series().astype(str)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df_clean, groups=df_clean["semantic_group_id"]))
    train_df = df_clean.iloc[train_idx].reset_index(drop=True)
    temp_df = df_clean.iloc[temp_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["semantic_group_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    print(f"Data Split: Train={len(train_df)} ({len(train_df)/len(df_clean):.1%}), "
          f"Val={len(val_df)} ({len(val_df)/len(df_clean):.1%}), "
          f"Heldout Test={len(test_df)} ({len(test_df)/len(df_clean):.1%})")

    # Evaluation test sets
    df_chall = pd.read_csv("data/challenge_test_set.csv")
    df_chall_bin = df_chall[df_chall["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].copy().reset_index(drop=True)
    df_chall_border = df_chall[df_chall["Ambiguity_Label"] == "Borderline"].copy().reset_index(drop=True)

    df_adv = pd.read_csv("data/adversarial_test_set.csv")
    df_adv = df_adv.rename(columns={"expected_label": "label"}).reset_index(drop=True)

    # Ground truth binary targets (1 = Ambiguous, 0 = Not Ambiguous)
    y_train = (train_df["label"] == "Ambiguous").astype(int)
    y_val = (val_df["label"] == "Ambiguous").astype(int)
    y_test = (test_df["label"] == "Ambiguous").astype(int)
    y_chall = (df_chall_bin["Ambiguity_Label"] == "Ambiguous").astype(int)
    y_adv = (df_adv["label"] == "Ambiguous").astype(int)

    # 2. Extract & Cache Features
    print("\n[2/6] Extracting & caching features...")
    fe = FeatureExtractor()
    sbert = SBERTExtractor()

    # Linguistic features
    X_train_ling = get_cached_ling_features(train_df, "requirement", fe, "train_v4")
    X_val_ling = get_cached_ling_features(val_df, "requirement", fe, "val_v4")
    X_test_ling = get_cached_ling_features(test_df, "requirement", fe, "test_v4")
    X_chall_ling = get_cached_ling_features(df_chall_bin, "Requirement_Text", fe, "chall_bin")
    X_chall_border_ling = get_cached_ling_features(df_chall_border, "Requirement_Text", fe, "chall_border")
    X_adv_ling = get_cached_ling_features(df_adv, "requirement", fe, "adv")

    # SBERT features
    X_train_sbert = get_cached_embeddings(train_df["requirement"].tolist(), sbert, "train_v4")
    X_val_sbert = get_cached_embeddings(val_df["requirement"].tolist(), sbert, "val_v4")
    X_test_sbert = get_cached_embeddings(test_df["requirement"].tolist(), sbert, "test_v4")
    X_chall_sbert = get_cached_embeddings(df_chall_bin["Requirement_Text"].tolist(), sbert, "chall_bin")
    X_chall_border_sbert = get_cached_embeddings(df_chall_border["Requirement_Text"].tolist(), sbert, "chall_border")
    X_adv_sbert = get_cached_embeddings(df_adv["requirement"].tolist(), sbert, "adv")

    # Concatenate Hybrid matrices
    X_train_hybrid = pd.concat([X_train_sbert, X_train_ling], axis=1)
    X_val_hybrid = pd.concat([X_val_sbert, X_val_ling], axis=1)
    X_test_hybrid = pd.concat([X_test_sbert, X_test_ling], axis=1)
    X_chall_hybrid = pd.concat([X_chall_sbert, X_chall_ling], axis=1)
    X_chall_border_hybrid = pd.concat([X_chall_border_sbert, X_chall_border_ling], axis=1)
    X_adv_hybrid = pd.concat([X_adv_sbert, X_adv_ling], axis=1)

    # Standard scaling fitted EXCLUSIVELY on train_df
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_hybrid), columns=X_train_hybrid.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_hybrid), columns=X_val_hybrid.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_hybrid), columns=X_test_hybrid.columns)
    X_chall_scaled = pd.DataFrame(scaler.transform(X_chall_hybrid), columns=X_chall_hybrid.columns)
    X_chall_border_scaled = pd.DataFrame(scaler.transform(X_chall_border_hybrid), columns=X_chall_border_hybrid.columns)
    X_adv_scaled = pd.DataFrame(scaler.transform(X_adv_hybrid), columns=X_adv_hybrid.columns)

    # 3. TF-IDF Representation (Word + Char n-grams, fitted EXCLUSIVELY on train_df)
    print("\n[3/6] Building classical TF-IDF baseline feature matrix (Word 1-2 & Char 3-5 n-grams)...")
    tfidf_union = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=2500, sublinear_tf=True)),
    ])
    X_train_tfidf = tfidf_union.fit_transform(train_df["requirement"])
    X_val_tfidf = tfidf_union.transform(val_df["requirement"])
    X_test_tfidf = tfidf_union.transform(test_df["requirement"])
    X_chall_tfidf = tfidf_union.transform(df_chall_bin["Requirement_Text"])
    X_adv_tfidf = tfidf_union.transform(df_adv["requirement"])

    # 4. Define Candidate Classifiers
    candidates = {
        "Model A (Baseline Hybrid + Calibrated L1 LR)": {
            "type": "hybrid",
            "base": LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", random_state=42),
            "calib_method": "sigmoid",
        },
        "Model A2 (Hybrid + Calibrated L2 LR)": {
            "type": "hybrid",
            "base": LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", random_state=42),
            "calib_method": "sigmoid",
        },
        "Model B (Hybrid + Calibrated Linear SVM)": {
            "type": "hybrid",
            "base": LinearSVC(C=0.1, class_weight="balanced", random_state=42),
            "calib_method": "sigmoid",
        },
        "Model C (TF-IDF + Calibrated Logistic Regression)": {
            "type": "tfidf",
            "base": LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", random_state=42),
            "calib_method": "sigmoid",
        },
        "Model C2 (TF-IDF + Calibrated Linear SVM)": {
            "type": "tfidf",
            "base": LinearSVC(C=0.1, class_weight="balanced", random_state=42),
            "calib_method": "sigmoid",
        },
        "Model D (Hybrid + Calibrated Random Forest)": {
            "type": "hybrid",
            "base": RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=42),
            "calib_method": "sigmoid",
        },
    }

    results = []

    print("\n[4/6] Training & Evaluating Candidates...")
    for model_name, cfg in candidates.items():
        is_hybrid = cfg["type"] == "hybrid"
        X_tr = X_train_scaled if is_hybrid else X_train_tfidf
        X_va = X_val_scaled if is_hybrid else X_val_tfidf
        X_te = X_test_scaled if is_hybrid else X_test_tfidf
        X_ch = X_chall_scaled if is_hybrid else X_chall_tfidf
        X_ad = X_adv_scaled if is_hybrid else X_adv_tfidf

        base_clf = cfg["base"]
        base_clf.fit(X_tr, y_train)

        # Calibrate using held-out validation set
        calibrator = CalibratedClassifierCV(base_clf, method=cfg["calib_method"], cv="prefit")
        calibrator.fit(X_va, y_val)

        # Predict probabilities of Ambiguous (class index 1)
        # Check class index
        amb_class_idx = np.where(calibrator.classes_ == 1)[0][0]

        probs_val = calibrator.predict_proba(X_va)[:, amb_class_idx]
        probs_test = calibrator.predict_proba(X_te)[:, amb_class_idx]
        probs_chall = calibrator.predict_proba(X_ch)[:, amb_class_idx]
        probs_adv = calibrator.predict_proba(X_ad)[:, amb_class_idx]

        m_val = compute_all_metrics(y_val, probs_val)
        m_test = compute_all_metrics(y_test, probs_test)
        m_chall = compute_all_metrics(y_chall, probs_chall)
        m_adv = compute_all_metrics(y_adv, probs_adv)

        # Borderline evaluation if hybrid
        if is_hybrid:
            probs_border = calibrator.predict_proba(X_chall_border_scaled)[:, amb_class_idx]
            border_review_count = sum(1 for p in probs_border if 0.25 < p < 0.75)
            border_review_pct = border_review_count / len(probs_border) * 100.0
        else:
            border_review_count = 0
            border_review_pct = 0.0

        res_entry = {
            "Model": model_name,
            "Feature_Type": cfg["type"],
            # Held-out Test
            "Heldout_Acc": m_test["accuracy"] * 100.0,
            "Heldout_Bal_Acc": m_test["balanced_accuracy"] * 100.0,
            "Heldout_Macro_F1": m_test["macro_f1"] * 100.0,
            "Heldout_F1_Amb": m_test["f1_amb"] * 100.0,
            "Heldout_Recall_Amb": m_test["recall_amb"] * 100.0,
            "Heldout_Precision_Amb": m_test["precision_amb"] * 100.0,
            "Heldout_MCC": m_test["mcc"],
            "Heldout_Brier": m_test["brier_score"],
            "Heldout_Strict_3Way_Acc": m_test["strict_acc"] * 100.0,
            "Heldout_Deferral_Rate": m_test["deferral_rate"] * 100.0,
            "Heldout_Selective_Acc": m_test["selective_acc"] * 100.0,
            # Challenge Test
            "Challenge_Acc": m_chall["accuracy"] * 100.0,
            "Challenge_Bal_Acc": m_chall["balanced_accuracy"] * 100.0,
            "Challenge_Macro_F1": m_chall["macro_f1"] * 100.0,
            "Challenge_F1_Amb": m_chall["f1_amb"] * 100.0,
            "Challenge_Recall_Amb": m_chall["recall_amb"] * 100.0,
            "Challenge_Precision_Amb": m_chall["precision_amb"] * 100.0,
            "Challenge_MCC": m_chall["mcc"],
            "Challenge_Brier": m_chall["brier_score"],
            "Challenge_Strict_3Way_Acc": m_chall["strict_acc"] * 100.0,
            "Challenge_Deferral_Rate": m_chall["deferral_rate"] * 100.0,
            "Challenge_Selective_Acc": m_chall["selective_acc"] * 100.0,
            "Challenge_Borderline_Review_Rate": border_review_pct,
            # Adversarial Test
            "Adversarial_Acc": m_adv["accuracy"] * 100.0,
            "Adversarial_Macro_F1": m_adv["macro_f1"] * 100.0,
            "Adversarial_Brier": m_adv["brier_score"],
        }
        results.append(res_entry)

        print(f"\n--- {model_name} ---")
        print(f"  Heldout:    Acc={m_test['accuracy']*100:.2f}% | Macro-F1={m_test['macro_f1']*100:.2f}% | Amb-Rec={m_test['recall_amb']*100:.2f}% | MCC={m_test['mcc']:.4f} | Brier={m_test['brier_score']:.4f}")
        print(f"  Challenge:  Acc={m_chall['accuracy']*100:.2f}% | Macro-F1={m_chall['macro_f1']*100:.2f}% | Amb-Rec={m_chall['recall_amb']*100:.2f}% | MCC={m_chall['mcc']:.4f} | Selective-Acc={m_chall['selective_acc']*100:.2f}%")
        print(f"  Adversarial Acc={m_adv['accuracy']*100:.2f}% | Macro-F1={m_adv['macro_f1']*100:.2f}%")

    # 5. Export Benchmark Summary Table
    df_results = pd.DataFrame(results)
    benchmark_csv_path = "evaluation/verified_model_benchmark.csv"
    df_results.to_csv(benchmark_csv_path, index=False)
    print(f"\n[5/6] Saved benchmark results to {benchmark_csv_path}")

    # Also save detailed JSON for API & UI integration
    json_path = "evaluation/verified_benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[6/6] Saved detailed metrics JSON to {json_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (HELD-OUT CLEAN vs CHALLENGE vs ADVERSARIAL)")
    print("=" * 80)
    summary_cols = ["Model", "Heldout_Acc", "Heldout_Macro_F1", "Challenge_Acc", "Challenge_Macro_F1", "Challenge_Selective_Acc", "Adversarial_Acc"]
    print(df_results[summary_cols].to_string(index=False))

    return df_results


if __name__ == "__main__":
    run_benchmark()
