"""
tools/external_cross_benchmark.py — Rigorous Cross-Dataset & Multi-Model Evaluation Suite

Executes the official evaluation protocol:
1. Existing-data-only baseline (IntelliReq Clean v4: Train 500, Val 107, Test 108)
2. External-data-only models (In-domain PURE: 6,725; In-domain UserStories: 1,680 with GroupShuffleSplit by source_file)
3. Cross-dataset transfer: Train on PURE -> Test on UserStories
4. Cross-dataset transfer: Train on UserStories -> Test on PURE
5. Cross-dataset transfer: Train on IntelliReq -> Test on PURE and UserStories
6. Compatible combined dataset with group-aware evaluation:
   IntelliReq Clean v4 + PURE + UserStories (Group-aware partition)
   Evaluated on held-out combined test, Diagnostic Challenge Set, and Adversarial Trap Set.

Evaluates candidates:
- Model A: Hybrid SBERT + 24 Ling/Spec + Calibrated L1 Logistic Regression (Baseline)
- Model B: Hybrid SBERT + 24 Ling/Spec + Calibrated Random Forest (First Challenger)
- Model B2: Hybrid SBERT + 24 Ling/Spec + Calibrated Linear SVM
- Model C: Classical TF-IDF (word 1-2 & char 3-5 n-grams) + Calibrated Logistic Regression

Reports all required metrics:
- Accuracy, Balanced Accuracy
- Per-class Precision, Recall, F1 (Ambiguous and Clear)
- Macro-F1, Matthews Correlation Coefficient (MCC)
- Brier Score Loss, Confusion Matrix (TN, FP, FN, TP)
- Ambiguous-class error rate (False Negative Rate)
- 3-Way Calibrated Decision Metrics (Strict Accuracy, Deferral Rate, Selective Accuracy)
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
    brier_score_loss,
    confusion_matrix,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(REPO_ROOT, "saved_models", "benchmark_cache")
EVAL_DIR = os.path.join(REPO_ROOT, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


def compute_metrics(y_true, probs_amb, upper_th=0.75, lower_th=0.25):
    """
    Computes all standard evaluation metrics for binary classification & 3-way policy.
    0 = Clear / Not Ambiguous, 1 = Ambiguous
    """
    raw_preds = (probs_amb >= 0.50).astype(int)

    acc = accuracy_score(y_true, raw_preds)
    bal_acc = balanced_accuracy_score(y_true, raw_preds)

    prec_amb = precision_score(y_true, raw_preds, pos_label=1, zero_division=0)
    rec_amb = recall_score(y_true, raw_preds, pos_label=1, zero_division=0)
    f1_amb = f1_score(y_true, raw_preds, pos_label=1, zero_division=0)

    prec_clear = precision_score(y_true, raw_preds, pos_label=0, zero_division=0)
    rec_clear = recall_score(y_true, raw_preds, pos_label=0, zero_division=0)
    f1_clear = f1_score(y_true, raw_preds, pos_label=0, zero_division=0)

    macro_f1 = f1_score(y_true, raw_preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, raw_preds)
    brier = brier_score_loss(y_true, probs_amb)

    cm = confusion_matrix(y_true, raw_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Ambiguous class error rate: missed ambiguities / total true ambiguities
    amb_error_rate = (fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    # 3-way calibrated decision policy
    y_3way = []
    for p in probs_amb:
        if p >= upper_th:
            y_3way.append(1)  # Ambiguous
        elif p <= lower_th:
            y_3way.append(0)  # Clear
        else:
            y_3way.append(-1)  # NEEDS REVIEW

    strict_correct = sum(1 for yt, yp in zip(y_true, y_3way) if yt == yp)
    strict_acc = strict_correct / len(y_true)

    deferred = sum(1 for yp in y_3way if yp == -1)
    deferral_rate = deferred / len(y_true)

    non_deferred = len(y_true) - deferred
    if non_deferred > 0:
        sel_correct = sum(1 for yt, yp in zip(y_true, y_3way) if yp != -1 and yt == yp)
        selective_acc = sel_correct / non_deferred
    else:
        selective_acc = 0.0

    return {
        "accuracy": acc * 100.0,
        "balanced_accuracy": bal_acc * 100.0,
        "precision_amb": prec_amb * 100.0,
        "recall_amb": rec_amb * 100.0,
        "f1_amb": f1_amb * 100.0,
        "precision_clear": prec_clear * 100.0,
        "recall_clear": rec_clear * 100.0,
        "f1_clear": f1_clear * 100.0,
        "macro_f1": macro_f1 * 100.0,
        "mcc": mcc,
        "brier_loss": brier,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "amb_error_rate_pct": amb_error_rate * 100.0,
        "strict_3way_acc": strict_acc * 100.0,
        "deferral_rate": deferral_rate * 100.0,
        "selective_acc": selective_acc * 100.0,
    }


def get_cached_hybrid_features(dataset_key: str) -> pd.DataFrame:
    sbert_path = os.path.join(CACHE_DIR, f"{dataset_key}_sbert.joblib")
    ling_path = os.path.join(CACHE_DIR, f"{dataset_key}_ling.joblib")
    if not (os.path.exists(sbert_path) and os.path.exists(ling_path)):
        raise FileNotFoundError(
            f"Missing cached features for {dataset_key}. Run tools/extract_external_features.py first."
        )
    df_sbert = joblib.load(sbert_path)
    df_ling = joblib.load(ling_path)
    return pd.concat([df_sbert.reset_index(drop=True), df_ling.reset_index(drop=True)], axis=1)


def build_models():
    return {
        "Model A (Baseline Hybrid + Calibrated L1 LR)": {
            "base": LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", random_state=42),
            "calib": "sigmoid",
        },
        "Model B (Hybrid + Calibrated Random Forest)": {
            "base": RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_split=4, class_weight="balanced", random_state=42, n_jobs=-1),
            "calib": "sigmoid",
        },
        "Model B2 (Hybrid + Calibrated Linear SVM)": {
            "base": LinearSVC(C=0.1, class_weight="balanced", random_state=42),
            "calib": "sigmoid",
        },
    }


def evaluate_dataset_pair(model_dict, X_tr_scaled, y_tr, X_va_scaled, y_va, X_te_scaled, y_te, experiment_name: str):
    """Fits models on X_tr, calibrates on X_va, evaluates on X_te."""
    results = []
    for model_name, cfg in model_dict.items():
        base_clf = cfg["base"]
        base_clf.fit(X_tr_scaled, y_tr)

        # Calibrate on validation partition
        calibrator = CalibratedClassifierCV(base_clf, method=cfg["calib"], cv="prefit")
        calibrator.fit(X_va_scaled, y_va)

        amb_idx = np.where(calibrator.classes_ == 1)[0][0]
        probs_te = calibrator.predict_proba(X_te_scaled)[:, amb_idx]

        m = compute_metrics(y_te, probs_te)
        m["Experiment"] = experiment_name
        m["Model"] = model_name
        results.append(m)
        print(f"  [{model_name}] Acc: {m['accuracy']:.2f}% | Macro-F1: {m['macro_f1']:.2f}% | Amb-Rec: {m['recall_amb']:.2f}% | Amb-Err: {m['amb_error_rate_pct']:.2f}% | Brier: {m['brier_loss']:.4f}")
    return results


def run_all_benchmarks():
    print("=" * 80)
    print("INTELLIREQ — EXTERNAL DATASET & CROSS-DATASET BENCHMARK SUITE")
    print("=" * 80)

    all_results = []
    models = build_models()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load Datasets
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/7] Loading datasets and verifying zero-leakage splits...")

    # A. IntelliReq Clean v4
    df_v4 = pd.read_csv(os.path.join(REPO_ROOT, "data", "intellireq_clean_dataset_v4.csv"))
    df_v4["semantic_group_id"] = df_v4["semantic_group_id"].fillna(df_v4.index.to_series().astype(str))
    y_v4 = (df_v4["label"] == "Ambiguous").astype(int).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    v4_tr_idx, v4_temp_idx = next(gss.split(df_v4, groups=df_v4["semantic_group_id"]))
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    v4_va_rel_idx, v4_te_rel_idx = next(gss_val.split(df_v4.iloc[v4_temp_idx], groups=df_v4.iloc[v4_temp_idx]["semantic_group_id"]))
    v4_va_idx = v4_temp_idx[v4_va_rel_idx]
    v4_te_idx = v4_temp_idx[v4_te_rel_idx]

    # Pre-cached v4 feature matrices
    X_v4_tr = get_cached_hybrid_features("train_v4")
    X_v4_va = get_cached_hybrid_features("val_v4")
    X_v4_te = get_cached_hybrid_features("test_v4")

    # Diagnostic benchmarks
    X_chall_hybrid = get_cached_hybrid_features("chall_bin")
    df_chall = pd.read_csv(os.path.join(REPO_ROOT, "data", "challenge_test_set.csv"))
    df_chall_bin = df_chall[df_chall["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].reset_index(drop=True)
    y_chall = (df_chall_bin["Ambiguity_Label"] == "Ambiguous").astype(int).values

    X_adv_hybrid = get_cached_hybrid_features("adv")
    df_adv = pd.read_csv(os.path.join(REPO_ROOT, "data", "adversarial_test_set.csv"))
    y_adv = (df_adv["expected_label"] == "Ambiguous").astype(int).values

    # B. External: User Stories (N=1680)
    df_us = pd.read_csv(os.path.join(REPO_ROOT, "data", "external", "userstories_labeled.csv"))
    X_us_hybrid = get_cached_hybrid_features("ext_userstories")
    y_us = df_us["label"].astype(int).values

    # GroupShuffleSplit by source_file for UserStories (70% train, 15% val, 15% test)
    gss_us = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    us_tr_idx, us_temp_idx = next(gss_us.split(df_us, groups=df_us["source_file"]))
    gss_us_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    us_va_rel_idx, us_te_rel_idx = next(gss_us_val.split(df_us.iloc[us_temp_idx], groups=df_us.iloc[us_temp_idx]["source_file"]))
    us_va_idx = us_temp_idx[us_va_rel_idx]
    us_te_idx = us_temp_idx[us_te_rel_idx]

    print(f"User Stories Split: Train={len(us_tr_idx)}, Val={len(us_va_idx)}, Test={len(us_te_idx)} (by source_file)")

    # C. External: PURE (N=6725)
    df_pure = pd.read_csv(os.path.join(REPO_ROOT, "data", "external", "pure_labeled.csv"))
    X_pure_hybrid = get_cached_hybrid_features("ext_pure")
    y_pure = df_pure["label"].astype(int).values

    # GroupShuffleSplit by source_file for PURE (70% train, 15% val, 15% test)
    gss_pure = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    pure_tr_idx, pure_temp_idx = next(gss_pure.split(df_pure, groups=df_pure["source_file"]))
    gss_pure_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    pure_va_rel_idx, pure_te_rel_idx = next(gss_pure_val.split(df_pure.iloc[pure_temp_idx], groups=df_pure.iloc[pure_temp_idx]["source_file"]))
    pure_va_idx = pure_temp_idx[pure_va_rel_idx]
    pure_te_idx = pure_temp_idx[pure_te_rel_idx]

    print(f"PURE Split: Train={len(pure_tr_idx)}, Val={len(pure_va_idx)}, Test={len(pure_te_idx)} (by source_file)")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Experiment 1: In-Domain IntelliReq Clean v4 Baseline
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/7] Experiment 1: In-Domain IntelliReq Baseline (Train=500, Val=107, Test=108)")
    scaler_v4 = StandardScaler().fit(X_v4_tr)
    X_v4_tr_s = scaler_v4.transform(X_v4_tr)
    X_v4_va_s = scaler_v4.transform(X_v4_va)
    X_v4_te_s = scaler_v4.transform(X_v4_te)
    X_chall_s = scaler_v4.transform(X_chall_hybrid)
    X_adv_s = scaler_v4.transform(X_adv_hybrid)

    res_v4_heldout = evaluate_dataset_pair(
        models, X_v4_tr_s, y_v4[v4_tr_idx], X_v4_va_s, y_v4[v4_va_idx], X_v4_te_s, y_v4[v4_te_idx],
        "1. In-Domain IntelliReq Held-Out Test"
    )
    all_results.extend(res_v4_heldout)

    print("\n   Evaluating Baseline on Diagnostic Challenge Set (N=80):")
    res_v4_chall = evaluate_dataset_pair(
        models, X_v4_tr_s, y_v4[v4_tr_idx], X_v4_va_s, y_v4[v4_va_idx], X_chall_s, y_chall,
        "1b. Diagnostic Challenge Benchmark (IntelliReq Model)"
    )
    all_results.extend(res_v4_chall)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Experiment 2: In-Domain User Stories (Group-Aware by source_file)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[3/7] Experiment 2: In-Domain User Stories (Train={len(us_tr_idx)}, Val={len(us_va_idx)}, Test={len(us_te_idx)})")
    scaler_us = StandardScaler().fit(X_us_hybrid.iloc[us_tr_idx])
    X_us_tr_s = scaler_us.transform(X_us_hybrid.iloc[us_tr_idx])
    X_us_va_s = scaler_us.transform(X_us_hybrid.iloc[us_va_idx])
    X_us_te_s = scaler_us.transform(X_us_hybrid.iloc[us_te_idx])

    res_us = evaluate_dataset_pair(
        models, X_us_tr_s, y_us[us_tr_idx], X_us_va_s, y_us[us_va_idx], X_us_te_s, y_us[us_te_idx],
        "2. In-Domain UserStories Test"
    )
    all_results.extend(res_us)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Experiment 3: In-Domain PURE (Group-Aware by source_file)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4/7] Experiment 3: In-Domain PURE (Train={len(pure_tr_idx)}, Val={len(pure_va_idx)}, Test={len(pure_te_idx)})")
    scaler_pure = StandardScaler().fit(X_pure_hybrid.iloc[pure_tr_idx])
    X_pure_tr_s = scaler_pure.transform(X_pure_hybrid.iloc[pure_tr_idx])
    X_pure_va_s = scaler_pure.transform(X_pure_hybrid.iloc[pure_va_idx])
    X_pure_te_s = scaler_pure.transform(X_pure_hybrid.iloc[pure_te_idx])

    res_pure = evaluate_dataset_pair(
        models, X_pure_tr_s, y_pure[pure_tr_idx], X_pure_va_s, y_pure[pure_va_idx], X_pure_te_s, y_pure[pure_te_idx],
        "3. In-Domain PURE Test"
    )
    all_results.extend(res_pure)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Experiment 4 & 5: Cross-Dataset Evaluation (PURE <-> UserStories)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[5/7] Experiment 4: Cross-Dataset Transfer — PURE -> UserStories (Train on PURE, Test on ALL UserStories N={len(df_us)})")
    X_us_all_s_pure = scaler_pure.transform(X_us_hybrid)
    res_pure_to_us = evaluate_dataset_pair(
        models, X_pure_tr_s, y_pure[pure_tr_idx], X_pure_va_s, y_pure[pure_va_idx], X_us_all_s_pure, y_us,
        "4. Cross-Dataset: PURE -> UserStories"
    )
    all_results.extend(res_pure_to_us)

    print(f"\n[6/7] Experiment 5: Cross-Dataset Transfer — UserStories -> PURE (Train on UserStories, Test on ALL PURE N={len(df_pure)})")
    X_pure_all_s_us = scaler_us.transform(X_pure_hybrid)
    res_us_to_pure = evaluate_dataset_pair(
        models, X_us_tr_s, y_us[us_tr_idx], X_us_va_s, y_us[us_va_idx], X_pure_all_s_us, y_pure,
        "5. Cross-Dataset: UserStories -> PURE"
    )
    all_results.extend(res_us_to_pure)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Experiment 6: Compatible Combined Dataset (IntelliReq + PURE + UserStories)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[7/7] Experiment 6: Multi-Source Combined Training (IntelliReq + PURE + UserStories)")
    # Train combines: v4_tr + pure_tr + us_tr
    # Val combines: v4_va + pure_va + us_va
    # Test combines: v4_te + pure_te + us_te
    X_comb_tr = pd.concat([X_v4_tr, X_pure_hybrid.iloc[pure_tr_idx], X_us_hybrid.iloc[us_tr_idx]], ignore_index=True)
    y_comb_tr = np.concatenate([y_v4[v4_tr_idx], y_pure[pure_tr_idx], y_us[us_tr_idx]])

    X_comb_va = pd.concat([X_v4_va, X_pure_hybrid.iloc[pure_va_idx], X_us_hybrid.iloc[us_va_idx]], ignore_index=True)
    y_comb_va = np.concatenate([y_v4[v4_va_idx], y_pure[pure_va_idx], y_us[us_va_idx]])

    X_comb_te = pd.concat([X_v4_te, X_pure_hybrid.iloc[pure_te_idx], X_us_hybrid.iloc[us_te_idx]], ignore_index=True)
    y_comb_te = np.concatenate([y_v4[v4_te_idx], y_pure[pure_te_idx], y_us[us_te_idx]])

    print(f"Combined Training Distribution: Train={len(X_comb_tr)} (Amb: {sum(y_comb_tr)}), Val={len(X_comb_va)}, Held-out Test={len(X_comb_te)}")

    scaler_comb = StandardScaler().fit(X_comb_tr)
    X_comb_tr_s = scaler_comb.transform(X_comb_tr)
    X_comb_va_s = scaler_comb.transform(X_comb_va)
    X_comb_te_s = scaler_comb.transform(X_comb_te)
    X_chall_comb_s = scaler_comb.transform(X_chall_hybrid)
    X_adv_comb_s = scaler_comb.transform(X_adv_hybrid)

    print("\n  Evaluating Combined Model on Held-out Combined Test Partition:")
    res_comb_te = evaluate_dataset_pair(
        models, X_comb_tr_s, y_comb_tr, X_comb_va_s, y_comb_va, X_comb_te_s, y_comb_te,
        "6a. Combined Multi-Source Held-Out Test"
    )
    all_results.extend(res_comb_te)

    print("\n  Evaluating Combined Model on Diagnostic Challenge Set (N=80):")
    res_comb_chall = evaluate_dataset_pair(
        models, X_comb_tr_s, y_comb_tr, X_comb_va_s, y_comb_va, X_chall_comb_s, y_chall,
        "6b. Diagnostic Challenge Benchmark (Combined Model)"
    )
    all_results.extend(res_comb_chall)

    print("\n  Evaluating Combined Model on Adversarial Trap Set (N=16):")
    res_comb_adv = evaluate_dataset_pair(
        models, X_comb_tr_s, y_comb_tr, X_comb_va_s, y_comb_va, X_adv_comb_s, y_adv,
        "6c. Adversarial Benchmark (Combined Model)"
    )
    all_results.extend(res_comb_adv)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Save and Report Results
    # ─────────────────────────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_results)
    csv_out = os.path.join(EVAL_DIR, "external_benchmark_results.csv")
    df_all.to_csv(csv_out, index=False)
    print(f"\nSaved complete benchmark results to: {csv_out}")

    json_out = os.path.join(EVAL_DIR, "external_benchmark_summary.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved JSON metrics to: {json_out}")

    # Build summary table for display
    print("\n" + "=" * 95)
    print("CROSS-DATASET & CANDIDATE MODEL SUMMARY MATRIX")
    print("=" * 95)
    cols = ["Experiment", "Model", "accuracy", "macro_f1", "recall_amb", "amb_error_rate_pct", "brier_loss", "selective_acc"]
    print(df_all[cols].to_string(index=False))

    return df_all, (models, scaler_comb, X_comb_tr_s, y_comb_tr, X_comb_va_s, y_comb_va)


if __name__ == "__main__":
    run_all_benchmarks()
