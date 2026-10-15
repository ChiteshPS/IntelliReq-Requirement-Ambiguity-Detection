"""
tools/audit_and_experiment_external.py — Rigorous Cross-Dataset Audit and Benchmark Suite
Implements Steps 1, 3, and 4 for the IntelliReq External Dataset Integration.

Experiments:
1. Existing-data-only baseline (Model A & Model B)
2. External-data-only model (Model A & Model B on PURE)
3. Train on PURE-derived data, test on User Stories
4. Train on User Stories data, test on PURE
5. Compatible combined datasets with group-aware evaluation (Model A, Model B, and Model C)
"""

import os
import sys
import json
import re
import warnings
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor

CACHE_DIR = os.path.join(REPO_ROOT, "saved_models", "benchmark_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
EVAL_DIR = os.path.join(REPO_ROOT, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def compute_comprehensive_metrics(y_true: np.ndarray, probs_amb: np.ndarray, name: str = "") -> Dict[str, Any]:
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
    pr_auc = average_precision_score(y_true, probs_amb)
    brier = brier_score_loss(y_true, probs_amb)

    cm = confusion_matrix(y_true, raw_preds, labels=[1, 0])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])

    # Ambiguous class error rate: false negatives out of actual ambiguous
    amb_err_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "dataset": name,
        "total_samples": len(y_true),
        "accuracy": round(acc * 100.0, 2),
        "balanced_accuracy": round(bal_acc * 100.0, 2),
        "precision_amb": round(prec_amb * 100.0, 2),
        "recall_amb": round(rec_amb * 100.0, 2),
        "f1_amb": round(f1_amb * 100.0, 2),
        "precision_clear": round(prec_clear * 100.0, 2),
        "recall_clear": round(rec_clear * 100.0, 2),
        "f1_clear": round(f1_clear * 100.0, 2),
        "macro_f1": round(macro_f1 * 100.0, 2),
        "mcc": round(float(mcc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "ambiguous_error_rate": round(amb_err_rate * 100.0, 2),
    }


def audit_datasets():
    print("=" * 80)
    print("STEP 1: AUDIT OF EXTERNAL AND EXISTING DATASETS")
    print("=" * 80)

    pure_path = os.path.join(REPO_ROOT, "data", "external", "pure_labeled.csv")
    us_path = os.path.join(REPO_ROOT, "data", "external", "userstories_labeled.csv")
    clean_path = os.path.join(REPO_ROOT, "data", "intellireq_clean_dataset_v4.csv")
    chall_path = os.path.join(REPO_ROOT, "data", "challenge_test_set.csv")
    adv_path = os.path.join(REPO_ROOT, "data", "adversarial_test_set.csv")

    df_pure = pd.read_csv(pure_path)
    df_us = pd.read_csv(us_path)
    df_clean = pd.read_csv(clean_path)
    df_chall = pd.read_csv(chall_path)
    df_adv = pd.read_csv(adv_path)

    audit_summary = {
        "pure": {
            "source": "profzivkovic/software-requirements-ambiguity-detection-replication v2.0.0 (Zenodo 19746839)",
            "license": "MIT License",
            "rows": len(df_pure),
            "columns": list(df_pure.columns),
            "text_column": "requirement_text",
            "label_column": "label",
            "label_mapping": {1: "Ambiguous", 0: "Not Ambiguous"},
            "class_distribution": df_pure["label"].value_counts().to_dict(),
            "source_documents_count": df_pure["source_file"].nunique(),
            "exact_duplicates": int(df_pure.duplicated(subset=["requirement_text"], keep=False).sum()),
            "normalized_duplicates": int(df_pure["requirement_text"].apply(normalize_text).duplicated(keep=False).sum()),
            "duplicate_label_conflicts": int((df_pure.groupby("requirement_text")["label"].nunique() > 1).sum()),
            "top_reasons": df_pure[df_pure["label"] == 1]["reasons"].fillna("").value_counts().head(8).to_dict(),
        },
        "userstories": {
            "source": "profzivkovic/software-requirements-ambiguity-detection-replication v2.0.0 (Zenodo 19746839)",
            "license": "MIT License",
            "rows": len(df_us),
            "columns": list(df_us.columns),
            "text_column": "requirement_text",
            "label_column": "label",
            "label_mapping": {1: "Ambiguous", 0: "Not Ambiguous"},
            "class_distribution": df_us["label"].value_counts().to_dict(),
            "source_projects_count": df_us["source_file"].nunique(),
            "exact_duplicates": int(df_us.duplicated(subset=["requirement_text"], keep=False).sum()),
            "normalized_duplicates": int(df_us["requirement_text"].apply(normalize_text).duplicated(keep=False).sum()),
            "duplicate_label_conflicts": int((df_us.groupby("requirement_text")["label"].nunique() > 1).sum()),
            "top_reasons": df_us[df_us["label"] == 1]["reasons"].fillna("").value_counts().head(8).to_dict(),
        },
        "leakage_checks": {
            "pure_overlap_clean_v4": len(set(df_pure["requirement_text"].str.lower().str.strip()).intersection(set(df_clean["requirement"].str.lower().str.strip()))),
            "pure_overlap_challenge": len(set(df_pure["requirement_text"].str.lower().str.strip()).intersection(set(df_chall["Requirement_Text"].str.lower().str.strip()))),
            "pure_overlap_adversarial": len(set(df_pure["requirement_text"].str.lower().str.strip()).intersection(set(df_adv["requirement"].str.lower().str.strip()))),
            "us_overlap_clean_v4": len(set(df_us["requirement_text"].str.lower().str.strip()).intersection(set(df_clean["requirement"].str.lower().str.strip()))),
            "us_overlap_challenge": len(set(df_us["requirement_text"].str.lower().str.strip()).intersection(set(df_chall["Requirement_Text"].str.lower().str.strip()))),
            "us_overlap_adversarial": len(set(df_us["requirement_text"].str.lower().str.strip()).intersection(set(df_adv["requirement"].str.lower().str.strip()))),
            "pure_overlap_us": len(set(df_pure["requirement_text"].str.lower().str.strip()).intersection(set(df_us["requirement_text"].str.lower().str.strip()))),
        }
    }

    print("\nAudit Summary:")
    print(json.dumps(audit_summary, indent=2))

    audit_out_path = os.path.join(EVAL_DIR, "external_dataset_audit.json")
    with open(audit_out_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Audit saved to {audit_out_path}")

    return df_pure, df_us, df_clean, df_chall, df_adv


def load_cached_features(df: pd.DataFrame, text_col: str, cache_prefix: str, fe: FeatureExtractor, sbert: SBERTExtractor) -> pd.DataFrame:
    sbert_path = os.path.join(CACHE_DIR, f"{cache_prefix}_sbert.joblib")
    ling_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ling.joblib")

    if os.path.exists(sbert_path):
        df_sbert = joblib.load(sbert_path)
    else:
        print(f"Extracting SBERT embeddings for {cache_prefix} ({len(df)} texts)...")
        df_sbert = sbert.transform(df[text_col].tolist())
        joblib.dump(df_sbert, sbert_path)

    if os.path.exists(ling_path):
        df_ling = joblib.load(ling_path)
    else:
        print(f"Extracting linguistic features for {cache_prefix} ({len(df)} texts)...")
        df_ling = fe.process_dataframe(df, text_col).drop(columns=df.columns)
        joblib.dump(df_ling, ling_path)

    return pd.concat([df_sbert.reset_index(drop=True), df_ling.reset_index(drop=True)], axis=1)


def run_all_experiments():
    df_pure, df_us, df_clean, df_chall, df_adv = audit_datasets()

    print("\n" + "=" * 80)
    print("PREPARING FEATURE MATRICES & REPRODUCIBLE SPLITS")
    print("=" * 80)

    fe = FeatureExtractor()
    sbert = SBERTExtractor()

    # 1. Clean v4 splits (70/15/15 GroupShuffleSplit by semantic_group_id)
    df_clean["semantic_group_id"] = df_clean["semantic_group_id"].fillna(df_clean.index.to_series().astype(str))
    gss_clean = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr_clean_idx, temp_clean_idx = next(gss_clean.split(df_clean, groups=df_clean["semantic_group_id"]))
    clean_train_df = df_clean.iloc[tr_clean_idx].reset_index(drop=True)
    clean_temp_df = df_clean.iloc[temp_clean_idx].reset_index(drop=True)

    gss_clean_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_clean_idx, te_clean_idx = next(gss_clean_val.split(clean_temp_df, groups=clean_temp_df["semantic_group_id"]))
    clean_val_df = clean_temp_df.iloc[val_clean_idx].reset_index(drop=True)
    clean_test_df = clean_temp_df.iloc[te_clean_idx].reset_index(drop=True)

    X_clean_train = load_cached_features(clean_train_df, "requirement", "train_v4", fe, sbert)
    X_clean_val = load_cached_features(clean_val_df, "requirement", "val_v4", fe, sbert)
    X_clean_test = load_cached_features(clean_test_df, "requirement", "test_v4", fe, sbert)

    y_clean_train = (clean_train_df["label"] == "Ambiguous").astype(int).values
    y_clean_val = (clean_val_df["label"] == "Ambiguous").astype(int).values
    y_clean_test = (clean_test_df["label"] == "Ambiguous").astype(int).values

    # 2. Diagnostic Sets
    df_chall_bin = df_chall[df_chall["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].copy().reset_index(drop=True)
    X_chall = load_cached_features(df_chall_bin, "Requirement_Text", "chall_bin", fe, sbert)
    y_chall = (df_chall_bin["Ambiguity_Label"] == "Ambiguous").astype(int).values

    df_adv_bin = df_adv.rename(columns={"expected_label": "label"}).reset_index(drop=True)
    X_adv = load_cached_features(df_adv_bin, "requirement", "adv", fe, sbert)
    y_adv = (df_adv_bin["label"] == "Ambiguous").astype(int).values

    # 3. PURE Dataset splits (70/15/15 GroupShuffleSplit by source_file)
    gss_pure = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    p_tr_idx, p_temp_idx = next(gss_pure.split(df_pure, groups=df_pure["source_file"]))
    p_temp = df_pure.iloc[p_temp_idx].reset_index(drop=True)

    gss_pure_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    p_va_sub_idx, p_te_sub_idx = next(gss_pure_val.split(p_temp, groups=p_temp["source_file"]))

    pure_train_df = df_pure.iloc[p_tr_idx].reset_index(drop=True)
    pure_val_df = p_temp.iloc[p_va_sub_idx].reset_index(drop=True)
    pure_test_df = p_temp.iloc[p_te_sub_idx].reset_index(drop=True)

    # Cache full PURE features
    X_pure_all = load_cached_features(df_pure, "requirement_text", "ext_pure", fe, sbert)
    X_pure_train = X_pure_all.iloc[p_tr_idx].reset_index(drop=True)
    X_pure_val = X_pure_all.iloc[p_temp_idx].iloc[p_va_sub_idx].reset_index(drop=True)
    X_pure_test = X_pure_all.iloc[p_temp_idx].iloc[p_te_sub_idx].reset_index(drop=True)

    y_pure_train = pure_train_df["label"].astype(int).values
    y_pure_val = pure_val_df["label"].astype(int).values
    y_pure_test = pure_test_df["label"].astype(int).values
    y_pure_all = df_pure["label"].astype(int).values

    print(f"PURE Split: Train={len(pure_train_df)} ({pure_train_df['source_file'].nunique()} docs), "
          f"Val={len(pure_val_df)} ({pure_val_df['source_file'].nunique()} docs), "
          f"Test={len(pure_test_df)} ({pure_test_df['source_file'].nunique()} docs)")

    # 4. User Stories Dataset splits (70/15/15 GroupShuffleSplit by source_file)
    gss_us = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    u_tr_idx, u_temp_idx = next(gss_us.split(df_us, groups=df_us["source_file"]))
    u_temp = df_us.iloc[u_temp_idx].reset_index(drop=True)

    gss_us_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    u_va_sub_idx, u_te_sub_idx = next(gss_us_val.split(u_temp, groups=u_temp["source_file"]))

    us_train_df = df_us.iloc[u_tr_idx].reset_index(drop=True)
    us_val_df = u_temp.iloc[u_va_sub_idx].reset_index(drop=True)
    us_test_df = u_temp.iloc[u_te_sub_idx].reset_index(drop=True)

    X_us_all = load_cached_features(df_us, "requirement_text", "ext_userstories", fe, sbert)
    X_us_train = X_us_all.iloc[u_tr_idx].reset_index(drop=True)
    X_us_val = X_us_all.iloc[u_temp_idx].iloc[u_va_sub_idx].reset_index(drop=True)
    X_us_test = X_us_all.iloc[u_temp_idx].iloc[u_te_sub_idx].reset_index(drop=True)

    y_us_train = us_train_df["label"].astype(int).values
    y_us_val = us_val_df["label"].astype(int).values
    y_us_test = us_test_df["label"].astype(int).values
    y_us_all = df_us["label"].astype(int).values

    print(f"User Stories Split: Train={len(us_train_df)} ({us_train_df['source_file'].nunique()} projs), "
          f"Val={len(us_val_df)} ({us_val_df['source_file'].nunique()} projs), "
          f"Test={len(us_test_df)} ({us_test_df['source_file'].nunique()} projs)")

    all_experiment_records = []

    # Helper function to train, calibrate, and evaluate
    def run_experiment(exp_id: str, desc: str, model_type: str,
                       X_tr: pd.DataFrame, y_tr: np.ndarray,
                       X_va: pd.DataFrame, y_va: np.ndarray,
                       eval_targets: Dict[str, Tuple[pd.DataFrame, np.ndarray]]):
        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
        X_va_s = pd.DataFrame(scaler.transform(X_va), columns=X_va.columns)

        if model_type == "lr":
            base_clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", random_state=42)
            clf_name = "Model A (Calibrated Logistic Regression L1)"
        elif model_type == "rf":
            base_clf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=4, class_weight="balanced", random_state=42)
            clf_name = "Model B (Calibrated Random Forest)"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        base_clf.fit(X_tr_s, y_tr)
        calibrator = CalibratedClassifierCV(base_clf, method="sigmoid", cv="prefit")
        calibrator.fit(X_va_s, y_va)

        amb_idx = np.where(calibrator.classes_ == 1)[0][0]

        print(f"\n[{exp_id}] {desc} | {clf_name}")
        for target_name, (X_target, y_target) in eval_targets.items():
            X_target_s = pd.DataFrame(scaler.transform(X_target), columns=X_target.columns)
            probs = calibrator.predict_proba(X_target_s)[:, amb_idx]
            metrics = compute_comprehensive_metrics(y_target, probs, target_name)
            metrics["experiment_id"] = exp_id
            metrics["experiment_description"] = desc
            metrics["model"] = clf_name
            all_experiment_records.append(metrics)
            print(f"   -> {target_name:25s}: Acc={metrics['accuracy']:5.2f}% | BalAcc={metrics['balanced_accuracy']:5.2f}% | "
                  f"MacroF1={metrics['macro_f1']:5.2f}% | AmbRec={metrics['recall_amb']:5.2f}% | AmbErr={metrics['ambiguous_error_rate']:5.2f}% | "
                  f"MCC={metrics['mcc']:6.4f} | Brier={metrics['brier_score']:6.4f}")

    # -------------------------------------------------------------
    # EXPERIMENT 1: Existing-data-only baseline (Model A & Model B)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: EXISTING-DATA-ONLY BASELINE")
    print("=" * 80)
    eval_targets_exp1 = {
        "Heldout Clean Test (v4)": (X_clean_test, y_clean_test),
        "Challenge Benchmark": (X_chall, y_chall),
        "Adversarial Benchmark": (X_adv, y_adv),
        "PURE (External Heldout)": (X_pure_test, y_pure_test),
        "User Stories (External Heldout)": (X_us_test, y_us_test),
    }
    run_experiment("EXP-1A", "Existing-data-only baseline", "lr",
                   X_clean_train, y_clean_train, X_clean_val, y_clean_val, eval_targets_exp1)
    run_experiment("EXP-1B", "Existing-data-only baseline", "rf",
                   X_clean_train, y_clean_train, X_clean_val, y_clean_val, eval_targets_exp1)

    # -------------------------------------------------------------
    # EXPERIMENT 2: External-data-only model (PURE only, Model A & Model B)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: EXTERNAL-DATA-ONLY MODEL (PURE TRAINED)")
    print("=" * 80)
    eval_targets_exp2 = {
        "PURE (In-domain Heldout)": (X_pure_test, y_pure_test),
        "Heldout Clean Test (v4)": (X_clean_test, y_clean_test),
        "Challenge Benchmark": (X_chall, y_chall),
        "Adversarial Benchmark": (X_adv, y_adv),
        "User Stories (Cross-dataset)": (X_us_all, y_us_all),
    }
    run_experiment("EXP-2A", "External-data-only (PURE)", "lr",
                   X_pure_train, y_pure_train, X_pure_val, y_pure_val, eval_targets_exp2)
    run_experiment("EXP-2B", "External-data-only (PURE)", "rf",
                   X_pure_train, y_pure_train, X_pure_val, y_pure_val, eval_targets_exp2)

    # -------------------------------------------------------------
    # EXPERIMENT 3: Train on PURE-derived data and test on User Stories
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: TRAIN ON PURE, TEST ON USER STORIES (SRS -> Agile)")
    print("=" * 80)
    eval_targets_exp3 = {
        "User Stories (Full Out-of-Domain)": (X_us_all, y_us_all),
        "User Stories (Heldout Split)": (X_us_test, y_us_test),
    }
    run_experiment("EXP-3A", "Train PURE, Test User Stories", "lr",
                   X_pure_train, y_pure_train, X_pure_val, y_pure_val, eval_targets_exp3)
    run_experiment("EXP-3B", "Train PURE, Test User Stories", "rf",
                   X_pure_train, y_pure_train, X_pure_val, y_pure_val, eval_targets_exp3)

    # -------------------------------------------------------------
    # EXPERIMENT 4: Train on User Stories data and test on PURE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: TRAIN ON USER STORIES, TEST ON PURE (Agile -> SRS)")
    print("=" * 80)
    eval_targets_exp4 = {
        "PURE (Full Out-of-Domain)": (X_pure_all, y_pure_all),
        "PURE (Heldout Split)": (X_pure_test, y_pure_test),
    }
    run_experiment("EXP-4A", "Train User Stories, Test PURE", "lr",
                   X_us_train, y_us_train, X_us_val, y_us_val, eval_targets_exp4)
    run_experiment("EXP-4B", "Train User Stories, Test PURE", "rf",
                   X_us_train, y_us_train, X_us_val, y_us_val, eval_targets_exp4)

    # -------------------------------------------------------------
    # EXPERIMENT 5: Train using compatible combined datasets with group-aware evaluation
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: COMPATIBLE COMBINED DATASETS (CLEAN V4 + PURE)")
    print("=" * 80)
    # Combine clean v4 and PURE without contaminating validation and test splits
    # Training combines clean_train_df + pure_train_df
    # Validation combines clean_val_df + pure_val_df
    # Test combines clean_test_df + pure_test_df
    X_comb_train = pd.concat([X_clean_train, X_pure_train], axis=0).reset_index(drop=True)
    y_comb_train = np.concatenate([y_clean_train, y_pure_train])

    X_comb_val = pd.concat([X_clean_val, X_pure_val], axis=0).reset_index(drop=True)
    y_comb_val = np.concatenate([y_clean_val, y_pure_val])

    X_comb_test = pd.concat([X_clean_test, X_pure_test], axis=0).reset_index(drop=True)
    y_comb_test = np.concatenate([y_clean_test, y_pure_test])

    print(f"Combined Splits: Train={len(X_comb_train)} (Clean={len(clean_train_df)}, PURE={len(pure_train_df)}), "
          f"Val={len(X_comb_val)} (Clean={len(clean_val_df)}, PURE={len(pure_val_df)}), "
          f"Heldout Test={len(X_comb_test)} (Clean={len(clean_test_df)}, PURE={len(pure_test_df)})")

    eval_targets_exp5 = {
        "Combined Heldout Test": (X_comb_test, y_comb_test),
        "Clean v4 Heldout Test": (X_clean_test, y_clean_test),
        "PURE Heldout Test": (X_pure_test, y_pure_test),
        "Challenge Benchmark": (X_chall, y_chall),
        "Adversarial Benchmark": (X_adv, y_adv),
        "User Stories (Zero-shot Out-of-Domain)": (X_us_all, y_us_all),
    }

    run_experiment("EXP-5A", "Combined Dataset (Clean + PURE)", "lr",
                   X_comb_train, y_comb_train, X_comb_val, y_comb_val, eval_targets_exp5)
    run_experiment("EXP-5B", "Combined Dataset (Clean + PURE)", "rf",
                   X_comb_train, y_comb_train, X_comb_val, y_comb_val, eval_targets_exp5)

    # Export all experiment records to CSV and JSON
    df_exp = pd.DataFrame(all_experiment_records)
    csv_path = os.path.join(EVAL_DIR, "external_benchmark_summary.csv")
    json_path = os.path.join(EVAL_DIR, "external_benchmark_summary.json")

    df_exp.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_experiment_records, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results exported to {csv_path} and {json_path}")
    print("=" * 80)

    # Print summary table
    summary_cols = ["experiment_id", "model", "dataset", "accuracy", "balanced_accuracy", "macro_f1", "recall_amb", "ambiguous_error_rate", "mcc", "brier_score"]
    print(df_exp[summary_cols].to_string(index=False))

    return df_exp


if __name__ == "__main__":
    run_all_experiments()
