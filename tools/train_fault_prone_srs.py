"""
tools/train_fault_prone_srs.py — Training, Evaluation, and Cross-Dataset Benchmarking
on the Fault-Prone SRS Dataset (Kaggle: corpus4panwo/fault-prone-srs-dataset).

Implements:
- Task A: Six-class classification (lexical, syntactic, semantic, syntax, pragmatic, clean)
- Task B: Binary classification (clean -> NOT AMBIGUOUS; others -> AMBIGUOUS)
- Candidate Models:
  1. Baseline A1: Hybrid SBERT + 24 Handcrafted Features + Calibrated Logistic Regression
  2. Baseline A2: Hybrid SBERT + 24 Handcrafted Features + Random Forest
  3. Candidate B: Fine-Tuned DistilRoBERTa Sequence Classifier
- Group-aware, leak-free train/val/test partitions (70 / 15 / 15)
- Cross-dataset evaluation on Fault-Prone SRS test, Clean v4 test, Challenge set, and PURE.
"""

import os
import sys
import json
import re
import time
import random
import warnings
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

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
    brier_score_loss,
    confusion_matrix,
    classification_report,
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
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "saved_models", "distilroberta_fpsrs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES_6 = ["lexical", "syntactic", "semantic", "syntax", "pragmatic", "clean"]
LABEL_MAP_6_TO_NAME = {1: "lexical", 2: "syntactic", 3: "semantic", 4: "syntax", 5: "pragmatic", 6: "clean"}
LABEL_MAP_6_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


def compute_metrics_multiclass(y_true: np.ndarray, preds: np.ndarray, probs: np.ndarray = None, class_names: List[str] = None) -> Dict[str, Any]:
    acc = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, preds)

    cm = confusion_matrix(y_true, preds)
    per_class_f1 = f1_score(y_true, preds, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, preds, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, preds, average=None, zero_division=0)

    per_class_metrics = {}
    for i, c_name in enumerate(class_names or [f"class_{i}" for i in range(len(per_class_f1))]):
        per_class_metrics[c_name] = {
            "precision": round(float(per_class_prec[i]) * 100.0, 2),
            "recall": round(float(per_class_rec[i]) * 100.0, 2),
            "f1": round(float(per_class_f1[i]) * 100.0, 2),
        }

    return {
        "accuracy": round(float(acc) * 100.0, 2),
        "balanced_accuracy": round(float(bal_acc) * 100.0, 2),
        "macro_f1": round(float(macro_f1) * 100.0, 2),
        "weighted_f1": round(float(weighted_f1) * 100.0, 2),
        "mcc": round(float(mcc), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }


def compute_metrics_binary(y_true: np.ndarray, probs_amb: np.ndarray, thresh: float = 0.50) -> Dict[str, Any]:
    preds = (probs_amb >= thresh).astype(int)
    acc = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    prec_amb = precision_score(y_true, preds, pos_label=1, zero_division=0)
    rec_amb = recall_score(y_true, preds, pos_label=1, zero_division=0)
    f1_amb = f1_score(y_true, preds, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_true, preds)
    brier = brier_score_loss(y_true, probs_amb)

    cm = confusion_matrix(y_true, preds, labels=[1, 0])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])
    amb_err_rate = (fn / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0

    return {
        "accuracy": round(float(acc) * 100.0, 2),
        "balanced_accuracy": round(float(bal_acc) * 100.0, 2),
        "macro_f1": round(float(macro_f1) * 100.0, 2),
        "precision_amb": round(float(prec_amb) * 100.0, 2),
        "recall_amb": round(float(rec_amb) * 100.0, 2),
        "f1_amb": round(float(f1_amb) * 100.0, 2),
        "amb_error_rate_pct": round(float(amb_err_rate), 2),
        "mcc": round(float(mcc), 4),
        "brier_loss": round(float(brier), 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn
    }


def audit_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("=" * 80)
    print("STEP 1: AUDITING FAULT-PRONE SRS DATASET (Kaggle: corpus4panwo)")
    print("=" * 80)

    dataset_path = os.path.join(REPO_ROOT, "data", "external", "dataset.csv")
    df = pd.read_csv(dataset_path, encoding="latin1", header=None)
    df.columns = ["requirement_text", "raw_label"]

    total_rows = len(df)
    missing_vals = int(df["requirement_text"].isnull().sum())
    exact_dups = int(df.duplicated(subset=["requirement_text"], keep=False).sum())
    norm_dups = int(df["requirement_text"].apply(normalize_text).duplicated(keep=False).sum())

    # Conflicting labels check
    conflicts_6class = int((df.groupby("requirement_text")["raw_label"].nunique() > 1).sum())
    df["binary_label"] = (df["raw_label"] != 6).astype(int)  # 6=clean (0), 1-5=ambiguous (1)
    conflicts_binary = int((df.groupby("requirement_text")["binary_label"].nunique() > 1).sum())

    class_counts = df["raw_label"].value_counts().to_dict()
    named_counts = {LABEL_MAP_6_TO_NAME[k]: v for k, v in sorted(class_counts.items())}

    audit_meta = {
        "dataset_name": "Fault-Prone SRS Dataset",
        "official_source": "https://www.kaggle.com/datasets/corpus4panwo/fault-prone-srs-dataset",
        "associated_paper": "Fault-Prone Software Requirements Specification Detection Using Ensemble Learning for Edge/Cloud Applications (MDPI Applied Sciences, 2023, DOI: 10.3390/app13148368)",
        "total_rows": total_rows,
        "missing_values": missing_vals,
        "exact_duplicates": exact_dups,
        "normalized_duplicates": norm_dups,
        "conflicting_labels_6class": conflicts_6class,
        "conflicting_labels_binary": conflicts_binary,
        "classes_6": named_counts,
        "binary_distribution": {
            "ambiguous": int(sum(df["binary_label"] == 1)),
            "not_ambiguous": int(sum(df["binary_label"] == 0))
        }
    }

    audit_json_path = os.path.join(EVAL_DIR, "fault_prone_srs_audit.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_meta, f, indent=2)

    print(json.dumps(audit_meta, indent=2))
    print(f"\nAudit saved to {audit_json_path}")
    return df, audit_meta


def get_cached_hybrid_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sbert_cache = os.path.join(CACHE_DIR, "fpsrs_sbert.joblib")
    ling_cache = os.path.join(CACHE_DIR, "fpsrs_ling.joblib")

    fe = FeatureExtractor()
    sbert = SBERTExtractor()

    if os.path.exists(sbert_cache):
        print("Loading cached SBERT features for Fault-Prone SRS...")
        X_sbert = joblib.load(sbert_cache)
    else:
        print(f"Extracting SBERT embeddings for {len(df)} requirements...")
        t0 = time.time()
        X_sbert = sbert.transform(df["requirement_text"].tolist())
        joblib.dump(X_sbert, sbert_cache)
        print(f"Done in {time.time()-t0:.2f}s")

    if os.path.exists(ling_cache):
        print("Loading cached linguistic features for Fault-Prone SRS...")
        X_ling = joblib.load(ling_cache)
    else:
        print(f"Extracting 24 handcrafted linguistic features for {len(df)} requirements...")
        t0 = time.time()
        X_ling = fe.process_dataframe(df, "requirement_text").drop(columns=df.columns)
        joblib.dump(X_ling, ling_cache)
        print(f"Done in {time.time()-t0:.2f}s")

    return X_sbert, X_ling


def run_pipeline():
    df, audit_meta = audit_dataset()

    print("\n" + "=" * 80)
    print("STEP 2: PREPARING GROUP-AWARE SPLITS (LEAKAGE PREVENTION)")
    print("=" * 80)

    # Build robust semantic group identifier based on normalized text and domain keywords
    df["norm_text"] = df["requirement_text"].apply(normalize_text)

    # Domain keyword clustering for grouping
    domain_tags = []
    domain_patterns = [
        ("steam_boiler", r"steam boiler"),
        ("ambulance_las", r"ambulance|las\b"),
        ("library", r"library"),
        ("address_book", r"address book"),
        ("atm", r"atm\b"),
        ("inventory", r"inventory|store"),
        ("hospital", r"hospital|patient"),
        ("flight", r"flight|airline"),
        ("bank", r"bank|account"),
        ("university", r"university|course"),
        ("hotel", r"hotel|guest"),
    ]

    for text in df["norm_text"]:
        tag = None
        for name, pat in domain_patterns:
            if re.search(pat, text):
                tag = name
                break
        domain_tags.append(tag if tag else text[:30])

    df["group_id"] = domain_tags

    gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_idx, temp_idx = next(gss_train.split(df, groups=df["group_id"]))
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va_sub_idx, te_sub_idx = next(gss_val.split(temp_df, groups=temp_df["group_id"]))
    va_idx = temp_idx[va_sub_idx]
    te_idx = temp_idx[te_sub_idx]

    train_df = df.iloc[tr_idx].reset_index(drop=True)
    val_df = df.iloc[va_idx].reset_index(drop=True)
    test_df = df.iloc[te_idx].reset_index(drop=True)

    print(f"Partition distribution: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
          f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
          f"Held-out Test={len(test_df)} ({len(test_df)/len(df):.1%})")

    # Target arrays: Task A (6-class 0..5) and Task B (binary 0..1)
    y_train_6 = np.array([LABEL_MAP_6_TO_INDEX[l] for l in train_df["raw_label"]])
    y_val_6 = np.array([LABEL_MAP_6_TO_INDEX[l] for l in val_df["raw_label"]])
    y_test_6 = np.array([LABEL_MAP_6_TO_INDEX[l] for l in test_df["raw_label"]])

    y_train_bin = train_df["binary_label"].values
    y_val_bin = val_df["binary_label"].values
    y_test_bin = test_df["binary_label"].values

    # Feature extraction for hybrid classical models
    X_sbert_all, X_ling_all = get_cached_hybrid_features(df)
    X_hybrid_all = pd.concat([X_sbert_all.reset_index(drop=True), X_ling_all.reset_index(drop=True)], axis=1)

    X_train_h = X_hybrid_all.iloc[tr_idx].reset_index(drop=True)
    X_val_h = X_hybrid_all.iloc[va_idx].reset_index(drop=True)
    X_test_h = X_hybrid_all.iloc[te_idx].reset_index(drop=True)

    scaler = StandardScaler().fit(X_train_h)
    X_tr_s = scaler.transform(X_train_h)
    X_va_s = scaler.transform(X_val_h)
    X_te_s = scaler.transform(X_test_h)

    benchmark_results = []

    print("\n" + "=" * 80)
    print("STEP 3: TRAINING BASELINE HYBRID MODELS (LOGISTIC REGRESSION & RANDOM FOREST)")
    print("=" * 80)

    # --- Baseline A1: Calibrated Logistic Regression ---
    # Task A: 6-Class Multiclass LR
    lr_6 = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    lr_6.fit(X_tr_s, y_train_6)
    preds_lr_6 = lr_6.predict(X_te_s)
    m_lr_6 = compute_metrics_multiclass(y_test_6, preds_lr_6, class_names=CLASS_NAMES_6)
    m_lr_6["model"] = "Baseline A1: Hybrid SBERT + Ling + Logistic Regression"
    m_lr_6["task"] = "Task A: Six-Class Classification"
    benchmark_results.append(m_lr_6)
    print(f"Baseline A1 [6-Class LR] -> Acc: {m_lr_6['accuracy']:.2f}% | Macro-F1: {m_lr_6['macro_f1']:.2f}% | MCC: {m_lr_6['mcc']:.4f}")

    # Task B: Binary Calibrated LR
    lr_bin = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", random_state=SEED)
    lr_bin.fit(X_tr_s, y_train_bin)
    calib_lr = CalibratedClassifierCV(lr_bin, method="sigmoid", cv="prefit")
    calib_lr.fit(X_va_s, y_val_bin)
    probs_lr_bin = calib_lr.predict_proba(X_te_s)[:, 1]
    m_lr_bin = compute_metrics_binary(y_test_bin, probs_lr_bin)
    m_lr_bin["model"] = "Baseline A1: Hybrid SBERT + Ling + Calibrated L1 LR"
    m_lr_bin["task"] = "Task B: Binary Ambiguity Classification"
    benchmark_results.append(m_lr_bin)
    print(f"Baseline A1 [Binary LR] -> Acc: {m_lr_bin['accuracy']:.2f}% | BalAcc: {m_lr_bin['balanced_accuracy']:.2f}% | Macro-F1: {m_lr_bin['macro_f1']:.2f}% | Amb-Rec: {m_lr_bin['recall_amb']:.2f}% | Amb-Err: {m_lr_bin['amb_error_rate_pct']:.2f}% | Brier: {m_lr_bin['brier_loss']:.4f}")

    # --- Baseline A2: Random Forest ---
    # Task A: 6-Class RF
    rf_6 = RandomForestClassifier(n_estimators=150, max_depth=14, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_6.fit(X_tr_s, y_train_6)
    preds_rf_6 = rf_6.predict(X_te_s)
    m_rf_6 = compute_metrics_multiclass(y_test_6, preds_rf_6, class_names=CLASS_NAMES_6)
    m_rf_6["model"] = "Baseline A2: Hybrid SBERT + Ling + Random Forest"
    m_rf_6["task"] = "Task A: Six-Class Classification"
    benchmark_results.append(m_rf_6)
    print(f"Baseline A2 [6-Class RF] -> Acc: {m_rf_6['accuracy']:.2f}% | Macro-F1: {m_rf_6['macro_f1']:.2f}% | MCC: {m_rf_6['mcc']:.4f}")

    # Task B: Binary Calibrated RF
    rf_bin = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_split=4, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_bin.fit(X_tr_s, y_train_bin)
    calib_rf = CalibratedClassifierCV(rf_bin, method="sigmoid", cv="prefit")
    calib_rf.fit(X_va_s, y_val_bin)
    probs_rf_bin = calib_rf.predict_proba(X_te_s)[:, 1]
    m_rf_bin = compute_metrics_binary(y_test_bin, probs_rf_bin)
    m_rf_bin["model"] = "Baseline A2: Hybrid SBERT + Ling + Calibrated Random Forest"
    m_rf_bin["task"] = "Task B: Binary Ambiguity Classification"
    benchmark_results.append(m_rf_bin)
    print(f"Baseline A2 [Binary RF] -> Acc: {m_rf_bin['accuracy']:.2f}% | BalAcc: {m_rf_bin['balanced_accuracy']:.2f}% | Macro-F1: {m_rf_bin['macro_f1']:.2f}% | Amb-Rec: {m_rf_bin['recall_amb']:.2f}% | Amb-Err: {m_rf_bin['amb_error_rate_pct']:.2f}% | Brier: {m_rf_bin['brier_loss']:.4f}")

    print("\n" + "=" * 80)
    print("STEP 4: FINE-TUNING TRANSFORMER (CANDIDATE B: DISTILROBERTA)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # DataLoaders for Task A (6-class)
    train_ds_6 = TextDataset(train_df["requirement_text"], y_train_6, tokenizer)
    val_ds_6 = TextDataset(val_df["requirement_text"], y_val_6, tokenizer)
    test_ds_6 = TextDataset(test_df["requirement_text"], y_test_6, tokenizer)

    train_loader_6 = DataLoader(train_ds_6, batch_size=32, shuffle=True)
    val_loader_6 = DataLoader(val_ds_6, batch_size=32, shuffle=False)
    test_loader_6 = DataLoader(test_ds_6, batch_size=32, shuffle=False)

    # Class weights for loss
    class_counts_arr = np.bincount(y_train_6, minlength=6)
    weights_6 = torch.tensor(len(y_train_6) / (6.0 * np.maximum(class_counts_arr, 1)), dtype=torch.float).to(DEVICE)

    model_roberta_6 = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=6)
    model_roberta_6.to(DEVICE)

    optimizer_6 = AdamW(model_roberta_6.parameters(), lr=3e-5, weight_decay=0.01)
    criterion_6 = nn.CrossEntropyLoss(weight=weights_6)
    epochs = 2
    total_steps = len(train_loader_6) * epochs
    scheduler_6 = get_linear_schedule_with_warmup(optimizer_6, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    print("Training DistilRoBERTa on Task A (6-Class) for 2 epochs on CPU...")
    model_roberta_6.train()
    for ep in range(epochs):
        ep_loss = 0.0
        t0 = time.time()
        for b_idx, batch in enumerate(train_loader_6):
            optimizer_6.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model_roberta_6(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion_6(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_roberta_6.parameters(), max_norm=1.0)
            optimizer_6.step()
            scheduler_6.step()
            ep_loss += loss.item()
        print(f"  Epoch {ep+1}/{epochs} finished in {time.time()-t0:.1f}s | Avg Loss: {ep_loss/len(train_loader_6):.4f}")

    # Evaluate Candidate B on Task A Test
    model_roberta_6.eval()
    all_preds_6 = []
    with torch.no_grad():
        for batch in test_loader_6:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model_roberta_6(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds_6.extend(preds)

    m_roberta_6 = compute_metrics_multiclass(y_test_6, np.array(all_preds_6), class_names=CLASS_NAMES_6)
    m_roberta_6["model"] = "Candidate B: Fine-Tuned DistilRoBERTa (Transformer)"
    m_roberta_6["task"] = "Task A: Six-Class Classification"
    benchmark_results.append(m_roberta_6)
    print(f"\nCandidate B [6-Class DistilRoBERTa] -> Acc: {m_roberta_6['accuracy']:.2f}% | Macro-F1: {m_roberta_6['macro_f1']:.2f}% | MCC: {m_roberta_6['mcc']:.4f}")

    # Task B: Binary DistilRoBERTa Evaluation (Mapping classes 0..4 to 1 [Ambiguous] and 5 to 0 [Clear])
    preds_bin_from_roberta = np.array([1 if p < 5 else 0 for p in all_preds_6])
    m_roberta_bin = {
        "accuracy": round(float(accuracy_score(y_test_bin, preds_bin_from_roberta)) * 100.0, 2),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test_bin, preds_bin_from_roberta)) * 100.0, 2),
        "macro_f1": round(float(f1_score(y_test_bin, preds_bin_from_roberta, average="macro", zero_division=0)) * 100.0, 2),
        "precision_amb": round(float(precision_score(y_test_bin, preds_bin_from_roberta, pos_label=1, zero_division=0)) * 100.0, 2),
        "recall_amb": round(float(recall_score(y_test_bin, preds_bin_from_roberta, pos_label=1, zero_division=0)) * 100.0, 2),
        "f1_amb": round(float(f1_score(y_test_bin, preds_bin_from_roberta, pos_label=1, zero_division=0)) * 100.0, 2),
        "amb_error_rate_pct": round(float((1.0 - recall_score(y_test_bin, preds_bin_from_roberta, pos_label=1, zero_division=0))) * 100.0, 2),
        "mcc": round(float(matthews_corrcoef(y_test_bin, preds_bin_from_roberta)), 4),
        "model": "Candidate B: Fine-Tuned DistilRoBERTa (Mapped Binary)",
        "task": "Task B: Binary Ambiguity Classification"
    }
    benchmark_results.append(m_roberta_bin)
    print(f"Candidate B [Binary Mapped DistilRoBERTa] -> Acc: {m_roberta_bin['accuracy']:.2f}% | BalAcc: {m_roberta_bin['balanced_accuracy']:.2f}% | Macro-F1: {m_roberta_bin['macro_f1']:.2f}% | Amb-Rec: {m_roberta_bin['recall_amb']:.2f}% | Amb-Err: {m_roberta_bin['amb_error_rate_pct']:.2f}%")

    # Save transformer artifacts
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    model_roberta_6.save_pretrained(CHECKPOINT_DIR)
    with open(os.path.join(CHECKPOINT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": CLASS_NAMES_6, "mapping_to_binary": {c: "AMBIGUOUS" if c != "clean" else "NOT AMBIGUOUS" for c in CLASS_NAMES_6}}, f, indent=2)
    print(f"\nSaved Candidate B artifacts to {CHECKPOINT_DIR}")

    # -------------------------------------------------------------
    # STEP 5: CROSS-DATASET GENERALIZATION BENCHMARK
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: CROSS-DATASET EVALUATION MATRIX")
    print("=" * 80)

    # 1. Evaluate on Clean v4 Held-out Test
    df_clean_v4 = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    X_v4_test_sbert = joblib.load("saved_models/benchmark_cache/test_v4_sbert.joblib")
    X_v4_test_ling = joblib.load("saved_models/benchmark_cache/test_v4_ling.joblib")
    X_v4_test_h = pd.concat([X_v4_test_sbert.reset_index(drop=True), X_v4_test_ling.reset_index(drop=True)], axis=1)
    X_v4_te_s = scaler.transform(X_v4_test_h)

    # Get v4 test labels
    gss_v4 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    _, v4_temp_idx = next(gss_v4.split(df_clean_v4, groups=df_clean_v4["semantic_group_id"].fillna(df_clean_v4.index.to_series().astype(str))))
    gss_v4_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    _, v4_te_rel_idx = next(gss_v4_val.split(df_clean_v4.iloc[v4_temp_idx], groups=df_clean_v4.iloc[v4_temp_idx]["semantic_group_id"]))
    v4_test_df = df_clean_v4.iloc[v4_temp_idx[v4_te_rel_idx]].reset_index(drop=True)
    y_v4_test = (v4_test_df["label"] == "Ambiguous").astype(int).values

    # 2. Evaluate on Diagnostic Challenge Set
    df_chall = pd.read_csv("data/challenge_test_set.csv")
    df_chall_bin = df_chall[df_chall["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].reset_index(drop=True)
    X_chall_sbert = joblib.load("saved_models/benchmark_cache/chall_bin_sbert.joblib")
    X_chall_ling = joblib.load("saved_models/benchmark_cache/chall_bin_ling.joblib")
    X_chall_h = pd.concat([X_chall_sbert.reset_index(drop=True), X_chall_ling.reset_index(drop=True)], axis=1)
    X_chall_s = scaler.transform(X_chall_h)
    y_chall_test = (df_chall_bin["Ambiguity_Label"] == "Ambiguous").astype(int).values

    # 3. Evaluate on PURE Held-Out Test
    df_pure = pd.read_csv("data/external/pure_labeled.csv")
    X_pure_sbert = joblib.load("saved_models/benchmark_cache/ext_pure_sbert.joblib")
    X_pure_ling = joblib.load("saved_models/benchmark_cache/ext_pure_ling.joblib")
    X_pure_h = pd.concat([X_pure_sbert.reset_index(drop=True), X_pure_ling.reset_index(drop=True)], axis=1)
    X_pure_s = scaler.transform(X_pure_h)
    y_pure_test = df_pure["label"].astype(int).values

    cross_eval_sets = {
        "Held-out Fault-Prone SRS Test": (X_te_s, y_test_bin),
        "IntelliReq Clean v4 Test": (X_v4_te_s, y_v4_test),
        "Diagnostic Challenge Set (Exposed)": (X_chall_s, y_chall_test),
        "PURE Benchmark Dataset": (X_pure_s, y_pure_test)
    }

    cross_results = []
    print("\nCross-Dataset Generalization of Fault-Prone SRS Trained Models:")
    for set_name, (X_eval, y_eval) in cross_eval_sets.items():
        # Baseline A1 (LR)
        probs_lr = calib_lr.predict_proba(X_eval)[:, 1]
        m_lr = compute_metrics_binary(y_eval, probs_lr)
        m_lr["eval_set"] = set_name
        m_lr["model"] = "Baseline A1: Calibrated L1 LR (FPSRS Trained)"
        cross_results.append(m_lr)

        # Baseline A2 (RF)
        probs_rf = calib_rf.predict_proba(X_eval)[:, 1]
        m_rf = compute_metrics_binary(y_eval, probs_rf)
        m_rf["eval_set"] = set_name
        m_rf["model"] = "Baseline A2: Calibrated RF (FPSRS Trained)"
        cross_results.append(m_rf)

        print(f"[{set_name:35s}] LR Acc: {m_lr['accuracy']:5.2f}% (F1: {m_lr['macro_f1']:5.2f}%) | RF Acc: {m_rf['accuracy']:5.2f}% (F1: {m_rf['macro_f1']:5.2f}%)")

    # Save benchmark outputs
    benchmark_json_path = os.path.join(EVAL_DIR, "fault_prone_srs_benchmark.json")
    benchmark_csv_path = os.path.join(EVAL_DIR, "fault_prone_srs_benchmark.csv")
    cross_json_path = os.path.join(EVAL_DIR, "fault_prone_srs_cross_dataset.json")

    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    pd.DataFrame(benchmark_results).to_csv(benchmark_csv_path, index=False)

    with open(cross_json_path, "w", encoding="utf-8") as f:
        json.dump(cross_results, f, indent=2)

    print(f"\nAll benchmark results serialized to {benchmark_json_path} and {cross_json_path}")
    return benchmark_results, cross_results


if __name__ == "__main__":
    run_pipeline()
