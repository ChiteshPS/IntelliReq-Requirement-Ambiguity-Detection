"""
tools/train_distilbert_candidate.py — Fine-Tuned Transformer Candidate (Model C)

Trains a sequence classification transformer (DistilBERT) on the exact same
group-aware split of IntelliReq clean v4 dataset (500 train, 107 val, 108 held-out test),
and evaluates on:
1. Held-Out Clean Test Set (N=108)
2. Independent Diagnostic Challenge Set (N=80)
3. Adversarial Test Set (N=16)

Compares performance directly against Model A (Hybrid Logistic Regression)
and Model B (Hybrid Random Forest).
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import GroupShuffleSplit
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
EVAL_DIR = os.path.join(REPO_ROOT, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class RequirementDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
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
            "label": torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(y_true, probs_amb, upper_th=0.75, lower_th=0.25):
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
    amb_error_rate = (fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    y_3way = []
    for p in probs_amb:
        if p >= upper_th:
            y_3way.append(1)
        elif p <= lower_th:
            y_3way.append(0)
        else:
            y_3way.append(-1)

    strict_acc = sum(1 for yt, yp in zip(y_true, y_3way) if yt == yp) / len(y_true)
    deferred = sum(1 for yp in y_3way if yp == -1)
    deferral_rate = deferred / len(y_true)
    non_deferred = len(y_true) - deferred
    selective_acc = (sum(1 for yt, yp in zip(y_true, y_3way) if yp != -1 and yt == yp) / non_deferred) if non_deferred > 0 else 0.0

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


def predict_probs(model, dataloader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


def main():
    print("=" * 80)
    print("MODEL C: FINE-TUNED TRANSFORMER CANDIDATE (DistilBERT)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("\n[1/5] Loading canonical splits...")
    df_clean = pd.read_csv(os.path.join(REPO_ROOT, "data", "intellireq_clean_dataset_v4.csv"))
    df_clean["semantic_group_id"] = df_clean["semantic_group_id"].fillna(df_clean.index.to_series().astype(str))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df_clean, groups=df_clean["semantic_group_id"]))
    train_df = df_clean.iloc[train_idx].reset_index(drop=True)
    temp_df = df_clean.iloc[temp_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["semantic_group_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    df_chall = pd.read_csv(os.path.join(REPO_ROOT, "data", "challenge_test_set.csv"))
    df_chall_bin = df_chall[df_chall["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].reset_index(drop=True)

    df_adv = pd.read_csv(os.path.join(REPO_ROOT, "data", "adversarial_test_set.csv"))

    y_train = (train_df["label"] == "Ambiguous").astype(int).values
    y_val = (val_df["label"] == "Ambiguous").astype(int).values
    y_test = (test_df["label"] == "Ambiguous").astype(int).values
    y_chall = (df_chall_bin["Ambiguity_Label"] == "Ambiguous").astype(int).values
    y_adv = (df_adv["expected_label"] == "Ambiguous").astype(int).values

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Heldout={len(test_df)}, Challenge={len(df_chall_bin)}, Adv={len(df_adv)}")

    # 2. Tokenizer & Datasets
    print("\n[2/5] Initializing DistilBERT tokenizer and data loaders...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = RequirementDataset(train_df["requirement"], y_train, tokenizer)
    val_ds = RequirementDataset(val_df["requirement"], y_val, tokenizer)
    test_ds = RequirementDataset(test_df["requirement"], y_test, tokenizer)
    chall_ds = RequirementDataset(df_chall_bin["Requirement_Text"], y_chall, tokenizer)
    adv_ds = RequirementDataset(df_adv["requirement"], y_adv, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    chall_loader = DataLoader(chall_ds, batch_size=32, shuffle=False)
    adv_loader = DataLoader(adv_ds, batch_size=16, shuffle=False)

    # 3. Model setup
    print("\n[3/5] Loading DistilBERT for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # 4. Training loop
    print(f"\n[4/5] Training DistilBERT ({epochs} epochs, {total_steps} steps)...")
    t0 = time.time()
    best_val_f1 = 0.0
    best_weights_path = os.path.join(EVAL_DIR, "distilbert_best_val.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        probs_val = predict_probs(model, val_loader, device)
        m_val = compute_metrics(y_val, probs_val)
        print(f"  Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {m_val['accuracy']:.2f}% | Val Macro-F1: {m_val['macro_f1']:.2f}%")

        if m_val["macro_f1"] > best_val_f1:
            best_val_f1 = m_val["macro_f1"]
            torch.save(model.state_dict(), best_weights_path)

    print(f"Training completed in {time.time() - t0:.2f}s. Loading best weights (Val Macro-F1: {best_val_f1:.2f}%)...")
    model.load_state_dict(torch.load(best_weights_path))

    # 5. Evaluation
    print("\n[5/5] Evaluating Best DistilBERT Checkpoint...")
    probs_te = predict_probs(model, test_loader, device)
    probs_ch = predict_probs(model, chall_loader, device)
    probs_ad = predict_probs(model, adv_loader, device)

    m_heldout = compute_metrics(y_test, probs_te)
    m_chall = compute_metrics(y_chall, probs_ch)
    m_adv = compute_metrics(y_adv, probs_ad)

    print("\n--- Model C (Fine-Tuned DistilBERT) Results ---")
    print(f"  Held-Out Test:  Acc={m_heldout['accuracy']:.2f}% | Macro-F1={m_heldout['macro_f1']:.2f}% | Amb-Rec={m_heldout['recall_amb']:.2f}% | Brier={m_heldout['brier_loss']:.4f}")
    print(f"  Challenge Test: Acc={m_chall['accuracy']:.2f}% | Macro-F1={m_chall['macro_f1']:.2f}% | Amb-Rec={m_chall['recall_amb']:.2f}% | Brier={m_chall['brier_loss']:.4f}")
    print(f"  Adversarial:    Acc={m_adv['accuracy']:.2f}% | Macro-F1={m_adv['macro_f1']:.2f}%")

    result_payload = {
        "model_name": "Model C (Fine-Tuned DistilBERT Transformer)",
        "heldout_test": m_heldout,
        "challenge_test": m_chall,
        "adversarial_test": m_adv,
    }

    out_file = os.path.join(EVAL_DIR, "transformer_candidate_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()
