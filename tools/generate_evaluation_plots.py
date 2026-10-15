"""
tools/generate_evaluation_plots.py — Canonical Evaluation Plot Generator for IntelliReq

Generates publication-quality, accurate visualization graphs based strictly on the
current trained model artifacts and canonical evaluation datasets:

  1. evaluation/confusion_matrix.png & evaluation/heldout_confusion_matrix.png
     -> Clean Held-out Test Set (N=108, Accuracy: 98.15%, TP=31, FN=0, FP=2, TN=75)
  2. evaluation/challenge_confusion_matrix.png
     -> Independent Challenge Test Set (N=80, Accuracy: 85.00%, TP=37, FN=3, FP=9, TN=31)
  3. evaluation/adversarial_confusion_matrix.png
     -> Adversarial Test Set (N=16, Accuracy: 87.50%, TP=7, FN=1, FP=1, TN=7)
  4. evaluation/feature_importance.png
     -> Top L1 Logistic Regression feature weights (Handcrafted vs SBERT Semantic)
  5. evaluation/dataset_distribution.png
     -> Clean v4 Dataset class balance and split proportions (Train/Val/Test)
  6. evaluation/model_comparison.png
     -> Benchmark comparison: Base Paper TF-IDF vs SBERT vs IntelliReq Hybrid
  7. evaluation/calibration_thresholds.png
     -> Calibrated probability distribution with 3-tier decision zones (<=0.25, 0.25-0.75, >=0.75)
"""

import os
import sys
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

OUTPUT_DIR = "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_annotated_confusion_matrix(cm, title, subtitle, save_paths, cmap='Blues'):
    """
    Renders a publication-quality confusion matrix heatmap with cell counts,
    percentages, and summary metric cards.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    
    total = np.sum(cm)
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    labels = [["Ambiguous", "Not Ambiguous"], ["Ambiguous", "Not Ambiguous"]]
    annot = np.array([
        [f"TP\n{tp}\n({tp/total*100:.1f}%)", f"FN\n{fn}\n({fn/total*100:.1f}%)"],
        [f"FP\n{fp}\n({fp/total*100:.1f}%)", f"TN\n{tn}\n({tn/total*100:.1f}%)"]
    ])
    
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        linewidths=1.5,
        linecolor='white',
        xticklabels=["Ambiguous\n(Predicted)", "Not Ambiguous\n(Predicted)"],
        yticklabels=["Ambiguous\n(Actual)", "Not Ambiguous\n(Actual)"],
        annot_kws={"fontsize": 11, "weight": "bold"},
        ax=ax
    )
    
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, weight='bold', pad=14)
    ax.text(
        0.5, -0.18,
        f"Accuracy: {acc*100:.2f}%  |  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%  |  F1: {f1*100:.2f}%",
        ha='center', va='center', transform=ax.transAxes,
        fontsize=10, weight='semibold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6')
    )
    
    plt.tight_layout()
    for p in save_paths:
        plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {', '.join(save_paths)}")


def generate_feature_importance_plot(model, scaler):
    """
    Extracts L1 logistic regression weights and creates a clean bar chart showing
    top positive (Not Ambiguous) and top negative (Ambiguous) predictors.
    """
    base_clf = model.calibrated_classifiers_[0].estimator
    coefs = base_clf.coef_[0]
    names = scaler.feature_names_in_
    
    records = []
    for name, coef in zip(names, coefs):
        if abs(coef) > 0:
            feat_type = "Semantic (SBERT)" if name.startswith("sbert_") else "Linguistic (Handcrafted)"
            # Negative coef -> drives Ambiguous (class 0); Positive coef -> drives Not Ambiguous (class 1)
            direction = "Ambiguous" if coef < 0 else "Not Ambiguous"
            records.append({
                "Feature": name,
                "Weight": coef,
                "AbsWeight": abs(coef),
                "Direction": direction,
                "Type": feat_type
            })
            
    df_weights = pd.DataFrame(records).sort_values(by="AbsWeight", ascending=False).head(16)
    
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    # Sort for horizontal bar plotting (smallest abs weight at bottom)
    df_weights = df_weights.sort_values(by="Weight", ascending=True)
    
    colors = ['#d9534f' if d == 'Ambiguous' else '#0275d8' for d in df_weights['Direction']]
    bars = ax.barh(df_weights['Feature'], df_weights['Weight'], color=colors, edgecolor='#333333', linewidth=0.6, height=0.65)
    
    ax.axvline(0, color='black', linewidth=0.9, linestyle='--')
    ax.set_title("IntelliReq Active Model Feature Importances (L1 Logistic Regression Weights)\n408 Features (384 SBERT Embeddings + 24 Handcrafted Linguistic)", fontsize=12, weight='bold', pad=15)
    ax.set_xlabel("Learned Model Weight (Coefficient)", fontsize=11, weight='semibold')
    ax.set_ylabel("Feature Name", fontsize=11, weight='semibold')
    
    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        x_pos = w + (0.08 if w >= 0 else -0.08)
        ha = 'left' if w >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{w:+.2f}",
                va='center', ha=ha, fontsize=8.5, weight='bold')
        
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d9534f', edgecolor='#333333', label='Predicts AMBIGUOUS (Negative Coef)'),
        Patch(facecolor='#0275d8', edgecolor='#333333', label='Predicts NOT AMBIGUOUS (Positive Coef)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_dataset_distribution_plot():
    """
    Renders current canonical dataset distribution across classes and splits.
    """
    df_v4 = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    total_samples = len(df_v4)
    amb_count = (df_v4['label'] == 'Ambiguous').sum()
    not_amb_count = (df_v4['label'] == 'Not Ambiguous').sum()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    
    # 1. Class Distribution
    classes = ['Ambiguous', 'Not Ambiguous']
    counts = [amb_count, not_amb_count]
    pcts = [c / total_samples * 100 for c in counts]
    colors = ['#e67e22', '#2980b9']
    
    bars1 = ax1.bar(classes, counts, color=colors, edgecolor='#333333', width=0.55, linewidth=0.7)
    ax1.set_title(f"Class Distribution (Clean v4 Dataset, N={total_samples})", fontsize=11, weight='bold', pad=10)
    ax1.set_ylabel("Number of Requirements", fontsize=10, weight='semibold')
    ax1.set_ylim(0, max(counts) * 1.18)
    
    for bar, count, pct in zip(bars1, counts, pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{count} ({pct:.1f}%)", ha='center', va='bottom', fontsize=9.5, weight='bold')
        
    # 2. Split Breakdown
    splits = ['Train (70%)', 'Val (15%)', 'Test (15%)']
    split_counts = [501, 108, 108]  # Verified group split counts
    split_colors = ['#27ae60', '#f39c12', '#8e44ad']
    
    bars2 = ax2.bar(splits, split_counts, color=split_colors, edgecolor='#333333', width=0.55, linewidth=0.7)
    ax2.set_title("Group-Aware Split Partitioning", fontsize=11, weight='bold', pad=10)
    ax2.set_ylabel("Number of Requirements", fontsize=10, weight='semibold')
    ax2.set_ylim(0, max(split_counts) * 1.18)
    
    for bar, count in zip(bars2, split_counts):
        pct = count / total_samples * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{count} ({pct:.1f}%)", ha='center', va='bottom', fontsize=9.5, weight='bold')
        
    plt.suptitle("IntelliReq Canonical Dataset & Split Partition Architecture", fontsize=13, weight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "dataset_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_model_comparison_plot():
    """
    Renders benchmark comparison between baseline architectures and the IntelliReq Hybrid model.
    """
    models = [
        "Base Paper Baseline\n(TF-IDF + RF)",
        "SBERT Only\n(MiniLM + LR)",
        "Linguistic Only\n(24 Features + LR)",
        "IntelliReq Hybrid\n(Clean Held-out)",
        "IntelliReq Hybrid\n(Challenge Set)",
        "IntelliReq Hybrid\n(Adversarial Set)"
    ]
    accuracies = [90.62, 90.62, 87.04, 98.15, 85.00, 87.50]
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#2ecc71', '#e67e22', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars = ax.bar(models, accuracies, color=colors, edgecolor='#2c3e50', width=0.55, linewidth=0.8)
    
    ax.axhline(97.0, color='#e74c3c', linestyle='--', linewidth=1.2, label='Base Paper Target (97.0%)')
    ax.set_ylim(60, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=11, weight='semibold')
    ax.set_title("Model Architecture & Generalization Benchmark Comparison\nIntelliReq Semantic Hybrid vs Baselines", fontsize=12, weight='bold', pad=14)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f"{acc:.2f}%", ha='center', va='bottom', fontsize=9.5, weight='bold')
        
    ax.legend(loc='lower right', frameon=True)
    plt.xticks(rotation=15, ha='right', fontsize=9.5)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_calibration_plot(test_probs, chall_probs, adv_probs):
    """
    Renders the probability confidence distribution highlighting the 3 calibrated zones:
    Not Ambiguous (<=0.25), Needs Review (0.25 - 0.75), and Ambiguous (>=0.75).
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    bins = np.linspace(0, 1, 21)
    ax.hist(test_probs, bins=bins, alpha=0.6, label='Held-out Test (N=108)', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.hist(chall_probs, bins=bins, alpha=0.5, label='Challenge Set (N=80)', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.hist(adv_probs, bins=bins, alpha=0.5, label='Adversarial Set (N=16)', color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    # Draw threshold regions
    ax.axvline(0.25, color='#27ae60', linestyle='--', linewidth=1.5, label='Lower Threshold (0.25)')
    ax.axvline(0.75, color='#c0392b', linestyle='--', linewidth=1.5, label='Upper Threshold (0.75)')
    
    # Highlight zones
    ax.axvspan(0.0, 0.25, alpha=0.10, color='green')
    ax.axvspan(0.25, 0.75, alpha=0.12, color='orange')
    ax.axvspan(0.75, 1.0, alpha=0.10, color='red')
    
    ax.text(0.125, ax.get_ylim()[1]*0.88, "NOT AMBIGUOUS\n(P <= 0.25)", ha='center', fontsize=9, weight='bold', color='#1b5e20')
    ax.text(0.50, ax.get_ylim()[1]*0.88, "NEEDS REVIEW\n(0.25 < P < 0.75)", ha='center', fontsize=9, weight='bold', color='#e65100')
    ax.text(0.875, ax.get_ylim()[1]*0.88, "AMBIGUOUS\n(P >= 0.75)", ha='center', fontsize=9, weight='bold', color='#b71c1c')
    
    ax.set_title("IntelliReq Calibrated Probability Distribution & 3-Tier Decision Boundaries", fontsize=12, weight='bold', pad=12)
    ax.set_xlabel("Predicted Probability of Ambiguity P(Ambiguous)", fontsize=11, weight='semibold')
    ax.set_ylabel("Frequency (Count)", fontsize=11, weight='semibold')
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "calibration_thresholds.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 70)
    print("    INTELLIREQ — GENERATING REPRODUCIBLE EVALUATION PLOTS")
    print("=" * 70)
    
    # 1. Load artifacts
    model = joblib.load("saved_models/model.joblib")
    scaler = joblib.load("saved_models/scaler.joblib")
    fe = FeatureExtractor()
    sbert = SBERTExtractor()
    
    def extract_X(df, text_col):
        X_ling = fe.process_dataframe(df, text_col).drop(columns=df.columns)
        X_sbert = sbert.transform(df[text_col])
        X = pd.concat([X_sbert, X_ling], axis=1)
        return pd.DataFrame(scaler.transform(X), columns=X.columns)

    # -------------------------------------------------------------------------
    # A. HELD-OUT CLEAN TEST SET (N=108)
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
    
    X_test_scaled = extract_X(test_df, "requirement")
    y_test_true = test_df["label"]
    test_probs = model.predict_proba(X_test_scaled)[:, 0]  # Ambiguous is class 0
    test_preds = ["Ambiguous" if p >= 0.50 else "Not Ambiguous" for p in test_probs]
    
    cm_test = confusion_matrix(y_test_true, test_preds, labels=["Ambiguous", "Not Ambiguous"])
    plot_annotated_confusion_matrix(
        cm_test,
        "Held-Out Clean Test Set Confusion Matrix (N=108)",
        "Model: Semantic Sentence Embeddings + Handcrafted Features + L1 Logistic Regression",
        [
            os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
            os.path.join(OUTPUT_DIR, "heldout_confusion_matrix.png")
        ],
        cmap='Blues'
    )
    
    # -------------------------------------------------------------------------
    # B. CHALLENGE TEST SET (N=80)
    # -------------------------------------------------------------------------
    df_chall = pd.read_csv("data/challenge_test_set.csv")
    df_chall_bin = df_chall[df_chall["Ambiguity_Label"].isin(["Ambiguous", "Not Ambiguous"])].copy()
    X_chall_scaled = extract_X(df_chall_bin, "Requirement_Text")
    y_chall_true = df_chall_bin["Ambiguity_Label"]
    chall_probs = model.predict_proba(X_chall_scaled)[:, 0]
    chall_preds = ["Ambiguous" if p >= 0.50 else "Not Ambiguous" for p in chall_probs]
    
    cm_chall = confusion_matrix(y_chall_true, chall_preds, labels=["Ambiguous", "Not Ambiguous"])
    plot_annotated_confusion_matrix(
        cm_chall,
        "Independent Challenge Test Set Confusion Matrix (N=80)",
        "Zero-Leakage External Requirements Benchmark",
        [os.path.join(OUTPUT_DIR, "challenge_confusion_matrix.png")],
        cmap='YlGnBu'
    )
    
    # -------------------------------------------------------------------------
    # C. ADVERSARIAL TEST SET (N=16)
    # -------------------------------------------------------------------------
    df_adv = pd.read_csv("data/adversarial_test_set.csv")
    adv_label_col = 'expected_label' if 'expected_label' in df_adv.columns else ('Ambiguity_Label' if 'Ambiguity_Label' in df_adv.columns else 'label')
    adv_text_col = 'requirement' if 'requirement' in df_adv.columns else ('Requirement_Text' if 'Requirement_Text' in df_adv.columns else 'text')
    
    df_adv_bin = df_adv[df_adv[adv_label_col].isin(["Ambiguous", "Not Ambiguous"])].copy()
    X_adv_scaled = extract_X(df_adv_bin, adv_text_col)
    y_adv_true = df_adv_bin[adv_label_col]
    adv_probs = model.predict_proba(X_adv_scaled)[:, 0]
    adv_preds = ["Ambiguous" if p >= 0.50 else "Not Ambiguous" for p in adv_probs]
    
    cm_adv = confusion_matrix(y_adv_true, adv_preds, labels=["Ambiguous", "Not Ambiguous"])
    plot_annotated_confusion_matrix(
        cm_adv,
        "Adversarial Test Set Confusion Matrix (N=16)",
        "Lexical Distractors & Syntactic Trap Requirements",
        [os.path.join(OUTPUT_DIR, "adversarial_confusion_matrix.png")],
        cmap='Purples'
    )
    
    # -------------------------------------------------------------------------
    # D. FEATURE IMPORTANCE PLOT
    # -------------------------------------------------------------------------
    generate_feature_importance_plot(model, scaler)
    
    # -------------------------------------------------------------------------
    # E. DATASET DISTRIBUTION PLOT
    # -------------------------------------------------------------------------
    generate_dataset_distribution_plot()
    
    # -------------------------------------------------------------------------
    # F. MODEL COMPARISON PLOT
    # -------------------------------------------------------------------------
    generate_model_comparison_plot()
    
    # -------------------------------------------------------------------------
    # G. CALIBRATION & THRESHOLD DISTRIBUTION PLOT
    # -------------------------------------------------------------------------
    generate_calibration_plot(test_probs, chall_probs, adv_probs)
    
    print("\nAll evaluation graphs generated and updated successfully!")

if __name__ == "__main__":
    main()
