"""
evaluation/visualizer.py — Publication-Quality Visualizer for IntelliReq

Provides plotting utilities for dataset distributions, confusion matrices,
feature importances, model benchmark comparisons, and calibration distributions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, output_dir: str = "evaluation"):
        """
        Initializes the Visualizer with standard seaborn/matplotlib themes.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    def plot_dataset_distribution(self, df: pd.DataFrame, label_col: str = 'label', filename: str = 'dataset_distribution.png'):
        """
        Creates a bar chart showing the balance between Ambiguous vs Not Ambiguous.
        """
        plt.figure(figsize=(7, 5), dpi=300)
        ax = sns.countplot(data=df, x=label_col, palette=["#e67e22", "#2980b9"])
        plt.title('Dataset Distribution: Ambiguous vs Not Ambiguous', fontsize=12, weight='bold', pad=12)
        plt.xlabel('Ground Truth Label', fontsize=10, weight='semibold')
        plt.ylabel('Number of Requirements', fontsize=10, weight='semibold')
        
        total = len(df)
        for p in ax.patches:
            h = int(p.get_height())
            pct = (h / total) * 100
            ax.annotate(f'{h} ({pct:.1f}%)', (p.get_x() + p.get_width() / 2., h + (total * 0.01)),
                        ha='center', va='bottom', fontsize=9.5, weight='bold')
                        
        plt.ylim(0, max([p.get_height() for p in ax.patches]) * 1.15)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_confusion_matrix(self, cm, title: str = "IntelliReq Hybrid Confusion Matrix", 
                              subtitle: str = "", filename: str = "confusion_matrix.png",
                              labels: list = None, cmap: str = 'Blues'):
        """
        Generates a publication-quality heatmap representation of a Confusion Matrix.
        """
        if labels is None:
            labels = ['Ambiguous', 'Not Ambiguous']
            
        cm_array = np.array(cm)
        total = np.sum(cm_array)
        
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        
        tp = cm_array[0, 0] if cm_array.shape == (2, 2) else 0
        fn = cm_array[0, 1] if cm_array.shape == (2, 2) else 0
        fp = cm_array[1, 0] if cm_array.shape == (2, 2) else 0
        tn = cm_array[1, 1] if cm_array.shape == (2, 2) else 0
        
        acc = (tp + tn) / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        annot = np.array([
            [f"TP\n{tp}\n({tp/total*100:.1f}%)", f"FN\n{fn}\n({fn/total*100:.1f}%)"],
            [f"FP\n{fp}\n({fp/total*100:.1f}%)", f"TN\n{tn}\n({tn/total*100:.1f}%)"]
        ]) if cm_array.shape == (2, 2) else True
        
        sns.heatmap(cm_array, annot=annot, fmt="", cmap=cmap,
                    xticklabels=[f"{l}\n(Pred)" for l in labels], 
                    yticklabels=[f"{l}\n(Actual)" for l in labels],
                    linewidths=1.2, linecolor='white',
                    annot_kws={"fontsize": 11, "weight": "bold"}, ax=ax)
                    
        full_title = f"{title}\n{subtitle}" if subtitle else title
        ax.set_title(full_title, fontsize=12, weight='bold', pad=12)
        ax.set_xlabel('Predicted Label', fontsize=10, weight='semibold')
        ax.set_ylabel('True Label', fontsize=10, weight='semibold')
        
        if cm_array.shape == (2, 2):
            ax.text(
                0.5, -0.18,
                f"Accuracy: {acc*100:.2f}%  |  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%  |  F1: {f1*100:.2f}%",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9.5, weight='semibold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6')
            )
            
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 16, 
                                filename: str = "feature_importance.png"):
        """
        Creates a horizontal bar chart of top positive (Not Ambiguous) and top negative (Ambiguous)
        learned feature weights.
        """
        df_top = importance_df.sort_values(by="Weight", ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        colors = ['#d9534f' if w < 0 else '#0275d8' for w in df_top['Weight']]
        bars = ax.barh(df_top['Feature'], df_top['Weight'], color=colors, edgecolor='#333333', linewidth=0.6, height=0.65)
        
        ax.axvline(0, color='black', linewidth=0.9, linestyle='--')
        ax.set_title("IntelliReq Active Model Feature Importances (L1 Regularized Logistic Weights)", fontsize=12, weight='bold', pad=14)
        ax.set_xlabel("Learned Coefficient (Weight)", fontsize=10, weight='semibold')
        ax.set_ylabel("Feature Name", fontsize=10, weight='semibold')
        
        for bar in bars:
            w = bar.get_width()
            x_pos = w + (0.08 if w >= 0 else -0.08)
            ha = 'left' if w >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{w:+.2f}",
                    va='center', ha=ha, fontsize=8.5, weight='bold')
            
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d9534f', edgecolor='#333333', label='Drives AMBIGUOUS (Negative Coef)'),
            Patch(facecolor='#0275d8', edgecolor='#333333', label='Drives NOT AMBIGUOUS (Positive Coef)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # When executed directly, invoke the canonical generator to update all plots
    print("Executing plot generator...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tools.generate_evaluation_plots import main as run_generator
    run_generator()
