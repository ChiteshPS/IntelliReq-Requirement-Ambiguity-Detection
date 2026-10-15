import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss
import seaborn as sns

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from models.predictor import ModelPredictor

def evaluate_dataset(df, dataset_name, predictor, scaler, feat_extractor, sbert_extractor):
    print(f"\n--- EVALUATING {dataset_name.upper()} ---")
    
    # Rename columns consistently
    if 'Requirement_Text' in df.columns:
        df = df.rename(columns={'Requirement_Text': 'requirement'})
    if 'Ambiguity_Label' in df.columns:
        df = df.rename(columns={'Ambiguity_Label': 'expected_label'})
    if 'label' in df.columns and 'expected_label' not in df.columns:
        df = df.rename(columns={'label': 'expected_label'})
        
    X_ling = feat_extractor.process_dataframe(df, 'requirement').drop(columns=df.columns)
    X_sbert = sbert_extractor.transform(df['requirement'])
    
    X_final = pd.concat([X_sbert, X_ling], axis=1)
    X_scaled = pd.DataFrame(scaler.transform(X_final), columns=X_final.columns)
    
    preds, probs = predictor.predict(X_scaled, original_features=X_ling)
    
    df['predicted_label'] = preds
    amb_col = [c for c in probs.columns if 'Ambiguous' in c and 'Not' not in c][0]
    df['probability'] = probs[amb_col] / 100.0
    
    # Separate borderline from strict evaluation
    eval_df = df[df['expected_label'].isin(["Ambiguous", "Not Ambiguous"])].copy()
    borderline_df = df[df['expected_label'] == "Borderline"].copy()
    
    # Strictly Evaluate Binary performance
    if not eval_df.empty:
        y_true = eval_df['expected_label']
        raw_preds = ["Ambiguous" if p >= 0.50 else "Not Ambiguous" for p in eval_df['probability']]
        y_pred_strict = [p if p in ["Ambiguous", "Not Ambiguous"] else "Incorrect" for p in eval_df['predicted_label']]
        
        raw_acc = accuracy_score(y_true, raw_preds)
        strict_acc = accuracy_score(y_true, y_pred_strict)
        brier = brier_score_loss((y_true == "Ambiguous").astype(int), eval_df['probability'])
        
        print(f"Raw Binary Accuracy:    {raw_acc*100:.2f}%")
        print(f"Strict Calibrated Acc:  {strict_acc*100:.2f}%")
        print(f"Brier Score Loss:       {brier:.4f}")
        
        cm = confusion_matrix(y_true, raw_preds, labels=["Ambiguous", "Not Ambiguous"])
        report = classification_report(y_true, raw_preds, zero_division=0, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(f'evaluation/{dataset_name}_classification_report.csv')
        
        tp, fn = cm[0, 0], cm[0, 1]
        fp, tn = cm[1, 0], cm[1, 1]
        total_eval = len(y_true)
        
        annot = np.array([
            [f"TP\n{tp}\n({tp/total_eval*100:.1f}%)", f"FN\n{fn}\n({fn/total_eval*100:.1f}%)"],
            [f"FP\n{fp}\n({fp/total_eval*100:.1f}%)", f"TN\n{tn}\n({tn/total_eval*100:.1f}%)"]
        ])
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        sns.heatmap(cm, annot=annot, fmt="", cmap='Blues',
                    linewidths=1.2, linecolor='white',
                    xticklabels=["Ambiguous", "Not Ambiguous"], 
                    yticklabels=["Ambiguous", "Not Ambiguous"],
                    annot_kws={"fontsize": 11, "weight": "bold"},
                    ax=ax)
        ax.set_title(f"{dataset_name.capitalize()} Set Confusion Matrix (N={total_eval})\nRaw Acc: {raw_acc*100:.2f}% | Strict: {strict_acc*100:.2f}%",
                     fontsize=11, weight='bold', pad=10)
        ax.set_ylabel('True Label', fontsize=10, weight='semibold')
        ax.set_xlabel('Predicted Label', fontsize=10, weight='semibold')
        plt.tight_layout()
        plt.savefig(f'evaluation/{dataset_name}_confusion_matrix.png', dpi=300)
        if dataset_name == "heldout":
            plt.savefig('evaluation/confusion_matrix.png', dpi=300)
        plt.close()
    else:
        raw_acc = 0
        brier = 0
        
    # Evaluate Borderline performance
    if not borderline_df.empty:
        print("\nBorderline Performance:")
        borderline_counts = borderline_df['predicted_label'].value_counts()
        print(borderline_counts)
        correct_borderline = borderline_counts.get("NEEDS REVIEW", 0)
        print(f"Correctly flagged NEEDS REVIEW: {correct_borderline}/{len(borderline_df)}")
        
    return df, raw_acc, brier

def run_evaluation():
    print("="*60)
    print("MASTER EVALUATION PIPELINE")
    print("="*60)

    os.makedirs('evaluation', exist_ok=True)

    print("\n[1/5] Loading Model Artifacts...")
    try:
        model = joblib.load('saved_models/model.joblib')
        scaler = joblib.load('saved_models/scaler.joblib')
        with open('saved_models/metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("ERROR: Run train.py first to generate model artifacts.")
        return

    predictor = ModelPredictor(
        trained_model=model, 
        upper_threshold=metadata.get('upper_threshold', 0.75), 
        lower_threshold=metadata.get('lower_threshold', 0.25)
    )
    
    feat_extractor = FeatureExtractor()
    sbert_extractor = SBERTExtractor()
    
    # 2. Held-out Clean Test Set
    print("\n[2/5] Evaluating Held-out Clean Test Set...")
    try:
        from sklearn.model_selection import GroupShuffleSplit
        df_clean = pd.read_csv('data/intellireq_clean_dataset_v4.csv')
        df_clean['semantic_group_id'] = df_clean['semantic_group_id'].fillna(df_clean.index.to_series().astype(str))
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, temp_idx = next(gss.split(df_clean, groups=df_clean['semantic_group_id']))
        temp_df = df_clean.iloc[temp_idx].reset_index(drop=True)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df['semantic_group_id']))
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)
        df_heldout_res, heldout_acc, heldout_brier = evaluate_dataset(test_df, "heldout", predictor, scaler, feat_extractor, sbert_extractor)
    except Exception as e:
        print(f"Warning: could not evaluate heldout set: {e}")

    # 3. Challenge Set
    print("\n[3/5] Evaluating Independent Challenge Set...")
    df_chall = pd.read_csv('data/challenge_test_set.csv')
    df_chall_res, chall_acc, chall_brier = evaluate_dataset(df_chall, "challenge", predictor, scaler, feat_extractor, sbert_extractor)
    
    # 4. Adversarial Set
    print("\n[4/5] Evaluating Adversarial Test Set...")
    df_adv = pd.read_csv('data/adversarial_test_set.csv')
    df_adv_res, adv_acc, adv_brier = evaluate_dataset(df_adv, "adversarial", predictor, scaler, feat_extractor, sbert_extractor)

    # 5. Error Analysis
    print("\n[5/5] Generating Error Analysis...")
    def get_error_type(r):
        if r['expected_label'] == 'Borderline':
            if r['predicted_label'] == 'NEEDS REVIEW': return 'Correctly Flagged Borderline'
            return 'Missed Borderline'
        elif r['expected_label'] == r['predicted_label']:
            return 'Correct'
        elif r['predicted_label'] == 'NEEDS REVIEW':
            return 'Underconfident Correct Prediction'
        elif r['expected_label'] == 'Ambiguous':
            return 'False Negative'
        else:
            return 'False Positive'
            
    df_chall_res['error_type'] = df_chall_res.apply(get_error_type, axis=1)
    errors = df_chall_res[df_chall_res['error_type'].isin(['False Negative', 'False Positive', 'Missed Borderline', 'Underconfident Correct Prediction'])]
    errors.to_csv('evaluation/error_analysis.csv', index=False)
    
    with open('evaluation/error_analysis.md', 'w') as f:
        f.write("# Error Analysis Report (Challenge Set)\n\n")
        for i, row in errors.iterrows():
            f.write(f"**Requirement:** {row['requirement']}\n")
            f.write(f"- Expected: {row['expected_label']} | Predicted: {row['predicted_label']} | Prob: {row['probability']:.4f}\n")
            f.write(f"- Error Type: {row['error_type']}\n\n")

    print("\nComplete! See evaluation/ folder for reports and plots.")

if __name__ == "__main__":
    run_evaluation()
