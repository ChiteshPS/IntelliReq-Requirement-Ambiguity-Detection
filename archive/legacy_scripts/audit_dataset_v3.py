import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

def get_stats(df):
    vague_lexicon = ["fast", "quickly", "reasonable", "efficient", "efficiently", "adequate", "high", "many", "massive", "heavy", "secure", "reliable", "appropriate", "robust"]
    subjective_lexicon = ["intuitive", "easy", "best", "satisfaction", "fairly"]
    tech_lexicon = ["aes-256", "database", "api", "memory", "server", "endpoint"]
    
    num_regex = re.compile(r'\b\d+\b')
    unit_regex = re.compile(r'\b(seconds|milliseconds|ms|gb|tb|gbps|hours)\b', re.IGNORECASE)
    pct_regex = re.compile(r'%')
    thresh_regex = re.compile(r'(>|<|minimum|maximum|at least|no more than)')
    
    amb = df[df['label'] == 'Ambiguous'].copy()
    not_amb = df[df['label'] == 'Not Ambiguous'].copy()
    
    def calc(subset):
        if len(subset) == 0: return {}
        texts = subset['requirement'].str.lower()
        lens = texts.apply(lambda x: len(x.split()))
        
        return {
            'count': len(subset),
            'len_mean': lens.mean(),
            'len_median': lens.median(),
            'len_min': lens.min(),
            'len_max': lens.max(),
            'num%': texts.apply(lambda x: bool(num_regex.search(x))).mean()*100,
            'unit%': texts.apply(lambda x: bool(unit_regex.search(x))).mean()*100,
            'pct%': texts.apply(lambda x: bool(pct_regex.search(x))).mean()*100,
            'thresh%': texts.apply(lambda x: bool(thresh_regex.search(x))).mean()*100,
            'vague%': texts.apply(lambda x: any(w in x for w in vague_lexicon)).mean()*100,
            'subj%': texts.apply(lambda x: any(w in x for w in subjective_lexicon)).mean()*100,
            'tech%': texts.apply(lambda x: any(w in x for w in tech_lexicon)).mean()*100,
        }

    return {'Ambiguous': calc(amb), 'Not Ambiguous': calc(not_amb)}

def run_baselines(train_df, test_df):
    y_train = train_df['label']
    y_test = test_df['label']
    
    # 1. Majority
    maj = y_train.mode()[0]
    maj_preds = [maj] * len(y_test)
    acc_maj = accuracy_score(y_test, maj_preds)*100
    
    # 2. Length-only (Predict Not Ambiguous if length > 10)
    len_preds = test_df['requirement'].apply(lambda x: "Not Ambiguous" if len(x.split()) > 10 else "Ambiguous")
    acc_len = accuracy_score(y_test, len_preds)*100
    
    # 3. Num/Unit/Pct Rule
    num_regex = re.compile(r'\b\d+\b|%|\b(seconds|ms|gb|tb|hours)\b')
    nup_preds = test_df['requirement'].str.lower().apply(lambda x: "Not Ambiguous" if num_regex.search(x) else "Ambiguous")
    acc_nup = accuracy_score(y_test, nup_preds)*100
    
    # 4. Vague Keyword Rule
    vague_lexicon = ["fast", "quickly", "reasonable", "efficient", "adequate", "high", "many", "massive", "secure", "reliable"]
    vague_preds = test_df['requirement'].str.lower().apply(lambda x: "Ambiguous" if any(w in x for w in vague_lexicon) else "Not Ambiguous")
    acc_vague = accuracy_score(y_test, vague_preds)*100
    
    # 5. TF-IDF + LR
    tfidf = TfidfVectorizer()
    X_tr_tf = tfidf.fit_transform(train_df['requirement'])
    X_te_tf = tfidf.transform(test_df['requirement'])
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_tr_tf, y_train)
    acc_tfidf = accuracy_score(y_test, lr.predict(X_te_tf))*100
    
    return {
        "Majority": acc_maj,
        "Length-only": acc_len,
        "Number/Unit/Pct-only": acc_nup,
        "Vague-keyword-only": acc_vague,
        "TF-IDF": acc_tfidf
    }

def audit_dataset_v3():
    df = pd.read_csv('data/intellireq_clean_dataset_v3.csv')
    stats = get_stats(df)
    
    # GSS split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['pair_id']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    baselines = run_baselines(train_df, test_df)
    
    # Check overlap
    texts = set(df['requirement'].str.lower().str.strip().tolist())
    try:
        chall = pd.read_csv('data/challenge_test_set.csv')
        chall_set = set(chall['Requirement_Text'].str.lower().str.strip().tolist())
        chall_overlap = len(texts.intersection(chall_set))
    except: chall_overlap = 0
    try:
        adv = pd.read_csv('data/adversarial_test_set.csv')
        adv_set = set(adv['requirement'].str.lower().str.strip().tolist())
        adv_overlap = len(texts.intersection(adv_set))
    except: adv_overlap = 0

    # Write report
    with open('docs/DATASET_V3_SHORTCUT_AUDIT.md', 'w') as f:
        f.write("# Dataset v3 Shortcut Audit\n\n")
        f.write(f"## 1. Overview\n- Total Rows: {len(df)}\n")
        f.write(f"- Exact Duplicates: {df.duplicated(subset=['requirement']).sum()}\n")
        f.write(f"- Normalized Duplicates: {df['requirement'].str.lower().str.strip().duplicated().sum()}\n")
        f.write(f"- Challenge Overlap: {chall_overlap}\n- Adversarial Overlap: {adv_overlap}\n\n")
        
        f.write("## 2. Class Balance & Structural Stats\n")
        f.write("### Ambiguous\n")
        for k, v in stats['Ambiguous'].items():
            f.write(f"- {k}: {v:.2f}\n")
        f.write("\n### Not Ambiguous\n")
        for k, v in stats['Not Ambiguous'].items():
            f.write(f"- {k}: {v:.2f}\n")
            
        f.write("\n## 3. Diagnostic Baseline Classifiers\n")
        f.write("Evaluated on group-aware internal split to detect remaining shortcuts:\n")
        for k, v in baselines.items():
            f.write(f"- {k}: {v:.2f}%\n")
            
    print("="*60)
    print("DATASET v3 GENERATION COMPLETE")
    print(f"Total rows: {len(df)}")
    print(f"Class balance: Ambiguous {stats['Ambiguous']['count']} | Not Ambiguous {stats['Not Ambiguous']['count']}")
    print(f"Shortcuts: Length (Amb {stats['Ambiguous']['len_mean']:.1f} vs Not Amb {stats['Not Ambiguous']['len_mean']:.1f})")
    print(f"Shortcuts: Units (Amb {stats['Ambiguous']['unit%']:.1f}% vs Not Amb {stats['Not Ambiguous']['unit%']:.1f}%)")
    
    strongest = max(baselines.items(), key=lambda x: x[1])
    print(f"Strongest remaining shortcut: {strongest[0]} ({strongest[1]:.2f}%)")
    print(f"Challenge overlap: {chall_overlap} | Adversarial overlap: {adv_overlap}")
    if strongest[1] > 90:
        print("Status: NOT READY FOR RETRAINING (TF-IDF > 90% implies structural shortcut persists).")
    else:
        print("Status: READY FOR RETRAINING (Shortcuts successfully broken).")
    print("="*60)

if __name__ == "__main__":
    audit_dataset_v3()
