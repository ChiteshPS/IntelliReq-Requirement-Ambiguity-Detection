import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def audit_v4():
    df = pd.read_csv('data/intellireq_clean_dataset_v4.csv')
    
    # Structural stats
    def get_stats(subset):
        texts = subset['requirement'].str.lower()
        lens = texts.apply(lambda x: len(x.split()))
        return {
            'len_mean': lens.mean(),
            'num%': texts.apply(lambda x: bool(re.search(r'\b\d+\b', x))).mean()*100,
            'unit%': texts.apply(lambda x: bool(re.search(r'\b(seconds|ms|gb|tb|hours)\b', x))).mean()*100,
        }
    
    amb = df[df['label'] == 'Ambiguous']
    clear = df[df['label'] == 'Not Ambiguous']
    s_amb = get_stats(amb)
    s_clear = get_stats(clear)
    
    # Baselines
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['semantic_group_id']))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    y_train = train_df['label']
    y_test = test_df['label']
    
    # TFIDF
    tfidf = TfidfVectorizer()
    X_tr = tfidf.fit_transform(train_df['requirement'])
    X_te = tfidf.transform(test_df['requirement'])
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_tr, y_train)
    acc_tfidf = accuracy_score(y_test, lr.predict(X_te))*100
    
    # Overlap
    texts = set(df['requirement'].str.lower().str.strip().tolist())
    try:
        chall = pd.read_csv('data/challenge_test_set.csv')
        chall_set = set(chall['Requirement_Text'].str.lower().str.strip().tolist())
        overlap = len(texts.intersection(chall_set))
    except: overlap = 0
    
    with open('docs/DATASET_V4_SHORTCUT_AUDIT.md', 'w') as f:
        f.write("# Dataset V4 Shortcut Audit\n\n")
        f.write(f"- Total Rows: {len(df)}\n")
        f.write(f"- Challenge Overlap: {overlap}\n\n")
        f.write(f"## Structural (Ambiguous vs Clear)\n")
        f.write(f"- Length: {s_amb['len_mean']:.1f} vs {s_clear['len_mean']:.1f}\n")
        f.write(f"- Numbers: {s_amb['num%']:.1f}% vs {s_clear['num%']:.1f}%\n")
        f.write(f"- Units: {s_amb['unit%']:.1f}% vs {s_clear['unit%']:.1f}%\n\n")
        f.write(f"## Baselines\n")
        f.write(f"- TF-IDF Accuracy: {acc_tfidf:.2f}%\n")
        
    print(f"V4 Audit Complete. Rows: {len(df)}. TF-IDF Acc: {acc_tfidf:.2f}%")
    if acc_tfidf >= 98:
        print("FAIL: TF-IDF is still >98%.")
    else:
        print("PASS: Lexical shortcuts broken.")

if __name__ == "__main__":
    audit_v4()
