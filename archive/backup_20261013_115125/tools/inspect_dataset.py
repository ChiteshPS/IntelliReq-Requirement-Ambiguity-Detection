import os
import pandas as pd
import numpy as np
import hashlib

def print_header(title):
    print("\n" + "="*80)
    print(f" {title.upper()} ".center(80, "="))
    print("="*80 + "\n")

def get_norm_text(text):
    if not isinstance(text, str): return ""
    import re
    return re.sub(r'\s+', ' ', text.lower().strip())

def inspect_dataset():
    print_header("DATASET AUDIT (PHASE 2 & 3)")
    
    datasets_to_check = [
        'data/ambiguity_dataset_v2.csv',
        'data/challenge_test_set.csv',
        'data/intellireq_clean_dataset.csv'
    ]
    
    inventory = []
    
    for path in datasets_to_check:
        if not os.path.exists(path):
            print(f"SKIPPED: {path} not found.")
            continue
            
        print(f"--- INVENTORY: {path} ---")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
            
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("Missing Values:")
        print(df.isnull().sum())
        
        # Identify text/label columns
        text_col = None
        for col in ['Requirement_Text', 'Requirement', 'text', 'requirement']:
            if col in df.columns: text_col = col; break
            
        label_col = None
        for col in ['Ambiguity_Label', 'label', 'expected_label']:
            if col in df.columns: label_col = col; break
            
        if text_col:
            df['norm_text'] = df[text_col].apply(get_norm_text)
            unique_texts = df['norm_text'].nunique()
            dups = len(df) - unique_texts
            print(f"Unique Requirements: {unique_texts}")
            print(f"Exact Normalized Duplicates: {dups}")
            
            dup_records = df[df.duplicated(subset=['norm_text'], keep=False)]
            if not dup_records.empty and label_col:
                conflict = dup_records.groupby('norm_text')[label_col].nunique()
                contradictory = conflict[conflict > 1]
                print(f"Contradictory Labels on Duplicates: {len(contradictory)}")
            
        if label_col:
            print(f"Labels:\n{df[label_col].value_counts()}")
            
        if 'domain' in df.columns:
            print(f"Domains:\n{df['domain'].value_counts()}")
            
        if 'source_type' in df.columns:
            print(f"Sources:\n{df['source_type'].value_counts()}")
            
        print("\n")

if __name__ == "__main__":
    inspect_dataset()
