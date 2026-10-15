import pandas as pd
import json
import os

def inspect_pipeline():
    print("="*80)
    print(" PIPELINE & LEAKAGE INSPECTION (PHASE 3)".center(80, "="))
    print("="*80 + "\n")
    
    # Check if challenge set data leaked into training data
    try:
        df_chall = pd.read_csv('data/challenge_test_set.csv')
        df_clean = pd.read_csv('data/intellireq_clean_dataset.csv')
        
        c_req = set(df_chall.get('Requirement_Text', pd.Series()).str.lower().str.strip().tolist())
        tr_req = set(df_clean.get('requirement', pd.Series()).str.lower().str.strip().tolist())
        
        overlap = c_req.intersection(tr_req)
        print(f"Challenge Leakage into Train Data: {len(overlap)} records")
        if overlap:
            print(f"Leaked examples: {list(overlap)[:3]}")
            
    except Exception as e:
        print(f"Error checking leakage: {e}")

if __name__ == "__main__":
    inspect_pipeline()
