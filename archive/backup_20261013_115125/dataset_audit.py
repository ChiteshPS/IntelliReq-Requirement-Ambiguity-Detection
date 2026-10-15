import os
import pandas as pd
import json

def audit_datasets():
    print("="*60)
    print("PHASE 1: DATASET AUDIT")
    print("="*60)
    
    os.makedirs('dataset_audit', exist_ok=True)
    
    datasets_to_check = [
        'data/ambiguity_dataset_v2.csv',
        'dataset/software_requirements.csv'
    ]
    
    inventory = []
    
    for dpath in datasets_to_check:
        if not os.path.exists(dpath):
            print(f"Skipping {dpath} (not found)")
            continue
            
        print(f"\nAuditing: {dpath}")
        try:
            df = pd.read_csv(dpath)
        except Exception as e:
            print(f"Error reading {dpath}: {e}")
            continue
            
        # Standardize column name for requirement text
        text_col = None
        for col in ['Requirement_Text', 'Requirement', 'text']:
            if col in df.columns:
                text_col = col
                break
                
        if not text_col:
            # Fallback if no header
            df = pd.read_csv(dpath, names=['Requirement_Text'])
            text_col = 'Requirement_Text'
            
        label_col = 'Ambiguity_Label' if 'Ambiguity_Label' in df.columns else None
        
        # 1. Basics
        total_records = len(df)
        
        # 2. Duplicates
        # Normalize text only for duplicate detection (lowercase, strip, remove extra spaces)
        df['norm_text'] = df[text_col].astype(str).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
        unique_records = df['norm_text'].nunique()
        duplicate_count = total_records - unique_records
        
        duplicates_df = df[df.duplicated(subset=['norm_text'], keep=False)].sort_values(by='norm_text')
        
        ambiguous_count = df[label_col].value_counts().get('Ambiguous', 0) if label_col else 0
        not_ambiguous_count = df[label_col].value_counts().get('Not Ambiguous', 0) if label_col else 0
        
        # Check contradictory labels
        contradictory = []
        if label_col:
            grouped = df.groupby('norm_text')[label_col].nunique()
            contradictory_texts = grouped[grouped > 1].index.tolist()
            if contradictory_texts:
                contradictory = df[df['norm_text'].isin(contradictory_texts)]
                
        inventory.append({
            'dataset': dpath,
            'total_records': total_records,
            'unique_records': unique_records,
            'duplicate_count': duplicate_count,
            'ambiguous_count': ambiguous_count,
            'not_ambiguous_count': not_ambiguous_count,
            'contradictory_labels': len(contradictory) if isinstance(contradictory, pd.DataFrame) else 0
        })
        
        # Save specific outputs
        base_name = os.path.basename(dpath).split('.')[0]
        if not duplicates_df.empty:
            duplicates_df.to_csv(f"dataset_audit/{base_name}_duplicates.csv", index=False)
        if isinstance(contradictory, pd.DataFrame) and not contradictory.empty:
            contradictory.to_csv(f"dataset_audit/{base_name}_contradictory_labels.csv", index=False)
            
        if label_col:
            df[label_col].value_counts().reset_index().to_csv(f"dataset_audit/{base_name}_label_dist.csv", index=False)
            
        if 'Source' in df.columns:
            df['Source'].value_counts().reset_index().to_csv(f"dataset_audit/{base_name}_source_dist.csv", index=False)

    # Save inventory
    inv_df = pd.DataFrame(inventory)
    inv_df.to_csv('dataset_audit/dataset_inventory.csv', index=False)
    
    # Generate Markdown Report
    with open('dataset_audit/dataset_audit_report.md', 'w') as f:
        f.write("# Dataset Audit Report\n\n")
        for i, row in inv_df.iterrows():
            f.write(f"## {row['dataset']}\n")
            f.write(f"- Total Records: {row['total_records']}\n")
            f.write(f"- Unique Records: {row['unique_records']}\n")
            f.write(f"- Duplicate Count: {row['duplicate_count']}\n")
            f.write(f"- Ambiguous: {row['ambiguous_count']}\n")
            f.write(f"- Not Ambiguous: {row['not_ambiguous_count']}\n")
            f.write(f"- Contradictory Labels: {row['contradictory_labels']}\n\n")
            
    print("\nDataset audit complete. Check dataset_audit/ folder.")

if __name__ == "__main__":
    audit_datasets()
