import pandas as pd
import os

datasets = [
    'dataset/software_requirements.csv',
    'software_requirements_extended.csv',
    'dataset/FR_NFR_Dataset.xlsx',
    'dataset/Cornelius_FR_NFR_Curated_Mendeley_Dataset_2025.xlsx',
    'dataset/Empirical-Data-Package_SLR_QualityAssessment_AI4RE.xlsx'
]

for d in datasets:
    if not os.path.exists(d):
        print(f"\n--- {d} NOT FOUND ---")
        continue
    print(f"\n--- {d} ---")
    try:
        df = pd.read_csv(d) if d.endswith('.csv') else pd.read_excel(d)
        print(f"Rows: {df.shape[0]}, Cols: {df.shape[1]}")
        print(f"Columns: {list(df.columns)}")
        print(f"Duplicates: {df.duplicated().sum()}")
        print(f"Missing:\n{df.isnull().sum().to_dict()}")
        for c in df.columns:
            if df[c].nunique() < 30:
                print(f"Value counts for {c}:\n{df[c].value_counts().to_dict()}")
    except Exception as e:
        print(f"Error reading {d}: {e}")
