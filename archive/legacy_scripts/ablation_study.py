import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading data...")
df = pd.read_csv('data/ambiguity_dataset.csv')
X = df['Requirement_Text']
y = df['Ambiguity_Label']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Extracting features...")
feat_extractor = FeatureExtractor()
sbert_extractor = SBERTExtractor()

X_train_ling = feat_extractor.process_dataframe(pd.DataFrame(X_train_raw, columns=['Requirement_Text']), 'Requirement_Text').drop(columns=['Requirement_Text']).reset_index(drop=True)
X_test_ling = feat_extractor.process_dataframe(pd.DataFrame(X_test_raw, columns=['Requirement_Text']), 'Requirement_Text').drop(columns=['Requirement_Text']).reset_index(drop=True)

X_train_sbert = sbert_extractor.transform(X_train_raw.tolist())
X_test_sbert = sbert_extractor.transform(X_test_raw.tolist())

X_train_hybrid = pd.concat([X_train_ling, X_train_sbert], axis=1)
X_test_hybrid = pd.concat([X_test_ling, X_test_sbert], axis=1)

print("\n--- ABLATION STUDY RESULTS ---")
# 1. Linguistic Only
lr_ling = LogisticRegression(max_iter=1000)
lr_ling.fit(X_train_ling, y_train)
acc_ling = accuracy_score(y_test, lr_ling.predict(X_test_ling))
print(f"Linguistic Features Only Accuracy: {acc_ling:.4f}")

# 2. SBERT Only
lr_sbert = LogisticRegression(max_iter=1000)
lr_sbert.fit(X_train_sbert, y_train)
acc_sbert = accuracy_score(y_test, lr_sbert.predict(X_test_sbert))
print(f"SBERT Semantic Features Only Accuracy: {acc_sbert:.4f}")

# 3. Hybrid
lr_hybrid = LogisticRegression(max_iter=1000)
lr_hybrid.fit(X_train_hybrid, y_train)
acc_hybrid = accuracy_score(y_test, lr_hybrid.predict(X_test_hybrid))
print(f"Hybrid (SBERT + Linguistic) Accuracy: {acc_hybrid:.4f}")
