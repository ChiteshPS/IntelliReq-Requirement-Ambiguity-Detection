# IntelliReq — Baseline Audit Report Before Master Build Changes

**Date:** 2026-10-13  
**Project:** IntelliReq — Smart Software Requirement Analyzer  
**Research Title:** Enhanced Ambiguity Detection in Software Requirements Using Semantic Sentence Embeddings and Hybrid Machine Learning  
**Environment:** Python 3.10.11 on Windows (AMD64)  
**Evaluation Status:** Baseline Verified (103/103 pytest tests passing, 56/56 acceptance gate checks passing)

---

## 1. Existing Architecture & Technologies

The IntelliReq system operates as a hybrid natural-language processing pipeline combining deep semantic embeddings with handcrafted rule-based linguistic features:

1. **Input Preprocessing & Ingestion:**
   - Single requirement sentences entered via CLI (`main.py`) or Streamlit UI (`app.py`).
   - Batch `.txt` files (one requirement per non-empty line).
   - Batch `.xlsx` files (with columns matching `Requirement`, `Requirement_Text`, or `requirement`).
   - Strict format gating: `.csv`, `.pdf`, `.docx`, `.json` are explicitly rejected to preserve parsing integrity.
   - Minimum requirement threshold: non-empty string with $\ge 3$ words (`is_valid_requirement`).

2. **Feature Extraction Pipeline:**
   - **Semantic Features (384 dimensions):** Pretrained Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`), bundled locally in `saved_models/all-MiniLM-L6-v2/` for 100% offline, deterministic inference.
   - **Handcrafted Features (24 dimensions):** Linguistic and domain-specificity feature extractor (`feature_engineering/feature_extractor.py`) calculating:
     - Structural/Lexical: `Token_Count`, `Word_Count`, `Sentence_Count`, `Char_Length`, `Avg_Token_Length`, `Lexical_Diversity`.
     - Ambiguity Cues: `Vague_Term_Count`, `Subjective_Term_Count`, `Approximate_Quantity_Count`, `Undefined_Reference_Count`, `Modal_Verb_Count`, `Passive_Voice_Count`, `Linguistic_Ambiguity_Score`.
     - Specificity & Metrics: `Numeric_Count`, `Percentage_Count`, `TimeUnit_Count`, `SizeUnit_Count`, `Capacity_Count`, `Threshold_Count`, `Format_Count`, `Condition_Count`, `Range_Count`, `Specificity_Score`, `Has_Testable_Metric`.
   - **Total Feature Space:** $384 + 24 = 408$ dimensions.

3. **Classification & Calibration:**
   - **StandardScaler:** Standardizes the 408-dimensional hybrid feature vector.
   - **Base Classifier:** $L_1$-regularized Logistic Regression (`solver='liblinear'`, `class_weight='balanced'`, `random_state=42`).
   - **Probability Calibration:** Platt scaling (`CalibratedClassifierCV(method='sigmoid', cv='prefit')`), fitted strictly on the validation set.
   - **Decision Policy (3-tier guardrails):**
     - $P(\text{Ambiguous}) \ge 0.75 \implies \text{AMBIGUOUS}$
     - $P(\text{Ambiguous}) \le 0.25 \implies \text{NOT AMBIGUOUS}$
     - $0.25 < P(\text{Ambiguous}) < 0.75 \implies \text{NEEDS REVIEW}$

4. **Explanation Engine (`rule_engine/explanation_engine.py`):**
   - Heuristic rule-based explanation extractor generating diagnosis reasons, missing specifications, and suggested rewrites.
   - **Isolation Guarantee:** The explanation engine never overrides the model's calibrated prediction.

---

## 2. Current Model and Artifact Versions

| Artifact | Path | Size | Description |
|---|---|---|---|
| Active Model | `saved_models/model.joblib` | 15.4 KB | `CalibratedClassifierCV` wrapping $L_1$ Logistic Regression |
| Scaler | `saved_models/scaler.joblib` | 15.6 KB | `StandardScaler` (408 features) |
| Metadata | `saved_models/metadata.json` | 453 B | Model build `v2.1.0`, timestamp, threshold definitions |
| Offline SBERT | `saved_models/all-MiniLM-L6-v2/` | ~87 MB | Local offline Sentence-BERT model weights |
| Legacy Classifier | `saved_models/primary_classifier.pkl` | 52.0 KB | Legacy archive model from previous iteration |
| Legacy Scaler | `saved_models/feature_scaler.pkl` | 15.5 KB | Legacy archive scaler |

---

## 3. Dataset Sizes, Class Distributions, and Data Sources

| Dataset File | Rows | Ambiguous | Not Ambiguous | Role / Purpose |
|---|---|---|---|---|
| `data/intellireq_clean_dataset_v4.csv` | 715 | 217 (30.35%) | 498 (69.65%) | Canonical training, validation, and held-out test dataset |
| `data/challenge_test_set.csv` | 80 | 40 (50.00%) | 40 (50.00%) | Independent out-of-domain evaluation set |
| `data/adversarial_test_set.csv` | 16 | 8 (50.00%) | 8 (50.00%) | Adversarial syntactic traps & distractor requirements |
| `data/intellireq_clean_dataset_v3.csv` | 654 | - | - | Previous intermediate clean iteration |
| `data/ambiguity_dataset_v2.csv` | 654 | - | - | Legacy raw dataset before duplicate pruning |

---

## 4. Train / Validation / Test Split Strategy

- **Dataset:** `data/intellireq_clean_dataset_v4.csv` (715 rows total).
- **Partitioning Algorithm:** Group-Aware Split using `GroupShuffleSplit(random_state=42)` on `semantic_group_id`.
  - Guarantees that near-duplicate syntactic variations (generated templates sharing underlying clause structures) are strictly isolated within a single partition, completely preventing data leakage across splits.
- **Partition Proportions:**
  - **Training Set (70%):** 500 rows (153 Ambiguous, 347 Not Ambiguous).
  - **Validation Set (15%):** 107 rows (33 Ambiguous, 74 Not Ambiguous) — used for model selection, threshold validation, and Platt probability calibration.
  - **Held-Out Test Set (15%):** 108 rows (31 Ambiguous, 77 Not Ambiguous) — held out until final evaluation.
- **Zero Leakage:**
  - Train $\leftrightarrow$ Challenge overlap: **0 rows (0.0%)**.
  - Train $\leftrightarrow$ Adversarial overlap: **0 rows (0.0%)**.

---

## 5. Internal, Challenge, and Adversarial Evaluation Results

Audited using `tools/scientific_evaluation.py` on the active model:

```text
================================================================================
                         SUMMARY COMPARISON TABLE
================================================================================
                Dataset Raw Binary Acc Macro-F1  Brier Coverage Selective Acc Strict 3-Way Acc
  Held-out Test (N=108)         98.15%   97.78% 0.0120   95.37%       100.00%           95.37%
Challenge Binary (N=80)         85.00%   84.92% 0.1023   83.75%        92.54%           77.50%
     Adversarial (N=16)         87.50%   87.50% 0.0636   68.75%       100.00%           68.75%
================================================================================
```

### Breakdown by Dataset:
1. **Held-Out Clean Test Set ($N=108$):**
   - Raw Accuracy: **98.15%** (106 / 108 correct)
   - Confusion Matrix: $TP=31, FN=0, FP=2, TN=75$
   - Precision (Ambiguous): 93.94% | Recall: 100.00% | F1: 96.88%
   - Precision (Not Ambiguous): 100.00% | Recall: 97.40% | F1: 98.68%
   - Brier Score Loss: 0.0120
   - Deferral Rate: 4.63% (5 / 108 deferred to `NEEDS REVIEW`)
   - Selective Accuracy (on non-deferred decisions): **100.00%**
   - Strict 3-Way Accuracy (penalizing deferrals as errors): **95.37%**

2. **Independent Challenge Set ($N=80$):**
   - Raw Accuracy: **85.00%** (68 / 80 correct)
   - Confusion Matrix: $TP=37, FN=3, FP=9, TN=31$
   - Precision (Ambiguous): 80.43% | Recall: 92.50% | F1: 86.05%
   - Precision (Not Ambiguous): 91.18% | Recall: 77.50% | F1: 83.78%
   - Brier Score Loss: 0.1023
   - Deferral Rate: 16.25% (13 / 80 deferred to `NEEDS REVIEW`)
   - Selective Accuracy: **92.54%**
   - Strict 3-Way Accuracy: **77.50%**
   - *Baseline Comparison:* On this exact same challenge set, the Base Paper's TF-IDF + Random Forest model achieves only **50.00% accuracy and 0.00% recall** because it relies strictly on exact word unigram matches.

3. **Adversarial Test Set ($N=16$):**
   - Raw Accuracy: **87.50%** (14 / 16 correct)
   - Confusion Matrix: $TP=7, FN=1, FP=1, TN=7$
   - Precision: 87.50% | Recall: 87.50% | F1: 87.50%
   - Brier Score Loss: 0.0636
   - Deferral Rate: 31.25% (5 / 16 deferred to `NEEDS REVIEW`)
   - Selective Accuracy: **100.00%**
   - Strict 3-Way Accuracy: **68.75%**

---

## 6. Existing Bugs, Inconsistent Documentation, and Technical Debt

1. **Streamlit Configuration:** No `.streamlit/config.toml` was defined; theme defaults fell back to browser-detected modes.
2. **Batch UI UX:** In the batch processing view, users need clear column mapping and sheet selection if an Excel workbook contains multiple worksheets.
3. **Session State Tracking:** In the Dashboard, real-time counters of user session activity (requirements evaluated during the current browser session, recent history table) need explicit session-state tracking.
4. **Documentation Sync:** Documentation across `docs/` contained multiple historical markdown reports from earlier development stages that must be unified with the master specification.
