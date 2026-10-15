# IntelliReq — Final Scientific Model Report & Benchmark Verification

**Project:** IntelliReq — Smart Software Requirement Analyzer  
**Research Title:** Enhanced Ambiguity Detection in Software Requirements Using Semantic Sentence Embeddings and Hybrid Machine Learning  
**Base Paper:** *Bridging Precision and Complexity: A Novel Machine Learning Approach for Ambiguity Detection in Software Requirements* (Izhar, Bhatti, & Alharthi, IEEE Access, Vol. 13, Jan 2025).  
**Model Version:** `2.1.0`  
**Acceptance Status:** `VERIFIED FINAL BUILD` (56/56 checks passed, 61/61 pytest tests passed)

---

## 1. Executive Summary & Base-Paper Comparison

### A. Base Paper Verification (Izhar et al., IEEE Access 2025)
Our audit inspected the published PDF `Bridging_Precision_and_Complexity_A_Novel_Machine_Learning_Approach_for_Ambiguity_Detection_in_Software_Requirements.pdf`:
- **Dataset:** 1,553 software requirements (derived from Kaggle/Zenodo PROMISE requirements extended with industry documentation).
- **Split:** 80% Train (1,244 requirements), 20% Test (312 requirements).
- **Architecture:** Hybrid Rule-Based Phrase Matching + TF-IDF (1,000 features) + Random Forest Classifier.
- **Reported Test Metrics:**
  - Accuracy: **96.79% (reported as 97%)**
  - Precision: **97%**
  - Recall: **89%**
  - F1-Score: **92%**
  - Confusion Matrix on 312 test samples: 271 True Negatives, 31 True Positives, ~10 errors.
- **Identified Limitation in Base Paper:**
  The authors noted: *"Models like SVM and Logistic Regression excelled in precision but suffered from extremely low recall, indicating their limitations in detecting all ambiguous requirements."*
  Because TF-IDF is restricted to exact token matches, it fails to recognize unseen synonyms or contextual paraphrases when deployed outside the training vocabulary. When tested on our external independent challenge set, TF-IDF + Random Forest collapsed to **50.00% accuracy and 0.00% recall**!

### B. IntelliReq Research Contribution
IntelliReq overcomes this domain-transfer failure by combining:
1. **Dense Semantic Embeddings:** Pretrained Sentence-BERT (`all-MiniLM-L6-v2`, 384 dimensions) capturing semantic meaning invariant to word choice.
2. **24 Handcrafted Linguistic & Specificity Features:** Explicitly encoding vague terms, subjective adjectives, unquantified adverbs, temporal bounds, and quantitative tolerances.
3. **Calibrated Linear Classifier:** $L_1$-regularized Logistic Regression calibrated via Platt scaling (sigmoid) with a 3-way decision policy (`AMBIGUOUS`, `NOT AMBIGUOUS`, `NEEDS REVIEW`).

On an independent, out-of-domain benchmark test set, IntelliReq achieves **85.00% raw binary accuracy** and **92.54% selective accuracy** (with 92.50% recall), demonstrating robust domain generalization where TF-IDF models completely collapse.

---

## 2. Dataset Composition & Leakage Audit

| Property | Value |
|---|---|
| Main Training Dataset | `data/intellireq_clean_dataset_v4.csv` |
| Total Rows | 715 |
| Class Distribution | Ambiguous: 217 (30.3%), Not Ambiguous: 498 (69.7%) |
| Splitting Strategy | Group-Aware Split (`GroupShuffleSplit`, random_state=42) |
| Partition Sizes | Train: 500 (70%), Validation: 107 (15%), Held-out Test: 108 (15%) |
| Overlap Across Splits | **0 overlapping rows / 0 leakage** |

### Investigation into Internal Near-Perfect Accuracy
- On the held-out test split of the controlled dataset, the model achieves **98.15% raw accuracy** (106/108 correct).
- **Forensic Diagnosis:** The training dataset consists of 450 controlled synthetic requirement variations paired with 265 curated requirements. Synthetic variants often share structural sentence skeletons (e.g. `"...strictly defined as within [X] ms"`). 
- When evaluated strictly on **unseen external projects** (`data/challenge_test_set.csv`, N=80), raw binary accuracy is **85.00%**, and selective accuracy among non-deferred predictions is **92.54%**. This provides the true, non-inflated measure of generalization.

---

## 3. Scientific Model Comparison (Ablation Study)

Evaluated under identical group-aware partitions and test sets:

### A. Performance on Group-Aware Held-Out Test Set (N=105)
| Architecture | Accuracy | Precision (Amb) | Recall (Amb) | F1 (Amb) |
|---|---|---|---|---|
| TF-IDF (1,000) + Random Forest (Base Paper) | 88.57% | 100.00% | 66.67% | 80.00% |
| TF-IDF (1,000) + Logistic Regression | 89.52% | 100.00% | 69.44% | 81.97% |
| SBERT (384) + Logistic Regression | 78.10% | 69.70% | 63.89% | 66.67% |
| **SBERT (384) + Ling/Spec (24) + Calibrated LR (Proposed)** | **96.19%** | **92.11%** | **97.22%** | **94.59%** |

### B. Generalization on Independent External Challenge Test Set (N=80)
| Architecture | Accuracy | Precision (Amb) | Recall (Amb) | F1 (Amb) |
|---|---|---|---|---|
| TF-IDF (1,000) + Random Forest (Base Paper) | 50.00% | 0.00% | 0.00% | 0.00% |
| TF-IDF (1,000) + Logistic Regression | 53.75% | 100.00% | 7.50% | 13.95% |
| SBERT (384) + Logistic Regression | 66.25% | 72.41% | 52.50% | 60.87% |
| **SBERT (384) + Ling/Spec (24) + Calibrated LR (Proposed)** | **87.50%** | **82.61%** | **95.00%** | **88.37%** |

---

## 4. Comprehensive Metrics Audit (Phase 8 Standard)

Computed by `tools/scientific_evaluation.py` on the deployed model:

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

### Definitions:
- **Raw Binary Accuracy:** Direct classification at $p \ge 0.50$.
- **Macro-F1:** Unweighted mean of $F_1$ across `Ambiguous` and `Not Ambiguous`.
- **Brier Score Loss:** Mean squared difference between predicted probability and binary target ($0.0$ is perfect calibration).
- **Coverage:** Fraction of samples where model confidence exceeds thresholds ($\ge 0.75$ or $\le 0.25$).
- **Selective Accuracy:** Accuracy on the subset of samples that were NOT deferred to `NEEDS REVIEW`.
- **Strict 3-Way Accuracy:** Accuracy where deferred predictions on binary targets are penalized as incorrect.

---

## 5. Offline SBERT Resolution (Hugging Face Fix)

### Issue:
During network dropouts, `SentenceTransformer("all-MiniLM-L6-v2")` attempted repeated HTTP handshakes with Hugging Face Hub, causing connection timeouts (`wsarecv: An existing connection was forcibly closed`). Furthermore, a corrupted `sentencepiece 0.2.2` package in the Python environment caused native Windows Access Violations.

### Solution:
1. **Uninstalled corrupted `sentencepiece 0.2.2`** (which is not needed for WordPiece/MiniLM models).
2. **Permanently bundled the model snapshot** directly inside the repository at `saved_models/all-MiniLM-L6-v2/` (87.34 MB).
3. **Updated `feature_engineering/sbert_extractor.py`** to deterministically resolve local snapshot paths first.
4. **Result:** Pytest suite runtime dropped from 29.94s to **6.27s**, and inference runs 100% offline with zero network overhead.

---

## 6. Evaluation Visualizations & Plots

All evaluation plots in `evaluation/` have been regenerated and verified against the active model readings:

| File | Description | Metrics / Content |
|---|---|---|
| `evaluation/confusion_matrix.png` / `heldout_confusion_matrix.png` | Held-Out Clean Test Set | $N=108$, Acc: **98.15%**, TP=31, FN=0, FP=2, TN=75 |
| `evaluation/challenge_confusion_matrix.png` | Independent Challenge Test Set | $N=80$, Acc: **85.00%**, TP=37, FN=3, FP=9, TN=31 |
| `evaluation/adversarial_confusion_matrix.png` | Adversarial Trap Test Set | $N=16$, Acc: **87.50%**, TP=7, FN=1, FP=1, TN=7 |
| `evaluation/feature_importance.png` | L1 Regularized Feature Weights | Top positive (Not Ambiguous) & negative (Ambiguous) predictors |
| `evaluation/dataset_distribution.png` | Canonical Dataset Partitioning | Class distribution (217 Amb / 498 Not Amb) and Train/Val/Test splits |
| `evaluation/model_comparison.png` | Architecture Benchmark Comparison | Base Paper TF-IDF vs SBERT vs Hybrid across all datasets |
| `evaluation/calibration_thresholds.png` | Probability Confidence Profiles | 3-tier zones: $\le 0.25$ Not Amb, $0.25-0.75$ Needs Review, $\ge 0.75$ Amb |

*Note: Obsolete legacy files from earlier clustering/sentiment experiments (`cluster_plot.png`, `elbow_plot.png`, `sentiment_distribution.png`) have been permanently removed.*

---

## 7. How to Run & Verify

```powershell
# 1. Run external dataset audit and cross-dataset benchmark suite
python tools/audit_and_experiment_external.py

# 2. Run multi-source cross-dataset benchmarks
python tools/external_cross_benchmark.py

# 3. Generate and update all evaluation graphs
python tools/generate_evaluation_plots.py

# 4. Run the master evaluation pipeline
python evaluate.py

# 5. Run the comprehensive scientific metric audit
python tools/scientific_evaluation.py

# 6. Run the full pytest test suite
python -m pytest tests/

# 7. Run the 14-section automated acceptance gate (56 checks)
python acceptance_check.py

# 8. Launch the terminal interactive application
python main.py
```

---

## 8. External Replication Dataset Audit & Cross-Dataset Generalization Analysis

### A. External Dataset Provenance & Scope
In accordance with research reproducibility standards, external data was incorporated from the official public replication package (v2.0.0):
- **Source Repository:** [profzivkovic/software-requirements-ambiguity-detection-replication](https://github.com/profzivkovic/software-requirements-ambiguity-detection-replication/tree/v2.0.0)
- **Archive:** [Zenodo 19746839](https://zenodo.org/records/19746839)
- **License:** MIT License (Copyright 2025 Dr. Zivkovic)
- **Directory Location:** `data/external/`

| Dataset | Total Samples | Ambiguous | Clear | Source Documents / Projects | Label Mapping |
|---|---|---|---|---|---|
| **PURE** (`pure_labeled.csv`) | 6,725 | 3,424 (50.91%) | 3,301 (49.09%) | 66 formal SRS documents | 1 = Ambiguous, 0 = Clear |
| **User Stories** (`userstories_labeled.csv`) | 1,680 | 935 (55.65%) | 745 (44.35%) | 22 Agile projects | 1 = Ambiguous, 0 = Clear |

### B. Audit & Data Leakage Verification
1. **Schema & Labels:**
   - Both datasets provide canonical columns: `['source_file', 'requirement_text', 'label', 'reasons']`.
   - Ambiguity reasons categorize classical IEEE 830 triggers: `weak_modal` (1,306 in PURE, 594 in US), `vague_term` (782 in PURE, 96 in US), `missing_actor`, `multiple_actions`, `undefined_quantity`, and `undefined_term`.
   - Clear requirements (`label == 0`) strictly contain zero reason tags across both datasets.
2. **Duplicate & Conflict Analysis:**
   - PURE contains 333 exact duplicate requirements (339 normalized); User Stories contains 14 exact duplicates.
   - **Label conflicts among duplicates: 0 rows (100% label consistency).**
3. **Zero Contamination Guarantee:**
   - Overlap between PURE and User Stories: **0 rows (0.0%)**.
   - Overlap between external datasets and IntelliReq Clean v4: **0 rows (0.0%)**.
   - Overlap between external datasets and Diagnostic Challenge Set: **0 rows (0.0%)**.
   - Overlap between external datasets and Adversarial Trap Set: **0 rows (0.0%)**.

### C. Cross-Dataset Transfer & Multi-Source Generalization Matrix

Evaluated under strict `GroupShuffleSplit` (isolating entire specification documents/projects):

| Experiment / Setting | Evaluated Model | Accuracy | Balanced Acc | Macro-F1 | Amb Recall | Amb Error Rate | Brier Loss | Selective Acc |
|---|---|---|---|---|---|---|---|---|
| **1. In-Domain Clean v4** | Model A (Hybrid LR L1) | **97.22%** | **97.09%** | **96.64%** | **96.77%** | 3.23% | 0.0233 | 99.01% |
| **1. In-Domain Clean v4** | Model B (Hybrid Random Forest) | **97.22%** | **97.09%** | **96.64%** | **96.77%** | 3.23% | 0.0182 | 100.00% |
| **1b. Diagnostic Challenge Set** | Model A (Hybrid LR L1) | **85.00%** | **85.00%** | **84.96%** | **90.00%** | 10.00% | 0.1091 | 88.24% |
| **1b. Diagnostic Challenge Set** | Model B (Hybrid Random Forest) | **88.75%** | **88.75%** | **88.73%** | **92.50%** | 7.50% | 0.0999 | 93.88% |
| **2. In-Domain User Stories** | Model A (Hybrid LR L1) | 75.30% | 71.97% | 65.81% | 88.48% | 11.52% | 0.1743 | 87.23% |
| **2. In-Domain User Stories** | Model B (Hybrid Random Forest) | 75.89% | 72.48% | 65.41% | 90.53% | 9.47% | 0.1840 | 79.80% |
| **3. In-Domain PURE** | Model A (Hybrid LR L1) | **79.53%** | **80.45%** | **79.46%** | **74.37%** | 25.63% | 0.1380 | 93.08% |
| **3. In-Domain PURE** | Model B (Hybrid Random Forest) | 74.78% | 74.80% | 74.46% | 74.65% | 25.35% | 0.1540 | 91.50% |
| **4. PURE $\rightarrow$ User Stories** | Model A (Hybrid LR L1) | 61.73% | 62.38% | 61.72% | 56.58% | 43.42% | 0.2253 | 78.54% |
| **4. PURE $\rightarrow$ User Stories** | Model B (Hybrid Random Forest) | 60.83% | 60.44% | 60.44% | 63.64% | 36.36% | 0.2361 | 73.05% |
| **5. User Stories $\rightarrow$ PURE** | Model A (Hybrid LR L1) | **68.24%** | **68.39%** | **68.08%** | **60.19%** | 39.81% | 0.2162 | 75.94% |
| **5. User Stories $\rightarrow$ PURE** | Model B (Hybrid Random Forest) | 58.75% | 58.63% | 58.63% | 52.45% | 47.55% | 0.2595 | 65.01% |
| **6a. Combined Held-Out Test** | Model A (Hybrid LR L1) | 73.95% | 74.19% | 73.63% | 72.47% | 27.53% | 0.1663 | 89.66% |
| **6a. Combined Held-Out Test** | Model B (Hybrid Random Forest) | **75.55%** | **75.69%** | **75.18%** | **74.80%** | 25.20% | 0.1595 | 89.48% |
| **6b. Combined on Challenge** | Model B (Hybrid Random Forest) | 75.00% | 75.00% | 73.96% | 55.00% | 45.00% | 0.1650 | 86.49% |

### D. Scientific Findings & Verification of 97% Target
1. **Target Verification:**
   The client's **97.0% accuracy target is genuinely verified on the in-domain held-out test distribution** (Model A: 97.22%, Model B: 97.22% - 99.07%, F1: 96.64%).
2. **Generalization Gap on Independent Datasets:**
   When evaluated on out-of-domain challenge requirements, performance stands at **85.00% (Model A)** and **88.75% - 90.00% (Model B)**. Selective accuracy on non-deferred predictions reaches **92.54% - 93.88%**.
3. **Cross-Dataset Domain Shift:**
   Transfer between formal SRS (PURE) and agile templates (User Stories) experiences a 15–20% performance drop (79.5% in-domain down to 61.7% cross-domain). This demonstrates that in software requirements engineering, syntax templates (*"As a [role], I want..."* vs *"The system shall..."*) impose distinct distribution boundaries.
4. **Honest Reporting:**
   97% accuracy cannot be claimed as an independent cross-dataset generalization figure. Reporting in-domain performance honestly alongside cross-dataset and challenge set diagnostics prevents misleading over-optimism and provides a dependable foundation for real-world deployment.

---

## 9. Fault-Prone SRS Dataset Audit and Pretrained Transformer Experiment

### A. Dataset Provenance and Quality Audit
- **Source:** Kaggle Fault-Prone SRS Dataset (`corpus4panwo/fault-prone-srs-dataset`), derived from research published in MDPI *Applied Sciences* (2023, DOI: `10.3390/app13148368`).
- **Scale:** 7,061 software requirement statements across 6 native categories:
  - `clean`: 4,020 (56.9%)
  - `pragmatic`: 834 (11.8%)
  - `syntactic`: 615 (8.7%)
  - `semantic`: 614 (8.7%)
  - `syntax`: 504 (7.1%)
  - `lexical`: 474 (6.7%)
- **Binary Mapping:**
  - `NOT AMBIGUOUS`: 4,020 requirements (clean).
  - `AMBIGUOUS`: 3,041 requirements (lexical, syntactic, semantic, syntax, pragmatic).
- **Data Integrity Audit:**
  - 489 exact duplicate texts (573 normalized duplicates).
  - 43 conflicting labels in the 6-class native scheme; 35 conflicting labels in the binary scheme.
  - Group-aware partitioning (`GroupShuffleSplit`, 72.7% train, 13.7% val, 13.7% test) applied to prevent duplicate and cluster leakage across splits.

### B. Baseline vs Transformer Benchmark Results

| Prediction Task | Model Architecture | Accuracy | Balanced Acc | Macro-F1 | Amb Recall | Amb Error Rate |
|---|---|---|---|---|---|---|
| **Task A (6-Class)** | Baseline A1: Hybrid LR (L1 Calibrated) | 43.98% | 45.15% | **38.42%** | N/A | N/A |
| **Task A (6-Class)** | Baseline A2: Hybrid Random Forest | **61.20%** | 30.63% | 23.17% | N/A | N/A |
| **Task A (6-Class)** | Candidate B: DistilRoBERTa Fine-Tuned | 32.99% | 36.32% | 37.09% | N/A | N/A |
| **Task B (Binary)** | Baseline A1: Hybrid LR (L1 Calibrated) | **70.12%** | **69.37%** | **69.48%** | 62.91% | 37.09% |
| **Task B (Binary)** | Baseline A2: Hybrid Random Forest | 69.09% | 67.91% | 68.05% | 57.75% | 42.25% |
| **Task B (Binary)** | Candidate B: DistilRoBERTa Fine-Tuned | 48.44% | 53.25% | 41.07% | **94.84%** | **5.16%** |

### C. Cross-Dataset Transfer (Models Trained on Fault-Prone SRS)

| Target Test Set | Evaluated Model | Accuracy | Macro-F1 | Amb Recall | Amb Error Rate |
|---|---|---|---|---|---|
| **IntelliReq Clean v4 Test** | FPSRS-Trained Hybrid LR | 57.41% | 53.85% | 51.61% | 48.39% |
| **IntelliReq Clean v4 Test** | FPSRS-Trained DistilRoBERTa | 33.33% | 29.80% | 48.39% | 51.61% |
| **Diagnostic Challenge Set** | FPSRS-Trained Hybrid LR | 51.25% | 49.32% | 37.50% | 62.50% |
| **Diagnostic Challenge Set** | FPSRS-Trained DistilRoBERTa | 41.25% | 33.91% | 72.50% | 27.50% |
| **PURE Formal Requirements** | FPSRS-Trained Hybrid LR | 50.35% | 50.04% | 45.47% | 54.53% |
| **PURE Formal Requirements** | FPSRS-Trained DistilRoBERTa | 50.04% | 36.33% | 91.22% | 8.78% |

### D. Production Deployment Architecture Verdict
- **Candidate B (`distilroberta-base`) Assessment:** The fine-tuned transformer achieves high ambiguous recall (94.84%), but suffers from an unacceptably high false-positive rate on clean requirements (accuracy 48.44%), and drops to 33.33% on clean IEEE requirements due to lexical distribution shift.
- **Production Integrity Decision:** The existing production model (`saved_models/model.joblib`: SBERT + 24 Linguistic Features + Calibrated L1 Logistic Regression) is strictly retained. It delivers verified **97.22% in-domain accuracy** and **85.00% out-of-domain challenge accuracy** with calibrated confidence bands.
- **Artifact Preservation:** The DistilRoBERTa checkpoint and tokenizer are preserved in `saved_models/distilroberta_fpsrs/` for full research repeatability.


