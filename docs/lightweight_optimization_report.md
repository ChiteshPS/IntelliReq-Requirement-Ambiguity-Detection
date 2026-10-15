# IntelliReq — Final Fast and Reliable Model Optimization Report

## Executive Summary

This report documents the final, fast, and reliable model optimization for the **IntelliReq** Software Requirement Ambiguity Detection System. Designed specifically for a sole-developer workflow under tight execution constraints, this optimization:
1. **Preserved the Production Baseline:** Kept the primary calibrated L1 Logistic Regression model at `saved_models/model.joblib` and its associated artifacts (`scaler.joblib`, `metadata.json`, and `saved_models/all-MiniLM-L6-v2`).
2. **Maintained the Challenger:** Retained the hybrid Calibrated Random Forest model as a benchmark challenger.
3. **Prevented Data Contamination:** Performed automated quality and overlap audits across all six requirement corpora without directly merging external datasets whose operational definitions of ambiguity diverge.
4. **Lightweight Validation-Only Exploration:** Leveraged cached feature embeddings (`saved_models/benchmark_cache/`) to evaluate candidate regularizations ($C \in \{0.5, 1.0, 2.0\}$) and threshold policies ($0.70/0.30$ through $0.90/0.10$) strictly on training and validation splits.
5. **Untouched Test Verification:** Evaluated the final models on the untouched held-out clean test partition (`test_v4`), verifying that the client's **97.0% accuracy target is genuinely achieved in-domain** (**98.15% – 99.07% Accuracy**, **100.0% Selective Accuracy**).
6. **Application Integrity:** Confirmed that all FastAPI endpoints, React components, 119/119 unit/regression tests, and 56/56 acceptance checks pass without any breaking changes.

---

## 1. Production Model Selection and Architecture Rationale

### Selected Production Engine
- **Model:** `saved_models/model.joblib` (Calibrated Logistic Regression with L1 Penalty, Sigmoid Platt Scaling via 5-fold CV).
- **Backbone Embeddings:** Sentence-Transformers `all-MiniLM-L6-v2` (384 dense semantic dimensions).
- **Linguistic Rules:** 24 handcrafted features (vague terms, subjective adjectives, measurable metric/unit constraints, standard format indicators, readability indices).
- **Total Representation:** 408 standardized features scaled via `saved_models/scaler.joblib`.
- **Primary Training Corpus:** `data/intellireq_clean_dataset_v4.csv` (715 clean requirements partitioned via `GroupShuffleSplit` on `semantic_group_id`: 500 Train, 107 Validation, 108 Test).

### Why Baseline LR (L1) Remains Selected Over All Challengers:
| Criterion | Baseline A1: Calibrated L1 LR | Challenger: Calibrated Random Forest | Candidate: DistilRoBERTa Transformer |
|---|---|---|---|
| **In-Domain Test Accuracy** | **98.15% – 99.07%** | 97.22% | 48.44% (on FPSRS) / 33.33% (on v4) |
| **In-Domain Balanced Acc** | **99.35%** | 95.16% | 53.25% |
| **In-Domain Ambiguous Recall** | **100.00%** (0 false negatives) | 90.32% (3 false negatives) | 94.84% (FPSRS) |
| **In-Domain Selective Accuracy** | **100.00%** (Coverage: 93.52%) | 98.96% (Coverage: 88.89%) | N/A (binary only) |
| **Out-of-Domain Challenge Acc** | **85.00% – 87.50%** | 78.75% – 90.00% | 41.25% |
| **Challenge Selective Acc** | **92.54% – 94.64%** (Coverage: 70.0%) | 100.00% (Coverage: 41.2% - over-defers!) | N/A |
| **Calibration Quality (Brier)** | **0.0168 – 0.0233** (Well-calibrated) | 0.0314 – 0.0341 | 0.4497 (Severe overconfidence) |
| **CPU Inference Latency** | **< 15 ms / sentence** | ~25 ms / sentence | > 350 ms / sentence |

**Conclusion:** The Calibrated L1 Logistic Regression model delivers the highest test accuracy (99.07%), perfect ambiguous recall (zero missed ambiguous requirements on the held-out set), superior probability calibration, and rapid CPU execution without over-deferring to human review.

---

## 2. Automated Dataset Quality & Overlap Audit

To avoid contaminating the operational criteria of IntelliReq, external corpora (PURE, User Stories, and Fault-Prone SRS) were audited automatically and maintained as independent benchmark sets:

| Dataset | Total Rows | Missing Texts | Exact Duplicates | Normalized Duplicates | Conflicting Labels | Operational Definition |
|---|---|---|---|---|---|---|
| **IntelliReq Clean v4** | 715 | 0 | 0 | 0 | 0 | Primary: Ground-truth verifiable IEEE requirements |
| **Diagnostic Challenge Set** | 100 | 0 | 0 | 0 | 0 | Diagnostic: Subtle IEEE 830 traps & non-functional requirements |
| **Adversarial Trap Set** | 16 | 0 | 0 | 0 | 0 | Stress test: Syntactic distractors & counter-examples |
| **PURE Formal SRS** | 6,725 | 0 | 191 | 191 | 0 | External: IEEE SRS documents with defect reason tags |
| **Agile User Stories** | 1,680 | 0 | 7 | 7 | 0 | External: Agile Connextra template user stories |
| **Fault-Prone SRS (Kaggle)** | 7,061 | 0 | 251 | 251 | **49** | External: 6-class fault-prone requirements (severe label noise) |

### Dataset Overlap Matrix (Normalized Text Intersection):
- **Clean v4 vs Diagnostic Challenge Set:** **0 rows (Zero Leakage Verified)**
- **Clean v4 vs Adversarial Trap Set:** **0 rows (Zero Leakage Verified)**
- **User Stories vs Clean v4 / Challenge:** **0 rows (Zero Contamination)**
- **Fault-Prone SRS vs Clean v4 / Challenge:** **0 rows (Zero Contamination)**
- **PURE vs Clean v4:** 1 accidental generic string (*"The system shall be reliable."*).
- **Quality Audit Artifact:** Saved in [`evaluation/data_quality_audit_summary.json`](file:///c:/Ambiguity%20detection/evaluation/data_quality_audit_summary.json).

---

## 3. Error Analysis Synthesis (Root Causes of FP / FN)

Reviewing [`evaluation/error_analysis.md`](file:///c:/Ambiguity%20detection/evaluation/error_analysis.md) on the diagnostic challenge set identified the key linguistic failure modes:

1. **False Positives (Clean Requirements Flagged as Ambiguous):**
   - *Example:* *"The system shall complete authentication before granting access to protected resources."* (Predicted $P = 51.5\%$)
   - *Root Cause:* Temporal prepositions (*"before"*, *"when"*) triggered mild ambiguity probability in pure text embeddings despite representing valid procedural logic.
   - *Mitigation:* The three-tier decision policy accurately catches these in the `NEEDS REVIEW` band ($25\% \le P \le 75\%$), preventing incorrect outright rejections.
2. **False Negatives (Ambiguous Requirements Missed):**
   - *Example:* *"Access must be strictly controlled."* (Predicted $P = 7.43\%$)
   - *Root Cause:* Short imperative clauses with strong modal verbs (*"must"*) without adjectives can superficially resemble concise clear requirements.
   - *Mitigation:* The rule-based explanation engine identifies the subjective qualifier (*"strictly"*) and flags it for review.
3. **Borderline Requirements:**
   - Soft modals (*"may"*, *"should"*, *"could"*) mixed with explicit numerical bounds (*"The system may cache results for up to 5 minutes."*) create conflicting evidence, precisely validating the necessity of human deferral.

---

## 4. Hyperparameter and Decision Threshold Optimization

Using cached features (`train_v4` and `val_v4`), we tested candidate classifiers and threshold configurations strictly on validation data (keeping the test set sealed):

### Validation Classifier Comparison (Validation Set: 107 rows):
| Candidate Configuration | Val Accuracy | Balanced Acc | Macro-F1 | Brier Loss | Selective Acc (Coverage) |
|---|---|---|---|---|---|
| **Baseline LR (L1, $C=1.0$)** | 98.13% | 98.29% | 97.84% | 0.0178 | 100.0% (95.3% coverage) |
| **Candidate LR (L1, $C=0.5$)** | 98.13% | 98.29% | 97.84% | 0.0188 | 99.02% (95.3% coverage) |
| **Candidate LR (L1, $C=2.0$)** | 98.13% | 98.29% | 97.84% | **0.0168** | 100.0% (95.3% coverage) |
| **Candidate LR (L2, $C=1.0$)** | 94.39% | 93.94% | 93.43% | 0.0278 | 99.00% (93.5% coverage) |
| **Challenger RF ($n=100$)** | 95.33% | 94.38% | 94.38% | 0.0341 | 98.00% (93.5% coverage) |
| **Challenger RF ($n=200$, depth=12)** | 96.26% | 95.54% | 95.54% | 0.0330 | 97.98% (92.5% coverage) |
| **Challenger RF ($n=300$, leaf=2)** | 96.26% | 95.54% | 95.54% | 0.0314 | 98.98% (91.6% coverage) |

### Decision Threshold Sensitivity Analysis (Validation Set):
| Upper Threshold | Lower Threshold | Strict 3-Way Acc | Selective Accuracy | Coverage | Deferral Rate |
|---|---|---|---|---|---|
| 0.70 | 0.30 | 96.26% | 98.10% | 98.1% | 1.9% |
| **0.75** | **0.25** | **95.33%** | **100.00%** | **95.3%** | **4.7%** |
| 0.80 | 0.20 | 92.52% | 100.00% | 92.5% | 7.5% |
| 0.85 | 0.15 | 85.98% | 100.00% | 86.0% | 14.0% |
| 0.90 | 0.10 | 83.18% | 100.00% | 83.2% | 16.8% |

**Threshold Decision:** The default threshold pair **(0.75 / 0.25)** achieves **100.0% Selective Accuracy** with **95.3% coverage** on validation data. Widening thresholds further (e.g. 0.85/0.15) increases deferrals to 14.0% without improving selective accuracy, confirming that 0.75/0.25 is optimal.

---

## 5. Final Evaluation on Untouched Held-Out Test Data

Once validation tuning concluded, the models were evaluated on the sealed held-out test split (108 items):

| Metric | Production Baseline (L1 LR) | Challenger (Random Forest) |
|---|---|---|
| **Raw Binary Accuracy** | **99.07%** (107 / 108 correct) | 97.22% (105 / 108 correct) |
| **Balanced Accuracy** | **99.35%** | 95.16% |
| **Macro-F1** | **98.88%** | 96.50% |
| **Ambiguous Recall** | **100.00%** (31 / 31 detected) | 90.32% (28 / 31 detected) |
| **Ambiguous Error Rate (FNR)** | **0.00%** | 9.68% |
| **Strict 3-Way Accuracy** | **93.52%** | 87.96% |
| **Selective Accuracy** | **100.00%** | 98.96% |
| **Prediction Coverage** | **93.52%** (6.48% deferred to review) | 88.89% (11.11% deferred) |
| **Confusion Matrix** | $TP=31, FN=0, FP=1, TN=76$ | $TP=28, FN=3, FP=0, TN=77$ |

---

## 6. External Cross-Dataset Performance Summary

When evaluated on independent out-of-domain and external corpora:
- **Diagnostic Challenge Benchmark (80 binary rows):**
  - Raw Accuracy: **85.00% – 87.50%**
  - Ambiguous Recall: **90.00%**
  - Selective Accuracy: **92.54% – 94.64%** (Coverage: 70.0%)
- **Adversarial Stress Test (16 rows):**
  - Raw Accuracy: **87.50%**
  - Selective Accuracy: **100.00%** (Coverage: 68.75%)
- **PURE Formal SRS (6,725 rows):** **79.53% Accuracy** (74.37% Amb Recall)
- **User Stories (1,680 rows):** **75.30% Accuracy** (88.48% Amb Recall)
- **Fault-Prone SRS (7,061 rows):** **70.12% Accuracy** (62.91% Amb Recall)

### 97% Accuracy Target Verification:
- **In-Domain Clean Data:** **YES, VERIFIED (98.15% – 99.07% Accuracy, 100.0% Selective Accuracy)**.
- **Cross-Dataset Generalization:** **NO**. External transfer accuracy ranges from 70% to 87.5% due to legitimate domain and annotation differences between IEEE SRS documents, agile user stories, and Kaggle fault categories.

---

## 7. Exact Commands to Run the Application

### 1. Run Automated Test Suites:
```powershell
# Run full 119-test pytest suite
python -m pytest tests/

# Run complete 56-check acceptance gate
python acceptance_check.py
```

### 2. Run the Optimization & Audit Pipeline:
```powershell
# Run automated dataset quality audit and lightweight optimization
python tools/lightweight_optimization.py

# Check standalone accuracy across all internal splits
python tools/check_accuracy.py
```

### 3. Start the FastAPI Backend:
```powershell
# Starts the API server on http://localhost:8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the React Frontend:
```powershell
# From the frontend directory:
cd frontend
npm run build   # Validates TypeScript & builds production bundle
npm run dev     # Starts Vite development server on http://localhost:5173
```

### 5. Run the CLI Interface:
```powershell
# Single interactive requirement analysis
python main.py

# Direct analysis via terminal argument
python main.py "The system shall respond within 150 ms."
```
