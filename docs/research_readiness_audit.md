# IntelliReq — Research Readiness & Scientific Audit Report

## 1. Executive Summary & Research Positioning
- **Project Name:** IntelliReq — Smart Software Requirement Analyzer
- **Research Title:** Enhanced Ambiguity Detection in Software Requirements Using Semantic Sentence Embeddings and Hybrid Machine Learning
- **Model Version:** v2.1.0 (Production Verified Build)
- **Base Paper Reference:** *Bridging Precision and Complexity: A Novel Machine Learning Approach for Ambiguity Detection in Software Requirements* (Izhar, Bhatti, & Alharthi, IEEE Access, Vol. 13, Jan 2025).

The base paper reported approximately ~96.8% (rounded to 97%) accuracy using 1,000-dimensional TF-IDF unigram/bigram features paired with Random Forest on controlled template datasets. 

Our investigation demonstrated that pure TF-IDF features suffer from extreme out-of-domain fragility: when evaluated on our zero-overlap independent challenge benchmark of 80 industrial software requirements, TF-IDF + Logistic Regression collapsed to **50.00% accuracy and 4.76% F1**, failing entirely to recognize semantic synonyms outside its closed training vocabulary.

IntelliReq resolves this generalization gap through a hybrid 408-dimensional feature architecture combining:
1. **Dense 384-dimensional contextual sentence embeddings** from Sentence-BERT (`all-MiniLM-L6-v2`)
2. **24 handcrafted linguistic and specificity cues** (measuring lexical ambiguity, fuzzy modifiers, passive voice, modal auxiliary verbs, and testable unit bounds)
3. **L1-regularized sparse Logistic Regression** with Platt sigmoid calibration
4. **Three-tier guarded decision policy:** $P \ge 0.75$ (`AMBIGUOUS`), $P \le 0.25$ (`NOT AMBIGUOUS`), and $0.25 < P < 0.75$ (`NEEDS REVIEW`).

---

## 2. Empirical Verification on Identical Controlled Splits

To ensure scientific honesty without data leakage, all models were evaluated on identical, strictly frozen held-out and out-of-domain test sets:

| Model Architecture | Held-Out Test (N=108) Accuracy | Held-Out F1 (Ambiguous) | Challenge Set (N=80) Accuracy | Challenge F1 (Ambiguous) | Adversarial Set (N=16) Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | **99.07%** | 98.36% | **50.00%** | 4.76% | **50.00%** |
| **SBERT Only + Logistic Regression** | 93.52% | 88.89% | 75.00% | 75.61% | 87.50% |
| **Linguistic Only + Logistic Regression** | 92.59% | 88.24% | 73.75% | 71.23% | 68.75% |
| **IntelliReq Hybrid (SBERT + Ling + L1 LR)** | **98.15%** | **96.88%** | **85.00%** | **86.05%** | **87.50%** |

### Key Research Insights:
1. **TF-IDF Overfitting:** High internal held-out accuracy (99.07%) is misleading because TF-IDF memorizes exact vocabulary tokens. On unseen industrial specifications, it drops by nearly 50 percentage points.
2. **Hybrid Advantage:** The IntelliReq Hybrid architecture beats pure SBERT by **+10.0%** on the challenge set and pure Linguistic features by **+11.25%**, confirming that semantic geometry and structural regex cues provide complementary predictive signals.
3. **Generalization Over Baseline:** On external specifications, IntelliReq achieves an absolute **+35.0% accuracy improvement** over the base paper's TF-IDF formulation.
4. **Selective Accuracy:** When uncertain requirements in the $[0.25, 0.75]$ interval are deferred to senior software engineers, selective non-deferred accuracy reaches **92.54%** on the challenge benchmark.

---

## 3. Data Integrity & Leakage Verification

| Audit Check | Status | Verification Detail |
| :--- | :--- | :--- |
| **Exact Duplicate Rows** | PASSED (0) | 100% unique text across 715 samples in Clean v4 dataset |
| **Cross-Split Overlap** | PASSED (0) | 0 items shared between Train, Val, Test, Challenge, and Adversarial |
| **Contradictory Labels** | PASSED (0) | All contradictory pairs resolved; conflicting semantics removed |
| **Template Pair Leakage** | PASSED (0) | Syntactic contrastive pairs kept together in the same partition |
| **Data Normalization Isolation** | PASSED | `StandardScaler` fitted strictly on Training data only |
| **Calibration Isolation** | PASSED | Platt scaling fitted on Validation partition only; Test untouched |

---

## 4. Software Quality & System Health

- **Unit & Integration Tests:** 105 / 105 passed (`pytest tests/`)
- **System Acceptance Checks:** 56 / 56 passed (`python acceptance_check.py`)
- **Offline Inference:** Verified 100% offline using locally bundled Sentence-BERT model (`saved_models/all-MiniLM-L6-v2`)
- **Supported User Input Formats:**
  1. Single Requirement (Terminal & Streamlit UI)
  2. Plain Text (`.txt`) — one requirement per non-empty line
  3. Excel (`.xlsx`) — requirement text column with multi-sheet and column selection
- **Unsupported Formats:** Explicitly rejects `.csv`, `.pdf`, `.docx`, and `.json` with actionable guidance.
