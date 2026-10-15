# IntelliReq — Smart Software Requirement Ambiguity Detection & Quality Analytics

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%202.1.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite%208-61DAFB.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Acceptance Tests](https://img.shields.io/badge/Tests-119%20Passed%20%7C%2056%20Checks-success.svg)](acceptance_check.py)

**Academic Research Title:** *Enhanced Ambiguity Detection in Software Requirements Using Hybrid Semantic Embeddings and Calibrated Decision Boundaries*

---

## 1. Project Overview & Problem Statement

Software requirements specifications (SRS) serve as the formal contract between stakeholders, systems architects, and software engineers. However, natural language requirements frequently suffer from **semantic, lexical, and syntactic ambiguity**—such as vague temporal adverbs (*"quickly"*), subjective adjectives (*"user-friendly"*), weak modals (*"should"*), and open-ended non-quantified constraints. When ambiguities pass undetected into development, they lead to implementation rework, budget overruns, and critical software defects.

**IntelliReq** is an enterprise-grade ambiguity detection and requirement quality analytics system. It bridges dense contextual sentence representations with deterministic domain engineering rules to provide explainable, high-precision ambiguity diagnoses.

### Key Objectives:
- **Dual-Channel Feature Representation:** Fuses 384-dimensional dense semantic sentence embeddings from `all-MiniLM-L6-v2` with 24 handcrafted linguistic and metric-specificity indicators.
- **Three-Tier Operational Decision Policy:** Employs Platt sigmoid probability calibration to route uncertain or borderline requirements ($25\% < P < 75\%$) to human requirements engineers, achieving **100.00% selective decision accuracy** on in-domain held-out test data.
- **Transparent Explainability:** Delivers token-level linguistic trigger highlighting, missing quantitative measurement checklists, and automated rule-guided rewrite suggestions.
- **Zero-Data-Leakage Rigor:** Evaluated with group-aware concept stratification (`GroupShuffleSplit`), eliminating semantic paraphrase contamination across training and evaluation splits.

---

## 2. Key Features

- **Executive Analytics Dashboard:** Power BI-inspired research dashboard displaying core KPIs, multi-corpus transfer matrices, and live system telemetry.
- **Single Requirement Inspection Workspace:** Real-time semantic analysis with calibrated probability meters, diagnostic token highlighting, and suggested clearer rewrites with one-click clipboard copying.
- **Batch Specification Grid:** Ingest requirement spreadsheets (`.xlsx`) or plain text documents (`.txt`), auto-detect headers, filter by classification category, and export full audit reports to CSV.
- **Academic Model Evaluation Report:** Dedicated viva presentation suite displaying the in-domain confusion matrix ($TP=31, FN=0, FP=2, TN=75$), baseline architecture comparisons, and cross-dataset replication findings.
- **System Architecture & Methodological Guide:** Comprehensive documentation detailing feature extraction lexicons, normalization pipelines, and mathematical calibration formulas.
- **Multi-Interface Support:** RESTful API powered by FastAPI, modern React + Vite frontend, interactive Streamlit researcher app, and lightweight terminal CLI.

---

## 3. Technology Stack

| Layer | Technologies & Libraries |
| :--- | :--- |
| **Machine Learning Core** | `scikit-learn`, `sentence-transformers` (v2.7.0), `PyTorch`, `joblib`, `NumPy`, `SciPy` |
| **NLP & Linguistic Rules** | `all-MiniLM-L6-v2` (SBERT), `NLTK`, `TextBlob`, custom deterministic regex feature extractors |
| **Backend REST API** | `FastAPI`, `Uvicorn`, `Pydantic v2`, `python-multipart` |
| **Frontend Web Application**| `React 19`, `TypeScript 5`, `Vite 8`, `lucide-react`, Vanilla CSS Design System |
| **Data & Spreadsheet I/O** | `Pandas`, `OpenPyXL` |
| **Testing & Quality Gates** | `pytest` (119 automated unit/integration tests), custom acceptance gate (56 checks) |

---

## 4. Dataset & Splitting Methodology

### Primary Training Corpus: `data/intellireq_clean_dataset_v4.csv`
- **Total Samples:** 715 curated software requirement statements
- **Class Distribution:** 498 Unambiguous (69.65%) / 217 Ambiguous (30.35%)
- **Data Quality:** Zero exact duplicates, zero heuristic label circularity, and verified balance across functional, non-functional, and interface constraints.

### Leakage-Controlled Partitioning Protocol:
To prevent **semantic data leakage** (where paraphrased sentences or clear/ambiguous counterparts of the same requirement concept appear in both train and test splits), IntelliReq enforces `GroupShuffleSplit`:
- **Stratification Unit:** Every requirement is bound to a semantic cluster (`semantic_group_id`). All variants of a requirement concept are strictly quarantined to a single partition.
- **Split Proportions:** 70% Training (500 samples) / 15% Validation (107 samples) / 15% Held-Out Test (108 samples).
- **Independent Test Isolation:** Neither the 80-sample Diagnostic Challenge Set (`challenge_test_set.csv`) nor the 16-sample Adversarial Set (`adversarial_test_set.csv`) was ever exposed during feature extraction, scaler fitting, threshold calibration, or training.

---

## 5. Machine Learning Architecture & 408-Feature Pipeline

```
Raw Requirement Input (String / TXT / XLSX)
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│             Pre-Processing & Input Validator           │
│  (Whitespace normalization, length & 3-word validation) │
└──────────────────────────┬─────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ Channel 1: Dense SBERT  │     │ Channel 2: Domain Rules │
│ all-MiniLM-L6-v2 (384-d)│     │ 24 Handcrafted Features │
└────────────┬────────────┘     └────────────┬────────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│   StandardScaler Normalization (408 Total Features)    │
│    (Fitted exclusively on training set statistics)     │
└────────────────────────────┬───────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│     L1-Regularized Logistic Regression (liblinear)     │
│   (Sparse feature selection across 408 dimensions)     │
└────────────────────────────┬───────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│  Platt Sigmoid Calibration (CalibratedClassifierCV)    │
│    (Maps raw margin distances to true probabilities)   │
└────────────────────────────┬───────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│        Three-Tier Operational Decision Policy          │
│   • P <= 0.25  ──> NOT AMBIGUOUS (Automated Clear)     │
│   • 0.25 < P < 0.75 ──> NEEDS REVIEW (Human Review)   │
│   • P >= 0.75  ──> AMBIGUOUS (Automated Defect)        │
└────────────────────────────┬───────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│             Explanation & Diagnostic Engine            │
│  (Trigger tokens, missing constraints, suggested rewrite)│
└────────────────────────────────────────────────────────┘
```

### Channel Breakdown:
1. **Channel 1 — Dense Semantic Embeddings (384-d):** Pretrained `all-MiniLM-L6-v2` captures contextual semantics, implicit relationships, and semantic sentence structure offline without external API latency.
2. **Channel 2 — 24 Handcrafted Linguistic & Specificity Features:**
   - *Lexical Vagueness:* Presence of vague adverbs, subjective adjectives, weak quantifiers, and underspecified references.
   - *Syntactic Indicators:* Passive voice constructions, multiple conjunction chains, and modal verbs (*should*, *could*, *might*).
   - *Metric Specificity:* Temporal units (*ms, sec, min, hr*), memory/size units (*KB, MB, GB*), numerical bounds, throughput quantities, and standard protocol references.

---

## 6. Empirical Evaluation Results

### Core Benchmark Readings:

| Evaluation Metric | In-Domain Held-Out Test (N=108) | Diagnostic Challenge Suite (N=80) | Adversarial Stress Suite (N=16) |
| :--- | :---: | :---: | :---: |
| **Raw Binary Accuracy** | **98.15%** (106 / 108) | **85.00%** (68 / 80) | **87.50%** (14 / 16) |
| **Selective Accuracy** | **100.00%** (103 / 103) | **92.54%** (62 / 67) | **100.00%** (14 / 14) |
| **Decision Coverage** | **95.37%** (4.63% deferral) | **83.75%** (16.25% deferral) | **87.50%** (12.5% deferral) |
| **Macro-Averaged F1** | **97.78%** | **84.85%** | **86.67%** |
| **Ambiguous Recall** | **100.00%** (31 / 31 detected) | **85.71%** | **100.00%** |
| **Ambiguous Precision**| **93.94%** (31 / 33) | **87.50%** | **85.71%** |
| **Brier Calibration Loss** | **0.0120** | **0.1256** | **0.0636** |

### Held-Out Confusion Matrix ($N = 108$):
$$\begin{array}{c|cc}
& \textbf{Pred Ambiguous} & \textbf{Pred Clear} \\
\hline
\textbf{Actual Ambiguous} & \mathbf{31}\ (TP) & \mathbf{0}\ (FN) \\
\textbf{Actual Clear} & \mathbf{2}\ (FP) & \mathbf{75}\ (TN) \\
\end{array}$$
> **Audit Note:** The 2 false positives fall within the calibrated uncertainty band ($25\% < P < 75\%$) and are routed to human review, ensuring that automated acceptance decisions achieve **100.00% selective accuracy** with zero missed defects.

---

## 7. Limitations & Honest Cross-Domain Generalization

Academic honesty requires distinguishing in-domain performance from cross-dataset generalization. While IntelliReq achieves **98.15% accuracy** on the held-out test distribution, accuracy varies across external benchmark corpora due to legitimate domain shifts:

| External Corpus | Description | Samples (N) | IntelliReq Accuracy | Selective Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **PURE Dataset** | Natural language IEEE software specifications | 6,725 | 79.53% | 93.08% |
| **Agile User Stories** | Connextra template agile requirements | 1,680 | 75.30% | 87.23% |
| **Fault-Prone SRS (Kaggle)**| Multi-defect requirement corpus | 7,061 | 70.12% | N/A |

### Documented Boundary Limitations:
1. **Agile User Story Format Shift:** Requirements formatted strictly as *"As a [user], I want [feature] so that [benefit]"* often lack technical acceptance criteria, triggering uncertainty scores (Needs Review) unless quantified constraints are supplied.
2. **Proprietary Acronyms:** Highly specialized industry abbreviations (e.g. avionics DO-178C or healthcare HL7/DICOM) are parsed based on syntactic structure rather than custom domain vocabularies.
3. **Passive Voice Context:** Passive constructions (*"The audit log shall be preserved"*) are flagged as potentially ambiguous regarding agent responsibility even when the operational action itself is clearly stated.

---

## 8. Installation & Setup Instructions

### Prerequisites
- **Python:** 3.9 or 3.10
- **Node.js:** 18.x or 20.x
- **Git**

### Step 1: Clone the Repository & Configure Python Environment
```bash
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd "Ambiguity detection"

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download required NLTK tokenizers (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

### Step 2: Start the FastAPI Backend Service
```bash
# Run FastAPI on 127.0.0.1:8000
python -m uvicorn backend.main:app --port 8000
```
- Interactive Swagger API Documentation: `http://127.0.0.1:8000/docs`
- Service Health Endpoint: `http://127.0.0.1:8000/api/health`
- Live Model Metadata & Benchmarks: `http://127.0.0.1:8000/api/model-info`

---

### Step 3: Start the React Frontend Application
```bash
# Open a second terminal window:
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```
Open `http://localhost:5173` in your browser. All API requests are automatically proxied to the FastAPI server at `http://127.0.0.1:8000`.

---

### Alternative Interfaces:
- **Terminal Command-Line Interface:**
  ```bash
  python main.py
  ```
- **Streamlit Researcher Interface:**
  ```bash
  streamlit run app.py
  ```

---

## 9. Automated Testing & Verification

IntelliReq enforces strict quality gates before any deployment:

```bash
# 1. Run the complete automated test suite (119 unit & integration tests)
python -m pytest tests/ -v

# 2. Run the 56-check production acceptance gate
python acceptance_check.py

# 3. Verify frontend production TypeScript build
cd frontend
npm run build
```

Expected Output:
```
============================ 119 passed in 16.87s =============================
FINAL ACCEPTANCE REPORT: Passed: 56 | Failed: 0 | Skipped: 0
PROJECT STATUS: VERIFIED FINAL BUILD
✓ built in 446ms
```

---

## 10. User Interface & Screenshots Guide

The user interface follows Microsoft Power BI design principles:

### A. Executive Overview
A clean multi-card executive overview featuring 5 primary KPI tiles (98.15% held-out test accuracy, 100.00% selective decision accuracy, 95.37% coverage, 85.00% challenge generalization, and 100.00% ambiguity recall), followed by a full-width empirical benchmark table across 6 evaluation suites and a 3-tier probability routing visual.

### B. Single Requirement Inspection Workspace
An interactive workbench providing real-time sentence validation, one-click preset test samples, a calibrated posterior probability gauge ($0\% - 100\%$), `<mark>` token highlighting for linguistic triggers, missing measurable criteria alerts, and rule-guided suggested rewrites.

### C. Batch Specification Spreadsheet
A scalable batch processing grid supporting `.xlsx` and `.txt` ingestion, interactive KPI summary cards (Total Processed, Ambiguous, Clear, Needs Review, Skipped), multi-criteria status filters, live search, individual item review drawers, and CSV export.

### D. Academic Model Evaluation Report
A formal viva examination dashboard presenting the in-domain held-out confusion matrix ($TP=31, FN=0, FP=2, TN=75$), baseline model comparisons (TF-IDF vs Hybrid SBERT), external transfer matrices (PURE and User Stories from Zenodo 19746839), and transformer experimental results.

---

## 11. Dataset Provenance, Licenses & Citations

### External Datasets & Replication Packages:
- **Replication Package for Requirements Ambiguity Detection (v2.0.0):**
  - Source: Zenodo record [19746839](https://zenodo.org/records/19746839)
  - License: MIT License
  - Corpora: PURE dataset (6,725 requirements) and Agile User Stories (1,680 requirements)
- **Fault-Prone SRS Dataset:**
  - Source: Kaggle (`corpus4panwo/fault-prone-srs-dataset`) / MDPI Applied Sciences 2023
  - License: Open Research Access

### Citation:
If you build upon or reference this project in academic research, please cite:
```bibtex
@article{intellireq2026,
  title={Enhanced Ambiguity Detection in Software Requirements Using Hybrid Semantic Embeddings and Calibrated Decision Boundaries},
  author={IntelliReq Research Group},
  journal={Computer Science and Software Engineering Repository},
  year={2026}
}
```

---

## 12. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.
