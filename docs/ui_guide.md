# IntelliReq — User Interface & Interactive Workspace Guide

## 1. Overview
The IntelliReq User Interface is a research-grade web application built in Streamlit. It provides an intuitive, accessible environment for software engineers and requirements analysts to audit requirement statements, batch-process documents, inspect empirical model evaluations, and analyze error cases.

---

## 2. Launching the Web Application in Windows PowerShell

Run the following command from the project root (`D:\Ambiguity detection`):

```powershell
streamlit run app.py
```

The application will launch locally at:
`http://localhost:8501`

---

## 3. Navigation & Workspace Breakdown

### Page 1: 📊 Dashboard
- **Telemetry & Session Activity:** Tracks requirements audited during the active session with counts of `Ambiguous`, `Not Ambiguous`, and `NEEDS REVIEW`. Shows an elegant empty state when no audits have been performed.
- **Recent Analysis History:** An interactive table showing the requirement statement, model decision, probability, and review flag.
- **Architectural Pipeline Workflow:** Visual step-by-step flowchart of the 5-stage processing pipeline.
- **Model Metadata:** Quick inspection of model version (v2.1.0), embedding dimension (384 SBERT + 24 Ling = 408 total), classifier (L1 Logistic Regression), and Platt calibration thresholds.

### Page 2: 🔍 Single Requirement Analysis
- **Text Area:** Large input box with real-time word and character count indicators (minimum 3 words required).
- **Preset Quick Starts:** Four clickable sample presets:
  - Vague Quality (🔴 Ambiguous)
  - Measurable SLA (🟢 Not Ambiguous)
  - Borderline Mechanism (🟡 Needs Review)
  - Vague Temporal (⏱️ Ambiguous)
- **Calibrated Posterior Probability Gauge:** Visual meter displaying ambiguity likelihood with color-coded guardrails.
- **Interactive Token Highlighter:** Real-time visual badge highlighting of detected vague words, fuzzy modifiers, and unquantified adjectives.
- **Rule Engine Diagnosis & Checklist:** Clear explanation of missing criteria (e.g., missing numerical latency target, test units).
- **Suggested Rule-Based Rewrite:** Heuristic structural suggestions without fabricating business constraints.
- **Report Export & Actions:**
  - Copy analytical summary to clipboard via code block.
  - "Mark for Manual Review" button to route borderline requirements to the senior engineer review queue.

### Page 3: 📁 Batch File Analysis
- **Supported Formats:**
  - `.txt`: One requirement sentence per non-empty line.
  - `.xlsx`: Excel workbook containing a requirement text column.
  - *CSV, PDF, and DOCX are explicitly rejected with informative error dialogs.*
- **Multi-Sheet & Column Selector:** For Excel workbooks, dynamically selects sheets and requirement columns with a 5-row live preview.
- **Progress Tracking:** Progress bar indicates batch inference status.
- **Interactive Results Table:** Filter by decision (`Ambiguous`, `Not Ambiguous`, `NEEDS REVIEW`, `SKIPPED`).
- **Download Excel Report:** One-click download of `.xlsx` workbook containing both a Summary metrics sheet and a complete row-by-row Results sheet.

### Page 4: 📈 Model Evaluation
- Reads real generated evaluation artifacts directly from disk (`evaluation/` directory).
- Tab 1: **Benchmark Comparison** (TF-IDF vs SBERT-only vs Linguistic-only vs IntelliReq Hybrid).
- Tab 2: **Held-Out Clean Test Set** (N=108, Confusion Matrix, Classification Report).
- Tab 3: **Independent Challenge Benchmark** (N=80, Zero-leakage out-of-domain evaluation).
- Tab 4: **Adversarial Robustness Set** (N=16, Syntactic traps).
- Tab 5: **Feature Weights & Dataset Distributions**.

### Page 5: 🔬 Dataset & Error Analysis
- **Dataset Partition Statistics:** Clean v4 (715), Train (500), Val (107), Test (108), Challenge (80), Adversarial (16).
- **Quality Audit Badges:** Zero duplicate rows, zero cross-split leakage, zero contradictory labels.
- **Interactive Failure Taxonomy:** Filterable table of 38 error cases from `evaluation/error_analysis.csv` (False Positives, False Negatives, Underconfident Review cases, Missed Borderlines) with qualitative takeaways.

### Page 6: 📖 About & Methodology
- Detailed explanation of requirement ambiguity types.
- Mathematical description of Platt probability calibration.
- Base paper reproduction and generalization gap analysis.
- Operational boundaries and offline guarantees.
