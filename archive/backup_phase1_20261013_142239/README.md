# IntelliReq — Smart Software Requirement Analyzer

**Research Title:** Enhanced Ambiguity Detection in Software Requirements Using Semantic Sentence Embeddings and Hybrid Machine Learning

---

## Overview

IntelliReq is a machine-learning–based command-line tool that analyses software requirements for ambiguity. It classifies each requirement as:

| Output | Meaning |
|---|---|
| `AMBIGUOUS` | Ambiguity probability ≥ **0.75** |
| `NOT AMBIGUOUS` | Ambiguity probability ≤ **0.25** |
| `NEEDS REVIEW` | Probability between 0.25 and 0.75 — defer to a human reviewer |

---

## Architecture

```
User Input
    │
    ▼
[Input Parser] ─── Single sentence / TXT / XLSX only
    │
    ▼
[FeatureExtractor] ─── 24 handcrafted linguistic + specificity features
[SBERTExtractor]   ─── 384-dim pretrained all-MiniLM-L6-v2 embeddings
    │
    ▼
[StandardScaler]   ─── fitted on training data only (408 features total)
    │
    ▼
[CalibratedClassifierCV] ─── Logistic Regression L1, sigmoid calibration
    │
    ▼
[ModelPredictor]   ─── threshold policy (0.75 / 0.25)
    │
    ▼
[ExplanationEngine] ─── supporting evidence (never overrides classifier)
    │
    ▼
Terminal output / XLSX results file
```

---

## Accepted Input Formats

The CLI accepts **three input methods only**:

1. A **single requirement sentence** typed at the terminal prompt
2. A **`.txt` file** — one requirement per non-empty line
3. An **`.xlsx` file** — must contain a column named `Requirement`, `Requirement_Text`, or `requirement`

> **PDF, DOCX, CSV and all other formats are not supported.**

---

## Setup (Windows PowerShell)

```powershell
# Python 3.10 or later required
python --version

# Install dependencies
pip install -r requirements.txt

# Download NLTK tokenizer data (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Run the Application

```powershell
python main.py
```

You will see:

```
======================================================================
   INTELLIREQ — SMART SOFTWARE REQUIREMENT ANALYZER
======================================================================

  1. Analyze one requirement sentence
  2. Analyze requirements from a TXT file
  3. Analyze requirements from an XLSX file
  4. Exit
```

Batch results are saved to the `outputs/` directory as XLSX files.

---

## Train the Model

```powershell
python train.py
```

This will:
1. Load `data/intellireq_clean_dataset_v4.csv` (715 rows, group-aware split 70/15/15)
2. Check for leakage against the challenge and adversarial sets
3. Extract SBERT + linguistic/specificity features (408 total)
4. Compare Logistic Regression L1, L2, and Random Forest on the validation set
5. Calibrate the best model using sigmoid/Platt scaling on the held-out validation set
6. Save `saved_models/model.joblib`, `scaler.joblib`, and `metadata.json`

---

## Run Tests

```powershell
python -m pytest tests/test_intellireq.py -v
```

Tests cover: input validation, feature dimensions and order, TXT/XLSX parsing,
unsupported format rejection, threshold boundary conditions, explanation engine
isolation, dataset leakage, end-to-end regression, batch order, duplicates,
unicode, and model metadata consistency.

---

## Evaluate the Model

```powershell
# Evaluate on held-out Challenge Set and Adversarial Set
python evaluate.py
```

Reports are written to `evaluation/`.

---

## Run Acceptance Gate

```powershell
python acceptance_check.py
```

Executes all 14 mandatory acceptance checks and prints:

```
PROJECT STATUS: VERIFIED FINAL BUILD
```
or
```
PROJECT STATUS: NOT READY — SEE FAILED CHECKS
```

---

## Dataset

| File | Rows | Purpose |
|---|---|---|
| `data/intellireq_clean_dataset_v4.csv` | 715 | Training / validation / internal test |
| `data/challenge_test_set.csv` | 100 | Independent held-out challenge set |
| `data/adversarial_test_set.csv` | 16 | Independent adversarial set |

Training labels: `Ambiguous` (217) and `Not Ambiguous` (498).
Group-aware `GroupShuffleSplit` ensures paired requirements stay in the same split.

---

## Key Model Metrics

| Dataset | Accuracy | Notes |
|---|---|---|
| Internal validation (hybrid) | ~100% | Controlled pairs, in-distribution |
| Challenge Set | ~77.5% | Independent, unseen data |
| Adversarial Set | ~68.75% | Deliberately tricky examples |

The **challenge set accuracy is the honest generalization metric**. Internal accuracy is
expected to be high because the training data uses controlled pairs.

---

## Thresholds

| Condition | Decision |
|---|---|
| Ambiguity probability ≥ **0.75** | `AMBIGUOUS` |
| Ambiguity probability ≤ **0.25** | `NOT AMBIGUOUS` |
| Between 0.25 and 0.75 | `NEEDS REVIEW` |

Thresholds were set based on calibration diagnostics on the validation set.
They were **not** tuned against the challenge or adversarial sets.

---

## Limitations

- The model is trained on a controlled synthetic dataset. Domain-specific acronyms or novel
  phrasings may not be well-represented.
- Ambiguity is assessed per sentence; the model cannot consider preceding context.
- The challenge set accuracy (~77.5%) reflects genuine out-of-distribution difficulty.
- No public, human-annotated ambiguity-only software requirements dataset was available;
  the dataset uses controlled pairs with documented generation methodology.

---

## File Structure

```
main.py                         ← Terminal CLI (single sentence / TXT / XLSX)
train.py                        ← Reproducible training pipeline
evaluate.py                     ← Evaluation on held-out sets
acceptance_check.py             ← Automated acceptance gate (14 checks)
requirements.txt                ← Python dependencies

feature_engineering/
    feature_extractor.py        ← 24 handcrafted features
    sbert_extractor.py          ← 384-dim SBERT embeddings

models/
    predictor.py                ← Threshold-based prediction

preprocessing/
    input_parser.py             ← TXT and XLSX parsing only

rule_engine/
    explanation_engine.py       ← Evidence and rewrites (no prediction override)

data/
    intellireq_clean_dataset_v4.csv
    challenge_test_set.csv
    adversarial_test_set.csv

saved_models/
    model.joblib
    scaler.joblib
    metadata.json

tests/
    test_intellireq.py          ← Comprehensive test suite

docs/
    final_model_report.md
    annotation_guidelines.md

outputs/                        ← Generated batch XLSX results
evaluation/                     ← Metrics, plots, error analysis
archive/                        ← Timestamped backups of previous versions
```
