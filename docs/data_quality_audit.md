# IntelliReq — Comprehensive Data Quality and Leakage Audit

**Audit Date:** 2026-10-13  
**Auditor:** Senior ML Engineer & Research Auditor  
**Primary Dataset:** `data/intellireq_clean_dataset_v4.csv`  
**External Benchmarks:** `data/challenge_test_set.csv` (N=80), `data/adversarial_test_set.csv` (N=16)

---

## 1. Executive Summary

| Check Category | Target Criteria | Audit Result | Status |
|---|---|---|---|
| **Exact Duplicates** | 0 exact duplicate text rows | 0 duplicates found (715/715 unique) | **PASS** |
| **Missing Values** | 0 null/NaN entries in critical fields | 0 nulls across all 11 columns | **PASS** |
| **Contradictory Labels** | 0 identical texts with differing labels | 0 contradictions found | **PASS** |
| **Train $\leftrightarrow$ Challenge Leakage** | 0 overlapping requirement sentences | 0 overlapping rows (0.00%) | **PASS** |
| **Train $\leftrightarrow$ Adversarial Leakage** | 0 overlapping requirement sentences | 0 overlapping rows (0.00%) | **PASS** |
| **Challenge $\leftrightarrow$ Adversarial Overlap**| 0 overlapping benchmark sentences | 0 overlapping rows (0.00%) | **PASS** |
| **Splitting Isolation** | Group-aware isolation of templates | `GroupShuffleSplit` on `semantic_group_id` | **PASS** |

---

## 2. Dataset Composition & Source Breakdown

`data/intellireq_clean_dataset_v4.csv` comprises **715 verified requirements**:

| Source Segment | Sample Count | Class Breakdown | Role in Mitigation |
|---|---|---|---|
| `v4_hard_clear` | 287 | Not Ambiguous: 287 | Non-ambiguous requirements with explicit numeric and functional criteria |
| `software_requirements_extended` | 265 | Ambiguous: 70, Not Ambiguous: 195 | Curated real-world industrial software requirements |
| `v4_hard_amb` | 147 | Ambiguous: 147 | Hard lexical ambiguity examples (subtle vagueness, unquantified bounds) |
| `v4_len_break` | 16 | Not Ambiguous: 16 | Length-balancing requirements preventing sentence length shortcuts |
| **Total Canonical** | **715** | **Ambiguous: 217 (30.3%), Not Ambiguous: 498 (69.7%)** | Complete canonical training corpus |

---

## 3. Split Partitioning Strategy & Manifest

To ensure zero leakage between training, validation, and held-out test sets:
- **Algorithm:** Scikit-learn `GroupShuffleSplit(random_state=42)` partitioned on `semantic_group_id`.
- **Isolation Principle:** Any related syntactic paraphrases or paired variations share the same group ID and are locked into the same split partition.
- **Partition Distribution:**
  - **Training Set (70%):** 500 rows ($N_{\text{amb}}=153$, $N_{\text{not}}=347$).
  - **Validation Set (15%):** 107 rows ($N_{\text{amb}}=33$, $N_{\text{not}}=74$) — Used solely for model comparison and Platt probability calibration.
  - **Held-Out Test Set (15%):** 108 rows ($N_{\text{amb}}=31$, $N_{\text{not}}=77$) — Frozen until final evaluation.

---

## 4. Benchmark Dataset Independence

1. **Independent Challenge Set (`data/challenge_test_set.csv`, N=80):**
   - Composed of 40 Ambiguous and 40 Not Ambiguous industrial requirements sourced from external projects (medical devices, avionics, defense, and fintech).
   - Confirmed zero lexical or semantic overlap with the training set.

2. **Adversarial Test Set (`data/adversarial_test_set.csv`, N=16):**
   - Composed of 8 Ambiguous and 8 Not Ambiguous syntactically deceptive sentences.
   - Designed to challenge models relying on simplistic keyword heuristics (e.g., numbers that do not constitute testable criteria).
   - Confirmed zero overlap with the training set.

---

## 5. Preprocessing & Leakage Boundary Rules

1. `StandardScaler` is fitted **exclusively on the training split** (`X_train`) and applied as a transform to validation and test splits.
2. Feature selection and model hyperparameter selection are conducted on validation data only.
3. Probability calibration (`CalibratedClassifierCV`) is fitted **strictly on the validation split** (`X_val`).
4. Final held-out test, challenge, and adversarial splits remain completely unobserved during all training and tuning phases.
