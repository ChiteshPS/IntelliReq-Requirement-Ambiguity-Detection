# Dataset v3 Shortcut Audit

## 1. Overview
- Total Rows: 559
- Exact Duplicates: 0
- Normalized Duplicates: 0
- Challenge Overlap: 0
- Adversarial Overlap: 0

## 2. Class Balance & Structural Stats
### Ambiguous
- count: 241.00
- len_mean: 11.29
- len_median: 10.00
- len_min: 5.00
- len_max: 32.00
- num%: 76.76
- unit%: 60.17
- pct%: 26.97
- thresh%: 0.00
- vague%: 56.85
- subj%: 6.64
- tech%: 41.49

### Not Ambiguous
- count: 318.00
- len_mean: 11.36
- len_median: 11.00
- len_min: 5.00
- len_max: 16.00
- num%: 94.97
- unit%: 74.84
- pct%: 41.82
- thresh%: 27.67
- vague%: 5.03
- subj%: 0.00
- tech%: 38.36

## 3. Diagnostic Baseline Classifiers
Evaluated on group-aware internal split to detect remaining shortcuts:
- Majority: 56.14%
- Length-only: 74.85%
- Number/Unit/Pct-only: 65.50%
- Vague-keyword-only: 65.50%
- TF-IDF: 99.42%
