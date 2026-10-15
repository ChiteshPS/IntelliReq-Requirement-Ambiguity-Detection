# IntelliReq — Model Evaluation Report

## 1. Overview
This document contains the verified experimental evaluation of IntelliReq (v2.1.0) across three independent evaluation benchmarks:
1. **Held-Out Clean Test Set (N=108):** In-domain held-out test partition isolated using group-aware stratified sampling.
2. **Independent Challenge Test Set (N=80):** Out-of-domain industrial software requirements gathered from diverse real-world systems (defense, fintech, medical devices, logistics).
3. **Adversarial Benchmark (N=16):** Syntactically tricky requirements featuring lexical distractors and misleading numeric tokens.

---

## 2. Quantitative Performance Metrics

### A. Comprehensive Benchmark Summary
| Metric | Held-Out Clean Test (N=108) | Independent Challenge (N=80) | Adversarial Robustness (N=16) |
| :--- | :---: | :---: | :---: |
| **Raw Binary Accuracy** | **98.15%** | **85.00%** | **87.50%** |
| **Strict 3-Way Accuracy** | 95.37% | 77.50% | 68.75% |
| **Selective Non-Deferred Accuracy** | **98.11%** | **92.54%** | **84.62%** |
| **Coverage (Non-Deferred Rate)** | 98.15% | 83.75% | 81.25% |
| **Deferral / Review Rate** | 1.85% | 16.25% | 18.75% |
| **Precision (Ambiguous)** | 93.94% | 80.43% | 87.50% |
| **Recall (Ambiguous)** | 100.00% | 92.50% | 87.50% |
| **Macro-F1 Score** | 97.78% | 84.76% | 87.50% |
| **Brier Calibration Loss** | **0.0120** | **0.1023** | **0.0636** |

---

## 3. Confusion Matrices

### Held-Out Clean Test Set (N=108)
```
                  Predicted Ambiguous    Predicted Not Ambiguous
Actual Ambiguous          31                        0
Actual Not Ambiguous       2                       75
```
- **True Positives (Ambiguous):** 31
- **False Negatives:** 0 (100% Recall on ambiguous requirements)
- **False Positives:** 2
- **True Negatives:** 75

### Independent Challenge Benchmark (N=80)
```
                  Predicted Ambiguous    Predicted Not Ambiguous
Actual Ambiguous          37                        3
Actual Not Ambiguous       9                       31
```
- **True Positives (Ambiguous):** 37
- **False Negatives:** 3 (92.5% Recall on unseen out-of-domain ambiguity)
- **False Positives:** 9 (11.25% Overflagging of technical architecture terms)
- **True Negatives:** 31

### Adversarial Benchmark (N=16)
```
                  Predicted Ambiguous    Predicted Not Ambiguous
Actual Ambiguous           7                        1
Actual Not Ambiguous       1                        7
```
- **True Positives:** 7
- **False Negatives:** 1
- **False Positives:** 1
- **True Negatives:** 7

---

## 4. Feature Ablation Analysis

| Architecture | Total Features | Held-Out Acc | Challenge Acc | Adversarial Acc | Challenge Amb. F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LR** | 1,000 | 99.07% | 50.00% | 50.00% | 4.76% |
| **SBERT Only + LR** | 384 | 93.52% | 75.00% | 87.50% | 75.61% |
| **Linguistic Only + LR** | 24 | 92.59% | 73.75% | 68.75% | 71.23% |
| **IntelliReq Hybrid** | **408** | **98.15%** | **85.00%** | **87.50%** | **86.05%** |

### Findings:
1. Pure SBERT embeddings capture semantic similarity but miss structural units (e.g. `latency under 100 ms`).
2. Pure linguistic features detect explicit regex triggers but fail on synonymous paraphrases.
3. Combining SBERT (384D) + Linguistic (24D) yields the highest out-of-domain transferability (85.00%), a **+10% to +11.25% gain** over individual modalities.

---

## 5. Probability Calibration & Three-Tier Decision Policy
Raw logistic regression logits are mapped to true posterior probabilities using Platt scaling:
$$P(\text{Ambiguous} \mid x) = \frac{1}{1 + \exp(-(A \cdot f(x) + B))}$$

Decision policy:
- $P \ge 0.75$: Flagged as **`AMBIGUOUS`**
- $P \le 0.25$: Certified as **`NOT AMBIGUOUS`**
- $0.25 < P < 0.75$: Deferred as **`NEEDS REVIEW`**

The low Brier score (0.0120 on held-out, 0.1023 on challenge) confirms reliable uncertainty calibration.
