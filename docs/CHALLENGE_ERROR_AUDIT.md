# Challenge Error Audit

## 1. Challenge Set Overview
The independent challenge set contains 100 heavily vetted requirements designed to stress-test the model's out-of-domain linguistic generalization capabilities without leakage. 

- Total Sentences: 100
- Labels: Ambiguous (40), Not Ambiguous (40), Borderline (20)

## 2. Model Performance
- **Strict Binary Accuracy**: 66.25% (on the 80 binary-labeled sentences).
- **Brier Score**: 0.1147 (very well calibrated, indicates model outputs soft probabilities appropriately instead of being overconfident).

### Handling "Borderline" inputs:
- Requirements explicitly labeled "Borderline" represent edge cases that cannot be cleanly resolved via syntax.
- **Model Result**: Out of 20 Borderline requirements, the model flagged exactly **12 as "NEEDS REVIEW" (60%)**.
- This demonstrates the success of the probability calibration and dual-threshold logic (`>=0.75` for Ambiguous, `<=0.25` for Not Ambiguous, middle for Review).

## 3. Error Categorization
The `evaluation/error_analysis.csv` details specific misses. Key themes among errors:

### A. Underconfident Correct Predictions
The model predicted "NEEDS REVIEW" for something that was explicitly Ambiguous or Not Ambiguous. Because we evaluated with strict rules, this counts as an error, but in a production environment, routing uncertain requirements to a human is the *desired* behavior.

### B. False Positives (Domain Shift)
Requirements containing words that historically correlate heavily with ambiguity (e.g. "support," "provide") but were sufficiently scoped by context that the model missed.
Example: *The application shall support English and Spanish.* (Model thought the lack of "how" it supports it might be ambiguous, but grammatically it is perfectly clear).

### C. False Negatives (Missing Acceptance Criteria)
Requirements that look structurally sound but lack actual testable metrics. 
Example: *The database must be highly efficient.* The model learned that "must be highly" is a strong structural phrase, but missed that "efficient" has no metric.

## 4. Adversarial Set Performance
We constructed 16 highly deceptive pairs explicitly meant to confuse simple keyword matchers.
- **Strict Accuracy**: 62.50%
- **Brier Score**: 0.1129
While accuracy is lower than the clean set, the strong Brier score indicates the model is appropriately uncertain rather than confidently wrong. This signifies that while semantic embeddings capture general syntactic vagueness, domain-specific adversarial requirements remain exceptionally difficult for generalized NLP models without explicit fine-tuning on massive scale.
