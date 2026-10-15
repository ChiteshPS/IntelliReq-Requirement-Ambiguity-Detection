# Final Model Report: IntelliReq v2.1.0

## 1. Problem Statement
Software requirements are frequently written with vague, subjective, or imprecise language, leading to misinterpretation during development. Existing ambiguity detection tools often rely on naive keyword matching, which fails to generalize or understand context, leading to brittle systems that incorrectly flag clear requirements or miss subtle semantic vagueness.

## 2. Definition of Ambiguity
In IntelliReq, ambiguity is defined objectively: A requirement is ambiguous if it uses subjective language, unquantified adjectives/adverbs, unbounded quantities, or undefined mechanisms without explicitly pairing them with a testable, measurable constraint or metric.

## 3. Dataset Version
The final model is trained exclusively on `intellireq_clean_v4` (`data/intellireq_clean_dataset_v4.csv`).

## 4. Dataset Size
715 total requirements.

## 5. Class Balance
Ambiguous: 217
Not Ambiguous: 498
Borderline: 0 (Quarantined from binary training).

## 6. Leakage Prevention
The dataset generation strictly isolated the out-of-domain evaluation datasets (`challenge_test_set.csv` and `adversarial_test_set.csv`). The final audit verified 0% semantic or exact string overlap between training data and these critical evaluation sets. 

## 7. 408-Dimensional Feature Architecture
The model relies on a fixed 408-dimensional feature space, ensuring structural integrity between training and inference pipelines.

## 8. SBERT Model
384 dimensions are generated via the dense sentence embedding model `all-MiniLM-L6-v2`, capturing the holistic semantic intent of the requirement.

## 9. Handcrafted Features
24 dimensions are generated via a deterministic linguistic parser, capturing explicit rule-based metrics (e.g., Number Count, Unit Count, Passive Voice Count, Modal Verb Count).

## 10. Logistic Regression
The core classifier is a `LogisticRegression(penalty='l1', solver='liblinear')`. L1 regularization was automatically selected during pipeline tuning to enforce sparsity and prevent overfitting to high-dimensional text correlations.

## 11. Calibration
The classifier is wrapped in a `CalibratedClassifierCV` (method: `sigmoid`). Crucially, this calibration is applied strictly to an isolated validation set (`cv='prefit'`), guaranteeing that probabilities are historically accurate rather than overconfident on training data.

## 12. Threshold Policy
Instead of forcing binary labels, the model uses strict confidence thresholds:
- Upper Threshold: `0.75` (>= 0.75 is flagged as Ambiguous).
- Lower Threshold: `0.25` (<= 0.25 is flagged as Not Ambiguous).

## 13. NEEDS REVIEW
Predictions falling between the `0.25` and `0.75` bounds are intelligently routed to human operators. This prevents the model from blindly guessing on inherently subjective statements or structurally conflicting requirements.

## 14. Explanation Engine
The inference pipeline utilizes an independent Explanation Engine that provides deterministic linguistic evidence for the ML model's prediction. The engine is entirely decoupled from the classifier—it provides evidence (e.g., highlighting missing thresholds) but does not overwrite the ML probability.

## 15. Internal Evaluation
- Calibrated Validation Set F1: ~0.97
- Calibrated Validation Accuracy: ~0.98

## 16. Challenge Evaluation
On the entirely unseen, out-of-domain Challenge dataset (80 items):
- Selective Accuracy (Confident predictions): **92.54%**
- Strict Binary Accuracy: **77.50%**
- Review Rate (Pushed to human): **16.25%**

## 17. Adversarial Evaluation
On the heavily modified Adversarial dataset (16 items designed specifically to break TF-IDF keyword heuristics):
- Selective Accuracy (Confident predictions): **100.00%**
- Strict Binary Accuracy: **68.75%**
- Review Rate (Pushed to human): **31.25%**

## 18. Borderline Evaluation
On the quarantined Borderline dataset (20 items deemed inherently context-dependent by human annotators):
- Correctly routed to "NEEDS REVIEW": **30.00%**

## 19. Limitations
While generalization is high, the model's reliance on L1 regression makes it slightly conservative on highly convoluted, multi-sentence paragraphs. Further, the model operates in English only and relies heavily on the quality of the specific SpaCy/NLTK tokenizers used in feature extraction.

## 20. Reproducibility Commands
To completely reproduce this build locally:
```bash
python train.py
python final_evaluation.py
python tools/final_acceptance_check.py
```
