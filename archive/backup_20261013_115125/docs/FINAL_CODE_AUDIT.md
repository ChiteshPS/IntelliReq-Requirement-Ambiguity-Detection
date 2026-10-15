# Final Code Audit

## 1. Current Architecture
- SBERT (`all-MiniLM-L6-v2`) providing 384 dimensions.
- Linguistic/Specificity feature extraction providing 24 dimensions.
- Standard scaler.
- Calibrated Classifier CV wrapping a base estimator.

## 2. Current Dataset actually used by train.py
- Currently uses `intellireq_clean_dataset.csv` (391 rows).
- Includes GroupShuffleSplit based on `pair_id`.

## 3. Current feature dimensions
- 408 dimensions total.

## 4. Current model actually selected
- Logistic Regression (L2) wrapped in CalibratedClassifierCV.

## 5. Current calibration implementation
- `CalibratedClassifierCV` (Sigmoid).
- **Flaw:** It is currently fit on the train set and relies on internal cross-validation (`cv=5`), or uses `cv='prefit'` improperly. We must switch to using a completely isolated validation set for calibration to avoid overconfident predictions on unseen data.

## 6. Current threshold implementation
- >=0.75 Ambiguous.
- <=0.25 Not Ambiguous.
- Hardcoded in evaluation but metadata supports `upper_threshold` and `lower_threshold`.

## 7. Current inference path
- Text -> FeatureExtractor (24 dims) -> SBERTExtractor (384 dims) -> Scaler -> LogisticRegression (Predict Proba) -> Threshold logic.

## 8. Current evaluation path
- `evaluate.py` does strict binary scoring, penalizing "NEEDS REVIEW". Needs to be decoupled into `Selective Accuracy` and `Coverage`.

## 9. Current tests
- `pytest` covers 5 basic tests, needs expansion to 25 distinct verification cases.

## 10. Current bugs/inconsistencies
- Tautological shortcuts in synthetic data (length/units).
- Calibration overlaps with training.
- Explanation engine sometimes overrides ML prediction or confuses itself with hard rules.
- Main.py CLI crashes if column headers in CSVs aren't strictly `Requirement_Text`.

## 11. Stale documentation/metrics
- Metrics claiming 100% or 97% are stale or based on dataset shortcuts.

## 12. Files that must be replaced
- `generate_dataset.py` (needs to become `generate_dataset_v4.py`).
- `train.py` (needs correct validation calibration).
- `evaluate.py` (needs to become `final_evaluation.py`).
- `main.py` (needs robust handling).

## 13. Files that must remain backward compatible
- SBERT extractor and models structure.
