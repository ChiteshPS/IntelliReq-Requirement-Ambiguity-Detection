# Feature Pipeline Audit

## 1. Feature Dimension Verification
The serialized model (`saved_models/model.joblib`) combined with the `StandardScaler` requires exactly **408 input features**.

This confirms that the model architecture actively utilizes:
- **384 features** from the `all-MiniLM-L6-v2` SBERT embeddings.
- **24 handcrafted features** from the Linguistic and Specificity extraction logic (which rely on JSON configurations like `ambiguity_lexicon.json` and `specificity_lexicon.json`).

## 2. Train vs. Inference Consistency
The pipeline was traced across both training and inference.

**Training Path (`train.py`)**:
1. Requirements are separated by `GroupShuffleSplit`.
2. Handcrafted features extracted via `FeatureExtractor.process_dataframe()`.
3. SBERT features extracted via `SBERTExtractor.transform()`.
4. The two dataframes are concatenated into a single 408-dimension dataframe.
5. `StandardScaler` is fitted and applied.
6. `LogisticRegression(penalty='l2')` is trained, followed by `CalibratedClassifierCV`.

**Inference Path (`main.py` & `evaluate.py`)**:
1. Requirement text provided.
2. Handcrafted features extracted via `FeatureExtractor`.
3. SBERT features extracted via `SBERTExtractor`.
4. Concatenated to 408 dimensions.
5. Loaded `StandardScaler` transforms the features.
6. Loaded `CalibratedClassifierCV` predicts probabilities.

**Conclusion**: The feature construction path is perfectly mirrored between training and inference. The 408-dimensional pipeline is consistent. No mismatched feature generation occurs.
