# Dataset Shortcut Audit

## 1. Internal Dataset Statistics
An analysis of the 391-row clean synthetic dataset reveals massive structural differences between the "Ambiguous" and "Not Ambiguous" classes, creating extreme lexical shortcuts that bypass the need for semantic understanding.

### Length Shortcuts
- **Ambiguous Average Length**: 7.2 words
- **Not Ambiguous Average Length**: 14.8 words
*(The model can achieve high accuracy simply by predicting long sentences as Not Ambiguous.)*

### Numeric & Unit Shortcuts
- **Contains Numerics**: 80.1% (Ambiguous) vs 98.5% (Not Ambiguous)
- **Contains Units**: 0.0% (Ambiguous) vs 61.4% (Not Ambiguous)
- **Contains Percentages**: 0.0% (Ambiguous) vs 12.6% (Not Ambiguous)
*(The synthetic dataset generation completely failed to include clear requirements without units, or ambiguous requirements with misleading units. Any requirement with a unit or percentage is 100% guaranteed to be Not Ambiguous in this dataset.)*

### Vague Lexicon Shortcuts
- **Contains Vague Words**: 42.6% (Ambiguous) vs 0.0% (Not Ambiguous)
*(The dataset contains zero negative examples of clear requirements that use vague-looking words like "efficient" in a strictly scoped context.)*

## 2. Baseline Model Performance
To prove that these structural flaws allow trivial classification, we evaluated four baselines on the same internal dataset split:

1. **Majority Class**: 50.00%
2. **Simple Numeric Rule** (If contains a number -> Not Ambiguous): **66.67%**
3. **Simple Vague Rule** (If contains vague term -> Ambiguous): **70.00%**
4. **TF-IDF + Logistic Regression**: **98.33%**

## 3. Conclusion
The internal 100% test accuracy is an illusion. 
Because a naive TF-IDF model (which ignores word order and semantic meaning completely) achieves 98.33%, the internal dataset is trivially solvable using surface-level vocabulary (e.g., presence of "seconds" or "percent") and sentence length. The 408-dimensional SBERT + Linguistic model simply memorized these trivial structural boundaries instead of learning what ambiguity actually means, leading to the severe 66.25% generalization drop on the challenge set where these shortcuts do not perfectly align.
