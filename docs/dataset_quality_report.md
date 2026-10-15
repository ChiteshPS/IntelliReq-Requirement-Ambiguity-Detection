# IntelliReq Dataset Quality Report

## Overview
The original project utilized a heavily synthetic 308-row dataset where the target label was deterministically bound to a small rule-based keyword detector. This resulted in data leakage and an artificially inflated test accuracy.

## The V2 Dataset Workflow (`ambiguity_dataset_v2.csv`)
To provide a research-defensible foundation for the new hybrid model, a new 634-row dataset was constructed. 

### Dataset Composition
- **Total Requirements**: 634
- **Ambiguous Requirements**: 422
- **Not Ambiguous Requirements**: 212

### Data Sourcing & Label Generation
- **Original Source Pool**: 610 requirements were sourced from public benchmark datasets (PROMISE, PURE).
- **Regex Pseudo-Labeling**: Initial pseudo-labels were generated using a comprehensive linguistic pattern matcher (looking for vague words, subjective adjectives, missing constraints) balanced against strict specificity rules (presence of numbers, percentages, time units).
- **Controlled Semantic Pairs (24 examples)**: To ensure the model learns semantics and not just structural keyword memorization, we injected controlled adversarial pairs.
  - *Example Pair A*: "The system shall be fast." (Ambiguous)
  - *Example Pair B*: "The system shall return search results within 2 seconds." (Not Ambiguous)
  These pairs force the SBERT embeddings to separate based on the explicit presence of a testable constraint, rather than the general topic of the requirement.

## Challenge Test Set (`challenge_test_set.csv`)
- **Total Requirements**: 100 
  - 40 Ambiguous
  - 40 Not Ambiguous
  - 20 Borderline
- **Purpose**: This dataset serves as a strictly held-out evaluation set. It is NEVER used for training, threshold tuning, or calibration. 
- **Leakage Prevention**: Stratified splitting (`Source_Label`) ensures that controlled pairs and edge-cases are evenly distributed during training without exposing test data to the training loop.
