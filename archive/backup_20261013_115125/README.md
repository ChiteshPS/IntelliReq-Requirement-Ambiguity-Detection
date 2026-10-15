# IntelliReq — Smart Software Requirement Analyzer
**Research Title**: Enhanced Ambiguity Detection in Software Requirements Using Semantic Sentence Embeddings and Hybrid Machine Learning

## 1. Project Overview
IntelliReq is a final-year B.Tech Information Technology project that provides a scientifically defensible machine learning system to detect ambiguous software requirements. It rejects simple keyword-matching in favor of dense semantic analysis combined with strict testability constraints.

## 2. Features
- **Semantic Understanding**: Uses a pre-trained SBERT (`all-MiniLM-L6-v2`) to capture context.
- **Linguistic & Specificity Guardrails**: Extracts explicit constraints (e.g., formats, time units, thresholds) to override false-positive semantic flags.
- **Explanation Engine**: Generates human-readable feedback explaining exactly *why* a requirement is ambiguous and proposes a measurable rewrite.
- **Data Leakage Prevention**: Built on a strict 70/15/15 validation loop with a completely isolated 100-row Challenge Test Set.

## 3. Architecture
USER INPUT -> Input Validation -> [SBERT 384-D Embedding] + [Linguistic Analysis] + [Specificity Analysis] -> Feature Fusion -> StandardScaler -> Calibrated Logistic Regression (L1 Penalty) -> Explanation Engine -> FINAL PREDICTION.

## 4. Installation
```bash
# Require python >= 3.8
pip install -r requirements.txt
```
*(Dependencies: pandas, numpy<2.0, scikit-learn, sentence-transformers==2.7.0)*

## 5. Training
To recreate the model from the raw dataset, run:
```bash
python train_pipeline.py
```
This script splits the data, extracts features, performs an ablation study, and saves the final models to the `saved_models/` directory.

## 6. Single Requirement Analysis
Run the main CLI:
```bash
python main.py
```
Select Option 1. 
**Example Input**: `The system shall respond quickly.`
**Output**: AMBIGUOUS (Reason: "quickly" does not define a measurable response-time limit).

## 7. CSV Analysis
Run the main CLI, and select Option 2.
Provide a path to a CSV (must contain a `Requirement` or `Requirement_Text` column). The system will generate a `batch_results.csv` containing predictions, probabilities, and explanations for every row.

## 8. Evaluation
To run the final held-out evaluation on the strictly isolated 100-row challenge set:
```bash
python evaluate_challenge.py
```
This produces the final `Challenge Set Accuracy` and generates an error analysis CSV.

## 9. Model Comparison
The `evaluation/model_comparison.csv` generated during training details an ablation study comparing TF-IDF, SBERT, Random Forest, and Hybrid architectures.

## 10. Limitations
- **Dataset Bias**: The system relies on a pseudo-labeled and controlled-pair dataset. Ground-truth expert labeling across specific domains (e.g., aerospace, finance) would improve robustness.
- **Context Ignorance**: Ambiguity is determined per-sentence. The model cannot read preceding context.
