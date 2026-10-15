# Data Leakage Audit

## 1. Initial Findings
Upon conducting the Phase 3 audit, it was discovered that **8 records** from the `challenge_test_set.csv` had previously leaked into the training data generation pipeline (`generate_dataset.py`). 

### Examples of leaked data:
1. "The system shall calculate the tax rate based on the user's billing ZIP code."
2. "The system shall generate a monthly PDF report on the 1st of every month."
3. "The web application shall pass W3C HTML5 validation without errors."

These leaked records artificially inflated the performance of previous models when evaluated against the challenge set.

Additionally, the original dataset contained exactly duplicated templates (normalized) that were scattered indiscriminately across Train and Test splits before `GroupShuffleSplit` was introduced, leading to "Pair Leakage".

## 2. Leakage Quarantine Process
1. **Rebuilt Generation Script**: The dataset generator was completely overhauled. It now explicitly loads the `challenge_test_set.csv` and cross-references all newly generated requirements against the challenge data. Any overlapping string is automatically discarded.
2. **Normalized Deduplication**: Generated datasets are deduplicated dynamically to prevent exact string matches.
3. **Adversarial Set Segregation**: We created an explicit `adversarial_test_set.csv` manually. It is entirely partitioned from the training process.
4. **Group-Aware Pair Hashing**: A `PAIR-XXXXXX` ID was assigned to every ambiguous/clear generation pair, and `GroupShuffleSplit` strictly enforces that paired rows stay exclusively in the same dataset partition (Train, Val, or Test).

## 3. Automated Quality Gate
We implemented a strict leakage verification gate inside `train.py`. The training process actively cross-references the training dataframe against both the Challenge and Adversarial datasets.
- **Trigger**: If even one challenge/adversarial text appears in the training data, the pipeline prints `LEAKAGE FATAL ERROR` and triggers `sys.exit(1)`.
- **Status**: The pipeline currently reports **Leakage check PASS: No external evaluation data leaked into training.**
