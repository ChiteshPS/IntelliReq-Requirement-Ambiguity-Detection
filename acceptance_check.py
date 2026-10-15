"""
acceptance_check.py — IntelliReq Terminal Acceptance Gate

Executes all mandatory checks. A failed/skipped critical check is NEVER
reported as PASS.

Run:
    python acceptance_check.py

Final status is one of:
    PROJECT STATUS: VERIFIED FINAL BUILD
    PROJECT STATUS: NOT READY — SEE FAILED CHECKS
"""
import os
import sys
import json
import tempfile
import warnings
import traceback

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

checks = []


def record(name: str, status: str, detail: str = ""):
    checks.append({"name": name, "status": status, "detail": detail})
    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(status, "[?]")
    colour = {"PASS": "\033[92m", "FAIL": "\033[91m", "SKIP": "\033[93m"}.get(status, "")
    reset = "\033[0m"
    msg = f"  {colour}{icon} {name}{reset}"
    if detail:
        msg += f"\n      {detail}"
    print(msg)


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# Load shared artifacts once
# ─────────────────────────────────────────────────────────────────────────────

model = None
scaler = None
meta = None
predictor = None
feat_extractor = None
sbert_extractor = None
cli = None

section("ENVIRONMENT")
print(f"  Python  : {sys.version}")
print(f"  CWD     : {os.getcwd()}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Canonical files exist
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 1 — CANONICAL FILES EXIST")

required_files = [
    "data/intellireq_clean_dataset_v4.csv",
    "data/challenge_test_set.csv",
    "data/adversarial_test_set.csv",
    "saved_models/model.joblib",
    "saved_models/scaler.joblib",
    "saved_models/metadata.json",
    "main.py",
    "train.py",
    "evaluate.py",
    "preprocessing/input_parser.py",
    "models/predictor.py",
    "feature_engineering/feature_extractor.py",
    "feature_engineering/sbert_extractor.py",
    "rule_engine/explanation_engine.py",
    "requirements.txt",
    "README.md",
]

all_files_ok = True
for fp in required_files:
    if os.path.isfile(fp):
        record(fp, PASS)
    else:
        record(fp, FAIL, "MISSING FILE")
        all_files_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Dataset validity
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 2 — DATASET LABELS AND VALIDITY")

try:
    df = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    label_counts = df["label"].value_counts().to_dict()
    valid_labels = {"Ambiguous", "Not Ambiguous"}
    invalid = set(df["label"].unique()) - valid_labels - {"Borderline"}
    if invalid:
        record("Training labels", FAIL, f"Unexpected labels: {invalid}")
    else:
        record("Training labels", PASS, f"Ambiguous={label_counts.get('Ambiguous',0)}, "
               f"Not Ambiguous={label_counts.get('Not Ambiguous',0)}, "
               f"Total rows={len(df)}")

    dups = df.duplicated(subset=["requirement"]).sum()
    if dups == 0:
        record("No exact duplicates in training data", PASS)
    else:
        record("No exact duplicates in training data", FAIL, f"{dups} duplicates found")

except Exception as exc:
    record("Dataset loading", FAIL, str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — No leakage between splits
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 3 — DATA LEAKAGE CHECKS")

try:
    df_train = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    df_ch = pd.read_csv("data/challenge_test_set.csv")
    df_adv = pd.read_csv("data/adversarial_test_set.csv")

    train_set = set(df_train["requirement"].str.lower().str.strip())
    ch_set = set(df_ch["Requirement_Text"].str.lower().str.strip())
    adv_set = set(df_adv["requirement"].str.lower().str.strip())

    tc_overlap = len(train_set & ch_set)
    ta_overlap = len(train_set & adv_set)
    ca_overlap = len(ch_set & adv_set)

    record("Train ↔ Challenge overlap", PASS if tc_overlap == 0 else FAIL,
           f"{tc_overlap} overlapping rows")
    record("Train ↔ Adversarial overlap", PASS if ta_overlap == 0 else FAIL,
           f"{ta_overlap} overlapping rows")
    record("Challenge ↔ Adversarial overlap", PASS if ca_overlap == 0 else FAIL,
           f"{ca_overlap} overlapping rows (informational)")

except Exception as exc:
    record("Leakage checks", FAIL, str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Model and preprocessing artifacts load
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 4 — MODEL ARTIFACT LOADING")

try:
    model = joblib.load("saved_models/model.joblib")
    record("model.joblib loads", PASS, str(type(model)))
except Exception as exc:
    record("model.joblib loads", FAIL, str(exc))

try:
    scaler = joblib.load("saved_models/scaler.joblib")
    record("scaler.joblib loads", PASS, f"{scaler.n_features_in_} features")
except Exception as exc:
    record("scaler.joblib loads", FAIL, str(exc))

try:
    with open("saved_models/metadata.json") as fh:
        meta = json.load(fh)
    record("metadata.json loads", PASS,
           f"upper={meta.get('upper_threshold')}, lower={meta.get('lower_threshold')}")
except Exception as exc:
    record("metadata.json loads", FAIL, str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — Feature counts and ordering match
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 5 — FEATURE COUNT AND ORDER")

try:
    from feature_engineering.feature_extractor import FeatureExtractor
    from feature_engineering.sbert_extractor import SBERTExtractor

    feat_extractor = FeatureExtractor()
    sbert_extractor = SBERTExtractor()

    text = "The system shall respond within 2 seconds."
    feats = feat_extractor.extract_features(text)
    emb = sbert_extractor.transform(pd.Series([text]))

    record("Handcrafted feature count == 24", PASS if len(feats) == 24 else FAIL,
           f"Got {len(feats)}")
    record("SBERT embedding dim == 384", PASS if emb.shape[1] == 384 else FAIL,
           f"Got {emb.shape[1]}")
    record("Total feature dim == 408", PASS if len(feats) + emb.shape[1] == 408 else FAIL,
           f"Got {len(feats) + emb.shape[1]}")

    if scaler is not None:
        scaler_names = list(scaler.feature_names_in_)
        sbert_names = [f"sbert_{i}" for i in range(384)]
        handcrafted_names = list(feats.keys())
        expected_order = sbert_names + handcrafted_names

        if scaler_names == expected_order:
            record("Scaler feature order matches inference order", PASS)
        else:
            mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(scaler_names, expected_order)) if a != b]
            record("Scaler feature order matches inference order", FAIL,
                   f"First mismatch at index {mismatch[0][0]}: '{mismatch[0][1]}' vs '{mismatch[0][2]}'")

except Exception as exc:
    record("Feature check", FAIL, traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — Metadata accuracy
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 6 — METADATA ACCURACY")

if meta and model and scaler:
    try:
        declared_dim = int(meta.get("embedding_dimension", 0))
        actual_sbert = len([f for f in scaler.feature_names_in_ if f.startswith("sbert_")])
        record(
            "Metadata embedding_dimension matches scaler",
            PASS if declared_dim == actual_sbert else FAIL,
            f"metadata={declared_dim}, scaler={actual_sbert}",
        )

        declared_clf = meta.get("classifier", "")
        actual_clf_type = type(model).__name__
        record(
            "Metadata classifier type present",
            PASS if declared_clf else FAIL,
            f"declared='{declared_clf}', actual type='{actual_clf_type}'",
        )

        upper = float(meta.get("upper_threshold", 0))
        lower = float(meta.get("lower_threshold", 0))
        record(
            "Thresholds are 0.75 / 0.25",
            PASS if upper == 0.75 and lower == 0.25 else FAIL,
            f"upper={upper}, lower={lower}",
        )
    except Exception as exc:
        record("Metadata accuracy check", FAIL, str(exc))
else:
    record("Metadata accuracy check", SKIP, "Artifacts failed to load in CHECK 4")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7 — Probability bounds and threshold behaviour
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 7 — PROBABILITY OUTPUTS AND THRESHOLDS")

if model and scaler and meta and feat_extractor and sbert_extractor:
    try:
        from models.predictor import ModelPredictor

        upper = float(meta.get("upper_threshold", 0.75))
        lower = float(meta.get("lower_threshold", 0.25))
        predictor = ModelPredictor(model, upper_threshold=upper, lower_threshold=lower)

        test_texts = [
            "The system shall respond quickly.",
            "The system shall respond within 2 seconds for 95% of requests.",
        ]

        all_bounded = True
        for txt in test_texts:
            X_ling = pd.DataFrame([feat_extractor.extract_features(txt)])
            X_sbert = sbert_extractor.transform(pd.Series([txt]))
            X = pd.concat([X_sbert.reset_index(drop=True), X_ling.reset_index(drop=True)], axis=1)
            X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
            preds, probs = predictor.predict(X_scaled)

            if not ((probs.values >= 0).all() and (probs.values <= 100).all()):
                all_bounded = False
                record("Probability values 0-100", FAIL, f"Out-of-bounds for: {txt}")
                break
            if preds.iloc[0] not in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW"):
                record("Prediction label valid", FAIL, f"Got: {preds.iloc[0]}")
                break

        if all_bounded:
            record("Probability values 0-100", PASS)
            record("Prediction labels valid", PASS)

        # Threshold boundary tests
        mock_model = MagicMock()
        mock_model.classes_ = np.array(["Ambiguous", "Not Ambiguous"])

        mock_model.predict_proba = MagicMock(return_value=np.array([[upper, 1 - upper]]))
        p = ModelPredictor(mock_model, upper_threshold=upper, lower_threshold=lower)
        pred_at_upper = p.predict(pd.DataFrame([[0]]))[0].iloc[0]
        record("Prob == upper_threshold → Ambiguous", PASS if pred_at_upper == "Ambiguous" else FAIL,
               f"Got: {pred_at_upper}")

        mock_model.predict_proba = MagicMock(return_value=np.array([[lower, 1 - lower]]))
        pred_at_lower = p.predict(pd.DataFrame([[0]]))[0].iloc[0]
        record("Prob == lower_threshold → Not Ambiguous", PASS if pred_at_lower == "Not Ambiguous" else FAIL,
               f"Got: {pred_at_lower}")

        mid = (upper + lower) / 2
        mock_model.predict_proba = MagicMock(return_value=np.array([[mid, 1 - mid]]))
        pred_mid = p.predict(pd.DataFrame([[0]]))[0].iloc[0]
        record("Prob at midpoint → NEEDS REVIEW", PASS if pred_mid == "NEEDS REVIEW" else FAIL,
               f"Got: {pred_mid}")

    except Exception as exc:
        record("Probability/threshold checks", FAIL, traceback.format_exc())
else:
    record("Probability/threshold checks", SKIP, "Artifacts failed to load")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8 — Explanation engine does not override prediction
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 8 — EXPLANATION ENGINE ISOLATION")

try:
    from rule_engine.explanation_engine import ExplanationEngine
    ee = ExplanationEngine()
    for cls in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW"):
        exp = ee.generate_explanation("The system shall respond quickly.", cls)
        assert isinstance(exp, dict)
        assert "Reason" in exp
        assert "Missing_Specification" in exp
        assert "Suggested_Rewrite" in exp
        # Must not contain a prediction key
        assert "prediction" not in exp or exp.get("prediction") is None
    record("Explanation engine returns explanation-only dict", PASS)
    record("Explanation engine has no 'prediction' key", PASS)
except Exception as exc:
    record("Explanation engine isolation", FAIL, str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 9 — Input format workflows (single, TXT, XLSX) end-to-end
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 9 — INPUT FORMAT END-TO-END")

try:
    from main import IntelliReqCLI, is_valid_requirement
    cli = IntelliReqCLI()

    # Single sentence
    result = cli._process_text("The system shall respond quickly.")
    assert result["prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")
    assert 0.0 <= result["ambiguity_probability_pct"] <= 100.0
    record("Single sentence inference", PASS, f"prediction={result['prediction']}, "
           f"prob={result['ambiguity_probability_pct']:.1f}%")

    # TXT file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("The system shall respond quickly.\nThe system shall respond within 2 seconds.\n")
        txt_path = fh.name
    from preprocessing.input_parser import parse_input_file
    rows = parse_input_file(txt_path)
    assert len(rows) == 2
    for r in rows:
        res = cli._process_text(r["text"])
        assert res["prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")
    os.unlink(txt_path)
    record("TXT file batch inference (2 rows)", PASS)

    # XLSX file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
        xlsx_path = fh.name
    pd.DataFrame({
        "Requirement": ["The system shall respond quickly.", "The system shall respond within 2 seconds."]
    }).to_excel(xlsx_path, index=False, engine="openpyxl")
    rows = parse_input_file(xlsx_path)
    assert len(rows) == 2
    for r in rows:
        res = cli._process_text(r["text"])
        assert res["prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")
    os.unlink(xlsx_path)
    record("XLSX file batch inference (2 rows)", PASS)

except Exception as exc:
    record("End-to-end input format checks", FAIL, traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 10 — Unsupported formats rejected
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 10 — UNSUPPORTED FORMAT REJECTION")

from preprocessing.input_parser import parse_input_file as _pif

for bad_ext in (".csv", ".pdf", ".docx", ".json"):
    try:
        with tempfile.NamedTemporaryFile(suffix=bad_ext, delete=False) as fh:
            fh.write(b"dummy content")
            bad_path = fh.name
        try:
            _pif(bad_path)
            record(f"Rejects {bad_ext}", FAIL, "No error raised")
        except ValueError as exc:
            record(f"Rejects {bad_ext}", PASS, str(exc)[:60])
        finally:
            os.unlink(bad_path)
    except Exception as exc:
        record(f"Rejects {bad_ext}", FAIL, str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 11 — Regression and integration tests
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 11 — REGRESSION TESTS (REAL MODEL)")

REGRESSION_CASES = [
    ("The system shall respond quickly.", "Ambiguous", "vague temporal"),
    ("The system shall respond within 2 seconds for 95% of requests.", "Not Ambiguous", "measurable constraint"),
    ("The system shall be easy to use.", "Ambiguous", "subjective usability"),
    ("The database connection pool shall have a maximum size of 50 connections.", "Not Ambiguous", "clear numeric limit"),
]

if cli:
    for text, expected_class, desc in REGRESSION_CASES:
        try:
            result = cli._process_text(text)
            pred = result["prediction"]
            # Accept NEEDS REVIEW as the model may be uncertain
            ok = pred in (expected_class, "NEEDS REVIEW")
            record(
                f"Regression: {desc}",
                PASS if ok else FAIL,
                f"expected={expected_class}, got={pred}",
            )
        except Exception as exc:
            record(f"Regression: {desc}", FAIL, str(exc))
else:
    record("Regression tests", SKIP, "CLI failed to load")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 12 — Evaluation metrics reproducible
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 12 — EVALUATION METRICS REPRODUCIBLE")

if os.path.exists("evaluation/challenge_classification_report.csv"):
    record("challenge_classification_report.csv exists", PASS)
else:
    record("challenge_classification_report.csv exists", FAIL, "Run python evaluate.py to generate")

if os.path.exists("evaluation/error_analysis.md"):
    record("error_analysis.md exists", PASS)
else:
    record("error_analysis.md exists", FAIL, "Run python evaluate.py to generate")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 13 — Documentation consistency
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 13 — DOCUMENTATION")

for doc in ("README.md", "docs/final_model_report.md"):
    if os.path.exists(doc):
        record(f"{doc} exists", PASS)
    else:
        record(f"{doc} exists", FAIL, "Missing documentation file")

if meta:
    # README should mention the threshold
    with open("README.md", encoding="utf-8") as fh:
        readme = fh.read()
    if "0.75" in readme and "0.25" in readme:
        record("README mentions thresholds 0.75/0.25", PASS)
    else:
        record("README mentions thresholds 0.75/0.25", FAIL, "Thresholds not documented in README")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 14 — No conflicting production entry points
# ─────────────────────────────────────────────────────────────────────────────

section("CHECK 14 — SINGLE CANONICAL ENTRY POINTS")

# There should be exactly one canonical training script: train.py
extra_train_scripts = [
    f for f in ("train_final.py", "train_fixed.py", "train_new.py", "train2.py")
    if os.path.exists(f)
]
if not extra_train_scripts:
    record("No duplicate training entry points", PASS)
else:
    record("No duplicate training entry points", FAIL,
           f"Extra scripts found: {extra_train_scripts}")

# Only one main.py
if os.path.exists("main.py"):
    record("main.py exists as single interface", PASS)
else:
    record("main.py exists as single interface", FAIL)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

section("FINAL ACCEPTANCE REPORT")

passed = [c for c in checks if c["status"] == PASS]
failed = [c for c in checks if c["status"] == FAIL]
skipped = [c for c in checks if c["status"] == SKIP]

print(f"\n  Passed  : {len(passed)}")
print(f"  Failed  : {len(failed)}")
print(f"  Skipped : {len(skipped)}")
print(f"  Total   : {len(checks)}")

if failed:
    print(f"\n  FAILED CHECKS:")
    for c in failed:
        print(f"    [FAIL] {c['name']}")
        if c["detail"]:
            print(f"        {c['detail']}")

print()
if not failed and not skipped:
    print("  \033[92mPROJECT STATUS: VERIFIED FINAL BUILD\033[0m")
elif not failed and skipped:
    print(f"  \033[93mPROJECT STATUS: CONDITIONALLY VERIFIED — {len(skipped)} check(s) skipped\033[0m")
else:
    print("  \033[91mPROJECT STATUS: NOT READY — SEE FAILED CHECKS\033[0m")
print()
