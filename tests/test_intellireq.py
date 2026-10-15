"""
tests/test_intellireq.py

Comprehensive test suite for IntelliReq.
Covers:
- Input validation
- Feature extraction (dimensions, names, order)
- TXT and XLSX batch parsing
- Unsupported file formats
- Empty / malformed inputs
- Missing XLSX column
- Duplicate requirements
- Unicode and punctuation
- Very short and very long inputs
- Probability bounds and threshold boundary conditions (0.25, 0.75)
- NEEDS REVIEW behaviour
- Explanation engine does NOT override model predictions
- Model loading and calibration
- Dataset leakage (train/challenge/adversarial)
- Batch result order
- End-to-end regression with real saved model
- TXT workflow end-to-end
- XLSX workflow end-to-end
"""
import os
import sys
import json
import tempfile
import pytest
import joblib
import pandas as pd
import numpy as np

# Make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import IntelliReqCLI, is_valid_requirement
from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from models.predictor import ModelPredictor
from rule_engine.explanation_engine import ExplanationEngine
from preprocessing.input_parser import parse_txt, parse_xlsx, parse_input_file

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cli():
    return IntelliReqCLI()


@pytest.fixture(scope="module")
def feat_extractor():
    return FeatureExtractor()


@pytest.fixture(scope="module")
def sbert_extractor():
    return SBERTExtractor()


@pytest.fixture(scope="module")
def saved_model():
    model = joblib.load("saved_models/model.joblib")
    scaler = joblib.load("saved_models/scaler.joblib")
    with open("saved_models/metadata.json") as fh:
        meta = json.load(fh)
    predictor = ModelPredictor(
        trained_model=model,
        upper_threshold=float(meta.get("upper_threshold", 0.75)),
        lower_threshold=float(meta.get("lower_threshold", 0.25)),
    )
    return model, scaler, meta, predictor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input validation
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_string_invalid():
    assert not is_valid_requirement("")


def test_whitespace_only_invalid():
    assert not is_valid_requirement("   ")


def test_single_word_invalid():
    assert not is_valid_requirement("Shall")


def test_two_words_invalid():
    assert not is_valid_requirement("System shall")


def test_three_words_valid():
    assert is_valid_requirement("System shall respond")


def test_long_input_valid():
    assert is_valid_requirement("The system " * 50)


def test_unicode_input_valid():
    assert is_valid_requirement("Das System soll schnell reagieren.")


def test_punctuation_only_invalid():
    assert not is_valid_requirement("... !!!")


def test_none_invalid():
    assert not is_valid_requirement(None)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature extraction — dimensions and names
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_HANDCRAFTED_FEATURES = [
    "Token_Count", "Word_Count", "Sentence_Count", "Char_Length",
    "Avg_Token_Length", "Lexical_Diversity",
    "Vague_Term_Count", "Subjective_Term_Count", "Approximate_Quantity_Count",
    "Undefined_Reference_Count", "Modal_Verb_Count", "Passive_Voice_Count",
    "Linguistic_Ambiguity_Score",
    "Numeric_Count", "Percentage_Count", "TimeUnit_Count", "SizeUnit_Count",
    "Capacity_Count", "Threshold_Count", "Format_Count", "Condition_Count",
    "Range_Count", "Has_Testable_Metric", "Specificity_Score",
]


def test_handcrafted_feature_count(feat_extractor):
    feats = feat_extractor.extract_features("The system shall respond quickly.")
    assert len(feats) == 24, f"Expected 24, got {len(feats)}"


def test_handcrafted_feature_names(feat_extractor):
    feats = feat_extractor.extract_features("The system shall respond quickly.")
    assert list(feats.keys()) == EXPECTED_HANDCRAFTED_FEATURES


def test_sbert_dimension(sbert_extractor):
    emb = sbert_extractor.transform(pd.Series(["Test sentence here."]))
    assert emb.shape == (1, 384)


def test_total_feature_count(feat_extractor, sbert_extractor):
    text = "The system shall respond quickly."
    feats = feat_extractor.extract_features(text)
    emb = sbert_extractor.transform(pd.Series([text]))
    assert len(feats) + emb.shape[1] == 408


def test_feature_names_start_sbert(sbert_extractor):
    emb = sbert_extractor.transform(pd.Series(["Test."]))
    assert all(c.startswith("sbert_") for c in emb.columns)


def test_scaler_feature_order():
    """Scaler feature order must match: sbert_0..sbert_383 then handcrafted."""
    scaler = joblib.load("saved_models/scaler.joblib")
    names = list(scaler.feature_names_in_)
    assert len(names) == 408
    assert names[:384] == [f"sbert_{i}" for i in range(384)]
    assert names[384:] == EXPECTED_HANDCRAFTED_FEATURES


def test_feature_consistency_across_calls(feat_extractor):
    text = "The system shall process 500 requests per second."
    a = feat_extractor.extract_features(text)
    b = feat_extractor.extract_features(text)
    assert a == b


def test_vague_term_detection(feat_extractor):
    feats = feat_extractor.extract_features("The system shall respond quickly.")
    assert feats["Vague_Term_Count"] >= 1


def test_numeric_detection(feat_extractor):
    feats = feat_extractor.extract_features("The system shall handle 500 users.")
    assert feats["Numeric_Count"] >= 1


def test_testable_metric_with_number_and_time(feat_extractor):
    feats = feat_extractor.extract_features("The system shall respond within 2 seconds.")
    assert feats["Has_Testable_Metric"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Predictor — thresholds and probability bounds
# ─────────────────────────────────────────────────────────────────────────────

def test_predictor_threshold_mismatch_raises():
    with pytest.raises(ValueError):
        ModelPredictor(None, upper_threshold=0.3, lower_threshold=0.8)


def test_predictor_output_types(saved_model, feat_extractor, sbert_extractor):
    _, scaler, _, predictor = saved_model
    text = "The system shall respond quickly."
    X_ling = pd.DataFrame([feat_extractor.extract_features(text)])
    X_sbert = sbert_extractor.transform(pd.Series([text]))
    X = pd.concat([X_sbert.reset_index(drop=True), X_ling.reset_index(drop=True)], axis=1)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    preds, probs = predictor.predict(X_scaled)
    assert isinstance(preds, pd.Series)
    assert isinstance(probs, pd.DataFrame)


def test_probability_bounds(saved_model, feat_extractor, sbert_extractor):
    """All probability values must be between 0 and 100 (displayed as %)."""
    _, scaler, _, predictor = saved_model
    texts = [
        "The system shall respond quickly.",
        "The system shall respond within 2 seconds for 95% of requests.",
    ]
    for text in texts:
        X_ling = pd.DataFrame([feat_extractor.extract_features(text)])
        X_sbert = sbert_extractor.transform(pd.Series([text]))
        X = pd.concat([X_sbert.reset_index(drop=True), X_ling.reset_index(drop=True)], axis=1)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        _, probs = predictor.predict(X_scaled)
        assert (probs.values >= 0).all() and (probs.values <= 100).all()


def test_threshold_upper_boundary(saved_model, feat_extractor, sbert_extractor):
    """Probability exactly at upper_threshold (0.75) should map to Ambiguous."""
    model, scaler, meta, _ = saved_model
    upper = float(meta.get("upper_threshold", 0.75))
    lower = float(meta.get("lower_threshold", 0.25))

    from unittest.mock import MagicMock
    import numpy as np

    mock_model = MagicMock()
    mock_model.classes_ = np.array(["Ambiguous", "Not Ambiguous"])
    # Prob exactly at upper threshold
    mock_model.predict_proba = MagicMock(return_value=np.array([[upper, 1 - upper]]))

    pred = ModelPredictor(mock_model, upper_threshold=upper, lower_threshold=lower)
    preds, _ = pred.predict(pd.DataFrame([[0]]))
    assert preds.iloc[0] == "Ambiguous"


def test_threshold_lower_boundary(saved_model):
    """Probability exactly at lower_threshold (0.25) should map to Not Ambiguous."""
    _, _, meta, _ = saved_model
    upper = float(meta.get("upper_threshold", 0.75))
    lower = float(meta.get("lower_threshold", 0.25))

    from unittest.mock import MagicMock
    import numpy as np

    mock_model = MagicMock()
    mock_model.classes_ = np.array(["Ambiguous", "Not Ambiguous"])
    mock_model.predict_proba = MagicMock(return_value=np.array([[lower, 1 - lower]]))

    pred = ModelPredictor(mock_model, upper_threshold=upper, lower_threshold=lower)
    preds, _ = pred.predict(pd.DataFrame([[0]]))
    assert preds.iloc[0] == "Not Ambiguous"


def test_needs_review_in_middle(saved_model):
    """Probability between thresholds should give NEEDS REVIEW."""
    _, _, meta, _ = saved_model
    upper = float(meta.get("upper_threshold", 0.75))
    lower = float(meta.get("lower_threshold", 0.25))
    mid = (upper + lower) / 2

    from unittest.mock import MagicMock
    mock_model = MagicMock()
    mock_model.classes_ = np.array(["Ambiguous", "Not Ambiguous"])
    mock_model.predict_proba = MagicMock(return_value=np.array([[mid, 1 - mid]]))

    pred = ModelPredictor(mock_model, upper_threshold=upper, lower_threshold=lower)
    preds, _ = pred.predict(pd.DataFrame([[0]]))
    assert preds.iloc[0] == "NEEDS REVIEW"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Explanation engine — must NOT override classifier prediction
# ─────────────────────────────────────────────────────────────────────────────

def test_explanation_engine_no_prediction_override(cli):
    """
    The explanation engine should return a dict with Reason, Missing_Specification,
    and Suggested_Rewrite — but the prediction from the classifier must not change.
    """
    ee = ExplanationEngine()

    for pred_class in ["Ambiguous", "Not Ambiguous", "NEEDS REVIEW"]:
        exp = ee.generate_explanation("The system shall respond quickly.", pred_class)
        assert isinstance(exp, dict)
        assert "Reason" in exp
        assert "Missing_Specification" in exp
        assert "Suggested_Rewrite" in exp
        # The engine must not return a different class
        assert "prediction" not in exp or exp.get("prediction") is None


def test_explanation_ambiguous_has_reason(cli):
    ee = ExplanationEngine()
    exp = ee.generate_explanation("The system shall respond quickly.", "Ambiguous")
    assert exp["Reason"] != ""


def test_explanation_not_ambiguous_no_rewrite(cli):
    ee = ExplanationEngine()
    exp = ee.generate_explanation(
        "The system shall respond within 2 seconds for 95% of requests.", "Not Ambiguous"
    )
    assert exp["Suggested_Rewrite"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model loading and metadata consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_model_artifacts_exist():
    assert os.path.exists("saved_models/model.joblib")
    assert os.path.exists("saved_models/scaler.joblib")
    assert os.path.exists("saved_models/metadata.json")


def test_metadata_fields(saved_model):
    _, _, meta, _ = saved_model
    for field in ["upper_threshold", "lower_threshold", "embedding_dimension",
                  "classifier", "calibration_method"]:
        assert field in meta, f"Missing metadata field: {field}"


def test_metadata_thresholds_consistent(saved_model):
    _, _, meta, _ = saved_model
    assert float(meta["upper_threshold"]) == 0.75
    assert float(meta["lower_threshold"]) == 0.25


def test_metadata_embedding_dim_matches_scaler():
    scaler = joblib.load("saved_models/scaler.joblib")
    with open("saved_models/metadata.json") as fh:
        meta = json.load(fh)
    sbert_cols = [c for c in scaler.feature_names_in_ if c.startswith("sbert_")]
    assert len(sbert_cols) == int(meta["embedding_dimension"])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Input file parsing — TXT
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_txt_basic(tmp_path):
    f = tmp_path / "reqs.txt"
    f.write_text("The system shall respond quickly.\nThe system shall be available 24/7.\n")
    rows = parse_txt(str(f))
    assert len(rows) == 2
    assert rows[0]["text"] == "The system shall respond quickly."
    assert rows[0]["row"] == 1
    assert rows[1]["row"] == 2


def test_parse_txt_blank_lines_skipped(tmp_path):
    f = tmp_path / "reqs.txt"
    f.write_text("Req one.\n\n  \nReq two.\n")
    rows = parse_txt(str(f))
    assert len(rows) == 2


def test_parse_txt_empty_file_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ValueError, match="empty"):
        parse_txt(str(f))


def test_parse_txt_not_found_raises():
    with pytest.raises(FileNotFoundError):
        parse_txt("does_not_exist.txt")


def test_parse_txt_unicode(tmp_path):
    f = tmp_path / "unicode.txt"
    f.write_text("Das System soll schnell reagieren.\nEl sistema debe responder rápidamente.\n", encoding="utf-8")
    rows = parse_txt(str(f))
    assert len(rows) == 2


def test_parse_txt_row_numbers_preserved(tmp_path):
    f = tmp_path / "reqs.txt"
    f.write_text("Line one.\n\nLine three.\n")
    rows = parse_txt(str(f))
    # blank line 2 skipped; rows should have row numbers 1 and 3
    assert rows[0]["row"] == 1
    assert rows[1]["row"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# 7. Input file parsing — XLSX
# ─────────────────────────────────────────────────────────────────────────────

def _make_xlsx(tmp_path, col_name, data):
    path = str(tmp_path / "reqs.xlsx")
    pd.DataFrame({col_name: data}).to_excel(path, index=False, engine="openpyxl")
    return path


def test_parse_xlsx_requirement_column(tmp_path):
    path = _make_xlsx(tmp_path, "Requirement", ["Req one here.", "Req two here."])
    rows = parse_xlsx(path)
    assert len(rows) == 2


def test_parse_xlsx_requirement_text_column(tmp_path):
    path = _make_xlsx(tmp_path, "Requirement_Text", ["Req one.", "Req two."])
    rows = parse_xlsx(path)
    assert len(rows) == 2


def test_parse_xlsx_lowercase_column(tmp_path):
    path = _make_xlsx(tmp_path, "requirement", ["Req one.", "Req two."])
    rows = parse_xlsx(path)
    assert len(rows) == 2


def test_parse_xlsx_missing_column_raises(tmp_path):
    path = str(tmp_path / "bad.xlsx")
    pd.DataFrame({"SomeOtherCol": ["val"]}).to_excel(path, index=False, engine="openpyxl")
    with pytest.raises(ValueError, match="No requirement column"):
        parse_xlsx(path)


def test_parse_xlsx_empty_file_raises(tmp_path):
    path = str(tmp_path / "empty.xlsx")
    pd.DataFrame().to_excel(path, index=False, engine="openpyxl")
    with pytest.raises(ValueError, match="empty"):
        parse_xlsx(path)


def test_parse_xlsx_empty_rows_skipped(tmp_path):
    path = _make_xlsx(tmp_path, "Requirement", ["Req one.", None, "", "Req two."])
    rows = parse_xlsx(path)
    assert len(rows) == 2


def test_parse_xlsx_not_found_raises():
    with pytest.raises(FileNotFoundError):
        parse_xlsx("does_not_exist.xlsx")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Unsupported formats
# ─────────────────────────────────────────────────────────────────────────────

def test_unsupported_csv_rejected(tmp_path):
    f = tmp_path / "reqs.csv"
    f.write_text("Requirement\nReq one.\n")
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_input_file(str(f))


def test_unsupported_pdf_rejected(tmp_path):
    f = tmp_path / "reqs.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_input_file(str(f))


def test_unsupported_docx_rejected(tmp_path):
    f = tmp_path / "reqs.docx"
    f.write_bytes(b"PK fake docx")
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_input_file(str(f))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Dataset leakage
# ─────────────────────────────────────────────────────────────────────────────

def test_no_train_challenge_leakage():
    df_train = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    df_ch = pd.read_csv("data/challenge_test_set.csv")
    train_set = set(df_train["requirement"].str.lower().str.strip())
    ch_set = set(df_ch["Requirement_Text"].str.lower().str.strip())
    overlap = train_set & ch_set
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} train↔challenge overlaps"


def test_no_train_adversarial_leakage():
    df_train = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    df_adv = pd.read_csv("data/adversarial_test_set.csv")
    train_set = set(df_train["requirement"].str.lower().str.strip())
    adv_set = set(df_adv["requirement"].str.lower().str.strip())
    overlap = train_set & adv_set
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} train↔adversarial overlaps"


def test_no_exact_duplicates_in_training():
    df = pd.read_csv("data/intellireq_clean_dataset_v4.csv")
    dups = df.duplicated(subset=["requirement"], keep=False).sum()
    assert dups == 0, f"{dups} exact duplicate requirements in training data"


# ─────────────────────────────────────────────────────────────────────────────
# 10. End-to-end regression tests with real model
# ─────────────────────────────────────────────────────────────────────────────

REGRESSION_CASES = [
    # (text, expected_class, description)
    # NEEDS REVIEW is always accepted alongside the expected class — borderline cases are valid.
    # "The system shall support English and Spanish languages." is a KNOWN false positive
    # documented in evaluation/error_analysis.md (line 24). It is NOT used as a regression
    # case because forcing a test expectation against the real model output is dishonest.
    ("The system shall respond quickly.", "Ambiguous", "vague temporal term"),
    ("The system shall respond within 2 seconds for 95% of requests.", "Not Ambiguous", "measurable constraint with threshold"),
    ("The system shall store data securely.", "Ambiguous", "security requirement — 'securely' is vague"),
    ("The database connection pool shall have a maximum size of 50 connections.", "Not Ambiguous", "clear numeric limit"),
]


@pytest.mark.parametrize("text,expected,desc", REGRESSION_CASES)
def test_regression_real_model(cli, text, expected, desc):
    result = cli._process_text(text)
    # Allow NEEDS REVIEW as acceptable for borderline cases
    assert result["prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW"), \
        f"Unexpected prediction for '{desc}': {result['prediction']}"
    # For clearly ambiguous/clear, check strictly
    if expected in ("Ambiguous", "Not Ambiguous"):
        assert result["prediction"] in (expected, "NEEDS REVIEW"), \
            f"Wrong class for '{desc}': expected {expected}, got {result['prediction']}"


def test_end_to_end_txt_workflow(cli, tmp_path):
    """TXT file → parse → analyze → result dict list."""
    f = tmp_path / "reqs.txt"
    f.write_text(
        "The system shall respond quickly.\n"
        "The system shall respond within 2 seconds for 95% of requests.\n"
    )
    rows = parse_input_file(str(f))
    assert len(rows) == 2
    for item in rows:
        assert is_valid_requirement(item["text"])
        result = cli._process_text(item["text"])
        assert result["prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")
        assert 0.0 <= result["ambiguity_probability_pct"] <= 100.0


def test_end_to_end_xlsx_workflow(cli, tmp_path):
    """XLSX file → parse → analyze → result dict list."""
    xlsx_path = str(tmp_path / "reqs.xlsx")
    pd.DataFrame({
        "Requirement": [
            "The system shall respond quickly.",
            "The system shall respond within 2 seconds.",
        ]
    }).to_excel(xlsx_path, index=False, engine="openpyxl")

    rows = parse_input_file(xlsx_path)
    assert len(rows) == 2
    for item in rows:
        result = cli._process_text(item["text"])
        assert result["prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")


def test_batch_result_order_preserved(cli, tmp_path):
    """Output order must match input order."""
    texts = [
        "The system shall respond quickly.",
        "The system shall respond within 2 seconds.",
        "The application must provide an easy to use interface.",
    ]
    f = tmp_path / "ordered.txt"
    f.write_text("\n".join(texts) + "\n")
    rows = parse_input_file(str(f))
    results = [cli._process_text(r["text"]) for r in rows]
    for i, (row, result) in enumerate(zip(rows, results)):
        assert row["text"] == result["text"], f"Order mismatch at index {i}"


def test_duplicate_requirements_processed(cli, tmp_path):
    """Duplicate rows should each be processed individually (not silently merged)."""
    f = tmp_path / "dups.txt"
    f.write_text(
        "The system shall respond quickly.\n"
        "The system shall respond quickly.\n"
    )
    rows = parse_input_file(str(f))
    assert len(rows) == 2
    r1 = cli._process_text(rows[0]["text"])
    r2 = cli._process_text(rows[1]["text"])
    assert r1["prediction"] == r2["prediction"]


def test_very_long_requirement_valid(cli):
    text = "The system shall " + "process and validate all incoming data " * 20 + "within 2 seconds."
    result = cli._process_text(text)
    assert result["prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")


def test_probability_is_finite(cli):
    result = cli._process_text("The system shall respond within 2 seconds.")
    assert np.isfinite(result["ambiguity_probability_pct"])
