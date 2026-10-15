import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
import joblib

from main import IntelliReqCLI, is_valid_requirement
from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from rule_engine.explanation_engine import ExplanationEngine
from rule_engine.rule_detector import RuleBasedDetector
from models.predictor import ModelPredictor

@pytest.fixture(scope="module")
def cli():
    return IntelliReqCLI()

@pytest.fixture(scope="module")
def ling_fe():
    return FeatureExtractor()
    
@pytest.fixture(scope="module")
def sbert_fe():
    return SBERTExtractor()

@pytest.fixture(scope="module")
def model_predictor():
    if os.path.exists('saved_models/model.joblib'):
        model = joblib.load('saved_models/model.joblib')
        return ModelPredictor(model)
    return None

# 1-3. Input Validation
def test_empty_input():
    assert not is_valid_requirement("")
    assert not is_valid_requirement("   ")
    assert not is_valid_requirement("A")

def test_long_input():
    assert is_valid_requirement("A " * 50)
    
def test_special_chars_input():
    assert is_valid_requirement("The system shall <html> respond </html> ©.")

# 4-6. Feature Extraction Dimensions
def test_ling_feature_dimension(ling_fe):
    feats = ling_fe.extract_features("The system shall respond quickly.")
    assert len(feats) == 24

def test_sbert_feature_dimension(sbert_fe):
    vec = sbert_fe.transform(pd.Series(["Test"]))
    assert vec.shape[1] == 384

def test_total_feature_dimension(ling_fe, sbert_fe):
    feats = ling_fe.extract_features("Test")
    vec = sbert_fe.transform(pd.Series(["Test"]))
    assert len(feats) + vec.shape[1] == 408

# 7-10. Linguistic Handcrafted Features
def test_feature_consistency(ling_fe):
    feats1 = ling_fe.extract_features("The system shall respond quickly.")
    feats2 = ling_fe.extract_features("The system shall process 500 requests per second.")
    assert len(feats1) == len(feats2)
    assert set(feats1.keys()) == set(feats2.keys())

def test_feature_numbers(ling_fe):
    feats = ling_fe.extract_features("The system shall wait 5 seconds.")
    assert feats['Numeric_Count'] > 0
    assert feats['TimeUnit_Count'] > 0

def test_feature_modals(ling_fe):
    feats = ling_fe.extract_features("The system may work.")
    assert feats['Modal_Verb_Count'] > 0

def test_feature_readability(ling_fe):
    feats = ling_fe.extract_features("Short text.")
    assert 'Char_Length' in feats

# 11-13. Rule Engine
def test_rule_detector_ambiguous():
    detector = RuleBasedDetector()
    res = detector.analyze_requirement("The system might fail.")
    assert res['Rule_Label'] == "Ambiguous"
    assert "might" in res['Matched_Ambiguous_Words']

def test_rule_detector_clear():
    detector = RuleBasedDetector()
    res = detector.analyze_requirement("The system shall return HTTP 200.")
    assert res['Rule_Label'] == "Not Ambiguous"
    assert len(res['Matched_Ambiguous_Words']) == 0

def test_explanation_engine():
    ee = ExplanationEngine()
    exp = ee.generate_explanation("The system shall run quickly.", "Ambiguous")
    assert "Reason" in exp
    assert "quickly" in exp['Reason'].lower() or "vague" in exp['Reason'].lower()

# 14-17. Predictor Logic
def test_predictor_bounds(model_predictor):
    if not model_predictor: pytest.skip("Model not trained.")
    import pandas as pd
    dummy_x = pd.DataFrame(np.random.rand(2, 408)) # Fake scaled features
    preds, probs = model_predictor.predict(dummy_x)
    assert len(preds) == 2
    assert probs.iloc[0].sum() > 99.0 # Should sum to approx 100%

def test_predictor_fallback(model_predictor):
    if not model_predictor: pytest.skip("Model not trained.")
    assert model_predictor.upper_threshold == 0.75
    assert model_predictor.lower_threshold == 0.25

def test_predictor_classes(model_predictor):
    if not model_predictor: pytest.skip("Model not trained.")
    assert "Ambiguous" in model_predictor.trained_model.classes_

def test_pipeline_regression(cli):
    # Tests the full inference pipeline (text -> prediction)
    if not os.path.exists('saved_models/model.joblib'): pytest.skip("Model not trained.")
    pred, conf, exp, feats = cli.process_text("The system shall process requests efficiently.")
    assert pred in ["Ambiguous", "Not Ambiguous", "NEEDS REVIEW"]
    assert 0.0 <= conf <= 100.0

# 18-20. Batch Processing Edge Cases
def test_batch_processing_mock(cli, tmp_path):
    df = pd.DataFrame({"Requirement": [
        "The system shall respond quickly.",
        "The application shall pass W3C HTML5 validation without errors."
    ]})
    out_path = tmp_path / "out.xlsx"
    if os.path.exists('saved_models/model.joblib'):
        res = cli._run_batch_dataframe(df, str(out_path))
        assert os.path.exists(out_path)
        assert len(res) == 2

def test_batch_malformed_xlsx(cli, tmp_path):
    df = pd.DataFrame({"WrongColumn": ["The system shall validate data securely."]})
    path = tmp_path / "bad.xlsx"
    df.to_excel(path, index=False)
    # Should throw KeyError for missing requirement header
    with pytest.raises(KeyError):
        cli._run_batch_dataframe(df, str(tmp_path / "out.xlsx"))

def test_batch_empty_xlsx(cli, tmp_path):
    df = pd.DataFrame(columns=["Requirement"])
    path = tmp_path / "empty.xlsx"
    df.to_excel(path, index=False)
    # Should throw ValueError for empty dataframe
    with pytest.raises(ValueError):
        cli._run_batch_dataframe(df, str(tmp_path / "out.xlsx"))

# 21-23. Metadata and Artifacts
def test_metadata_exists():
    if os.path.exists('saved_models/model.joblib'):
        assert os.path.exists('saved_models/metadata.json')

def test_metadata_contents():
    if os.path.exists('saved_models/metadata.json'):
        with open('saved_models/metadata.json', 'r') as f:
            meta = json.load(f)
            assert 'embedding_dimension' in meta
            assert meta['embedding_dimension'] == 384

def test_model_joblib():
    if os.path.exists('saved_models/model.joblib'):
        model = joblib.load('saved_models/model.joblib')
        assert hasattr(model, 'predict_proba')

# 24-25. Verification of Fresh Process Consistency
def test_fresh_process_consistency(cli):
    if not os.path.exists('saved_models/model.joblib'): pytest.skip("Model not trained.")
    # Ensures identical inputs give identical float outputs
    req = "The user interface shall be very fast."
    pred1, conf1, _, _ = cli.process_text(req)
    pred2, conf2, _, _ = cli.process_text(req)
    assert pred1 == pred2
    assert abs(conf1 - conf2) < 1e-6
