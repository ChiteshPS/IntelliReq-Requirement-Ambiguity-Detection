"""
tests/test_fastapi_backend.py — Automated Tests for FastAPI Backend Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["service_ready"] is True
    assert data["model_version"] == "2.1.0"


def test_model_info_endpoint(client):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "IntelliReq" in data["model_name"]
    assert data["total_features"] == 408
    assert data["thresholds"]["upper_ambiguous_threshold"] == 0.75
    assert data["thresholds"]["lower_clear_threshold"] == 0.25
    assert "dataset_provenance" in data
    assert "target_accuracy_claim" in data
    assert len(data["verified_benchmarks"]) > 0


def test_analyze_ambiguous_requirement(client):
    payload = {"requirement": "The system shall respond to user requests quickly."}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["predicted_class"] == "AMBIGUOUS"
    assert data["confidence_score"] >= 75.0
    assert "quickly" in [c.lower() for c in data["detected_cues"]]
    assert len(data["missing_specifications"]) > 0
    assert data["suggested_rewrite"] is not None
    assert "Suggestion only" in data["rewrite_disclaimer"]
    assert data["explanation"]["rule_based_evidence"] is not None


def test_analyze_clear_requirement(client):
    payload = {"requirement": "The system shall return search results within 2 seconds for 95% of requests."}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["predicted_class"] == "NOT AMBIGUOUS"
    assert data["confidence_score"] <= 25.0
    assert data["probabilities"]["Not Ambiguous"] >= 75.0


def test_analyze_invalid_short_requirement(client):
    payload = {"requirement": "Short text"}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert data["predicted_class"] == "INVALID"


def test_analyze_batch_endpoint(client):
    payload = {
        "requirements": [
            {"id": "REQ-1", "requirement": "The system shall respond to user requests quickly."},
            {"id": "REQ-2", "requirement": "The system shall return search results within 2 seconds for 95% of queries."},
            {"id": "REQ-3", "requirement": "Too short"},
        ]
    }
    response = client.post("/api/analyze-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["total_processed"] == 3
    assert data["summary"]["ambiguous_count"] == 1
    assert data["summary"]["not_ambiguous_count"] == 1
    assert data["summary"]["skipped_count"] == 1
    assert len(data["results"]) == 3
    assert data["results"][0]["id"] == "REQ-1"
    assert data["results"][0]["predicted_class"] == "AMBIGUOUS"
    assert data["results"][1]["id"] == "REQ-2"
    assert data["results"][1]["predicted_class"] == "NOT AMBIGUOUS"
    assert data["results"][2]["id"] == "REQ-3"
    assert data["results"][2]["is_valid"] is False


def test_analyze_batch_empty_raises(client):
    response = client.post("/api/analyze-batch", json={"requirements": []})
    assert response.status_code == 400


def test_missing_model_artifacts_error_handling(tmp_path):
    """Verify that RequirementService raises clear FileNotFoundError when model files are missing."""
    from services.requirement_service import RequirementService
    with pytest.raises(FileNotFoundError) as excinfo:
        RequirementService(model_dir=str(tmp_path))
    assert "Missing required model artifacts" in str(excinfo.value)


def test_deterministic_inference(client):
    """Verify that multiple consecutive API calls with the same requirement return bit-for-bit identical results."""
    text = "The system shall respond to user requests within 500 milliseconds for all standard queries."
    payload = {"requirement": text}
    
    first_resp = client.post("/api/analyze", json=payload).json()
    for _ in range(3):
        resp = client.post("/api/analyze", json=payload).json()
        assert resp["predicted_class"] == first_resp["predicted_class"]
        assert resp["confidence_score"] == pytest.approx(first_resp["confidence_score"], abs=1e-5)
        assert resp["probabilities"] == first_resp["probabilities"]
        assert resp["detected_cues"] == first_resp["detected_cues"]


def test_probability_bounds_and_class_mapping(client):
    """Verify probability percentages are in [0, 100] and class thresholds are correctly mapped."""
    test_cases = [
        "The system shall respond to user requests quickly.",
        "The system shall process 100 transactions per second under normal conditions.",
        "The background process takes approximately 10 minutes.",
    ]
    for text in test_cases:
        resp = client.post("/api/analyze", json={"requirement": text})
        assert resp.status_code == 200
        data = resp.json()
        prob_amb = data["confidence_score"]
        probs = data["probabilities"]
        
        # Check bounds
        assert 0.0 <= prob_amb <= 100.0
        assert 0.0 <= probs["Ambiguous"] <= 100.0
        assert 0.0 <= probs["Not Ambiguous"] <= 100.0
        assert pytest.approx(probs["Ambiguous"] + probs["Not Ambiguous"], abs=0.1) == 100.0
        
        # Check 3-tier class mapping
        if prob_amb >= 75.0:
            assert data["predicted_class"] == "AMBIGUOUS"
        elif prob_amb <= 25.0:
            assert data["predicted_class"] == "NOT AMBIGUOUS"
        else:
            assert data["predicted_class"] == "NEEDS REVIEW"


def test_api_prediction_matches_service_pipeline(client):
    """Verify strict consistency between FastAPI response and direct RequirementService output."""
    from backend.main import get_service
    service = get_service()
    
    requirement = "The mobile app shall authenticate users via biometric sensors within 300 ms."
    service_result = service.analyze_single(requirement)
    
    resp = client.post("/api/analyze", json={"requirement": requirement})
    assert resp.status_code == 200
    api_result = resp.json()
    
    assert api_result["is_valid"] == service_result["is_valid"]
    assert api_result["predicted_class"] == service_result["prediction"].upper()
    assert api_result["confidence_score"] == pytest.approx(service_result["probability_pct"], abs=1e-2)
    assert api_result["suggested_rewrite"] == service_result["suggested_rewrite"]


def test_batch_limit_exceeded_raises_400(client):
    """Verify batch request exceeding 500 items is safely rejected with status 400."""
    too_many = [{"id": f"REQ-{i}", "requirement": "The system shall respond in 1 second."} for i in range(501)]
    resp = client.post("/api/analyze-batch", json={"requirements": too_many})
    assert resp.status_code == 400
    assert "Batch limit exceeded" in resp.json()["detail"]


def test_invalid_json_request_raises_422(client):
    """Verify invalid JSON schema payload returns standard 422 Unprocessable Entity."""
    resp = client.post("/api/analyze", json={"wrong_field": "Some text"})
    assert resp.status_code == 422


def test_unsupported_file_upload_raises_400(client):
    """Verify uploading an unsupported file format like .csv or .pdf returns 400."""
    fake_csv = b"Requirement\nThe system shall respond quickly."
    files = {"file": ("requirements.csv", fake_csv, "text/csv")}
    resp = client.post("/api/analyze-file", files=files)
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]

