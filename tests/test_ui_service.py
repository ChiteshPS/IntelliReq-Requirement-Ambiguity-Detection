"""
tests/test_ui_service.py — Verification Tests for UI Service Layer

Tests the RequirementService for:
- Single prediction inference
- Invalid inputs and edge cases
- Batch parsing of TXT and XLSX files
- Unsupported formats rejection (.csv, .pdf, .docx)
- Explanation engine isolation
- Export to Excel
- Metadata and evaluation metric loading
"""

import os
import io
import pytest
import pandas as pd

from services.requirement_service import RequirementService, is_valid_requirement


@pytest.fixture(scope="module")
def service():
    return RequirementService()


# ── 1. Validation & Single Prediction Tests ──────────────────────────────────

def test_is_valid_requirement():
    assert not is_valid_requirement("")
    assert not is_valid_requirement("   ")
    assert not is_valid_requirement("Too short")
    assert not is_valid_requirement(None)
    assert is_valid_requirement("The system shall log errors.")
    assert is_valid_requirement("Users shall be able to log in.")


def test_analyze_single_invalid_input(service):
    res = service.analyze_single("Short")
    assert not res["is_valid"]
    assert res["prediction"] == "INVALID"
    assert "at least 3 words" in res["reason"]


def test_analyze_single_ambiguous(service):
    req = "The system shall be extremely fast and highly scalable under heavy load."
    res = service.analyze_single(req)
    assert res["is_valid"]
    assert res["prediction"] == "Ambiguous"
    assert res["probability_pct"] >= 75.0
    assert len(res["reason"]) > 0


def test_analyze_single_not_ambiguous(service):
    req = "The platform shall process exactly 500 transactions per second with a latency under 100 ms."
    res = service.analyze_single(req)
    assert res["is_valid"]
    assert res["prediction"] == "Not Ambiguous"
    assert res["probability_pct"] <= 25.0


def test_analyze_single_borderline(service):
    req = "The client application shall handle data securely during all sessions."
    res = service.analyze_single(req)
    assert res["is_valid"]
    # Model returns Ambiguous, Not Ambiguous, or NEEDS REVIEW
    assert res["prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")
    assert 0.0 <= res["probability_pct"] <= 100.0


def test_explanation_engine_isolation(service):
    req = "The system shall be user friendly."
    res = service.analyze_single(req)
    # Ensure explanation is dictionary-like and does not override model prediction
    assert "prediction" in res
    assert res["prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")


# ── 2. Batch Processing & Parsing Tests ──────────────────────────────────────

def test_analyze_batch_records(service):
    records = [
        {"row": 1, "text": "The platform shall process 100 requests per minute."},
        {"row": 2, "text": "Invalid"},  # Too short
        {"row": 3, "text": "The portal shall be user friendly and efficient."},
    ]
    df = service.analyze_batch_records(records)
    assert len(df) == 3
    assert df.iloc[0]["Prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")
    assert df.iloc[1]["Prediction"] == "SKIPPED"
    assert df.iloc[2]["Prediction"] in ("Ambiguous", "Not Ambiguous", "NEEDS REVIEW")


def test_parse_txt_file(service):
    txt_bytes = b"The API shall respond in 50 ms.\n\nThe system shall be fast.\n"
    records = service.parse_file_content(txt_bytes, "test.txt")
    assert len(records) == 2
    assert records[0]["text"] == "The API shall respond in 50 ms."
    assert records[1]["text"] == "The system shall be fast."


def test_parse_xlsx_file(service):
    # Create in-memory Excel workbook
    io_buf = io.BytesIO()
    test_df = pd.DataFrame({"Requirement": ["The module shall encrypt tokens.", "The service shall scale well."]})
    test_df.to_excel(io_buf, index=False, engine="openpyxl")
    
    records = service.parse_file_content(io_buf.getvalue(), "test.xlsx")
    assert len(records) == 2
    assert records[0]["text"] == "The module shall encrypt tokens."


def test_unsupported_file_formats(service):
    with pytest.raises(ValueError, match="Unsupported file format"):
        service.parse_file_content(b"some content", "data.csv")

    with pytest.raises(ValueError, match="Unsupported file format"):
        service.parse_file_content(b"some content", "specs.pdf")

    with pytest.raises(ValueError, match="Unsupported file format"):
        service.parse_file_content(b"some content", "specs.docx")


def test_export_to_excel(service):
    records = [
        {"row": 1, "text": "The system shall process 50 items within 2 seconds."},
    ]
    res_df = service.analyze_batch_records(records)
    excel_bytes = service.export_to_excel(res_df)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    # Verify the exported workbook can be read
    read_df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Analysis Results", engine="openpyxl")
    assert len(read_df) == 1
    assert "Prediction" in read_df.columns


# ── 3. Metadata & Evaluation Retrieval ───────────────────────────────────────

def test_get_model_metadata(service):
    meta = service.get_model_metadata()
    assert "model_version" in meta
    assert meta["model_version"] == "2.1.0"
    assert meta["embedding_dimension"] == 384
    assert meta["upper_threshold"] == 0.75
    assert meta["lower_threshold"] == 0.25


def test_get_evaluation_metrics(service):
    eval_data = service.get_evaluation_metrics()
    assert "verified_readings" in eval_data
    assert "heldout" in eval_data["verified_readings"]
    assert eval_data["verified_readings"]["heldout"]["raw_accuracy"] == 98.15
    assert eval_data["verified_readings"]["challenge"]["raw_accuracy"] == 85.00
    assert eval_data["verified_readings"]["adversarial"]["raw_accuracy"] == 87.50


# ── 4. Excel Multi-Sheet & Custom Column Selection Tests ──────────────────────

def test_excel_sheet_and_column_selection(service):
    # Create multi-sheet workbook in memory
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df1 = pd.DataFrame({"Notes": ["Skip me"], "Custom_Req": ["The system shall log all transactions within 500 ms."]})
        df2 = pd.DataFrame({"Other_Col": [1, 2, 3]})
        df1.to_excel(writer, sheet_name="Spec_Sheet", index=False)
        df2.to_excel(writer, sheet_name="MetaData", index=False)
    file_bytes = buf.getvalue()

    # Test sheet names
    sheet_names = service.get_excel_sheet_names(file_bytes)
    assert sheet_names == ["Spec_Sheet", "MetaData"]

    # Test columns
    cols = service.get_excel_columns(file_bytes, sheet_name="Spec_Sheet")
    assert cols == ["Notes", "Custom_Req"]

    # Test preview
    preview = service.get_excel_preview(file_bytes, sheet_name="Spec_Sheet", n_rows=1)
    assert len(preview) == 1
    assert "Custom_Req" in preview.columns

    # Test parse with explicit sheet and column
    records = service.parse_file_content(file_bytes, "test.xlsx", sheet_name="Spec_Sheet", column_name="Custom_Req")
    assert len(records) == 1
    assert "within 500 ms" in records[0]["text"]

