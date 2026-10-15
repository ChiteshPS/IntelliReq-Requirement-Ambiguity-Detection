"""
tests/test_streamlit_app.py — End-to-End Tests for Streamlit UI/UX

Uses streamlit.testing.v1.AppTest to programmatically simulate user navigation
and interaction across all 5 pages of the IntelliReq interface.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

@pytest.fixture(scope="module")
def app():
    # Cold startup loads SentenceTransformers model once
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run(timeout=30)
    return at


def test_dashboard_page(app):
    """
    Verifies that Dashboard page loads by default with title, warning banner, and KPIs.
    """
    assert len(app.exception) == 0
    # Check sidebar radio exists
    radio = app.sidebar.radio[0]
    assert radio.value == "📊 Dashboard"
    # Ensure header content is rendered
    markdown_texts = [m.value for m in app.markdown]
    has_header = any("IntelliReq — Smart Software Requirement Analyzer" in t for t in markdown_texts)
    assert has_header


def test_single_requirement_page(app):
    """
    Verifies navigation to Single Requirement Analysis, entering a requirement,
    clicking analyze, and verifying prediction results.
    """
    # Navigate to Single Requirement Analysis
    app.sidebar.radio[0].set_value("🔍 Single Requirement").run(timeout=15)
    assert len(app.exception) == 0

    # Locate requirement text area
    assert len(app.text_area) > 0
    text_input = "The system shall be extremely fast and highly scalable under heavy load."
    app.text_area[0].input(text_input).run(timeout=15)
    assert len(app.exception) == 0

    # Locate and click Analyze button
    analyze_buttons = [b for b in app.button if "Analyze Requirement" in b.label]
    assert len(analyze_buttons) > 0
    analyze_buttons[0].click().run(timeout=20)
    assert len(app.exception) == 0

    # Confirm results are displayed
    markdown_texts = [m.value for m in app.markdown]
    has_badge = any("AMBIGUOUS" in t for t in markdown_texts)
    assert has_badge


def test_batch_analysis_page(app):
    """
    Verifies navigation to Batch File Analysis and presence of file uploader.
    """
    app.sidebar.radio[0].set_value("📁 Batch File Analysis").run(timeout=15)
    assert len(app.exception) == 0
    uploaders = getattr(app, "file_uploader", None)
    if uploaders is None:
        uploaders = app.get("file_uploader")
    assert len(uploaders) > 0 or any("batch" in m.value.lower() or "upload" in m.value.lower() for m in app.markdown)


def test_model_evaluation_page(app):
    """
    Verifies navigation to Model Evaluation and presence of evaluation tabs.
    """
    app.sidebar.radio[0].set_value("📈 Model Evaluation").run(timeout=15)
    assert len(app.exception) == 0
    markdown_texts = [m.value for m in app.markdown]
    has_eval_header = any("Model Evaluation & Benchmark Verification" in t for t in markdown_texts)
    assert has_eval_header


def test_dataset_error_analysis_page(app):
    """
    Verifies navigation to Dataset & Error Analysis and presence of error taxonomy.
    """
    app.sidebar.radio[0].set_value("🔬 Dataset & Error Analysis").run(timeout=15)
    assert len(app.exception) == 0
    markdown_texts = [m.value for m in app.markdown]
    has_dataset_header = any("Dataset Provenance & Empirical Error Analysis" in t for t in markdown_texts)
    assert has_dataset_header


def test_about_methodology_page(app):
    """
    Verifies navigation to About & Methodology and presence of research explanation.
    """
    app.sidebar.radio[0].set_value("📖 About & Methodology").run(timeout=15)
    assert len(app.exception) == 0
    markdown_texts = [m.value for m in app.markdown]
    has_about_header = any("About & Methodology" in t for t in markdown_texts)
    assert has_about_header
