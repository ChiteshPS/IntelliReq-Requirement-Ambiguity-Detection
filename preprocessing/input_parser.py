"""
preprocessing/input_parser.py

Handles the two supported batch input formats:
  1. TXT  — one requirement per non-empty line
  2. XLSX — requirement text in a column named 'Requirement', 'Requirement_Text', or 'requirement'

Unsupported formats raise ValueError with a clear message.
"""
import os
import pandas as pd

ACCEPTED_XLSX_HEADERS = ["Requirement", "Requirement_Text", "requirement"]


def parse_txt(path: str) -> list[dict]:
    """
    Reads a plain-text file.  Each non-empty line is one requirement.
    Returns a list of dicts: {'row': int, 'text': str}
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    results = []
    for i, line in enumerate(lines, start=1):
        text = line.strip()
        if text:
            results.append({"row": i, "text": text})

    if not results:
        raise ValueError(f"TXT file is empty or contains only blank lines: {path}")

    return results


def parse_xlsx(path: str) -> list[dict]:
    """
    Reads an Excel (.xlsx) file.
    Looks for a requirement column by accepted names (case-insensitive).
    Returns a list of dicts: {'row': int, 'text': str}
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"Could not read Excel file '{path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Excel file is empty: {path}")

    # Find requirement column (case-insensitive match)
    col_map = {c.strip().lower(): c for c in df.columns}
    req_col = None
    for header in ACCEPTED_XLSX_HEADERS:
        if header.lower() in col_map:
            req_col = col_map[header.lower()]
            break

    if req_col is None:
        raise ValueError(
            f"No requirement column found in '{path}'.\n"
            f"  Expected one of: {ACCEPTED_XLSX_HEADERS}\n"
            f"  Found columns:   {list(df.columns)}"
        )

    results = []
    for i, val in enumerate(df[req_col], start=1):
        text = str(val).strip() if pd.notna(val) else ""
        if text and text.lower() not in ("nan", "none", ""):
            results.append({"row": i, "text": text})

    if not results:
        raise ValueError(
            f"Requirement column '{req_col}' has no non-empty rows in '{path}'."
        )

    return results


def parse_input_file(path: str) -> list[dict]:
    """
    Dispatcher: routes to parse_txt or parse_xlsx based on extension.
    Raises ValueError for unsupported formats.
    """
    if not isinstance(path, str):
        raise TypeError(f"Path must be a string, got {type(path)}")

    ext = os.path.splitext(path)[-1].lower()

    if ext == ".txt":
        return parse_txt(path)
    elif ext == ".xlsx":
        return parse_xlsx(path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            "Only .txt and .xlsx files are accepted."
        )
