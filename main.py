"""
main.py — IntelliReq Terminal Interface

Accepted input methods:
  1. A single requirement sentence typed in the terminal
  2. A .txt file (one requirement per non-empty line)
  3. An .xlsx file (column named Requirement / Requirement_Text / requirement)

Output for batch jobs is saved to: outputs/<filename>_results.xlsx
"""
import os
import sys
import json
import warnings
import pandas as pd
import joblib

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings("ignore")

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor
from models.predictor import ModelPredictor
from rule_engine.explanation_engine import ExplanationEngine
from preprocessing.input_parser import parse_input_file

OUTPUTS_DIR = "outputs"
ACCEPTED_SENTENCE_HEADERS = ["Requirement", "Requirement_Text", "requirement"]


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_requirement(text: str) -> bool:
    """
    A requirement must be a non-empty string with at least 3 words.
    Rejects None, empty strings, and trivially short fragments.
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    return bool(text) and len(text.split()) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# CLI class
# ─────────────────────────────────────────────────────────────────────────────

class IntelliReqCLI:
    def __init__(self):
        self._load_model()
        self.feat_extractor = FeatureExtractor()
        self.sbert_extractor = SBERTExtractor()
        self.explanation_engine = ExplanationEngine()
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

    def _load_model(self):
        model_path = "saved_models/model.joblib"
        scaler_path = "saved_models/scaler.joblib"
        meta_path = "saved_models/metadata.json"

        missing = [p for p in (model_path, scaler_path, meta_path) if not os.path.exists(p)]
        if missing:
            print("\nERROR: The following model files are missing:")
            for p in missing:
                print(f"  {p}")
            print("Run  python train.py  to generate model artifacts, then try again.")
            sys.exit(1)

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(meta_path, "r") as fh:
            self.metadata = json.load(fh)

        upper = float(self.metadata.get("upper_threshold", 0.75))
        lower = float(self.metadata.get("lower_threshold", 0.25))
        self.predictor = ModelPredictor(
            trained_model=self.model,
            upper_threshold=upper,
            lower_threshold=lower,
        )

    # ── Core inference ───────────────────────────────────────────────────────

    def _process_text(self, text: str) -> dict:
        """
        Full pipeline: text → features → scale → predict → explain.
        Returns a flat dict ready for display or tabular export.
        """
        ling_feats = self.feat_extractor.extract_features(text)
        X_ling = pd.DataFrame([ling_feats])
        X_sbert = self.sbert_extractor.transform(pd.Series([text]))

        X_final = pd.concat(
            [X_sbert.reset_index(drop=True), X_ling.reset_index(drop=True)],
            axis=1,
        )
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_final), columns=X_final.columns
        )

        preds, prob_df = self.predictor.predict(X_scaled)
        predicted_class = preds.iloc[0]

        # Identify the Ambiguous probability column (percentage)
        amb_col = next(
            (c for c in prob_df.columns if "Ambiguous" in c and "Not" not in c),
            prob_df.columns[0],
        )
        amb_pct = prob_df[amb_col].iloc[0]

        explanation = self.explanation_engine.generate_explanation(text, predicted_class)

        return {
            "text": text,
            "prediction": predicted_class,
            "ambiguity_probability_pct": round(amb_pct, 2),
            "reason": explanation["Reason"],
            "missing": explanation["Missing_Specification"],
            "suggested_rewrite": explanation["Suggested_Rewrite"],
        }

    def process_text(self, text: str):
        """
        Public stable prediction API.
        Returns:
            tuple: (prediction_label, ambiguity_probability_pct, explanation_dict, feature_dict)
        """
        result = self._process_text(text)
        explanation = {
            "Reason": result["reason"],
            "Missing_Specification": result["missing"],
            "Suggested_Rewrite": result["suggested_rewrite"],
        }
        feats = self.feat_extractor.extract_features(text)
        return (
            result["prediction"],
            result["ambiguity_probability_pct"],
            explanation,
            feats,
        )

    def _run_batch_dataframe(self, df: pd.DataFrame, out_path: str):
        """
        Batch processing helper for dataframes.
        Validates non-empty input and accepted requirement headers.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        req_col = next(
            (c for c in ACCEPTED_SENTENCE_HEADERS if c in df.columns),
            None,
        )
        if not req_col:
            raise KeyError(
                f"Missing requirement column. Expected one of: {ACCEPTED_SENTENCE_HEADERS}"
            )

        results = []
        for idx, row in df.iterrows():
            text = str(row[req_col]) if pd.notnull(row[req_col]) else ""
            if not is_valid_requirement(text):
                continue
            res = self._process_text(text)
            results.append({
                "Row": idx + 1,
                "Requirement": text,
                "Prediction": res["prediction"],
                "Ambiguity_Probability_%": res["ambiguity_probability_pct"],
                "Reason": res["reason"],
                "Suggested_Rewrite": res["suggested_rewrite"] or "",
            })

        out_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        if str(out_path).endswith(".xlsx"):
            out_df.to_excel(out_path, index=False)
        else:
            out_df.to_csv(out_path, index=False)
        return out_df

    # ── Display helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _print_separator():
        print("-" * 70)

    def _display_result(self, result: dict, row_id: str = None):
        self._print_separator()
        if row_id:
            print(f"  Row {row_id}")
        print(f"  Requirement       : {result['text']}")
        print(f"  Prediction        : {result['prediction'].upper()}")
        print(f"  Ambiguity Prob.   : {result['ambiguity_probability_pct']:.1f}%")
        decision = result["prediction"]
        if decision == "NEEDS REVIEW":
            print(f"  Decision Status   : [?] NEEDS HUMAN REVIEW")
        elif decision == "Ambiguous":
            print(f"  Decision Status   : [X] AMBIGUOUS")
        else:
            print(f"  Decision Status   : [OK] NOT AMBIGUOUS")
        print(f"  Reason            : {result['reason']}")
        if result["missing"]:
            print(f"  Missing Specs     : {'; '.join(result['missing'])}")
        if result["suggested_rewrite"]:
            print(f"  Suggested Rewrite : {result['suggested_rewrite']}")
        self._print_separator()

    # ── Menu ─────────────────────────────────────────────────────────────────

    def print_header(self):
        print("\n" + "=" * 70)
        print("   INTELLIREQ — SMART SOFTWARE REQUIREMENT ANALYZER")
        print("=" * 70)

    def main_menu(self):
        while True:
            self.print_header()
            print("\n  1. Analyze one requirement sentence")
            print("  2. Analyze requirements from a TXT file")
            print("  3. Analyze requirements from an XLSX file")
            print("  4. Exit\n")

            choice = input("  Select an option (1-4): ").strip()

            if choice == "1":
                self._analyze_single()
            elif choice == "2":
                self._analyze_file("txt")
            elif choice == "3":
                self._analyze_file("xlsx")
            elif choice == "4":
                print("\n  Goodbye!\n")
                break
            else:
                print("\n  Invalid choice. Please enter 1, 2, 3, or 4.")

    # ── Option 1: Single sentence ────────────────────────────────────────────

    def _analyze_single(self):
        print("\n  ANALYZE ONE REQUIREMENT")
        print("  (Enter 'back' to return to the menu)\n")

        while True:
            text = input("  Enter requirement:\n  > ").strip()
            if text.lower() in ("back", "b", "q", "quit"):
                break
            if not is_valid_requirement(text):
                print(
                    "\n  ⚠ Input too short or empty. "
                    "A requirement must have at least 3 words.\n"
                )
                continue
            try:
                result = self._process_text(text)
                self._display_result(result)
            except Exception as exc:
                print(f"\n  ERROR during analysis: {exc}\n")

    # ── Options 2 & 3: Batch file ────────────────────────────────────────────

    def _analyze_file(self, fmt: str):
        label = "TXT" if fmt == "txt" else "XLSX"
        print(f"\n  ANALYZE REQUIREMENTS FROM {label} FILE")
        print("  (Enter 'back' to return to the menu)\n")

        path_raw = input(f"  Enter path to {label} file:\n  > ").strip()

        # Strip surrounding quotes pasted from Explorer
        path = path_raw.strip('"').strip("'")

        if path.lower() in ("back", "b", "q", "quit"):
            return

        # Validate extension
        ext = os.path.splitext(path)[-1].lower()
        expected_ext = f".{fmt}"
        if ext != expected_ext:
            print(
                f"\n  ⚠ Expected a {expected_ext} file, got '{ext}'. "
                f"Only .txt and .xlsx are supported.\n"
            )
            return

        if not os.path.isfile(path):
            print(f"\n  ⚠ File not found: {path}\n")
            return

        # Parse the file
        try:
            rows = parse_input_file(path)
        except ValueError as exc:
            print(f"\n  ⚠ {exc}\n")
            return
        except FileNotFoundError as exc:
            print(f"\n  ⚠ {exc}\n")
            return

        print(f"\n  Found {len(rows)} requirement(s). Analyzing …\n")

        results = []
        skipped = 0
        for item in rows:
            text = item["text"]
            row_num = item["row"]

            if not is_valid_requirement(text):
                skipped += 1
                results.append(
                    {
                        "Row": row_num,
                        "Requirement": text,
                        "Prediction": "SKIPPED",
                        "Ambiguity_Probability_%": "",
                        "Decision_Status": "Too short / invalid",
                        "Reason": "",
                        "Missing_Specifications": "",
                        "Suggested_Rewrite": "",
                    }
                )
                continue

            try:
                r = self._process_text(text)
                decision = (
                    "NEEDS HUMAN REVIEW"
                    if r["prediction"] == "NEEDS REVIEW"
                    else r["prediction"].upper()
                )
                results.append(
                    {
                        "Row": row_num,
                        "Requirement": r["text"],
                        "Prediction": r["prediction"],
                        "Ambiguity_Probability_%": r["ambiguity_probability_pct"],
                        "Decision_Status": decision,
                        "Reason": r["reason"],
                        "Missing_Specifications": "; ".join(r["missing"]),
                        "Suggested_Rewrite": r["suggested_rewrite"] or "",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "Row": row_num,
                        "Requirement": text,
                        "Prediction": "ERROR",
                        "Ambiguity_Probability_%": "",
                        "Decision_Status": "Processing error",
                        "Reason": str(exc),
                        "Missing_Specifications": "",
                        "Suggested_Rewrite": "",
                    }
                )

        # Summary
        out_df = pd.DataFrame(results)
        total = len(out_df)
        amb = (out_df["Prediction"] == "Ambiguous").sum()
        not_amb = (out_df["Prediction"] == "Not Ambiguous").sum()
        review = (out_df["Prediction"] == "NEEDS REVIEW").sum()
        errors = (out_df["Prediction"] == "ERROR").sum()

        print(f"\n  ── SUMMARY ──────────────────────────────────────────────")
        print(f"  Total rows processed  : {total}")
        print(f"  Ambiguous             : {amb}")
        print(f"  Not Ambiguous         : {not_amb}")
        print(f"  Needs Review          : {review}")
        print(f"  Skipped (too short)   : {skipped}")
        if errors:
            print(f"  Errors                : {errors}")
        print(f"  ─────────────────────────────────────────────────────────")

        # Save output
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUTPUTS_DIR, f"{base_name}_results.xlsx")

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Processed",
                    "Ambiguous",
                    "Not Ambiguous",
                    "Needs Review",
                    "Skipped (too short)",
                    "Errors",
                ],
                "Count": [total, amb, not_amb, review, skipped, errors],
            }
        )

        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                out_df.to_excel(writer, sheet_name="Analysis Results", index=False)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
            print(f"\n  Results saved → {out_path}\n")
        except Exception as exc:
            print(f"\n  ⚠ Could not save XLSX ({exc}). Saving CSV instead.")
            csv_path = out_path.replace(".xlsx", ".csv")
            out_df.to_csv(csv_path, index=False)
            print(f"  Results saved → {csv_path}\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli = IntelliReqCLI()
    cli.main_menu()
