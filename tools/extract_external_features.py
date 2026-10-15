"""
tools/extract_external_features.py

Extracts and caches 384-d SBERT dense embeddings and 24 handcrafted linguistic/specificity
features for external datasets:
- data/external/userstories_labeled.csv (N=1,680)
- data/external/pure_labeled.csv (N=6,725)

Uses saved_models/all-MiniLM-L6-v2 offline model.
Caches matrices into saved_models/benchmark_cache/
"""

import os
import sys
import gc
import time
import joblib
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.sbert_extractor import SBERTExtractor

CACHE_DIR = os.path.join(REPO_ROOT, "saved_models", "benchmark_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def extract_sbert_batched(sbert_model, texts: list[str], chunk_size: int = 256) -> pd.DataFrame:
    all_embs = []
    total = len(texts)
    print(f"  Encoding {total} texts in chunks of {chunk_size}...")
    for i in range(0, total, chunk_size):
        chunk = texts[i : i + chunk_size]
        emb = sbert_model.model.encode(chunk, batch_size=64, show_progress_bar=False)
        all_embs.append(emb)
        if (i // chunk_size) % 5 == 0 or i + chunk_size >= total:
            print(f"    Encoded {min(i + chunk_size, total)} / {total}...")
        gc.collect()

    merged = np.vstack(all_embs)
    return pd.DataFrame(merged, columns=[f"sbert_{i}" for i in range(merged.shape[1])])


def extract_and_cache(df: pd.DataFrame, text_col: str, dataset_key: str, sbert: SBERTExtractor, fe: FeatureExtractor):
    sbert_cache = os.path.join(CACHE_DIR, f"{dataset_key}_sbert.joblib")
    ling_cache = os.path.join(CACHE_DIR, f"{dataset_key}_ling.joblib")

    texts = df[text_col].astype(str).tolist()

    # 1. SBERT embeddings
    if os.path.exists(sbert_cache):
        print(f"Loading cached SBERT embeddings for '{dataset_key}' from {sbert_cache}...")
        df_sbert = joblib.load(sbert_cache)
    else:
        print(f"\n[1/2] Extracting SBERT embeddings for '{dataset_key}' ({len(texts)} texts)...")
        t0 = time.time()
        df_sbert = extract_sbert_batched(sbert, texts, chunk_size=256)
        joblib.dump(df_sbert, sbert_cache, compress=3)
        print(f"Saved {sbert_cache} in {time.time()-t0:.2f}s (Shape: {df_sbert.shape})")

    # 2. Handcrafted features
    if os.path.exists(ling_cache):
        print(f"Loading cached linguistic features for '{dataset_key}' from {ling_cache}...")
        df_ling = joblib.load(ling_cache)
    else:
        print(f"\n[2/2] Extracting 24 handcrafted features for '{dataset_key}' ({len(df)} rows)...")
        t0 = time.time()
        df_ling = fe.process_dataframe(df, text_col).drop(columns=df.columns)
        joblib.dump(df_ling, ling_cache, compress=3)
        print(f"Saved {ling_cache} in {time.time()-t0:.2f}s (Shape: {df_ling.shape})")

    gc.collect()
    return df_sbert, df_ling


def main():
    print("=" * 70)
    print("EXTRACTING FEATURES FOR EXTERNAL REPLICATION DATASETS")
    print("=" * 70)

    print("Initializing feature extractors...")
    sbert = SBERTExtractor()
    fe = FeatureExtractor()

    # 1. User Stories
    us_path = os.path.join(REPO_ROOT, "data", "external", "userstories_labeled.csv")
    if os.path.exists(us_path):
        df_us = pd.read_csv(us_path)
        print(f"\nProcessing User Stories: {us_path} ({len(df_us)} rows)")
        extract_and_cache(df_us, "requirement_text", "ext_userstories", sbert, fe)

    # 2. PURE
    pure_path = os.path.join(REPO_ROOT, "data", "external", "pure_labeled.csv")
    if os.path.exists(pure_path):
        df_pure = pd.read_csv(pure_path)
        print(f"\nProcessing PURE: {pure_path} ({len(df_pure)} rows)")
        extract_and_cache(df_pure, "requirement_text", "ext_pure", sbert, fe)

    print("\n" + "=" * 70)
    print("EXTERNAL FEATURE EXTRACTION AND CACHING SUCCESSFUL!")
    print("=" * 70)


if __name__ == "__main__":
    main()
