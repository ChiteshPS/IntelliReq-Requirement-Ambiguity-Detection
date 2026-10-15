"""
feature_engineering/sbert_extractor.py

SBERT Feature Extractor with deterministic offline resolution:
1. Prefers bundled local model in saved_models/all-MiniLM-L6-v2
2. Checks user cache directory
3. Falls back to online load only if internet is available
4. Never silently substitutes random embeddings or unsupported models
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Suppress noisy download warnings when loading cached models
warnings.filterwarnings("ignore", category=FutureWarning)


def resolve_local_model_path(model_name: str = "all-MiniLM-L6-v2") -> str:
    """
    Resolves the local path for the embedding model to ensure 100% offline
    operation without Hugging Face timeouts or network checks.
    """
    candidates = [
        # 1. Project-bundled model
        os.path.join(os.path.dirname(__file__), "..", "saved_models", "all-MiniLM-L6-v2"),
        os.path.join(os.getcwd(), "saved_models", "all-MiniLM-L6-v2"),
        # 2. Local huggingface cache snapshot
        os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
        os.path.expanduser("~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2"),
    ]

    for path in candidates:
        if os.path.exists(path) and (
            os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "pytorch_model.bin"))
        ):
            return os.path.abspath(path)

    return model_name


class SBERTExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", force_offline: bool = False):
        """
        Initializes the SBERT Feature Extractor.
        Automatically uses local offline snapshot when available.
        """
        self.model_name = model_name
        self.resolved_path = resolve_local_model_path(model_name)
        self.is_local = os.path.isdir(self.resolved_path)

        try:
            # When a local directory is found, load strictly from disk
            self.model = SentenceTransformer(self.resolved_path)
        except Exception as exc:
            if force_offline or not self.is_local:
                raise RuntimeError(
                    f"CRITICAL: Failed to load SBERT model '{model_name}'. "
                    f"Resolved path: '{self.resolved_path}'. "
                    f"To use offline, ensure 'saved_models/all-MiniLM-L6-v2' is populated. "
                    f"Original error: {exc}"
                ) from exc
            # Fallback if local path had an issue
            self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self

    def transform(self, texts) -> pd.DataFrame:
        """
        Generates SBERT embeddings for a list/series of texts.
        Output dimension: (N, 384) with columns sbert_0 to sbert_383.
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        elif isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, show_progress_bar=False)

        # Ensure 2D numpy array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        df_embeddings = pd.DataFrame(
            embeddings,
            columns=[f"sbert_{i}" for i in range(embeddings.shape[1])]
        )
        return df_embeddings

    def fit_transform(self, texts) -> pd.DataFrame:
        return self.transform(texts)


if __name__ == "__main__":
    extractor = SBERTExtractor()
    sample_text = [
        "The system shall respond quickly.",
        "The system shall return a response within 2 seconds."
    ]
    df = extractor.transform(sample_text)
    print(f"Model Identifier : {extractor.model_name}")
    print(f"Resolved Path    : {extractor.resolved_path}")
    print(f"Is Local Snapshot: {extractor.is_local}")
    print(f"Embedding Shape  : {df.shape}")
    print(f"First 5 values   : {df.iloc[0, :5].values}")
