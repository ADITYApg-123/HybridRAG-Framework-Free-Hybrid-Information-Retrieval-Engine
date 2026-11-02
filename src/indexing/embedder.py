"""Dense embedding generator wrapping SentenceTransformers with explicit L2 normalization.

Ensures all output vectors are C-contiguous float32 arrays normalized to unit L2 length
(||v||_2 = 1.0) so that inner product in FAISS `IndexFlatIP` computes exact Cosine Similarity.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("hybrid_rag.indexing.embedder")


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 1D or 2D NumPy array along the last axis.

    Formula: v_hat = v / max(||v||_2, eps)
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return np.ascontiguousarray(arr / max(norm, eps), dtype=np.float32)
    if arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return np.ascontiguousarray(arr / norms, dtype=np.float32)
    raise ValueError(f"Expected 1D or 2D array for L2 normalization, got shape {arr.shape}")


class DenseEmbedder:
    """Wrapper around SentenceTransformer (`all-MiniLM-L6-v2`) with batching & L2 normalization."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info("Loading SentenceTransformer model '%s'...", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim: int = int(self.model.get_sentence_embedding_dimension() or 384)

    def embed_texts(
        self,
        texts: Sequence[str],
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a sequence of document chunks into a 2D float32 array of shape (N, D)."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        raw_vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vectors = np.asarray(raw_vectors, dtype=np.float32)
        if normalize:
            vectors = l2_normalize(vectors)
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Encode a single query string into a 2D float32 array of shape (1, D)."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
        return self.embed_texts([query.strip()], normalize=normalize, show_progress_bar=False)
