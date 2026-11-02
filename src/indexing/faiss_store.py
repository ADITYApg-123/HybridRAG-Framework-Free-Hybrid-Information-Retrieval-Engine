"""FAISS IndexFlatIP dense vector store for exact Cosine Similarity retrieval.

Enforces L2 vector normalization on both indexed document embeddings and runtime query
vectors so inner product (`faiss.IndexFlatIP`) is mathematically equivalent to Cosine Similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.indexing.embedder import l2_normalize

logger = logging.getLogger("hybrid_rag.indexing.faiss_store")


class VectorStore:
    """Wraps `faiss.IndexFlatIP` for dense vector indexing, persistence, and search."""

    def __init__(self, embedding_dim: int = 384) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")
        self.embedding_dim = embedding_dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)

    @property
    def ntotal(self) -> int:
        """Return total number of vectors currently stored in the FAISS index."""
        return int(self.index.ntotal)

    def build_index(self, embeddings: np.ndarray) -> None:
        """Reset and populate the FAISS IndexFlatIP with L2-normalized embeddings."""
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings of shape (N, {self.embedding_dim}), got {arr.shape}"
            )

        normalized = l2_normalize(arr)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(normalized)
        logger.info(
            "Built FAISS IndexFlatIP with %d vectors (dim=%d).",
            self.ntotal,
            self.embedding_dim,
        )

    def search(
        self, query_embedding: np.ndarray, top_k: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index for Top-K nearest neighbors by inner product (cosine).

        Returns:
            Tuple of (scores, indices), each of shape (1, k).
        """
        if self.ntotal == 0:
            raise RuntimeError("FAISS index is empty. Call build_index() or load() first.")

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dim {q.shape[1]} does not match index dim {self.embedding_dim}"
            )

        q_norm = l2_normalize(q)
        k = min(max(1, top_k), self.ntotal)
        scores, indices = self.index.search(q_norm, k)
        return scores, indices

    def search_ranked(
        self, query_embedding: np.ndarray, top_k: int = 60
    ) -> List[Tuple[int, float]]:
        """Return a sorted list of `(chunk_int_idx, cosine_score)` tuples for RRF."""
        scores, indices = self.search(query_embedding, top_k=top_k)
        results: List[Tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if int(idx) >= 0:
                results.append((int(idx), float(score)))
        return results

    def save(self, index_path: str | Path) -> None:
        """Write the FAISS index to disk."""
        out = Path(index_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out))
        logger.info("Saved FAISS index (%d vectors) to %s", self.ntotal, out)

    @classmethod
    def load(cls, index_path: str | Path) -> "VectorStore":
        """Load a persisted FAISS index from disk."""
        inp = Path(index_path)
        if not inp.exists():
            raise FileNotFoundError(f"FAISS index file not found: {inp}")
        loaded_index = faiss.read_index(str(inp))
        store = cls(embedding_dim=int(loaded_index.d))
        store.index = loaded_index
        logger.info("Loaded FAISS index (%d vectors, dim=%d) from %s", store.ntotal, store.embedding_dim, inp)
        return store


FAISSVectorStore = VectorStore
