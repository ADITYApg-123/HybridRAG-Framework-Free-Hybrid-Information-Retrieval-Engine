"""Unit tests for Reciprocal Rank Fusion (ranker.py) and HybridRetriever (hybrid_search.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.indexing.bm25_store import BM25Store
from src.indexing.faiss_store import VectorStore
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.ranker import reciprocal_rank_fusion, reciprocal_rank_fusion_detailed


class _DummyEmbedder:
    """Deterministic lightweight embedder for fast unit testing."""

    embedding_dim = 4

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        if "conv2d" in query.lower():
            return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        if "adamw" in query.lower():
            return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)


def test_rrf_boosts_consensus_documents() -> None:
    # Doc 2 is rank 2 in both Dense and Sparse; Doc 0 is rank 1 in Dense only; Doc 1 is rank 1 in Sparse only
    dense = [(0, 0.95), (2, 0.88), (3, 0.70)]
    sparse = [(1, 14.2), (2, 11.5), (4, 5.1)]

    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)
    # Score(2) = 1/62 + 1/62 = 2/62 = 0.032258 > Score(0) = 1/61 = 0.016393
    assert fused[0][0] == 2
    assert np.isclose(fused[0][1], 2.0 / 62.0)

    details = reciprocal_rank_fusion_detailed(dense, sparse, rrf_k=60, top_k=2)
    assert len(details) == 2
    assert details[0].doc_int_id == 2
    assert details[0].dense_rank == 2
    assert details[0].sparse_rank == 2


def test_rrf_ignores_zero_sparse_scores_by_default() -> None:
    dense = [(0, 0.80)]
    sparse = [(1, 0.0), (2, 0.0)]
    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)
    assert len(fused) == 1
    assert fused[0][0] == 0


def test_hybrid_retriever_facade() -> None:
    docs = [
        {
            "chunk_id": "doc0::chunk_0",
            "doc_id": "doc0",
            "int_id": 0,
            "text": "torch.nn.Conv2d applies 2D spatial convolution with groups and dilation.",
            "raw_text": "torch.nn.Conv2d applies 2D spatial convolution.",
            "module": "torch.nn",
            "symbol": "Conv2d",
            "section": "sig",
            "title": "Conv2d",
            "source_file": "conv.html",
        },
        {
            "chunk_id": "doc1::chunk_0",
            "doc_id": "doc1",
            "int_id": 1,
            "text": "torch.optim.AdamW decouples weight_decay from moment estimates.",
            "raw_text": "torch.optim.AdamW decouples weight_decay.",
            "module": "torch.optim",
            "symbol": "AdamW",
            "section": "math",
            "title": "AdamW",
            "source_file": "adamw.html",
        },
    ]

    vs = VectorStore(embedding_dim=4)
    vs.build_index(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32))

    bm25 = BM25Store()
    bm25.build_index([d["text"] for d in docs])

    retriever = HybridRetriever(
        embedder=_DummyEmbedder(),  # type: ignore[arg-type]
        faiss_store=vs,
        bm25_store=bm25,
        chunk_metadata=docs,
        rrf_k=60,
        top_k_fetch=10,
        top_k_final=2,
    )

    bundle = retriever.retrieve_all_modalities("torch.nn.Conv2d groups dilation", top_k=2)
    assert bundle["hybrid"][0]["doc_id"] == "doc0"
    assert bundle["dense"][0]["doc_id"] == "doc0"
    assert bundle["sparse"][0]["doc_id"] == "doc0"
    assert bundle["latency_ms"] >= 0.0

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], rrf_k=0)
