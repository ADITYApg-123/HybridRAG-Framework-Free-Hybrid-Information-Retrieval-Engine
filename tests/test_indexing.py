"""Unit tests for DenseEmbedder, VectorStore (FAISS), and BM25Store."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.indexing.bm25_store import BM25Store, technical_tokenize
from src.indexing.embedder import l2_normalize
from src.indexing.faiss_store import VectorStore


def test_l2_normalize_unit_vectors() -> None:
    raw = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]], dtype=np.float32)
    normed = l2_normalize(raw)
    norms = np.linalg.norm(normed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert np.allclose(normed[0], np.array([0.6, 0.8, 0.0], dtype=np.float32))


def test_faiss_vector_store_cosine_search_and_persistence(tmp_path: Path) -> None:
    store = VectorStore(embedding_dim=4)
    # Intentionally pass unnormalized vectors of different magnitudes to verify internal L2 normalization
    docs = np.array(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0],
            [7.0, 7.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    store.build_index(docs)
    assert store.ntotal == 3

    query = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    ranked = store.search_ranked(query, top_k=3)
    # Vector 2 ([7, 7, 0, 0]) has exact cosine similarity 1.0 with [1, 1, 0, 0]
    assert ranked[0][0] == 2
    assert np.isclose(ranked[0][1], 1.0, atol=1e-5)

    idx_file = tmp_path / "test.index"
    store.save(idx_file)
    reloaded = VectorStore.load(idx_file)
    assert reloaded.ntotal == 3
    reloaded_ranked = reloaded.search_ranked(query, top_k=1)
    assert reloaded_ranked[0][0] == 2


def test_technical_tokenizer_and_bm25_store(tmp_path: Path) -> None:
    tokens = technical_tokenize("class torch.nn.Conv2d(in_channels, padding='same', eps=1e-05)")
    assert "torch.nn.conv2d" in tokens
    assert "conv2d" in tokens
    assert "conv" in tokens
    assert "in_channels" in tokens
    assert "1e-05" in tokens

    corpus = [
        "torch.nn.Conv2d applies 2D convolution with in_channels and groups for depthwise convolution.",
        "torch.optim.AdamW decouples weight_decay from adaptive gradient moment updates.",
        "torch.cuda.empty_cache releases unused cached memory blocks from the caching allocator.",
    ]
    bm25 = BM25Store()
    bm25.build_index(corpus)
    assert bm25.ntotal == 3

    ranked = bm25.search_ranked("AdamW decoupled weight_decay", top_k=2)
    assert ranked[0][0] == 1
    assert ranked[0][1] > 0.0

    pkl_path = tmp_path / "bm25.pkl"
    bm25.save(pkl_path)
    loaded = BM25Store.load(pkl_path)
    assert loaded.search_ranked("empty_cache caching allocator", top_k=1)[0][0] == 2
