"""Reciprocal Rank Fusion (RRF) implementation from first principles.

Combines ranked retrieval lists from heterogeneous engines (Dense Cosine Similarity
from FAISS and Sparse lexical relevance from Okapi BM25) without score calibration:

    RRF_Score(d) = 1 / (rrf_k + rank_dense(d)) + 1 / (rrf_k + rank_sparse(d))

where ranks are 1-indexed (1, 2, 3, ...).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class FusedRankDetail:
    """Detailed rank and score attribution for a single fused document chunk."""

    doc_int_id: int
    rrf_score: float
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def reciprocal_rank_fusion(
    dense_results: Sequence[Tuple[int, float]],
    sparse_results: Sequence[Tuple[int, float]],
    rrf_k: int = 60,
    top_k: int | None = None,
    ignore_zero_sparse: bool = True,
) -> List[Tuple[int, float]]:
    """Fuse dense and sparse ranked `(doc_int_id, score)` lists via Reciprocal Rank Fusion.

    Args:
        dense_results: Ranked list of `(doc_int_id, cosine_score)` sorted descending.
        sparse_results: Ranked list of `(doc_int_id, bm25_score)` sorted descending.
        rrf_k: Smoothing constant `k` (default 60 per Cormack et al., SIGIR 2009).
        top_k: Optional cutoff for number of fused results returned.
        ignore_zero_sparse: If True, excludes sparse hits with score <= 0.0 from receiving
            rank points, since a BM25 score of 0.0 indicates zero lexical term overlap.

    Returns:
        List of `(doc_int_id, rrf_score)` sorted by `rrf_score` descending.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer.")

    details = reciprocal_rank_fusion_detailed(
        dense_results=dense_results,
        sparse_results=sparse_results,
        rrf_k=rrf_k,
        top_k=top_k,
        ignore_zero_sparse=ignore_zero_sparse,
    )
    return [(item.doc_int_id, item.rrf_score) for item in details]


def reciprocal_rank_fusion_detailed(
    dense_results: Sequence[Tuple[int, float]],
    sparse_results: Sequence[Tuple[int, float]],
    rrf_k: int = 60,
    top_k: int | None = None,
    ignore_zero_sparse: bool = True,
) -> List[FusedRankDetail]:
    """Compute RRF scores and retain per-engine rank and raw score attribution."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer.")

    rrf_scores: Dict[int, float] = defaultdict(float)
    dense_rank_map: Dict[int, int] = {}
    dense_score_map: Dict[int, float] = {}
    sparse_rank_map: Dict[int, int] = {}
    sparse_score_map: Dict[int, float] = {}

    for rank_idx, (doc_id, score) in enumerate(dense_results, start=1):
        doc_int = int(doc_id)
        if doc_int not in dense_rank_map:
            dense_rank_map[doc_int] = rank_idx
            dense_score_map[doc_int] = float(score)
            rrf_scores[doc_int] += 1.0 / (rrf_k + rank_idx)

    for rank_idx, (doc_id, score) in enumerate(sparse_results, start=1):
        doc_int = int(doc_id)
        if ignore_zero_sparse and float(score) <= 0.0:
            continue
        if doc_int not in sparse_rank_map:
            sparse_rank_map[doc_int] = rank_idx
            sparse_score_map[doc_int] = float(score)
            rrf_scores[doc_int] += 1.0 / (rrf_k + rank_idx)

    # Sort descending by RRF score; tie-break by best dense rank then doc_id
    sorted_doc_ids = sorted(
        rrf_scores.keys(),
        key=lambda d: (
            -rrf_scores[d],
            dense_rank_map.get(d, 10**9),
            sparse_rank_map.get(d, 10**9),
            d,
        ),
    )

    if top_k is not None and top_k > 0:
        sorted_doc_ids = sorted_doc_ids[:top_k]

    return [
        FusedRankDetail(
            doc_int_id=doc_id,
            rrf_score=float(rrf_scores[doc_id]),
            dense_rank=dense_rank_map.get(doc_id),
            dense_score=dense_score_map.get(doc_id),
            sparse_rank=sparse_rank_map.get(doc_id),
            sparse_score=sparse_score_map.get(doc_id),
        )
        for doc_id in sorted_doc_ids
    ]
