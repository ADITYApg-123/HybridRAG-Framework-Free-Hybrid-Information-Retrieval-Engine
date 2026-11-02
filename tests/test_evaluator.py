"""Unit tests for IR evaluation metrics (Recall@K, MRR, Precision@K) and IREvaluator."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.evaluator import IREvaluator, precision_at_k, recall_at_k, reciprocal_rank


def test_ir_math_metrics() -> None:
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]

    assert recall_at_k(retrieved, "doc_c", k=5) == 1.0
    assert recall_at_k(retrieved, "doc_c", k=2) == 0.0
    assert recall_at_k(retrieved, "doc_missing", k=5) == 0.0

    assert np.isclose(reciprocal_rank(retrieved, "doc_a"), 1.0)
    assert np.isclose(reciprocal_rank(retrieved, "doc_b"), 0.5)
    assert np.isclose(reciprocal_rank(retrieved, "doc_c"), 1.0 / 3.0)
    assert reciprocal_rank(retrieved, "doc_missing") == 0.0

    assert np.isclose(precision_at_k(retrieved, "doc_b", k=5), 0.2)
    assert precision_at_k(retrieved, "doc_missing", k=5) == 0.0

    with pytest.raises(ValueError):
        recall_at_k(retrieved, "doc_a", k=0)
    with pytest.raises(ValueError):
        precision_at_k(retrieved, "doc_a", k=0)


class _MockRetriever:
    """Mock retriever demonstrating complementary Sparse + Dense strengths fused by RRF."""

    def retrieve_all_modalities(
        self, query: str, top_k: int = 5, rrf_k: int = 60
    ) -> dict:
        if "lexical" in query:
            return {
                "latency_ms": 4.2,
                "sparse": [{"doc_id": "doc_lex"}, {"doc_id": "doc_other"}],
                "dense": [{"doc_id": "doc_other"}, {"doc_id": "doc_lex"}],
                "hybrid": [{"doc_id": "doc_lex"}, {"doc_id": "doc_other"}],
            }
        return {
            "latency_ms": 4.5,
            "sparse": [{"doc_id": "doc_other"}, {"doc_id": "doc_sem"}],
            "dense": [{"doc_id": "doc_sem"}, {"doc_id": "doc_other"}],
            "hybrid": [{"doc_id": "doc_sem"}, {"doc_id": "doc_other"}],
        }


def test_ir_evaluator_aggregation() -> None:
    evaluator = IREvaluator(retriever=_MockRetriever())  # type: ignore[arg-type]
    golden = [
        {"query_id": "Q1", "category": "lex", "query": "lexical query", "expected_doc_id": "doc_lex"},
        {"query_id": "Q2", "category": "sem", "query": "semantic query", "expected_doc_id": "doc_sem"},
    ]

    report = evaluator.evaluate_dataset(golden, top_k=2, rrf_k=60)
    assert report["num_queries"] == 2
    assert report["summary"]["sparse"]["recall@1"] == 0.5
    assert report["summary"]["dense"]["recall@1"] == 0.5
    assert report["summary"]["hybrid"]["recall@1"] == 1.0
    assert report["summary"]["hybrid"]["mrr"] == 1.0
    assert report["latency"]["under_50ms_target"] is True
