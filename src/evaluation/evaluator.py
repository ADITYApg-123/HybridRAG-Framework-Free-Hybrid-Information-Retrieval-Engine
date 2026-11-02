"""Information Retrieval evaluation module for HybridRAG (Recall@K, MRR, Precision@K, Latency).

Evaluates Sparse (BM25), Dense (FAISS), and Hybrid (RRF) retrieval pipelines against
the ground-truth Golden Dataset (`data/evaluation/golden_queries.json`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.config import load_config, resolve_project_path, setup_logging
from src.retrieval.hybrid_search import HybridRetriever

logger = logging.getLogger("hybrid_rag.evaluation.evaluator")


def recall_at_k(retrieved_doc_ids: Sequence[str], expected_doc_id: str, k: int = 5) -> float:
    """Return 1.0 if `expected_doc_id` appears within the top `k` retrieved doc_ids, else 0.0."""
    if k <= 0:
        raise ValueError("k must be positive.")
    top_slice = list(retrieved_doc_ids)[:k]
    return 1.0 if expected_doc_id in top_slice else 0.0


def reciprocal_rank(retrieved_doc_ids: Sequence[str], expected_doc_id: str) -> float:
    """Compute Reciprocal Rank (1 / rank) of the first matching `expected_doc_id` (0.0 if absent)."""
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == expected_doc_id:
            return 1.0 / float(idx)
    return 0.0


def precision_at_k(retrieved_doc_ids: Sequence[str], expected_doc_id: str, k: int = 5) -> float:
    """Compute Precision@K (proportion of top-K retrieved chunks matching `expected_doc_id`)."""
    if k <= 0:
        raise ValueError("k must be positive.")
    top_slice = list(retrieved_doc_ids)[:k]
    if not top_slice:
        return 0.0
    hits = sum(1 for doc_id in top_slice if doc_id == expected_doc_id)
    return float(hits) / float(k)


class IREvaluator:
    """Benchmarks BM25, FAISS, and Hybrid RRF retrieval over a Golden Dataset."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def evaluate_dataset(
        self,
        golden_queries: List[Dict[str, Any]],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """Run all Golden Queries and aggregate Recall@1, Recall@3, Recall@5, MRR, and Latency."""
        if not golden_queries:
            raise ValueError("golden_queries cannot be empty.")

        modalities = ("sparse", "dense", "hybrid")
        metrics_accum: Dict[str, Dict[str, List[float]]] = {
            m: {"recall@1": [], "recall@3": [], "recall@5": [], "mrr": [], "precision@5": []}
            for m in modalities
        }
        latencies_ms: List[float] = []
        per_query_records: List[Dict[str, Any]] = []

        # Warm-up query so model cache initialization does not skew latency measurement
        _ = self.retriever.retrieve_all_modalities(
            golden_queries[0]["query"], top_k=top_k, rrf_k=rrf_k
        )

        for item in golden_queries:
            qid = item.get("query_id", "unknown")
            query = item["query"]
            expected_doc_id = item["expected_doc_id"]
            category = item.get("category", "general")

            bundle = self.retriever.retrieve_all_modalities(
                query=query, top_k=max(top_k, 5), rrf_k=rrf_k
            )
            latencies_ms.append(float(bundle["latency_ms"]))

            q_record: Dict[str, Any] = {
                "query_id": qid,
                "category": category,
                "query": query,
                "expected_doc_id": expected_doc_id,
                "latency_ms": round(float(bundle["latency_ms"]), 3),
                "results": {},
            }

            for mod in modalities:
                doc_ids = [chunk["doc_id"] for chunk in bundle[mod]]
                r1 = recall_at_k(doc_ids, expected_doc_id, k=1)
                r3 = recall_at_k(doc_ids, expected_doc_id, k=3)
                r5 = recall_at_k(doc_ids, expected_doc_id, k=top_k)
                rr = reciprocal_rank(doc_ids, expected_doc_id)
                p5 = precision_at_k(doc_ids, expected_doc_id, k=top_k)

                metrics_accum[mod]["recall@1"].append(r1)
                metrics_accum[mod]["recall@3"].append(r3)
                metrics_accum[mod]["recall@5"].append(r5)
                metrics_accum[mod]["mrr"].append(rr)
                metrics_accum[mod]["precision@5"].append(p5)

                q_record["results"][mod] = {
                    "top_doc_ids": doc_ids[:top_k],
                    "recall@1": r1,
                    "recall@5": r5,
                    "reciprocal_rank": round(rr, 4),
                }

            per_query_records.append(q_record)

        summary: Dict[str, Any] = {}
        for mod in modalities:
            summary[mod] = {
                metric_name: round(float(np.mean(vals)), 4)
                for metric_name, vals in metrics_accum[mod].items()
            }

        latency_summary = {
            "mean_ms": round(float(np.mean(latencies_ms)), 2),
            "median_ms": round(float(np.median(latencies_ms)), 2),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
            "under_50ms_target": bool(float(np.mean(latencies_ms)) < 50.0),
        }

        return {
            "num_queries": len(golden_queries),
            "top_k": top_k,
            "rrf_k": rrf_k,
            "summary": summary,
            "latency": latency_summary,
            "per_query": per_query_records,
        }


def run_evaluation(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load Golden Queries, execute IREvaluator, and save benchmark results JSON."""
    cfg = load_config(config_path)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    golden_path = resolve_project_path(cfg["paths"]["golden_dataset_path"])
    results_path = resolve_project_path(cfg["paths"]["eval_results_path"])

    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at: {golden_path}")

    with golden_path.open("r", encoding="utf-8") as f:
        golden_queries: List[Dict[str, Any]] = json.load(f)

    retriever = HybridRetriever.from_config(config_path=config_path, auto_build=True)
    evaluator = IREvaluator(retriever=retriever)

    ret_cfg = cfg["retrieval"]
    report = evaluator.evaluate_dataset(
        golden_queries=golden_queries,
        top_k=int(ret_cfg.get("top_k_final", 5)),
        rrf_k=int(ret_cfg.get("rrf_k", 60)),
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Saved evaluation benchmark report to %s", results_path)
    for mod in ("sparse", "dense", "hybrid"):
        m = report["summary"][mod]
        logger.info(
            "%-7s | Recall@1: %.2f%% | Recall@5: %.2f%% | MRR: %.4f | Precision@5: %.4f",
            mod.upper(),
            m["recall@1"] * 100.0,
            m["recall@5"] * 100.0,
            m["mrr"],
            m["precision@5"],
        )
    logger.info(
        "Latency | Mean: %.2f ms | Median: %.2f ms | P95: %.2f ms (Target <50ms: %s)",
        report["latency"]["mean_ms"],
        report["latency"]["median_ms"],
        report["latency"]["p95_ms"],
        report["latency"]["under_50ms_target"],
    )
    return report


if __name__ == "__main__":
    run_evaluation()

