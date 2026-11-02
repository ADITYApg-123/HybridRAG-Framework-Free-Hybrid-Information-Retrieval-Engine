"""HybridRetriever Facade coordinating Dense (FAISS), Sparse (BM25), and RRF fusion.

Provides unified retrieval methods for:
- Pure Sparse lexical search (BM25)
- Pure Dense semantic vector search (FAISS IndexFlatIP)
- Hybrid Reciprocal Rank Fusion (RRF) search
- Single-pass multi-modality comparison with sub-50ms latency instrumentation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import load_config, resolve_project_path
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import DenseEmbedder
from src.indexing.faiss_store import VectorStore
from src.indexing.index_builder import build_and_save_indexes
from src.retrieval.ranker import reciprocal_rank_fusion_detailed

logger = logging.getLogger("hybrid_rag.retrieval.hybrid_search")


class HybridRetriever:
    """Facade orchestrating query embedding, FAISS search, BM25 search, and RRF."""

    def __init__(
        self,
        embedder: DenseEmbedder,
        faiss_store: VectorStore,
        bm25_store: BM25Store,
        chunk_metadata: List[Dict[str, Any]],
        rrf_k: int = 60,
        top_k_fetch: int = 60,
        top_k_final: int = 5,
    ) -> None:
        if faiss_store.ntotal != len(chunk_metadata) or bm25_store.ntotal != len(chunk_metadata):
            raise ValueError(
                f"Index alignment mismatch: FAISS={faiss_store.ntotal}, "
                f"BM25={bm25_store.ntotal}, Metadata={len(chunk_metadata)}"
            )

        self.embedder = embedder
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.chunk_metadata = chunk_metadata
        self.rrf_k = rrf_k
        self.top_k_fetch = top_k_fetch
        self.top_k_final = top_k_final

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        auto_build: bool = True,
    ) -> "HybridRetriever":
        """Load indexes and initialize HybridRetriever from `config/config.yaml`."""
        cfg = load_config(config_path)
        faiss_path = resolve_project_path(cfg["paths"]["faiss_index_path"])
        bm25_path = resolve_project_path(cfg["paths"]["bm25_index_path"])
        meta_path = resolve_project_path(cfg["paths"]["metadata_store_path"])

        model_name = cfg["model"]["embedding"]
        embedder = DenseEmbedder(model_name=model_name)

        if not (faiss_path.exists() and bm25_path.exists() and meta_path.exists()):
            if not auto_build:
                raise FileNotFoundError("Saved indexes not found and auto_build=False.")
            faiss_store, bm25_store, metadata = build_and_save_indexes(
                config_path=config_path, embedder=embedder
            )
        else:
            faiss_store = VectorStore.load(faiss_path)
            bm25_store = BM25Store.load(bm25_path)
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

        ret_cfg = cfg["retrieval"]
        return cls(
            embedder=embedder,
            faiss_store=faiss_store,
            bm25_store=bm25_store,
            chunk_metadata=metadata,
            rrf_k=int(ret_cfg.get("rrf_k", 60)),
            top_k_fetch=int(ret_cfg.get("top_k_fetch", 60)),
            top_k_final=int(ret_cfg.get("top_k_final", 5)),
        )

    def _format_single_modality(
        self, ranked_pairs: List[Tuple[int, float]], modality: str, top_k: int
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for rank_idx, (doc_int_id, score) in enumerate(ranked_pairs[:top_k], start=1):
            meta = dict(self.chunk_metadata[doc_int_id])
            meta.update(
                {
                    "rank": rank_idx,
                    "score": float(score),
                    "modality": modality,
                }
            )
            results.append(meta)
        return results

    def retrieve_dense(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Retrieve Top-K chunks using only FAISS Dense Cosine Similarity."""
        k = top_k or self.top_k_final
        q_emb = self.embedder.embed_query(query, normalize=True)
        dense_ranked = self.faiss_store.search_ranked(q_emb, top_k=max(k, self.top_k_fetch))
        return self._format_single_modality(dense_ranked, modality="dense", top_k=k)

    def retrieve_sparse(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Retrieve Top-K chunks using only Okapi BM25 Sparse lexical search."""
        k = top_k or self.top_k_final
        sparse_ranked = self.bm25_store.search_ranked(query, top_k=max(k, self.top_k_fetch))
        return self._format_single_modality(sparse_ranked, modality="sparse", top_k=k)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rrf_k: int | None = None,
        top_k_fetch: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve Top-K chunks via Hybrid Reciprocal Rank Fusion (BM25 + FAISS)."""
        bundle = self.retrieve_all_modalities(
            query=query,
            top_k=top_k or self.top_k_final,
            rrf_k=rrf_k or self.rrf_k,
            top_k_fetch=top_k_fetch or self.top_k_fetch,
        )
        return bundle["hybrid"]

    def retrieve_all_modalities(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int | None = None,
        top_k_fetch: int | None = None,
    ) -> Dict[str, Any]:

        """Run Dense, Sparse, and Hybrid RRF in one pass and record execution latency."""
        t0 = time.perf_counter()
        fetch_k = max(top_k, top_k_fetch or self.top_k_fetch)
        k_const = rrf_k or self.rrf_k

        q_emb = self.embedder.embed_query(query, normalize=True)
        dense_ranked = self.faiss_store.search_ranked(q_emb, top_k=fetch_k)
        sparse_ranked = self.bm25_store.search_ranked(query, top_k=fetch_k)

        fused_details = reciprocal_rank_fusion_detailed(
            dense_results=dense_ranked,
            sparse_results=sparse_ranked,
            rrf_k=k_const,
            top_k=top_k,
        )

        hybrid_results: List[Dict[str, Any]] = []
        for rank_idx, detail in enumerate(fused_details, start=1):
            meta = dict(self.chunk_metadata[detail.doc_int_id])
            meta.update(
                {
                    "rank": rank_idx,
                    "score": detail.rrf_score,
                    "rrf_score": detail.rrf_score,
                    "dense_rank": detail.dense_rank,
                    "dense_score": detail.dense_score,
                    "sparse_rank": detail.sparse_rank,
                    "sparse_score": detail.sparse_score,
                    "modality": "hybrid",
                }
            )
            hybrid_results.append(meta)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "Query '%s' executed in %.2f ms (top_k=%d, rrf_k=%d)",
            query,
            latency_ms,
            top_k,
            k_const,
        )
        return {
            "query": query,
            "sparse": self._format_single_modality(sparse_ranked, modality="sparse", top_k=top_k),
            "dense": self._format_single_modality(dense_ranked, modality="dense", top_k=top_k),
            "hybrid": hybrid_results,
            "latency_ms": latency_ms,
        }

