"""Offline dual index builder orchestrating Dense (FAISS) and Sparse (BM25) stores.

Ensures strict integer ID alignment (0 .. N-1) across:
1. FAISS `IndexFlatIP` vector store
2. BM25 `BM25Okapi` sparse store
3. JSON chunk metadata store (`chunk_metadata.json`)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import load_config, resolve_project_path, setup_logging
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import DenseEmbedder
from src.indexing.faiss_store import VectorStore
from src.ingestion.chunker import DocumentChunk, SentenceSlidingWindowChunker
from src.ingestion.pipeline import run_ingestion_pipeline

logger = logging.getLogger("hybrid_rag.indexing.index_builder")


def build_and_save_indexes(
    config_path: str | Path | None = None,
    embedder: DenseEmbedder | None = None,
) -> Tuple[VectorStore, BM25Store, List[Dict[str, Any]]]:
    """Build and persist aligned FAISS, BM25, and chunk metadata stores."""
    cfg = load_config(config_path)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    chunks_path = resolve_project_path(cfg["paths"]["processed_chunks_path"])
    faiss_path = resolve_project_path(cfg["paths"]["faiss_index_path"])
    bm25_path = resolve_project_path(cfg["paths"]["bm25_index_path"])
    metadata_path = resolve_project_path(cfg["paths"]["metadata_store_path"])

    if not chunks_path.exists():
        chunks: List[DocumentChunk] = run_ingestion_pipeline(config_path)
    else:
        chunks = SentenceSlidingWindowChunker.load_jsonl(chunks_path)

    if not chunks:
        raise RuntimeError("No chunks found to index.")

    texts = [c.text for c in chunks]
    model_name = cfg["model"]["embedding"]
    dense_embedder = embedder or DenseEmbedder(model_name=model_name)

    embeddings = dense_embedder.embed_texts(texts, normalize=True)
    faiss_store = VectorStore(embedding_dim=dense_embedder.embedding_dim)
    faiss_store.build_index(embeddings)
    faiss_store.save(faiss_path)

    bm25_store = BM25Store()
    bm25_store.build_index(texts)
    bm25_store.save(bm25_path)

    metadata_records: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        record = chunk.to_dict()
        record["int_id"] = idx
        metadata_records.append(record)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2, ensure_ascii=False)

    assert faiss_store.ntotal == bm25_store.ntotal == len(metadata_records), (
        "Index alignment failure between FAISS, BM25, and metadata records!"
    )
    logger.info(
        "Successfully built aligned FAISS + BM25 indexes over %d chunks.",
        len(metadata_records),
    )
    return faiss_store, bm25_store, metadata_records


if __name__ == "__main__":
    build_and_save_indexes()
