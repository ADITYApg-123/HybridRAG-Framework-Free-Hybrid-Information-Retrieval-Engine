"""End-to-end offline ingestion pipeline: corpus generation -> HTML parsing -> chunking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.config import load_config, resolve_project_path, setup_logging
from src.ingestion.chunker import DocumentChunk, SentenceSlidingWindowChunker
from src.ingestion.corpus_builder import build_raw_corpus
from src.ingestion.parser import HTMLDocParser

logger = logging.getLogger("hybrid_rag.ingestion.pipeline")


def run_ingestion_pipeline(config_path: str | Path | None = None) -> List[DocumentChunk]:
    """Execute the complete offline ingestion and chunking workflow."""
    cfg = load_config(config_path)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    raw_dir = resolve_project_path(cfg["paths"]["raw_data_dir"])
    processed_path = resolve_project_path(cfg["paths"]["processed_chunks_path"])

    # Ensure raw HTML corpus exists
    if not raw_dir.exists() or not list(raw_dir.glob("*.html")):
        build_raw_corpus(raw_dir)

    parser = HTMLDocParser(preserve_code_blocks=True)
    sections = parser.parse_directory(raw_dir)

    chunk_cfg = cfg["chunking"]
    chunker = SentenceSlidingWindowChunker(
        max_tokens=int(chunk_cfg.get("max_tokens", 190)),
        overlap_tokens=int(chunk_cfg.get("overlap_tokens", 35)),
        prepend_header=bool(chunk_cfg.get("prepend_header", True)),
    )

    chunks = chunker.chunk_sections(sections)
    chunker.save_jsonl(chunks, processed_path)
    return chunks


if __name__ == "__main__":
    run_ingestion_pipeline()
