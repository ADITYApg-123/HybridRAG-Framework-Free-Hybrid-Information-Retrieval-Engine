"""Code-aware and sentence-aware sliding window chunker for HybridRAG.

Implements SentenceSlidingWindowChunker from scratch without external RAG frameworks:
1. Keeps fenced ```python ... ``` code blocks intact as atomic units so periods inside
   API calls (e.g. `torch.nn.Conv2d`) or floats (`1e-5`) are never split mid-code.
2. Splits prose on sentence boundaries and accumulates sentences up to `max_tokens`.
3. Slides the window forward while retaining trailing sentences up to `overlap_tokens`.
4. Prepends hierarchical context headers (`[Module | Symbol | Section]`) so every chunk
   stays within `all-MiniLM-L6-v2`'s 256-token window while remaining self-contained.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.ingestion.parser import ParsedDocumentSection

logger = logging.getLogger("hybrid_rag.ingestion.chunker")

_CODE_BLOCK_PATTERN = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])")


@dataclass
class DocumentChunk:
    """Represents a single indexed chunk produced by SentenceSlidingWindowChunker."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    raw_text: str
    token_count: int
    module: str
    symbol: str
    section: str
    title: str
    source_file: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SentenceSlidingWindowChunker:
    """Sentence-boundary sliding window chunker with atomic code block preservation."""

    def __init__(
        self,
        max_tokens: int = 190,
        overlap_tokens: int = 35,
        prepend_header: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be >= 0 and strictly less than max_tokens.")

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.prepend_header = prepend_header

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count whitespace-delimited words/tokens in a string."""
        return len(text.split())

    def split_into_units(self, text: str) -> List[str]:
        """Split text into sentence units while keeping fenced code blocks intact."""
        if not text or not text.strip():
            return []

        parts = _CODE_BLOCK_PATTERN.split(text.strip())
        units: List[str] = []

        for part in parts:
            part_stripped = part.strip()
            if not part_stripped:
                continue
            if part_stripped.startswith("```") and part_stripped.endswith("```"):
                units.append(part_stripped)
            else:
                for paragraph in part_stripped.split("\n\n"):
                    para = paragraph.strip()
                    if not para:
                        continue
                    sentences = _SENTENCE_SPLIT_PATTERN.split(para)
                    for sent in sentences:
                        clean_sent = sent.strip()
                        if clean_sent:
                            units.append(clean_sent)

        return units

    def chunk_document(self, text: str, header_prefix: str = "") -> List[str]:
        """Chunk a document string using a sentence-aware sliding window."""
        units = self.split_into_units(text)
        if not units:
            return []

        header_str = f"{header_prefix.strip()}\n" if (self.prepend_header and header_prefix.strip()) else ""
        header_tokens = self.count_tokens(header_str)
        effective_budget = max(32, self.max_tokens - header_tokens)

        raw_chunks: List[str] = []
        current_buffer: List[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)

            if current_buffer and (current_tokens + unit_tokens > effective_budget):
                raw_chunks.append("\n".join(current_buffer).strip())

                # Build overlap window from trailing sentences of current_buffer
                overlap_buffer: List[str] = []
                overlap_count = 0
                for prev_unit in reversed(current_buffer):
                    prev_tokens = self.count_tokens(prev_unit)
                    if overlap_count + prev_tokens <= self.overlap_tokens or not overlap_buffer:
                        if prev_tokens < effective_budget:
                            overlap_buffer.insert(0, prev_unit)
                            overlap_count += prev_tokens
                    if overlap_count >= self.overlap_tokens:
                        break

                # Ensure window strictly advances if next unit alone overflows with overlap
                while overlap_buffer and (overlap_count + unit_tokens > effective_budget):
                    removed = overlap_buffer.pop(0)
                    overlap_count -= self.count_tokens(removed)

                current_buffer = overlap_buffer
                current_tokens = overlap_count

            current_buffer.append(unit)
            current_tokens += unit_tokens

        if current_buffer:
            final_text = "\n".join(current_buffer).strip()
            if not raw_chunks or final_text != raw_chunks[-1]:
                raw_chunks.append(final_text)

        if header_str:
            return [f"{header_str}{chunk}" for chunk in raw_chunks]
        return raw_chunks

    def chunk_sections(self, sections: List[ParsedDocumentSection]) -> List[DocumentChunk]:
        """Chunk a list of ParsedDocumentSection instances into indexed DocumentChunks."""
        all_chunks: List[DocumentChunk] = []

        for sec in sections:
            header = f"[Module: {sec.module} | Symbol: {sec.symbol} | Section: {sec.section}]"
            chunk_texts = self.chunk_document(sec.text, header_prefix=header)

            for idx, chunk_text in enumerate(chunk_texts):
                raw_without_header = chunk_text
                if self.prepend_header and chunk_text.startswith(header):
                    raw_without_header = chunk_text[len(header) :].lstrip("\n")

                chunk_id = f"{sec.doc_id}::chunk_{idx}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=sec.doc_id,
                        chunk_index=idx,
                        text=chunk_text,
                        raw_text=raw_without_header,
                        token_count=self.count_tokens(chunk_text),
                        module=sec.module,
                        symbol=sec.symbol,
                        section=sec.section,
                        title=sec.title,
                        source_file=sec.source_file,
                        metadata=dict(sec.metadata),
                    )
                )

        logger.info(
            "Chunked %d document sections into %d chunks (max_tokens=%d, overlap=%d).",
            len(sections),
            len(all_chunks),
            self.max_tokens,
            self.overlap_tokens,
        )
        return all_chunks

    @staticmethod
    def save_jsonl(chunks: List[DocumentChunk], output_path: str | Path) -> None:
        """Persist DocumentChunk list to a JSONL file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Saved %d chunks to %s", len(chunks), out)

    @staticmethod
    def load_jsonl(input_path: str | Path) -> List[DocumentChunk]:
        """Load DocumentChunk list from a JSONL file."""
        inp = Path(input_path)
        if not inp.exists():
            raise FileNotFoundError(f"Processed chunks JSONL not found: {inp}")
        chunks: List[DocumentChunk] = []
        with inp.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    chunks.append(DocumentChunk(**data))
        return chunks

