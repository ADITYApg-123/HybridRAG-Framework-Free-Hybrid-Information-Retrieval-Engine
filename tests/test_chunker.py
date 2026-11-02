"""Unit tests for SentenceSlidingWindowChunker (src/ingestion/chunker.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.chunker import SentenceSlidingWindowChunker
from src.ingestion.parser import ParsedDocumentSection


def test_chunker_validation() -> None:
    with pytest.raises(ValueError):
        SentenceSlidingWindowChunker(max_tokens=0, overlap_tokens=10)
    with pytest.raises(ValueError):
        SentenceSlidingWindowChunker(max_tokens=50, overlap_tokens=50)


def test_sentence_sliding_window_and_overlap() -> None:
    chunker = SentenceSlidingWindowChunker(max_tokens=40, overlap_tokens=12, prepend_header=False)
    text = (
        "Sentence one describes the input tensor dimensions for convolution. "
        "Sentence two explains how stride controls spatial downsampling of feature maps. "
        "Sentence three details dilation spacing between kernel filter elements. "
        "Sentence four covers depthwise separable convolutions when groups equals in_channels. "
        "Sentence five concludes with padding modes such as reflect and circular."
    )

    chunks = chunker.chunk_document(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunker.count_tokens(chunk) <= 40

    # Verify sliding window overlap: last sentence of chunk 0 should appear at start of chunk 1
    assert "Sentence two" in chunks[0] or "Sentence three" in chunks[0]
    overlap_words = set(chunks[0].split()) & set(chunks[1].split())
    assert len(overlap_words) >= 5


def test_atomic_code_block_preservation_and_header_injection(tmp_path: Path) -> None:
    chunker = SentenceSlidingWindowChunker(max_tokens=120, overlap_tokens=20, prepend_header=True)
    sec = ParsedDocumentSection(
        doc_id="torch_nn_conv2d::shape_and_depthwise_math",
        module="torch.nn",
        symbol="Conv2d",
        section="shape_and_depthwise_math",
        title="Conv2d Shape",
        text=(
            "Depthwise separable convolution uses groups=in_channels.\n"
            "```python\n"
            "depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)\n"
            "pointwise = nn.Conv2d(32, 64, kernel_size=1)\n"
            "```"
        ),
        source_file="torch_nn_conv2d.html",
    )

    chunks = chunker.chunk_sections([sec])
    assert len(chunks) == 1
    assert chunks[0].text.startswith("[Module: torch.nn | Symbol: Conv2d | Section: shape_and_depthwise_math]")
    assert "```python\ndepthwise = nn.Conv2d" in chunks[0].text

    out_jsonl = tmp_path / "chunks.jsonl"
    chunker.save_jsonl(chunks, out_jsonl)
    loaded = chunker.load_jsonl(out_jsonl)
    assert len(loaded) == 1
    assert loaded[0].chunk_id == chunks[0].chunk_id
