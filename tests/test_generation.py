"""Unit tests for PromptBuilder and GeminiLLMClient."""

from __future__ import annotations

import pytest

from src.generation.llm_client import GeminiLLMClient
from src.generation.prompt_builder import PromptBuilder


def test_prompt_builder_formats_contexts_and_enforces_cap() -> None:
    builder = PromptBuilder(max_context_chunks=2)
    chunks = [
        {"doc_id": "doc_1", "title": "Conv2d", "score": 0.0328, "raw_text": "Chunk 1 body."},
        {"doc_id": "doc_2", "title": "AdamW", "score": 0.0310, "raw_text": "Chunk 2 body."},
        {"doc_id": "doc_3", "title": "LSTM", "score": 0.0150, "raw_text": "Chunk 3 body."},
    ]

    prompt = builder.build_prompt("How does Conv2d work?", chunks)
    assert "Context 1 (Doc ID: doc_1" in prompt
    assert "0.0328" in prompt
    assert "Context 2 (Doc ID: doc_2" in prompt
    assert "Context 3" not in prompt
    assert "The retrieved documentation does not contain the answer." in prompt
    assert "User Question: How does Conv2d work?" in prompt

    with pytest.raises(ValueError):
        builder.build_prompt("", chunks)


def test_gemini_client_offline_grounded_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    client = GeminiLLMClient(api_key=None)
    assert client.is_configured is False

    chunks = [
        {
            "doc_id": "torch_nn_conv2d::signature_and_parameters",
            "title": "torch.nn.Conv2d Signature",
            "score": 0.0328,
            "raw_text": "padding='same' cannot be used with any stride value other than stride=1.",
        }
    ]
    res = client.generate_answer("What is the stride rule for padding='same'?", chunks)
    assert res["mode"] == "offline_fallback"
    assert "torch_nn_conv2d::signature_and_parameters" in res["answer"]
    assert "stride=1" in res["answer"]
