"""Context-Augmented Generation prompt builder for HybridRAG.

Formats Top-K retrieved chunks with relevance scores and enforces strict grounding
instructions to prevent hallucination and avoid the 'Lost in the Middle' phenomenon.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

SYSTEM_INSTRUCTION = (
    "You are an expert PyTorch technical assistant. Use ONLY the provided retrieved context "
    "to answer the user's question. Cite the relevant Context number or doc_id in your explanation. "
    "If the answer is not contained in the context, explicitly state "
    "\"The retrieved documentation does not contain the answer.\" Do not use outside knowledge."
)


class PromptBuilder:
    """Constructs strictly grounded prompts from Hybrid RRF retrieved context chunks."""

    def __init__(
        self,
        system_instruction: str = SYSTEM_INSTRUCTION,
        max_context_chunks: int = 5,
    ) -> None:
        if max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be a positive integer.")
        self.system_instruction = system_instruction.strip()
        self.max_context_chunks = max_context_chunks

    def build_context_block(self, chunks: Sequence[Dict[str, Any]]) -> str:
        """Format up to `max_context_chunks` into a numbered context block with scores."""
        if not chunks:
            return "No context chunks were retrieved."

        selected = list(chunks)[: self.max_context_chunks]
        blocks = []
        for idx, chunk in enumerate(selected, start=1):
            score = float(chunk.get("score", chunk.get("rrf_score", 0.0)))
            doc_id = chunk.get("doc_id", f"chunk_{idx}")
            title = chunk.get("title", "")
            text = chunk.get("raw_text") or chunk.get("text", "")
            header = f"Context {idx} (Doc ID: {doc_id} | Title: {title} | Relevance Score: {score:.4f}):"
            blocks.append(f"{header}\n{text.strip()}")

        return "\n\n".join(blocks)

    def build_prompt(self, query: str, chunks: Sequence[Dict[str, Any]]) -> str:
        """Build the complete grounded prompt string combining System, Contexts, and Query."""
        if not query or not query.strip():
            raise ValueError("User query cannot be empty.")

        context_block = self.build_context_block(chunks)
        return (
            f"System: {self.system_instruction}\n\n"
            f"{context_block}\n\n"
            f"User Question: {query.strip()}\n"
            f"Answer:"
        )
