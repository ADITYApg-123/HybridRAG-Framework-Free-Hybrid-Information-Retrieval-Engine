"""Gemini API client with exponential backoff and offline extractive fallback.

Uses the official `google-genai` SDK (`gemini-2.5-flash` by default) to generate
strictly grounded answers from the fused Hybrid RRF context chunks.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Sequence

from dotenv import load_dotenv

from src.generation.prompt_builder import SYSTEM_INSTRUCTION, PromptBuilder

load_dotenv()
logger = logging.getLogger("hybrid_rag.generation.llm_client")


class GeminiLLMClient:
    """Handles grounded answer generation via Google Gemini API with exponential backoff."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        max_retries: int = 3,
        base_delay_sec: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.base_delay_sec = base_delay_sec
        self.prompt_builder = PromptBuilder()

    @property
    def is_configured(self) -> bool:
        """Return True if a Gemini API key is present."""
        return bool(self.api_key and self.api_key.strip())

    def generate_answer(
        self,
        query: str,
        chunks: Sequence[Dict[str, Any]],
        allow_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Generate a grounded response from retrieved chunks using Gemini or extractive fallback."""
        prompt = self.prompt_builder.build_prompt(query=query, chunks=chunks)

        if not chunks:
            return {
                "answer": "The retrieved documentation does not contain the answer.",
                "model": self.model_name,
                "mode": "empty_context",
                "prompt": prompt,
            }

        if not self.is_configured:
            if allow_fallback:
                logger.info("No GEMINI_API_KEY configured; using deterministic grounded synthesis.")
                return {
                    "answer": self._extractive_grounded_fallback(query, chunks),
                    "model": "extractive-grounded-fallback (set GEMINI_API_KEY for live Gemini)",
                    "mode": "offline_fallback",
                    "prompt": prompt,
                }
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set GEMINI_API_KEY in environment or pass api_key."
            )

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                answer_text = (response.text or "").strip()
                if not answer_text:
                    answer_text = "The retrieved documentation does not contain the answer."
                return {
                    "answer": answer_text,
                    "model": self.model_name,
                    "mode": "gemini_api",
                    "prompt": prompt,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries:
                    sleep_sec = self.base_delay_sec * (2 ** (attempt - 1))
                    logger.warning(
                        "Gemini API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt,
                        self.max_retries,
                        exc,
                        sleep_sec,
                    )
                    time.sleep(sleep_sec)

        if allow_fallback:
            logger.warning("Gemini API retries exhausted (%s); falling back to extractive synthesis.", last_error)
            return {
                "answer": self._extractive_grounded_fallback(query, chunks),
                "model": f"extractive-fallback (API error: {type(last_error).__name__})",
                "mode": "offline_fallback",
                "prompt": prompt,
            }
        raise RuntimeError(f"Gemini API generation failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _extractive_grounded_fallback(
        query: str, chunks: Sequence[Dict[str, Any]]
    ) -> str:
        """Synthesize a clean, citation-backed grounded answer directly from top hybrid chunks."""
        top = chunks[0]
        doc_id = top.get("doc_id", "unknown")
        title = top.get("title", doc_id)
        raw_text = (top.get("raw_text") or top.get("text", "")).strip()

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        body_lines = [line for line in lines if line != title.strip()]
        primary_excerpt = "\n\n".join(body_lines) if body_lines else raw_text

        supporting_refs = [
            f"`{c.get('doc_id')}` (RRF={float(c.get('score', 0.0)):.4f})"
            for c in list(chunks)[:3]
        ]
        refs_str = ", ".join(supporting_refs)

        return (
            f"**Grounded Technical Answer (Source: `{doc_id}` — {title}):**\n\n"
            f"{primary_excerpt}\n\n"
            f"---\n"
            f"*Top Fused Citations:* {refs_str}"
        )
