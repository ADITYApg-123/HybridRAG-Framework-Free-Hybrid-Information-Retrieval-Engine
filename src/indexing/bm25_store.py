"""BM25 sparse retrieval index with code-aware technical tokenization.

Guarantees strict tokenizer parity between offline index construction and online query
execution while preserving exact API identifiers (`torch.nn.Conv2d`, `eps=1e-05`,
`clip_grad_norm_`) alongside split camelCase and snake_case sub-tokens.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger("hybrid_rag.indexing.bm25_store")

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
        "was", "were", "will", "with", "this", "these", "those", "what", "which",
        "who", "whom", "how", "when", "where", "why", "can", "should", "would",
        "could", "does", "do", "did", "into", "over", "such", "than", "then",
    }
)

_RAW_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*(?:=[A-Za-z0-9_.'\"-]+)?|1e-?\d+")
_CAMEL_SPLIT_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+[A-Za-z]*")


def technical_tokenize(text: str) -> List[str]:
    """Tokenize technical documentation or queries with identifier & sub-token expansion.

    Examples:
        "torch.nn.Conv2d(eps=1e-05)" ->
        ["torch.nn.conv2d", "torch", "nn", "conv2d", "conv", "2d", "eps=1e-05", "eps", "1e-05"]
    """
    if not text or not text.strip():
        return []

    tokens: List[str] = []
    for match in _RAW_TOKEN_RE.finditer(text):
        raw = match.group(0)
        lowered = raw.lower().strip(".")
        if not lowered:
            continue

        if lowered not in _STOPWORDS:
            tokens.append(lowered)

        # Expand dotted/underscored/assigned parts (e.g. torch.nn.Conv2d or padding='same')
        if any(sep in raw for sep in (".", "_", "=")):
            subparts = re.split(r"[._= '\"]+", raw)
            for sp in subparts:
                sp_low = sp.lower()
                if sp_low and sp_low != lowered and sp_low not in _STOPWORDS:
                    tokens.append(sp_low)
                # Further split camelCase inside subparts (e.g. Conv2d -> conv, 2d)
                camel_parts = _CAMEL_SPLIT_RE.findall(sp)
                if len(camel_parts) > 1:
                    for cp in camel_parts:
                        cp_low = cp.lower()
                        if cp_low and cp_low != sp_low and cp_low not in _STOPWORDS:
                            tokens.append(cp_low)
        else:
            camel_parts = _CAMEL_SPLIT_RE.findall(raw)
            if len(camel_parts) > 1:
                for cp in camel_parts:
                    cp_low = cp.lower()
                    if cp_low and cp_low != lowered and cp_low not in _STOPWORDS:
                        tokens.append(cp_low)

    return tokens


class BM25Store:
    """Wraps Okapi BM25 (`rank_bm25.BM25Okapi`) with deterministic technical tokenization."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    @property
    def ntotal(self) -> int:
        """Return total number of documents indexed in BM25."""
        return len(self.tokenized_corpus)

    def build_index(self, texts: Sequence[str]) -> None:
        """Tokenize corpus chunks and fit Okapi BM25 statistics."""
        if not texts:
            raise ValueError("Cannot build BM25 index from an empty text sequence.")

        self.tokenized_corpus = [technical_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        logger.info("Built BM25Okapi index over %d chunks.", len(self.tokenized_corpus))

    def search_ranked(self, query: str, top_k: int = 60) -> List[Tuple[int, float]]:
        """Query the BM25 index and return Top-K `(chunk_int_idx, bm25_score)` sorted descending."""
        if self.bm25 is None or not self.tokenized_corpus:
            raise RuntimeError("BM25 index is not initialized. Call build_index() or load() first.")

        query_tokens = technical_tokenize(query)
        if not query_tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float64)
        k = min(max(1, top_k), len(scores))

        # Sort descending by score, breaking ties deterministically by ascending index
        ranked_indices = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices]

    def save(self, index_path: str | Path) -> None:
        """Serialize BM25 store state to disk via pickle."""
        out = Path(index_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "tokenized_corpus": self.tokenized_corpus,
            "bm25": self.bm25,
        }
        with out.open("wb") as f:
            pickle.dump(payload, f)
        logger.info("Saved BM25 index (%d chunks) to %s", self.ntotal, out)

    @classmethod
    def load(cls, index_path: str | Path) -> "BM25Store":
        """Load serialized BM25 store from disk."""
        inp = Path(index_path)
        if not inp.exists():
            raise FileNotFoundError(f"BM25 pickle file not found: {inp}")
        with inp.open("rb") as f:
            payload = pickle.load(f)
        store = cls(k1=float(payload.get("k1", 1.5)), b=float(payload.get("b", 0.75)))
        store.tokenized_corpus = payload["tokenized_corpus"]
        store.bm25 = payload["bm25"]
        logger.info("Loaded BM25 index (%d chunks) from %s", store.ntotal, inp)
        return store
