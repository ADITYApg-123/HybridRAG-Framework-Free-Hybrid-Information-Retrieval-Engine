"""HybridRAG Interactive Streamlit Benchmarking UI.

Features:
- Sidebar controls for Top-K, RRF_k smoothing constant, preset technical queries,
  and optional Gemini API key override.
- 3-Column side-by-side comparison of:
    1. Top-K Sparse Chunks (Okapi BM25)
    2. Top-K Dense Chunks (FAISS IndexFlatIP + all-MiniLM-L6-v2)
    3. Top-K Hybrid Chunks (Reciprocal Rank Fusion with rank attribution)
- Grounded Gemini LLM response panel + live IR evaluation summary tab.
"""

from __future__ import annotations

import json
import streamlit as st

from src.config import load_config, resolve_project_path
from src.generation.llm_client import GeminiLLMClient
from src.retrieval.hybrid_search import HybridRetriever

st.set_page_config(
    page_title="HybridRAG — Framework-Free IR Engine",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading FAISS IndexFlatIP, BM25Okapi, and all-MiniLM-L6-v2...")
def get_retriever() -> HybridRetriever:
    """Cache the HybridRetriever singleton across Streamlit reruns."""
    return HybridRetriever.from_config(auto_build=True)


def load_benchmark_summary() -> dict | None:
    """Load precomputed 40-query Golden Dataset evaluation metrics if available."""
    cfg = load_config()
    eval_path = resolve_project_path(cfg["paths"]["eval_results_path"])
    if eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


PRESET_QUERIES = [
    "Why must scaler.unscale_(optimizer) be called before clip_grad_norm_ when using torch.amp.GradScaler?",
    "How to prevent unstable derivative explosions in deep sequence models by capping vector magnitude?",
    "div_factor=25.0 final_div_factor=10000.0",
    "Speeding up neural network matrix multiplications on Tensor Cores using half-precision floating point",
    "Why does calling tensor.view() after transpose() or permute() fail on non-contiguous tensors while tensor.reshape() succeeds?",
    "Difference between torch.cuda.memory_allocated() and torch.cuda.memory_reserved() and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
]


def render_chunk_card(chunk: dict, modality: str) -> None:
    """Render a single retrieved chunk card inside one of the 3 comparison columns."""
    rank = chunk.get("rank", 1)
    score = float(chunk.get("score", 0.0))
    doc_id = chunk.get("doc_id", "unknown")
    title = chunk.get("title", doc_id)
    raw_text = chunk.get("raw_text") or chunk.get("text", "")

    if modality == "hybrid":
        d_rank = chunk.get("dense_rank")
        s_rank = chunk.get("sparse_rank")
        d_str = f"#{d_rank}" if d_rank is not None else "—"
        s_str = f"#{s_rank}" if s_rank is not None else "—"
        badge = f"**Rank #{rank}** | `RRF = {score:.4f}` *(Dense: {d_str}, Sparse: {s_str})*"
    elif modality == "dense":
        badge = f"**Rank #{rank}** | `Cosine = {score:.4f}`"
    else:
        badge = f"**Rank #{rank}** | `BM25 = {score:.2f}`"

    with st.container(border=True):
        st.markdown(badge)
        st.caption(f"📄 `{doc_id}` — **{title}**")
        st.markdown(raw_text)


def main() -> None:
    cfg = load_config()
    retriever = get_retriever()
    benchmark_data = load_benchmark_summary()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Retrieval Hyperparameters")
        top_k = st.slider(
            "Top-K Chunks Displayed",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of top chunks shown per retrieval modality and injected into the LLM prompt.",
        )
        rrf_k = st.slider(
            "RRF Smoothing Constant (k)",
            min_value=1,
            max_value=120,
            value=int(cfg["retrieval"].get("rrf_k", 60)),
            help="Reciprocal Rank Fusion constant k in 1/(k + rank_dense) + 1/(k + rank_sparse).",
        )

        st.divider()
        st.subheader("🔑 Gemini Generation Config")
        api_key_input = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            help="Leave blank to use GEMINI_API_KEY from environment or deterministic grounded synthesis.",
        )

        if benchmark_data:
            st.divider()
            st.subheader("📊 Golden Dataset (40 Queries)")
            s_sum = benchmark_data["summary"]["sparse"]
            d_sum = benchmark_data["summary"]["dense"]
            h_sum = benchmark_data["summary"]["hybrid"]
            lat = benchmark_data["latency"]
            st.markdown(
                f"- **Hybrid RRF Recall@5:** `{h_sum['recall@5'] * 100:.1f}%` *(MRR: `{h_sum['mrr']:.4f}`)*\n"
                f"- **BM25 Sparse Recall@5:** `{s_sum['recall@5'] * 100:.1f}%` *(MRR: `{s_sum['mrr']:.4f}`)*\n"
                f"- **FAISS Dense Recall@5:** `{d_sum['recall@5'] * 100:.1f}%` *(MRR: `{d_sum['mrr']:.4f}`)*\n"
                f"- **Mean Query Latency:** `{lat['mean_ms']:.2f} ms` *(P95: `{lat['p95_ms']:.2f} ms`)*"
            )

    st.title("🔍 HybridRAG: Framework-Free Hybrid Information Retrieval Engine")
    st.caption(
        "Zero LangChain / LlamaIndex | Okapi BM25 (Sparse) + FAISS IndexFlatIP (`all-MiniLM-L6-v2` Dense) "
        "+ Reciprocal Rank Fusion (RRF) + Grounded Gemini Generation"
    )

    selected_preset = st.selectbox(
        "💡 Select a Preset Technical Query (or type your own below):",
        options=["(Custom Query)"] + PRESET_QUERIES,
        index=1,
    )
    default_query = PRESET_QUERIES[0] if selected_preset == "(Custom Query)" else selected_preset
    query = st.text_input("Enter a PyTorch technical question or API identifier:", value=default_query)

    if query.strip():
        bundle = retriever.retrieve_all_modalities(query=query.strip(), top_k=top_k, rrf_k=rrf_k)
        st.info(
            f"⚡ **Retrieval Latency:** `{bundle['latency_ms']:.2f} ms` | "
            f"**Indexed Chunks:** `{retriever.faiss_store.ntotal}` | "
            f"**RRF Formula:** `1 / ({rrf_k} + rank_dense) + 1 / ({rrf_k} + rank_sparse)`"
        )

        col_sparse, col_dense, col_hybrid = st.columns(3)

        with col_sparse:
            st.subheader(f"1️⃣ Top {top_k} BM25 (Sparse)")
            if not bundle["sparse"]:
                st.warning("No lexical token matches found in BM25.")
            for chunk in bundle["sparse"]:
                render_chunk_card(chunk, modality="sparse")

        with col_dense:
            st.subheader(f"2️⃣ Top {top_k} FAISS (Dense)")
            for chunk in bundle["dense"]:
                render_chunk_card(chunk, modality="dense")

        with col_hybrid:
            st.subheader(f"3️⃣ Top {top_k} RRF (Hybrid)")
            for chunk in bundle["hybrid"]:
                render_chunk_card(chunk, modality="hybrid")

        st.divider()
        st.subheader("🤖 Grounded Gemini Response (Generated from Hybrid RRF Contexts)")

        llm_client = GeminiLLMClient(
            model_name=cfg["model"].get("llm", "gemini-2.5-flash"),
            api_key=api_key_input.strip() if api_key_input else None,
            temperature=float(cfg["model"].get("temperature", 0.1)),
            max_output_tokens=int(cfg["model"].get("max_output_tokens", 1024)),
        )
        gen_result = llm_client.generate_answer(query=query.strip(), chunks=bundle["hybrid"])
        st.caption(f"Model: `{gen_result['model']}` | Mode: `{gen_result['mode']}`")
        st.markdown(gen_result["answer"])

        with st.expander("🔎 Inspect Exact Grounded Prompt Sent to LLM"):
            st.code(gen_result["prompt"], language="text")


if __name__ == "__main__":
    main()

