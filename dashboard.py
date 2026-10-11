"""Streamlit dashboard for interactive LLM evaluation."""

from __future__ import annotations

import time

import streamlit as st

from src.loader import load_data
from src.metrics import LLMEvaluator

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="BeyondChats Eval",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stTextArea textarea {font-size: 14px;}
        div[data-testid="stMetricValue"] {font-size: 24px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ LLM Quality Assurance Pipeline")


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    chat_file = "data/chat_history.json"
    ctx_file = "data/context_data.json"

    st.markdown("### 📂 Data Sources")
    st.info(f"**Active Files:**\n\n1. `{chat_file}`\n2. `{ctx_file}`")

    st.divider()

    threshold = st.slider("Faithfulness Threshold", 0.0, 1.0, 0.7, 0.05)
    st.caption(f"Scores below {threshold} are flagged as Hallucinations.")

    run_btn = st.button("🚀 Run Evaluation", type="primary")


# ── Cached model initialiser (outside the run block) ─────────────────
@st.cache_resource
def _get_evaluator() -> LLMEvaluator:
    return LLMEvaluator()


# ── Main Logic ───────────────────────────────────────────────────────
if run_btn:
    # 1. Load data
    with st.spinner("Parsing & Validating Input Files..."):
        dataset, logs = load_data(chat_file, ctx_file)

    # 2. Display validation logs
    if logs:
        with st.expander("📝 Input Validation Logs", expanded=False):
            for log in logs:
                if "Error" in log:
                    st.error(log)
                else:
                    st.warning(log)

    # 3. Check if data loaded correctly
    if not dataset:
        st.error("Evaluation stopped due to data errors.")
        st.stop()

    # 4. Initialise models
    with st.spinner("Initializing Local NLP Models..."):
        evaluator = _get_evaluator()

    # 5. Evaluate ALL test cases (not just the first)
    for case_idx, data_item in enumerate(dataset):
        if len(dataset) > 1:
            st.subheader(f"📋 Case {case_idx + 1} of {len(dataset)}")

        start_time = time.time()
        rel_score = evaluator.calc_relevance(data_item["query"], data_item["response"])
        faith_score = evaluator.calc_faithfulness(data_item["context"], data_item["response"])
        latency_ms = (time.time() - start_time) * 1000

        # 6. Results row
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Relevance", f"{rel_score:.2f}")
        with c2:
            color = "normal" if faith_score >= threshold else "inverse"
            st.metric("Faithfulness", f"{faith_score:.2f}", delta_color=color)
        with c3:
            st.metric("Latency", f"{latency_ms:.0f} ms")
        with c4:
            if faith_score < threshold:
                st.error("🚨 HALLUCINATION")
            else:
                st.success("✅ PASSED")

        # 7. Deep dive
        st.divider()
        st.subheader("🔍 Deep Dive Analysis")

        col_chat, col_context = st.columns([1, 1], gap="medium")

        # Escape $ signs so Streamlit doesn't interpret them as LaTeX
        display_response = data_item["response"].replace("$", r"\$")

        with col_chat:
            st.markdown("### 💬 Conversation")
            with st.container(border=True, height=500):
                with st.chat_message("user"):
                    st.write("**User Query:**")
                    st.write(data_item["query"])

                with st.chat_message("assistant"):
                    st.write("**AI Response:**")
                    st.markdown(display_response)

        with col_context:
            st.markdown("### 📚 Retrieved Context")
            with st.container(border=True, height=500):
                st.info(
                    "ℹ️ This is the 'Source of Truth' retrieved from the vector database."
                )

                chunks = data_item["context"].split("\n\n")
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        safe_chunk = chunk.strip().replace("$", r"\$")
                        st.markdown(f"**Chunk {i + 1}:**")
                        st.caption(safe_chunk)
                        st.divider()

else:
    st.info("👈 Click 'Run Evaluation' to start.")