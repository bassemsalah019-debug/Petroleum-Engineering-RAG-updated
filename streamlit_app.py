"""
streamlit_app.py — Multi-document RAG assistant for Petroleum Engineering.

Extends the original single-document RAG into a persistent multi-document
ChatPDF system. Users can upload PDFs, manage the knowledge base, and ask
questions across all indexed documents.

Usage:
    streamlit run streamlit_app.py
"""

import sys
import os
import logging
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import faiss

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "implementation"))

from pipeline_utils import (
    ensure_nltk_resources,
    generate_answer_hf,
)
from knowledge_base import KnowledgeBase

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
HF_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
HF_TOKEN = os.environ.get("GROQ_API_KEY")

PDF_PATH = PROJECT_ROOT / "geokniga-drillingengineeringprasslwl.pdf"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("petro_rag_streamlit")


# ---------------------------------------------------------------------------
# Cached resource: Embedding model
# ---------------------------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """Load the sentence-transformers model (cached across reruns)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Knowledge Base initialization (via session state)
# ---------------------------------------------------------------------------

def get_knowledge_base():
    """Get or create the KnowledgeBase instance in session state."""
    if "knowledge_base" not in st.session_state:
        ensure_nltk_resources()
        kb = KnowledgeBase()
        st.session_state.knowledge_base = kb
    return st.session_state.knowledge_base


def auto_migrate(kb):
    """Auto-migrate the existing textbook PDF if the knowledge base is empty."""
    if kb.is_empty() and PDF_PATH.exists():
        embedding_model = load_embedding_model()
        with st.spinner("📥 Migrating existing textbook into knowledge base..."):
            result = kb.migrate_existing_pdf(PDF_PATH, embedding_model)
        if result["success"]:
            st.success(f"✅ {result['message']}")
            st.rerun()
        elif "already" in result["message"].lower():
            pass  # Already migrated
        else:
            st.warning(f"⚠️ Migration: {result['message']}")


# ---------------------------------------------------------------------------
# Custom CSS — Petroleum Engineering RAG Theme
# ---------------------------------------------------------------------------

def inject_custom_css():
    """Inject custom CSS for a polished petroleum-themed UI."""
    st.markdown("""
    <style>
    /* ── Global ─────────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(180deg, #0C1220 0%, #111927 100%);
    }

    /* ── Hero Banner ────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0F1A2E 0%, #1A2744 40%, #243352 100%);
        border: 1px solid rgba(45, 212, 191, 0.25);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(45, 212, 191, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2DD4BF;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #9CA3AF;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(45, 212, 191, 0.15);
        border: 1px solid rgba(45, 212, 191, 0.3);
        color: #2DD4BF;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-bottom: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* ── Sidebar ────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1A2E 0%, #131B2E 100%);
        border-right: 1px solid rgba(45, 212, 191, 0.12);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
        color: #2DD4BF !important;
    }

    /* ── Chat Messages ──────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(45, 212, 191, 0.06);
        border-color: rgba(45, 212, 191, 0.12);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255, 255, 255, 0.03);
        border-color: rgba(255, 255, 255, 0.08);
    }

    /* ── Chat Input ─────────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        border: 1px solid rgba(45, 212, 191, 0.2);
        background: rgba(255, 255, 255, 0.04);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(45, 212, 191, 0.5);
        box-shadow: 0 0 0 2px rgba(45, 212, 191, 0.1);
    }

    /* ── Expanders ──────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
    }

    /* ── Status Cards (sidebar) ─────────────────────────────────── */
    .status-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .status-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6B7280;
        margin-bottom: 0.2rem;
    }
    .status-card .value {
        font-size: 0.9rem;
        color: #2DD4BF;
        font-weight: 600;
        font-family: 'Courier New', monospace;
    }

    /* ── Divider ────────────────────────────────────────────────── */
    .teal-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(45, 212, 191, 0.3), transparent);
        margin: 1rem 0;
    }

    /* ── Footer ─────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: #4B5563;
        font-size: 0.75rem;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 2rem;
    }

    /* ── How It Works Steps ─────────────────────────────────────── */
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        margin-bottom: 0.6rem;
    }
    .step-num {
        background: rgba(45, 212, 191, 0.15);
        color: #2DD4BF;
        font-size: 0.7rem;
        font-weight: 700;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .step-text {
        color: #9CA3AF;
        font-size: 0.82rem;
        line-height: 1.4;
    }

    /* ── Example Questions ──────────────────────────────────────── */
    .example-q {
        background: rgba(45, 212, 191, 0.06);
        border: 1px solid rgba(45, 212, 191, 0.12);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        color: #D1D5DB;
        font-size: 0.82rem;
        cursor: default;
    }

    /* ── Document Card ──────────────────────────────────────────── */
    .doc-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .doc-card .doc-name {
        color: #2DD4BF;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        word-break: break-all;
    }
    .doc-card .doc-meta {
        color: #6B7280;
        font-size: 0.72rem;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Petroleum RAG — Multi-Document Assistant",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# ── Initialize Knowledge Base ──────────────────────────────────────────────
kb = get_knowledge_base()
embedding_model = load_embedding_model()

# Auto-migrate existing PDF if KB is empty
auto_migrate(kb)

# Rebuild FAISS if needed (e.g., after deletion)
if kb.needs_faiss_rebuild:
    with st.spinner("🔄 Rebuilding search index..."):
        kb.rebuild_faiss(embedding_model)

# ── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🛢️ Multi-Document RAG</div>
    <div class="hero-title">Petroleum Engineering RAG</div>
    <p class="hero-subtitle">
        Multi-document ChatPDF — upload PDFs, ask questions, get answers grounded in your documents
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Knowledge Base Stats
    st.markdown("### 📊 Knowledge Base")
    stats = kb.get_stats()
    st.markdown(f"""
    <div class="status-card">
        <div class="label">Total Documents</div>
        <div class="value">{stats['total_documents']}</div>
    </div>
    <div class="status-card">
        <div class="label">Total Pages</div>
        <div class="value">{stats['total_pages']:,}</div>
    </div>
    <div class="status-card">
        <div class="label">Total Chunks</div>
        <div class="value">{stats['total_chunks']:,}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

    # Models
    st.markdown("### 🧠 Models")
    st.markdown(f"""
    <div class="status-card">
        <div class="label">Embedding Model</div>
        <div class="value">{EMBEDDING_MODEL_NAME}</div>
    </div>
    <div class="status-card">
        <div class="label">Vector Database</div>
        <div class="value">FAISS IndexFlatIP</div>
    </div>
    <div class="status-card">
        <div class="label">LLM Model</div>
        <div class="value">{HF_MODEL}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

    # Upload Section
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF to add it to the knowledge base",
    )

    if uploaded_file is not None:
        # Check if this file is already processed (by checking session state)
        upload_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
        if upload_key not in st.session_state:
            # Save to temp file and process
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.spinner(f"📄 Processing '{uploaded_file.name}'..."):
                result = kb.add_document(tmp_path, uploaded_file.name, embedding_model)

            # Clean up temp file
            os.unlink(tmp_path)

            if result["success"]:
                st.success(f"✅ {result['message']}")
                st.session_state[upload_key] = True
                st.rerun()
            else:
                st.warning(f"⚠️ {result['message']}")
                st.session_state[upload_key] = True

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

    # Uploaded Documents
    st.markdown("### 📚 Uploaded Documents")
    docs = kb.get_all_documents()

    if docs.empty:
        st.markdown("""
        <div style="color: #6B7280; font-size: 0.82rem; text-align: center; padding: 1rem 0;">
            No documents uploaded yet.
        </div>
        """, unsafe_allow_html=True)
    else:
        for _, doc in docs.iterrows():
            doc_id = doc["document_id"]
            doc_name = doc["document_name"]
            doc_pages = doc["num_pages"]
            doc_chunks = doc["num_chunks"]
            upload_time = doc.get("upload_time", "Unknown")

            # Format upload time
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(str(upload_time))
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_str = str(upload_time)[:16]

            file_size = doc.get("file_size", 0)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024*1024):.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.0f} KB"
            else:
                size_str = f"{file_size} B"

            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-name">📄 {doc_name}</div>
                <div class="doc-meta">
                    {doc_pages} pages · {doc_chunks} chunks · {size_str}<br>
                    Uploaded: {time_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # View metadata and Delete buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("ℹ️ Details", key=f"info_{doc_id[:12]}", use_container_width=True):
                    st.session_state[f"show_info_{doc_id[:12]}"] = True
            with col2:
                if st.button("🗑️ Delete", key=f"del_{doc_id[:12]}", use_container_width=True):
                    st.session_state[f"confirm_delete_{doc_id[:12]}"] = True

            # Show metadata if requested
            if st.session_state.get(f"show_info_{doc_id[:12]}", False):
                st.markdown(f"""
                **Document ID:** `{doc_id[:16]}...`<br>
                **File Name:** {doc_name}<br>
                **Pages:** {doc_pages}<br>
                **Chunks:** {doc_chunks}<br>
                **File Size:** {size_str}<br>
                **Uploaded:** {time_str}
                """, unsafe_allow_html=True)
                if st.button("Close", key=f"close_{doc_id[:12]}"):
                    st.session_state[f"show_info_{doc_id[:12]}"] = False
                    st.rerun()

            # Confirm deletion
            if st.session_state.get(f"confirm_delete_{doc_id[:12]}", False):
                st.warning(f"Delete **{doc_name}**? This will remove all {doc_chunks} chunks.")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Confirm", key=f"yes_{doc_id[:12]}", use_container_width=True):
                        result = kb.delete_document(doc_id)
                        if result["success"]:
                            st.success(result["message"])
                            # Clean up session state
                            for key_suffix in ["show_info_", "confirm_delete_"]:
                                st.session_state.pop(f"{key_suffix}{doc_id[:12]}", None)
                            st.rerun()
                        else:
                            st.error(result["message"])
                with col_b:
                    if st.button("❌ Cancel", key=f"no_{doc_id[:12]}", use_container_width=True):
                        st.session_state[f"confirm_delete_{doc_id[:12]}"] = False
                        st.rerun()

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

    # How It Works
    st.markdown("### 📖 How It Works")
    st.markdown("""
    <div class="step-item"><div class="step-num">1</div><div class="step-text">Upload PDF documents to the knowledge base</div></div>
    <div class="step-item"><div class="step-num">2</div><div class="step-text">Ask a question about your documents</div></div>
    <div class="step-item"><div class="step-num">3</div><div class="step-text">FAISS retrieves relevant chunks across all documents</div></div>
    <div class="step-item"><div class="step-num">4</div><div class="step-text">LLM generates an answer grounded in the evidence</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

    # Try Asking
    st.markdown("### 💡 Try Asking")
    st.markdown("""
    <div class="example-q">"What is drill string design?"</div>
    <div class="example-q">"How does rotary drilling work?"</div>
    <div class="example-q">"Explain well control methods"</div>
    <div class="example-q">"What are the types of drill bits?"</div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

    # Groq API key status
    if not HF_TOKEN:
        st.warning("**GROQ_API_KEY** not set. Add it as a secret for the LLM to work.")
    else:
        st.success("**GROQ_API_KEY** configured")

    st.markdown("""
    <div style="color: #4B5563; font-size: 0.75rem; line-height: 1.5;">
        <strong style="color: #6B7280;">Tech Stack</strong><br>
        Streamlit · FAISS · Sentence-Transformers · Groq Inference API
    </div>
    """, unsafe_allow_html=True)


# ── Main Area ──────────────────────────────────────────────────────────────

# Check if knowledge base has documents
if kb.is_empty():
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #6B7280;">
        <div style="font-size: 3rem; margin-bottom: 0.8rem;">📄</div>
        <div style="font-size: 1.2rem; color: #2DD4BF; font-weight: 600; margin-bottom: 0.5rem;">
            No Documents in Knowledge Base
        </div>
        <div style="font-size: 0.95rem; max-width: 480px; margin: 0 auto; line-height: 1.6;">
            Upload a PDF document using the sidebar to get started.
            The existing textbook will be automatically indexed on first run.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Chat History ────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome message on first load
    if not st.session_state.messages:
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem 1rem; color: #6B7280;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛢️</div>
            <div style="font-size: 1.1rem; color: #2DD4BF; font-weight: 600; margin-bottom: 0.3rem;">
                Welcome to Petroleum RAG
            </div>
            <div style="font-size: 0.9rem; max-width: 480px; margin: 0 auto; line-height: 1.6;">
                Ask any question about drilling engineering. Answers are generated from
                <strong>{stats['total_documents']}</strong> indexed document(s)
                ({stats['total_chunks']:,} chunks) using retrieval-augmented generation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        doc_name = src.get("document_name", "Unknown")
                        st.markdown(
                            f"- **{doc_name}** — Page {src['page']}, {src['chapter']}"
                        )
            if message.get("chunks"):
                with st.expander("🔍 Retrieved Chunks"):
                    for chunk in message["chunks"]:
                        doc_name = chunk.get("document_name", "Unknown")
                        st.markdown(
                            f"**{chunk['chunk_id']}** ({doc_name}, Page {chunk['page']}, "
                            f"Score: {chunk['score']:.4f})\n"
                            f"> {chunk['text']}"
                        )

    # ── Chat Input ──────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not HF_TOKEN:
                st.error(
                    "❌ **GROQ_API_KEY not set.** Please add your Groq API key as a secret in "
                    "Streamlit Cloud settings → Secrets.\n\n"
                    "Get a free key at: https://console.groq.com/keys"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "❌ GROQ_API_KEY not configured. Please add it in Streamlit Cloud secrets.",
                })
            else:
                with st.spinner("🔍 Searching documents and generating answer..."):
                    try:
                        # Use knowledge base's multi-document context builder
                        package = kb.build_context_package(
                            prompt,
                            embedding_model,
                            retrieval_k=10,
                            max_context_chunks=4,
                            max_chunks_per_page=2,
                            word_budget=220,
                        )

                        if package["num_sources"] == 0:
                            answer = "I couldn't find relevant information in the indexed documents for your question. Please try rephrasing or upload a relevant document."
                            st.markdown(answer)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                            })
                        else:
                            # Generate answer using the existing HF function
                            answer = generate_answer_hf(
                                prompt,
                                package["context_text"],
                                model_name=HF_MODEL,
                                hf_token=HF_TOKEN,
                            )

                            st.markdown(answer)

                            # Build sources list with document names
                            sources = []
                            if len(package["selected"]) > 0:
                                for _, row in package["selected"].iterrows():
                                    sources.append({
                                        "page": int(row["page_number"]),
                                        "chapter": row["chapter"],
                                        "chunk_id": row["chunk_id"],
                                        "document_name": row.get("document_name", "Unknown"),
                                    })

                            if sources:
                                with st.expander("📚 Sources"):
                                    for src in sources:
                                        st.markdown(
                                            f"- **{src['document_name']}** — "
                                            f"Page {src['page']}, {src['chapter']}"
                                        )

                            # Build retrieved chunks list with document names
                            retrieved_chunks = []
                            if len(package["candidates"]) > 0:
                                for _, row in package["candidates"].head(8).iterrows():
                                    retrieved_chunks.append({
                                        "chunk_id": row["chunk_id"],
                                        "page": int(row["page_number"]),
                                        "chapter": row["chapter"],
                                        "score": round(float(row["score"]), 4),
                                        "text": row["chunk_text"][:300] + ("..." if len(row["chunk_text"]) > 300 else ""),
                                        "document_name": row.get("document_name", "Unknown"),
                                    })

                            if retrieved_chunks:
                                with st.expander("🔍 Retrieved Chunks"):
                                    for chunk in retrieved_chunks:
                                        st.markdown(
                                            f"**{chunk['chunk_id']}** ({chunk['document_name']}, "
                                            f"Page {chunk['page']}, Score: {chunk['score']:.4f})\n"
                                            f"> {chunk['text']}"
                                        )

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                                "chunks": retrieved_chunks,
                            })

                    except Exception as e:
                        error_msg = f"❌ Error generating answer: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Petroleum Engineering RAG · Multi-Document Knowledge Base<br>
    Built with Streamlit · FAISS · Sentence-Transformers · Groq
</div>
""", unsafe_allow_html=True)
