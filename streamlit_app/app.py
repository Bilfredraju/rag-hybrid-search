import os

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

REQUEST_TIMEOUT = 60


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Universal AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .status-card {
        padding: 0.8rem;
        border-radius: 0.6rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.6rem;
    }

    .route-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background-color: #eef2ff;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }

    .source-card {
        padding: 0.7rem;
        border-left: 3px solid #6366f1;
        background-color: #f8fafc;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# API HELPERS
# ============================================================


def api_get(endpoint):
    """Perform a GET request against the backend API."""

    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


def api_post(endpoint, **kwargs):
    """Perform a POST request against the backend API."""

    response = requests.post(
        f"{API_URL}{endpoint}",
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    )

    response.raise_for_status()

    return response.json()


def api_delete(endpoint):
    """Perform a DELETE request against the backend API."""

    response = requests.delete(
        f"{API_URL}{endpoint}",
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# SOURCE DISPLAY
# ============================================================


def display_sources(sources):
    """Display document and web sources."""

    if not sources:
        return

    with st.expander(
        f"📚 Sources ({len(sources)})",
        expanded=False,
    ):

        for index, source in enumerate(sources, start=1):

            source_type = source.get(
                "type",
                "unknown",
            )

            source_name = source.get(
                "source",
                "Unknown source",
            )

            page = source.get("page")

            verified = source.get(
                "verified"
            )

            evidence_type = source.get(
                "evidence_type"
            )

            st.markdown(
                f"**{index}. {source_name}**"
            )

            if source_type == "document":

                if page is not None:
                    st.caption(
                        f"📄 Document • Page {page}"
                    )
                else:
                    st.caption(
                        "📄 Document"
                    )

            elif source_type == "web":

                if verified is True:
                    status = "✅ Verified web page"

                elif evidence_type == "search_fallback":
                    status = (
                        "⚠️ Search-result fallback "
                        "(page not verified)"
                    )

                else:
                    status = "🌐 Web evidence"

                st.caption(status)

                url = source.get("url")

                if url:
                    st.markdown(
                        f"[Open source]({url})"
                    )

            else:

                st.caption(
                    f"Source type: {source_type}"
                )

            st.divider()


# ============================================================
# ASSISTANT RESPONSE DISPLAY
# ============================================================


def display_assistant_response(data):
    """Display an assistant response."""

    answer = data.get(
        "answer",
        "No answer returned.",
    )

    route = data.get(
        "route",
        "unknown",
    )

    confidence = data.get(
        "confidence",
        {},
    )

    st.markdown(
        f"""
        <div class="route-badge">
            Route: {route.upper()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(answer)

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if confidence:

        confident = confidence.get(
            "confident"
        )

        score = confidence.get(
            "score"
        )

        reason = confidence.get(
            "reason"
        )

        if confident:

            if score is not None:

                st.caption(
                    f"🟢 Retrieval confidence: "
                    f"{float(score):.3f}"
                )

            else:

                st.caption(
                    "🟢 Retrieval confidence: sufficient"
                )

        else:

            st.caption(
                "🟡 Retrieval confidence: limited"
            )

        if reason:
            st.caption(
                f"Reason: {reason}"
            )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    display_sources(
        data.get("sources", [])
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Universal AI Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions across your documents, general AI knowledge,
    and current web research — all through one assistant.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 Universal AI Assistant")

    st.caption(
        "Document + General Knowledge + Live Web Research"
    )

    st.divider()

    # --------------------------------------------------------
    # Backend Status
    # --------------------------------------------------------

    st.subheader("System Status")

    try:

        health = api_get("/health")

        if health.get("status") == "healthy":

            st.success(
                "Backend connected"
            )

            st.caption(
                f"API version: "
                f"{health.get('version', 'unknown')}"
            )

        else:

            st.warning(
                "Backend reported an unhealthy state."
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Backend unavailable"
        )

        st.caption(
            f"Expected API: {API_URL}"
        )

    except requests.exceptions.Timeout:

        st.warning(
            "Backend health check timed out."
        )

    except Exception as exc:

        st.warning(
            f"Health check failed: {exc}"
        )

    st.divider()

    # --------------------------------------------------------
    # Upload PDF
    # --------------------------------------------------------

    st.subheader("📄 Add Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help=(
            "The PDF will be parsed, chunked, embedded, "
            "and added to the knowledge base."
        ),
    )

    if uploaded_file is not None:

        if st.button(
            "⬆️ Upload & Index",
            use_container_width=True,
        ):

            with st.spinner(
                "Uploading and indexing document..."
            ):

                try:

                    result = api_post(
                        "/documents/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                    )

                    st.success(
                        "Document indexed successfully."
                    )

                    document = result.get(
                        "document",
                        {},
                    )

                    if document:

                        st.caption(
                            f"Pages: "
                            f"{document.get('pages', '-')}"
                        )

                        st.caption(
                            f"Chunks: "
                            f"{document.get('chunks', '-')}"
                        )

                    st.rerun()

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Unable to connect to the backend."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "Document upload timed out."
                    )

                except requests.exceptions.HTTPError as exc:

                    st.error(
                        f"Upload failed: {exc}"
                    )

                except Exception as exc:

                    st.error(
                        f"Unexpected error: {exc}"
                    )

    st.divider()

    # --------------------------------------------------------
    # Knowledge Base
    # --------------------------------------------------------

    st.subheader("📚 Knowledge Base")

    try:

        document_response = api_get(
            "/documents"
        )

        documents = document_response.get(
            "documents",
            [],
        )

        document_count = document_response.get(
            "count",
            len(documents),
        )

        total_chunks = sum(
            int(document.get("chunks", 0))
            for document in documents
        )

        st.metric(
            "Indexed Documents",
            document_count,
        )

        st.metric(
            "Text Chunks",
            total_chunks,
        )

        if documents:

            with st.expander(
                "View documents",
                expanded=False,
            ):

                for document in documents:

                    filename = document.get(
                        "filename",
                        "Unknown",
                    )

                    pages = document.get(
                        "pages",
                        0,
                    )

                    chunks = document.get(
                        "chunks",
                        0,
                    )

                    st.markdown(
                        f"**📄 {filename}**"
                    )

                    st.caption(
                        f"{pages} pages • "
                        f"{chunks} chunks"
                    )

                    document_id = document.get(
                        "document_id"
                    )

                    if document_id:

                        if st.button(
                            "Delete",
                            key=f"delete_{document_id}",
                        ):

                            try:

                                api_delete(
                                    f"/documents/{document_id}"
                                )

                                st.success(
                                    f"Deleted {filename}"
                                )

                                st.rerun()

                            except Exception as exc:

                                st.error(
                                    f"Delete failed: {exc}"
                                )

                    st.divider()

    except Exception as exc:

        st.warning(
            f"Could not load documents: {exc}"
        )

    st.divider()

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    st.subheader("⚙️ Architecture")

    st.markdown(
        """
        **Routing**
        - Document RAG
        - General LLM
        - Current Web
        - Document + Web

        **Retrieval**
        - Semantic Search
        - BM25
        - Reciprocal Rank Fusion
        - CrossEncoder

        **Knowledge**
        - ChromaDB
        - Sentence Transformers

        **Generation**
        - Groq

        **Web Research**
        - Search
        - Source ranking
        - Concurrent fetching
        - Verified evidence
        - Fallback evidence
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Clear Chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "Universal AI Assistant v4.0"
    )

    st.caption(
        "FastAPI • Streamlit • Groq • ChromaDB"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant",
    )

    with st.chat_message(role):

        if role == "assistant":

            display_assistant_response(
                message.get(
                    "data",
                    {},
                )
            )

        else:

            st.write(
                message.get(
                    "content",
                    "",
                )
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your documents, AI, or current events..."
)


if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.write(question)

    # --------------------------------------------------------
    # Assistant request
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Thinking..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 200:

                    data = response.json()

                    display_assistant_response(
                        data
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "data": data,
                        }
                    )

                else:

                    try:
                        error_detail = response.json().get(
                            "detail",
                            response.text,
                        )
                    except Exception:
                        error_detail = response.text

                    error_message = (
                        f"Backend error "
                        f"({response.status_code}): "
                        f"{error_detail}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "data": {
                                "answer": error_message,
                                "route": "error",
                                "sources": [],
                            },
                        }
                    )

            except requests.exceptions.ConnectionError:

                error_message = (
                    "⚠️ Unable to connect to the backend.\n\n"
                    f"Please make sure FastAPI is running at:\n"
                    f"`{API_URL}`"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "data": {
                            "answer": error_message,
                            "route": "error",
                            "sources": [],
                        },
                    }
                )

            except requests.exceptions.Timeout:

                error_message = (
                    "⚠️ The request timed out.\n\n"
                    "The assistant may be performing "
                    "web research or processing a large request."
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "data": {
                            "answer": error_message,
                            "route": "error",
                            "sources": [],
                        },
                    }
                )

            except Exception as exc:

                error_message = (
                    f"⚠️ Unexpected error: {exc}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "data": {
                            "answer": error_message,
                            "route": "error",
                            "sources": [],
                        },
                    }
                )