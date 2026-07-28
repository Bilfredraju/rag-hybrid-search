import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "https://rag-hybrid-search-backend.onrender.com/ask")

st.set_page_config(
    page_title="Enterprise Hybrid RAG",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.title("🤖 Enterprise Hybrid RAG Assistant")

st.markdown("""
Search and interact with enterprise documents using an AI-powered
**Hybrid Retrieval-Augmented Generation (RAG)** system.

### Technology Stack

**Groq • ChromaDB • Sentence Transformers • Hybrid Search • CrossEncoder**
""")

st.markdown("---")

# -------------------------------------------------------
# Initialize Chat History
# -------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------------
# Display Previous Messages
# -------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("📄 Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"""
📄 **{source['source']}**

📑 **Page {source['page']}**
"""
                    )

# -------------------------------------------------------
# Chat Input
# -------------------------------------------------------

question = st.chat_input("Enter your question")

if question:

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # Assistant response

    with st.chat_message("assistant"):

        with st.spinner("🔍 Retrieving relevant documents..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=30
                )

                if response.status_code == 200:

                    data = response.json()

                    # FIX 3: use .get() to avoid KeyError if API response is malformed
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])

                    st.write(answer)

                    if sources:

                        with st.expander("📄 Sources"):

                            for source in sources:

                                st.markdown(
                                    f"""
📄 **{source['source']}**

📑 **Page {source['page']}**
"""
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                else:

                    error_msg = (
                        f"Server error ({response.status_code}). "
                        "Please try again."
                    )

                    st.error(error_msg)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg
                        }
                    )

            except requests.exceptions.ConnectionError:

                error_msg = (
                    "⚠️ Unable to connect to the backend.\n\n"
                    "Please make sure FastAPI is running on port 8000."
                )

                st.error(error_msg)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg
                    }
                )

            except requests.exceptions.Timeout:

                error_msg = (
                    "⚠️ Request timed out.\n\n"
                    "The server took too long to respond."
                )

                st.error(error_msg)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg
                    }
                )

            # FIX 1: catch-all so unexpected errors don't crash the demo with a raw traceback
            except Exception as e:

                error_msg = f"⚠️ Unexpected error: {str(e)}"

                st.error(error_msg)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg
                    }
                )

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.title("🤖 Enterprise RAG")
    st.caption("AI Document Intelligence")

    st.markdown("---")

    st.subheader("System Configuration")

    st.info(
        """
**LLM**
- Groq (Llama)

**Embeddings**
- Sentence Transformers

**Vector Database**
- ChromaDB

**Search**
- Hybrid Search
- BM25
- Semantic Search

**Ranking**
- Reciprocal Rank Fusion
- CrossEncoder
"""
    )

    st.markdown("---")

    st.subheader("Knowledge Base")

    # NOTE (FIX 2): still hardcoded for now — see note below
    st.write("📄 Indexed Documents: **3**")
    st.write("🧩 Text Chunks: **116**")

    st.markdown("---")

    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.caption("Enterprise Hybrid RAG v1.0")

    st.caption(
        """
Developed with

• FastAPI

• Groq

• ChromaDB

• Sentence Transformers

• BM25

• Reciprocal Rank Fusion

• CrossEncoder

• Streamlit
"""
    )