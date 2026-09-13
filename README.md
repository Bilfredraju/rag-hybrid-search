# Universal AI Knowledge Assistant

> **A production-style AI knowledge assistant that dynamically routes queries between document-grounded Hybrid RAG, general LLM reasoning, real-time web research, and combined document + web reasoning.**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![RAG](https://img.shields.io/badge/RAG-Hybrid-purple.svg)](https://github.com/Bilfredraju/rag-hybrid-search)

---

## 📌 Overview

The **Universal AI Knowledge Assistant** is an AI-powered question-answering system designed to handle different types of user queries intelligently.

Instead of treating every question as a document-search problem, the system first **classifies the user's query** and dynamically selects the appropriate reasoning path:

```text
                         User Query
                              │
                              ▼
                     ┌─────────────────┐
                     │  Query Router   │
                     └────────┬────────┘
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
        DOCUMENT          GENERAL          CURRENT
              │               │                │
              ▼               ▼                ▼
        Hybrid RAG       General LLM      Web Research
              │               │                │
              └───────────────┼────────────────┘
                              │
                              ▼
                         BOTH Route
                  Document + Web Evidence
                              │
                              ▼
                     Evidence-Aware Answer
```

The system therefore supports:

- 📄 **Document-grounded question answering**
- 🧠 **General LLM reasoning**
- 🌐 **Current web research**
- 🔀 **Document + web reasoning**
- 🔎 **Hybrid semantic + keyword retrieval**
- 📚 **Source-aware responses**
- 🛡️ **Retrieval confidence evaluation**
- ⚡ **Web caching and parallel source fetching**
- 🧩 **FastAPI backend**
- 💬 **Streamlit user interface**

---

# 🚀 Key Features

## 1. Intelligent Query Routing

The assistant automatically classifies questions into four routes:

| Route      | Purpose                                             |
| ---------- | --------------------------------------------------- |
| `DOCUMENT` | Answer using indexed documents                      |
| `GENERAL`  | Answer using general LLM knowledge                  |
| `CURRENT`  | Research current information from the web           |
| `BOTH`     | Combine document evidence with current web research |

Example:

```text
"Who manages Project Phoenix?"
        ↓
DOCUMENT
        ↓
Hybrid RAG
```

```text
"What is the capital of France?"
        ↓
GENERAL
        ↓
LLM
```

```text
"What is the latest AI news?"
        ↓
CURRENT
        ↓
Web Research
```

```text
"Based on Project Phoenix, how does it compare
with the latest AI assistant trends?"
        ↓
BOTH
        ↓
Document RAG + Web Research
```

---

# 🔎 Hybrid RAG Pipeline

The document retrieval pipeline combines multiple retrieval strategies.

```text
PDF Documents
     │
     ▼
PDF Parsing
     │
     ▼
Document Chunking
     │
     ▼
Sentence Transformer Embeddings
     │
     ▼
 ┌───────────────┬───────────────┐
 │               │               │
 ▼               ▼               │
Semantic Search  BM25 Search     │
 │               │               │
 └───────┬───────┘               │
         ▼                       │
 Reciprocal Rank Fusion          │
         │                       │
         ▼                       │
 Cross-Encoder Reranking         │
         │                       │
         ▼                       │
 Retrieval Confidence            │
         │                       │
         ▼                       │
      LLM Answer                 │
```

### Retrieval components

- **Semantic Search** — ChromaDB + Sentence Transformers
- **Keyword Search** — BM25
- **Rank Fusion** — Reciprocal Rank Fusion (RRF)
- **Reranking** — Cross-Encoder
- **Confidence Evaluation** — evidence-based retrieval filtering

---

# 🌐 Web Research Pipeline

For current-information queries, the assistant performs web research instead of relying only on the model's knowledge.

```text
User Query
    │
    ▼
Web Search
    │
    ▼
Multiple Search Results
    │
    ▼
Parallel Page Fetching
    │
    ▼
Content Extraction
    │
    ▼
Source Verification
    │
    ├── Verified Page Content
    │
    └── Search Result Fallback
    │
    ▼
Evidence-Aware LLM Response
```

The web research layer includes:

- Search result retrieval
- Parallel source fetching
- HTML content extraction
- Verified page evidence
- Search-result fallback handling
- Query-aware caching
- Source reliability tracking
- Domain failure tracking
- Configurable timeouts
- Graceful failure handling

---

# 🔀 Document + Web Reasoning

The `BOTH` route combines information from the user's documents with current web evidence.

Example:

```text
User:
"Based on Project Phoenix, how does it compare
with the latest AI assistant trends?"
```

The system:

1. Retrieves relevant Project Phoenix document evidence.
2. Searches the web for current AI assistant trends.
3. Fetches and verifies relevant pages.
4. Combines both evidence sources.
5. Generates a grounded comparison.

The response distinguishes between:

- **Document evidence**
- **Verified web evidence**
- **Search-result fallback evidence**

This helps prevent the system from treating an unverified search snippet as equivalent to verified webpage content.

---

# 🤖 LLM Layer

The system uses a Groq-hosted LLM for answer generation.

The generation layer supports:

- Document-grounded prompts
- General knowledge prompts
- Current web research prompts
- Combined document + web prompts
- Evidence-aware response generation
- Graceful handling of generation failures

---

# 📊 Evaluation

The project includes an evaluation framework covering:

- Routing accuracy
- Source accuracy
- Concept accuracy
- Answer success rate
- Overall success rate
- Average latency

The evaluation dataset includes:

- Document questions
- General knowledge questions
- Current web questions
- Document + web questions

### Current evaluation result

**Overall evaluation success: 100%**

The evaluated routing, source, concept, and answer-success metrics reached **100%** on the current evaluation dataset.

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │       User          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Streamlit UI     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Query Router     │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
        DOCUMENT                 GENERAL                CURRENT
              │                     │                     │
              ▼                     ▼                     ▼
        Hybrid RAG              Groq LLM             Web Research
              │                                           │
              └──────────────────┬────────────────────────┘
                                 │
                                 ▼
                              BOTH
                                 │
                                 ▼
                    Document + Web Evidence
                                 │
                                 ▼
                         Evidence-Aware LLM
                                 │
                                 ▼
                         Final Answer
```

---

# 🧰 Technology Stack

| Category          | Technology               |
| ----------------- | ------------------------ |
| Language          | Python                   |
| LLM               | Groq                     |
| RAG               | Hybrid RAG               |
| Vector Database   | ChromaDB                 |
| Embeddings        | Sentence Transformers    |
| Keyword Retrieval | BM25                     |
| Rank Fusion       | Reciprocal Rank Fusion   |
| Reranking         | Cross-Encoder            |
| PDF Processing    | PyMuPDF                  |
| Backend           | FastAPI                  |
| Frontend          | Streamlit                |
| Web Search        | DDGS                     |
| Web Extraction    | Requests + BeautifulSoup |
| Configuration     | python-dotenv            |
| Containerization  | Docker                   |
| CI                | GitHub Actions           |

---

# 📁 Project Structure

```text
rag-hybrid-search/
│
├── src/
│   ├── api/
│   ├── assistant/
│   ├── chunking/
│   ├── documents/
│   ├── embeddings/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── retrieval/
│   ├── router/
│   └── web/
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│
├── docs/
│
├── .github/
│   └── workflows/
│
├── .env.example
├── .gitignore
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Bilfredraju/rag-hybrid-search.git

cd rag-hybrid-search
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv

venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key

GROQ_MODEL=openai/gpt-oss-120b

EMBEDDING_MODEL=all-MiniLM-L6-v2

RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

CHROMA_COLLECTION=rag_documents
```

Additional retrieval and web-research settings are available in `.env.example`.

**Never commit your actual `.env` file or API keys.**

---

# ▶️ Running the Application

## Start FastAPI

```bash
uvicorn src.api.main:app --reload
```

API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## Start Streamlit

In another terminal:

```bash
streamlit run streamlit_app/app.py
```

Application:

```text
http://localhost:8501
```

---

# 📄 Document Management

The application supports PDF document management through the API and Streamlit interface.

Supported operations include:

- PDF upload
- PDF parsing
- Chunk creation
- Embedding generation
- Vector indexing
- Document listing
- Document deletion
- Retrieval index refresh

Uploaded documents are indexed into the hybrid retrieval pipeline.

---

# 🔌 API Endpoints

| Method   | Endpoint                   | Purpose                |
| -------- | -------------------------- | ---------------------- |
| `GET`    | `/`                        | API information        |
| `GET`    | `/health`                  | Health check           |
| `POST`   | `/ask`                     | Ask the AI assistant   |
| `POST`   | `/documents/upload`        | Upload and index a PDF |
| `GET`    | `/documents`               | List indexed documents |
| `DELETE` | `/documents/{document_id}` | Delete a document      |

---

# 🧪 Example Queries

### Document

```text
Who manages Project Phoenix?
```

### General

```text
What is the capital of France?
```

### Current Web

```text
What is the latest AI news?
```

### Document + Web

```text
Based on Project Phoenix, how does it compare
with the latest AI assistant trends?
```

---

# 🖥️ User Interface

The Streamlit application provides:

- Conversational chat interface
- Query route display
- Document upload
- Document management
- Source/evidence display
- Web evidence status
- API health monitoring
- Chat history
- Indexed document information

### Screenshots

Add screenshots here showing:

1. Streamlit home/chat interface
2. Document-grounded answer
3. Current web research answer
4. Document + web reasoning
5. FastAPI Swagger interface

Example:

```text
screenshots/
├── streamlit-home.png
├── document-answer.png
├── web-research.png
├── both-route.png
└── swagger-api.png
```

---

# 🐳 Docker

The project includes Docker support.

Build and run:

```bash
docker compose build

docker compose up
```

FastAPI:

```text
http://localhost:8000
```

Streamlit:

```text
http://localhost:8501
```

---

# 🔄 Continuous Integration

GitHub Actions validates the project through automated checks including:

- Dependency installation
- Python compilation
- Streamlit compilation
- Unit tests
- FastAPI import validation
- Docker image build

---

# 🎯 What This Project Demonstrates

This project demonstrates practical experience with:

- Retrieval-Augmented Generation
- Information retrieval
- Semantic search
- Keyword search
- Rank fusion
- Neural reranking
- LLM application development
- Query classification and routing
- Web research
- Evidence grounding
- API development
- Document processing
- Vector databases
- Caching
- Fault-tolerant system design
- Evaluation
- Docker
- CI/CD

---

# 🔮 Future Improvements

Potential future improvements include:

- Authentication and authorization
- Multi-user support
- Persistent conversation memory
- Advanced RAG evaluation dashboards
- Observability and monitoring
- Cloud deployment
- More robust web source verification
- Multimodal document support
- Agentic task execution

---

# 👨‍💻 Author

**Bilfredraju**

GitHub:

https://github.com/Bilfredraju

---

## 📜 License

This project is developed for educational and research purposes.
