# Enterprise Hybrid RAG System

An enterprise-grade Hybrid Retrieval-Augmented Generation (Hybrid RAG) system that combines semantic search, keyword search, reranking, and Large Language Models (LLMs) to provide accurate, context-aware answers from enterprise documents.

---

## Project Overview

This project enables intelligent document question answering by combining modern Retrieval-Augmented Generation (RAG) techniques with enterprise-grade search strategies.

Instead of relying only on vector search, this system combines multiple retrieval methods to improve accuracy and reduce hallucinations.

The application provides:

- Intelligent document retrieval
- Hybrid semantic + keyword search
- CrossEncoder reranking
- Context-aware answer generation using Groq LLM
- FastAPI backend
- Streamlit user interface
- Docker containerization
- GitHub Actions Continuous Integration (CI)

---

# Features

- PDF Document Processing
- Automatic Document Chunking
- Sentence Transformer Embeddings
- ChromaDB Vector Database
- BM25 Keyword Search
- Hybrid Search
- Reciprocal Rank Fusion (RRF)
- CrossEncoder Reranking
- Prompt Engineering
- Groq LLM Integration
- FastAPI REST API
- Streamlit Chat Interface
- Docker Support
- GitHub Actions CI

---

# Technology Stack

| Category         | Technology                               |
| ---------------- | ---------------------------------------- |
| Language         | Python 3.12                              |
| Backend          | FastAPI                                  |
| Frontend         | Streamlit                                |
| Vector Database  | ChromaDB                                 |
| Embedding Model  | Sentence Transformers (all-MiniLM-L6-v2) |
| Keyword Search   | BM25                                     |
| Rank Fusion      | Reciprocal Rank Fusion (RRF)             |
| Reranking        | CrossEncoder                             |
| LLM              | Groq (Llama 3.3 70B)                     |
| Containerization | Docker                                   |
| CI               | GitHub Actions                           |

---

# System Architecture

User

↓

Streamlit UI

↓

FastAPI

↓

Hybrid Retrieval Pipeline

├── Semantic Search (ChromaDB)

├── BM25 Search

↓

Reciprocal Rank Fusion

↓

CrossEncoder Reranker

↓

Prompt Builder

↓

Groq LLM

↓

Final Response + Sources

---

# Project Structure

```
rag-hybrid-search/

├── docs/
├── data/
├── src/
│   ├── api/
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── generation/
│   ├── pipeline/
│   └── evaluation/
│
├── streamlit_app/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone <repository-url>

cd rag-hybrid-search

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

---

# Run FastAPI

```bash
uvicorn src.api.main:app --reload
```

Swagger:

```
http://localhost:8000/docs
```

---

# Run Streamlit

```bash
streamlit run streamlit_app/app.py
```

---

# Run using Docker

Build:

```bash
docker compose build
```

Run:

```bash
docker compose up
```

FastAPI:

```
http://localhost:8000/docs
```

Streamlit:

```
http://localhost:8501
```

---

# CI Pipeline

GitHub Actions automatically performs:

- Checkout Repository
- Install Dependencies
- Compile Streamlit Application
- Run Unit Tests
- Validate FastAPI Import
- Build Docker Image

---

# Future Improvements

- Continuous Deployment (CD)
- Authentication & Authorization
- Multi-user Support
- Conversation Memory
- RAG Evaluation Dashboard
- Monitoring & Logging
- Cloud Deployment
- Document Versioning

---

# Screenshots

Add screenshots of:

- Streamlit Chat Interface
- Swagger UI
- GitHub Actions
- Docker Containers

---

# License

This project is developed for educational and research purposes.
