from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.assistant.assistant_pipeline import AssistantPipeline
from src.config import DOCS_DIR
from src.documents.document_manager import DocumentManager


# ============================================================
# API CONFIGURATION
# ============================================================

API_VERSION = "4.0.0"


app = FastAPI(
    title="Universal AI Assistant API",
    description=(
        "Production API for a universal AI assistant that supports "
        "document-grounded RAG, general LLM knowledge, current web "
        "research, and combined document + web reasoning."
    ),
    version=API_VERSION,
)


# ============================================================
# CORE SERVICES
# ============================================================

assistant = AssistantPipeline()
document_manager = DocumentManager()


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================


class QuestionRequest(BaseModel):
    """
    Request body for asking the assistant a question.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the Universal AI Assistant.",
        examples=[
            "Who manages Project Phoenix?",
            "What is the capital of France?",
            "What is the latest AI news?",
        ],
    )


class HealthResponse(BaseModel):
    """
    API health response.
    """

    status: str
    service: str
    version: str


class RootResponse(BaseModel):
    """
    API information response.
    """

    message: str
    version: str


class DocumentDeleteResponse(BaseModel):
    """
    Response returned after deleting a document.
    """

    message: str
    result: dict


# ============================================================
# ROOT / HEALTH
# ============================================================


@app.get(
    "/",
    response_model=RootResponse,
    tags=["System"],
)
def root():
    """
    Return basic API information.
    """

    return RootResponse(
        message="Universal AI Assistant API is running",
        version=API_VERSION,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health_check():
    """
    Health-check endpoint for monitoring and deployment systems.
    """

    return HealthResponse(
        status="healthy",
        service="universal-ai-assistant",
        version=API_VERSION,
    )


# ============================================================
# ASK ASSISTANT
# ============================================================


@app.post(
    "/ask",
    tags=["Assistant"],
)
def ask_question(data: QuestionRequest):
    """
    Ask the Universal AI Assistant a question.

    The AssistantPipeline automatically determines whether
    the query should use:

    - document retrieval
    - general LLM knowledge
    - current web research
    - document + web research
    """

    question = data.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    try:
        return assistant.ask(question)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Assistant request failed: {exc}",
        ) from exc


# ============================================================
# DOCUMENT MANAGEMENT
# ============================================================


@app.post(
    "/documents/upload",
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload and index a PDF document.

    The document is:

    1. Saved into the configured docs directory.
    2. Parsed into pages.
    3. Split into chunks.
    4. Embedded.
    5. Indexed in ChromaDB.
    6. Added to the document registry.
    7. Reloaded into the BM25 index.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    filename = Path(file.filename).name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:
        DOCS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = DOCS_DIR / filename

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        destination.write_bytes(content)

        document = document_manager.ingest_pdf(
            destination
        )

        assistant.refresh_indexes()

        return {
            "message": "Document uploaded and indexed successfully.",
            "document": document,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {exc}",
        ) from exc


@app.get(
    "/documents",
    tags=["Documents"],
)
def list_documents():
    """
    Return all indexed documents.
    """

    try:
        documents = document_manager.list_documents()

        return {
            "count": len(documents),
            "documents": documents,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {exc}",
        ) from exc


@app.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    tags=["Documents"],
)
def delete_document(
    document_id: str,
):
    """
    Delete an indexed document and refresh retrieval indexes.
    """

    document_id = document_id.strip()

    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="Document ID must not be empty.",
        )

    try:
        result = document_manager.delete_document(
            document_id
        )

        assistant.refresh_indexes()

        return DocumentDeleteResponse(
            message="Document deleted successfully.",
            result=result,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {exc}",
        ) from exc