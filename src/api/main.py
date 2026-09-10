from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import DOCS_DIR
from src.documents.document_manager import DocumentManager
from src.pipeline.rag_pipeline import RAGPipeline


app = FastAPI(
    title="Enterprise RAG API",
    description=(
        "Hybrid RAG API with PDF document management."
    ),
    version="2.0.0",
)


pipeline = RAGPipeline()
document_manager = DocumentManager()


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Enterprise RAG API is running",
        "version": "2.0.0",
    }


@app.post("/ask")
def ask_question(data: Question):
    """
    Ask a question against indexed documents.
    """

    if not data.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    try:
        return pipeline.ask(data.question)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Upload and index a PDF document.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:
        DOCS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = DOCS_DIR / Path(
            file.filename
        ).name

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        destination.write_bytes(content)

        document = (
            document_manager.ingest_pdf(
                destination
            )
        )

        # BM25 is loaded into memory when the
        # pipeline starts, so refresh it after
        # document ingestion.
        pipeline.refresh_indexes()

        return {
            "message": (
                "Document uploaded and indexed "
                "successfully."
            ),
            "document": document,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/documents")
def list_documents():
    """
    List all indexed documents.
    """

    try:
        return {
            "documents": (
                document_manager
                .list_documents()
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str
):
    """
    Delete an indexed document.
    """

    try:
        result = (
            document_manager
            .delete_document(
                document_id
            )
        )

        pipeline.refresh_indexes()

        return {
            "message": (
                "Document deleted successfully."
            ),
            "result": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )