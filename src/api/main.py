from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.rag_pipeline import RAGPipeline

app = FastAPI(title="Enterprise RAG API")

pipeline = RAGPipeline()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_question(data: Question):

    return pipeline.ask(data.question)