"""
Entry point for the FastAPI back‑end.

This module creates an application instance, ingests all documents on
startup and exposes an API for asking questions against the indexed
corpus.  It relies on the `RAGEngine` defined in `rag.py` to
perform ingestion, retrieval and answer generation.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

import os

from .rag import RAGEngine

# instantiate the RAG engine and FastAPI app
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')
DATA_DIR = os.path.abspath(DATA_DIR)

rag = RAGEngine(data_dir=DATA_DIR)
app = FastAPI(title="MDA Regulatory Chatbot API", description="Answer agriculture regulatory questions using RAG and Llama")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 5


@app.on_event("startup")
async def startup_event() -> None:
    """Load and index documents on application start."""
    try:
        rag.ingest()
    except Exception as exc:
        # Log the error; raising prevents the app from starting
        raise RuntimeError(f"Failed to ingest documents: {exc}")


@app.post("/ask")
async def ask(req: AskRequest) -> Dict[str, Any]:
    """Answer a question using the RAG pipeline."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")
    k = req.k or 5
    try:
        answer_text, sources = rag.answer(question, k=int(k))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "answer": answer_text,
        "sources": sources
    }
