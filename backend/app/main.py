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
from .chat_store import ChatStore

# Load environment variables from .env file
import dotenv
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    dotenv.load_dotenv(env_path)

# instantiate the RAG engine and FastAPI app
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
BACKEND_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "chat_history.db")

# Use the new Supabase-backed engine (no data_dir argument)
rag = RAGEngine()
chat_store = ChatStore(DB_PATH)
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
    user_id: str
    session_id: Optional[str] = None
    session_title: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    createdAt: str
    citations: Optional[List[Dict[str, Any]]] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    user_message: ChatMessage
    assistant_message: ChatMessage
    session: Dict[str, Any]


class SessionCreateRequest(BaseModel):
    title: Optional[str] = None


class SessionRenameRequest(BaseModel):
    title: str


class MemoryRequest(BaseModel):
    content: str


@app.on_event("startup")
async def startup_event() -> None:
    """Load and index documents on application start."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting RAG engine ingestion...")
    try:
        rag.ingest()
        logger.info("RAG engine ingestion completed successfully")
    except Exception as exc:
        logger.error(f"Failed to ingest documents: {exc}")
        # Log the error; raising prevents the app from starting
        raise RuntimeError(f"Failed to ingest documents: {exc}")


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> Dict[str, Any]:
    """Answer a question using the RAG pipeline."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Received question: {req.question[:100]}...")
    
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")
    k_raw = req.k if req.k is not None else 5
    try:
        k = int(k_raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="k must be a positive integer")
    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")
    user_id = req.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    logger.info(f"Processing request for user: {user_id}, k: {k}")
    chat_store.ensure_user(user_id)

    session_summary: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = req.session_id.strip() if req.session_id else None
    if session_id:
        try:
            session_summary = chat_store.get_session(user_id, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
    if session_id and session_summary:
        history = chat_store.get_recent_messages(session_id)
    else:
        session_title = req.session_title.strip() if req.session_title else question[:60]
        session_summary = chat_store.create_session(user_id, session_title or "New chat")
        session_id = session_summary["id"]
        history = []

    memories = chat_store.get_memory_text(user_id)

    logger.info("Calling RAG engine for answer generation...")
    try:
        answer_text, sources = rag.answer(
            question,
            k=k,
            conversation=history,
            memories=memories,
            filters=req.filters,
        )
        logger.info(f"RAG engine returned answer with {len(sources)} sources")
    except Exception as exc:
        logger.error(f"RAG engine error: {exc}")
        raise HTTPException(status_code=500, detail=f"RAG engine error: {str(exc)}")

    try:
        user_record = chat_store.append_message(session_id, "user", question)
        assistant_record = chat_store.append_message(session_id, "assistant", answer_text, citations=sources)
        session_summary = chat_store.get_session(user_id, session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "answer": answer_text,
        "sources": sources,
        "session_id": session_id,
        "user_message": user_record.to_dict(),
        "assistant_message": assistant_record.to_dict(),
        "session": {k: v for k, v in session_summary.items() if k != "messages"},
    }


@app.get("/users/{user_id}/sessions")
async def list_sessions(user_id: str) -> Dict[str, Any]:
    user_id = user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    chat_store.ensure_user(user_id)
    sessions = chat_store.list_sessions(user_id)
    return {"sessions": sessions}


@app.post("/users/{user_id}/sessions")
async def create_session(user_id: str, payload: SessionCreateRequest) -> Dict[str, Any]:
    user_id = user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    chat_store.ensure_user(user_id)
    title = payload.title or "New chat"
    session = chat_store.create_session(user_id, title)
    return {"session": session}


@app.get("/users/{user_id}/sessions/{session_id}")
async def get_session(user_id: str, session_id: str) -> Dict[str, Any]:
    user_id = user_id.strip()
    session_id = session_id.strip()
    if not user_id or not session_id:
        raise HTTPException(status_code=400, detail="user_id and session_id are required")
    try:
        session = chat_store.get_session(user_id, session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session}


@app.patch("/users/{user_id}/sessions/{session_id}")
async def rename_session(user_id: str, session_id: str, payload: SessionRenameRequest) -> Dict[str, Any]:
    user_id = user_id.strip()
    session_id = session_id.strip()
    if not user_id or not session_id:
        raise HTTPException(status_code=400, detail="user_id and session_id are required")
    try:
        session = chat_store.rename_session(user_id, session_id, payload.title)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"session": session}


@app.delete("/users/{user_id}/sessions/{session_id}")
async def delete_session(user_id: str, session_id: str) -> Dict[str, Any]:
    user_id = user_id.strip()
    session_id = session_id.strip()
    if not user_id or not session_id:
        raise HTTPException(status_code=400, detail="user_id and session_id are required")
    try:
        chat_store.delete_session(user_id, session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@app.get("/users/{user_id}/memories")
async def list_memories(user_id: str) -> Dict[str, Any]:
    user_id = user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    chat_store.ensure_user(user_id)
    memories = chat_store.list_memories(user_id)
    return {"memories": memories}


@app.post("/users/{user_id}/memories")
async def add_memory(user_id: str, payload: MemoryRequest) -> Dict[str, Any]:
    user_id = user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        memory = chat_store.add_memory(user_id, payload.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"memory": memory}


@app.delete("/users/{user_id}/memories/{memory_id}")
async def delete_memory(user_id: str, memory_id: str) -> Dict[str, Any]:
    user_id = user_id.strip()
    memory_id = memory_id.strip()
    if not user_id or not memory_id:
        raise HTTPException(status_code=400, detail="user_id and memory_id are required")
    try:
        chat_store.delete_memory(user_id, memory_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"ok": True}
