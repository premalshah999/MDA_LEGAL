"""Persistence layer for chat sessions and user memories."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from uuid import uuid4


def _utcnow() -> str:
    """Return an ISO-8601 timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat()


def _summarize(text: str, limit: int = 140) -> str:
    """Return a short snippet suitable for previews."""

    snippet = " ".join(text.strip().split())
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 1] + "…"


@dataclass
class ChatMessageRecord:
    """Representation of a single chat message."""

    id: str
    role: str
    content: str
    created_at: str
    citations: Optional[List[Dict[str, object]]] = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "createdAt": self.created_at,
        }
        if self.citations:
            data["citations"] = self.citations
        return data


class ChatStore:
    """SQLite-backed persistence for chat sessions and user memories."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Initialise schema
        with self._get_conn() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    snippet TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _get_conn(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- user helpers -----------------------------------------------------

    def ensure_user(self, user_id: str) -> None:
        ts = _utcnow()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (id, created_at) VALUES (?, ?)",
                (user_id, ts),
            )

    # -- session helpers --------------------------------------------------

    def _session_summary_from_row(self, row: sqlite3.Row) -> Dict[str, object]:
        return {
            "id": row["id"],
            "title": row["title"],
            "snippet": row["snippet"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def create_session(self, user_id: str, title: str) -> Dict[str, object]:
        session_id = str(uuid4())
        ts = _utcnow()
        title = title.strip() or "New chat"
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, title, snippet, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, user_id, title, None, ts, ts),
            )
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise RuntimeError("Failed to create session")
        return self._session_summary_from_row(row)

    def rename_session(self, user_id: str, session_id: str, title: str) -> Dict[str, object]:
        title = title.strip()
        if not title:
            raise ValueError("title must not be empty")
        ts = _utcnow()
        with self._get_conn() as conn:
            res = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (title, ts, session_id, user_id),
            )
            if res.rowcount == 0:
                raise KeyError("session not found")
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError("session not found")
        return self._session_summary_from_row(row)

    def delete_session(self, user_id: str, session_id: str) -> None:
        with self._get_conn() as conn:
            res = conn.execute("DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
            if res.rowcount == 0:
                raise KeyError("session not found")

    def list_sessions(self, user_id: str) -> List[Dict[str, object]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [self._session_summary_from_row(r) for r in rows]

    def get_session(self, user_id: str, session_id: str) -> Dict[str, object]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if row is None:
                raise KeyError("session not found")
            messages = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        session = self._session_summary_from_row(row)
        session["messages"] = [self._message_from_row(m).to_dict() for m in messages]
        return session

    def _message_from_row(self, row: sqlite3.Row) -> ChatMessageRecord:
        citations_json = row["citations"]
        citations: Optional[List[Dict[str, object]]]
        if citations_json:
            citations = json.loads(citations_json)
        else:
            citations = None
        return ChatMessageRecord(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            citations=citations,
        )

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, object]]] = None,
    ) -> ChatMessageRecord:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        msg_id = str(uuid4())
        ts = _utcnow()
        citations_json = json.dumps(citations or []) if citations else None
        with self._get_conn() as conn:
            res = conn.execute(
                "INSERT INTO messages (id, session_id, role, content, citations, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (msg_id, session_id, role, content, citations_json, ts),
            )
            if res.rowcount == 0:
                raise KeyError("session not found")
            snippet = None
            if role == "assistant":
                snippet = _summarize(content)
            conn.execute(
                "UPDATE sessions SET updated_at = ?, snippet = COALESCE(?, snippet) WHERE id = ?",
                (ts, snippet, session_id),
            )
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (msg_id,)).fetchone()
        if row is None:
            raise RuntimeError("failed to store message")
        return self._message_from_row(row)

    def get_recent_messages(self, session_id: str, limit: int = 6) -> List[Dict[str, object]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [self._message_from_row(r).to_dict() for r in reversed(rows)]

    # -- memory helpers ---------------------------------------------------

    def list_memories(self, user_id: str) -> List[Dict[str, object]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        memories: List[Dict[str, object]] = []
        for r in rows:
            memories.append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "createdAt": r["created_at"],
                    "updatedAt": r["updated_at"],
                }
            )
        return memories

    def add_memory(self, user_id: str, content: str) -> Dict[str, object]:
        content = content.strip()
        if not content:
            raise ValueError("content must not be empty")
        mem_id = str(uuid4())
        ts = _utcnow()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO memories (id, user_id, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (mem_id, user_id, content, ts, ts),
            )
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
        if row is None:
            raise RuntimeError("failed to create memory")
        return {
            "id": row["id"],
            "content": row["content"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def delete_memory(self, user_id: str, memory_id: str) -> None:
        with self._get_conn() as conn:
            res = conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            if res.rowcount == 0:
                raise KeyError("memory not found")

    def get_memory_text(self, user_id: str) -> List[str]:
        return [m["content"] for m in self.list_memories(user_id)]

