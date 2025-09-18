import { randomUUID } from "crypto";
import {
  ChatAskRequest,
  ChatAskResponse,
  ChatMemory,
  ChatMemoryResponse,
  ChatSessionCreateResponse,
  ChatSessionDetail,
  ChatSessionResponse,
  ChatSessionSummary,
  ListMemoriesResponse,
  ListSessionsResponse,
} from "@shared/api";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

async function backendFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const url = new URL(path, BACKEND_URL).toString();
  const headers: HeadersInit = {
    Accept: "application/json",
    ...(init.headers ?? {}),
  };
  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    let detail = `Backend request failed with status ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body && typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // ignore JSON parsing errors
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

function mapCitation(raw: any) {
  return {
    id: raw.id ?? raw.doc_id ?? raw.comar_number ?? randomUUID(),
    label: raw.label,
    pages: raw.pages,
    url: raw.url,
    docId: raw.doc_id,
    docTitle: raw.doc_title,
    comarNumber: raw.comar_number,
    comarDisplay: raw.comar_display,
    snippet: raw.snippet,
  };
}

function mapSessionSummary(raw: any): ChatSessionSummary {
  return {
    id: raw.id,
    title: raw.title,
    snippet: raw.snippet ?? null,
    createdAt: raw.createdAt ?? raw.created_at,
    updatedAt: raw.updatedAt ?? raw.updated_at,
  };
}

function mapMessage(raw: any) {
  return {
    id: raw.id,
    role: raw.role,
    content: raw.content,
    createdAt: raw.createdAt ?? raw.created_at,
    citations: Array.isArray(raw.citations) ? raw.citations.map(mapCitation) : undefined,
  };
}

export async function askBackend(userId: string, payload: ChatAskRequest): Promise<ChatAskResponse> {
  const body = {
    question: payload.question,
    k: payload.topK ?? 5,
    user_id: userId,
    session_id: payload.sessionId,
    session_title: payload.sessionTitle,
    filters: payload.filters,
  };
  const raw = await backendFetch<any>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return {
    answer: raw.answer,
    sources: Array.isArray(raw.sources) ? raw.sources.map(mapCitation) : [],
    sessionId: raw.session_id ?? raw.sessionId,
    userMessage: mapMessage(raw.user_message ?? raw.userMessage),
    assistantMessage: mapMessage(raw.assistant_message ?? raw.assistantMessage),
    session: mapSessionSummary(raw.session),
  };
}

export async function listSessions(userId: string): Promise<ChatSessionSummary[]> {
  const data = await backendFetch<ListSessionsResponse>(`/users/${encodeURIComponent(userId)}/sessions`);
  return Array.isArray(data.sessions) ? data.sessions.map(mapSessionSummary) : [];
}

export async function getSession(userId: string, sessionId: string): Promise<ChatSessionDetail> {
  const data = await backendFetch<ChatSessionResponse>(
    `/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(sessionId)}`,
  );
  const session = data.session;
  return {
    ...mapSessionSummary(session),
    messages: Array.isArray(session.messages) ? session.messages.map(mapMessage) : [],
  };
}

export async function createSession(userId: string, title?: string): Promise<ChatSessionSummary> {
  const data = await backendFetch<ChatSessionCreateResponse>(`/users/${encodeURIComponent(userId)}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return mapSessionSummary(data.session);
}

export async function renameSession(userId: string, sessionId: string, title: string): Promise<ChatSessionSummary> {
  const data = await backendFetch<ChatSessionCreateResponse>(
    `/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    },
  );
  return mapSessionSummary(data.session);
}

export async function removeSession(userId: string, sessionId: string): Promise<void> {
  await backendFetch(`/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

export async function listMemories(userId: string): Promise<ChatMemory[]> {
  const data = await backendFetch<ListMemoriesResponse>(`/users/${encodeURIComponent(userId)}/memories`);
  return Array.isArray(data.memories)
    ? data.memories.map((m) => ({
        id: m.id,
        content: m.content,
        createdAt: m.createdAt ?? (m as any).created_at,
        updatedAt: m.updatedAt ?? (m as any).updated_at,
      }))
    : [];
}

export async function addMemory(userId: string, content: string): Promise<ChatMemory> {
  const data = await backendFetch<ChatMemoryResponse>(`/users/${encodeURIComponent(userId)}/memories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  const m = data.memory;
  return {
    id: m.id,
    content: m.content,
    createdAt: m.createdAt ?? (m as any).created_at,
    updatedAt: m.updatedAt ?? (m as any).updated_at,
  };
}

export async function removeMemory(userId: string, memoryId: string): Promise<void> {
  await backendFetch(`/users/${encodeURIComponent(userId)}/memories/${encodeURIComponent(memoryId)}`, {
    method: "DELETE",
  });
}
