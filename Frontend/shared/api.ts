/**
 * Shared code between client and server
 * Useful to share types between client and server
 * and/or small pure JS functions that can be used on both client and server
 */

export interface DemoResponse {
  message: string;
}

export type Jurisdiction = "Maryland" | "Federal" | "Other";

export interface RegDocumentMeta {
  id: string;
  title: string;
  agency: string; // e.g., Maryland Dept of Agriculture
  jurisdiction: Jurisdiction;
  year: number;
  sourceUrl?: string;
}

export interface RegDocument extends RegDocumentMeta {
  text: string; // full plain text body indexed for search
}

export interface SearchFilters {
  jurisdiction?: Jurisdiction;
  agency?: string;
  yearFrom?: number;
  yearTo?: number;
  // New filters
  titleSubtitleChapter?: string; // e.g., Title 15 or Subtitle 26
  section?: string; // e.g., 15.20.13.02
}

export interface ListDocumentsResponse {
  documents: RegDocumentMeta[];
}

export interface AddDocumentRequest extends RegDocumentMeta {
  text: string;
}

export interface AddDocumentResponse {
  ok: true;
  document: RegDocumentMeta;
}

export interface ChatCitation {
  id: string;
  label: string;
  pages: string;
  url?: string;
  docId?: string;
  docTitle?: string;
  comarNumber?: string;
  comarDisplay?: string;
  snippet?: string;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  citations?: ChatCitation[];
}

export interface ChatSessionSummary {
  id: string;
  title: string;
  snippet?: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ChatSessionDetail extends ChatSessionSummary {
  messages: ChatMessage[];
}

export interface ChatAskRequest {
  question: string;
  filters?: SearchFilters;
  topK?: number;
  sessionId?: string;
  sessionTitle?: string;
}

export interface ChatAskResponse {
  answer: string;
  sources: ChatCitation[];
  sessionId: string;
  userMessage: ChatMessage;
  assistantMessage: ChatMessage;
  session: ChatSessionSummary;
}

export interface ListSessionsResponse {
  sessions: ChatSessionSummary[];
}

export interface ChatSessionResponse {
  session: ChatSessionDetail;
}

export interface ChatSessionCreateResponse {
  session: ChatSessionSummary;
}

export interface ChatMemory {
  id: string;
  content: string;
  createdAt: string;
  updatedAt: string;
}

export interface ListMemoriesResponse {
  memories: ChatMemory[];
}

export interface ChatMemoryResponse {
  memory: ChatMemory;
}
