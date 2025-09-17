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

export interface SearchRequest {
  query: string;
  filters?: SearchFilters;
  topK?: number;
}

export interface Snippet {
  text: string;
  start: number;
  end: number;
}

export interface ScoredDoc {
  meta: RegDocumentMeta;
  score: number;
  snippet: Snippet;
  highlights: string[]; // matched terms
}

export interface SearchResponse {
  answer: string; // deterministic synthesis from top matches
  hits: ScoredDoc[];
  tokenUsage?: {
    // reserved for future LLM integration
    inputTokens?: number;
    outputTokens?: number;
  };
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
