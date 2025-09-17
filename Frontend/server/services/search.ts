import { AddDocumentRequest, RegDocument, RegDocumentMeta, ScoredDoc, SearchFilters, SearchRequest, SearchResponse } from "@shared/api";
import { seedCorpus } from "../data/corpus";

let documents: RegDocument[] = [...seedCorpus];

export function listDocuments(): RegDocumentMeta[] {
  return documents.map(({ text, ...meta }) => meta);
}

export function addDocument(doc: AddDocumentRequest): RegDocumentMeta {
  const exists = documents.find((d) => d.id === doc.id);
  if (exists) {
    // Replace existing document
    documents = documents.map((d) => (d.id === doc.id ? { ...doc } : d));
  } else {
    documents.push({ ...doc });
  }
  const { text, ...meta } = doc;
  return meta;
}

const tokenizer = (s: string): string[] =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

function applyFilters(docs: RegDocument[], f?: SearchFilters): RegDocument[] {
  if (!f) return docs;
  return docs.filter((d) => {
    if (f.jurisdiction && d.jurisdiction !== f.jurisdiction) return false;
    if (f.agency && d.agency.toLowerCase().indexOf(f.agency.toLowerCase()) === -1) return false;
    if (typeof f.yearFrom === "number" && d.year < f.yearFrom) return false;
    if (typeof f.yearTo === "number" && d.year > f.yearTo) return false;
    return true;
  });
}

function computeTfIdfScores(query: string, docs: RegDocument[]): { doc: RegDocument; score: number; highlights: string[] }[] {
  const qTokens = Array.from(new Set(tokenizer(query)));
  if (qTokens.length === 0) return docs.map((doc) => ({ doc, score: 0, highlights: [] }));

  // Document frequencies
  const df = new Map<string, number>();
  for (const t of qTokens) df.set(t, 0);
  const docTokens: Map<string, number>[] = [];

  for (const d of docs) {
    const tokens = tokenizer(d.text);
    const counts = new Map<string, number>();
    for (const t of tokens) {
      if (!counts.has(t)) counts.set(t, 0);
      counts.set(t, (counts.get(t) || 0) + 1);
    }
    docTokens.push(counts);
    for (const t of qTokens) {
      if (counts.has(t)) df.set(t, (df.get(t) || 0) + 1);
    }
  }

  const N = docs.length;
  return docs.map((doc, i) => {
    const counts = docTokens[i];
    let score = 0;
    for (const t of qTokens) {
      const tf = counts.get(t) || 0;
      const idf = Math.log((N + 1) / ((df.get(t) || 0) + 1)) + 1; // smoothed
      score += tf * idf;
    }
    return { doc, score, highlights: qTokens.filter((t) => (counts.get(t) || 0) > 0) };
  });
}

function bestSnippet(text: string, terms: string[], windowSize = 240): { text: string; start: number; end: number } {
  if (terms.length === 0) return { text: text.slice(0, windowSize), start: 0, end: Math.min(windowSize, text.length) };
  const lower = text.toLowerCase();
  let bestStart = 0;
  let bestEnd = Math.min(windowSize, text.length);
  let bestHits = -1;
  for (let i = 0; i < lower.length; i += Math.max(20, Math.floor(windowSize / 4))) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(text.length, start + windowSize);
    const chunk = lower.slice(start, end);
    const hits = terms.reduce((acc, t) => acc + (chunk.indexOf(t) >= 0 ? 1 : 0), 0);
    if (hits > bestHits) {
      bestHits = hits;
      bestStart = start;
      bestEnd = end;
    }
  }
  const snippet = (bestStart > 0 ? "…" : "") + text.slice(bestStart, bestEnd) + (bestEnd < text.length ? "…" : "");
  return { text: snippet, start: bestStart, end: bestEnd };
}

function synthesizeAnswer(scored: { doc: RegDocument; score: number; highlights: string[] }[], topK: number, query: string): string {
  const top = scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ doc }) => doc);

  if (top.length === 0) {
    return `No matching regulations found for "${query}" in the current corpus.`;
  }

  const bullets = top
    .map((d, idx) => {
      const summary = d.text.replace(/\s+/g, " ").slice(0, 220);
      return `${idx + 1}. ${d.title} — ${summary}${summary.length >= 220 ? "…" : ""}`;
    })
    .join("\n");

  return `Here is what the corpus says about "${query}":\n\n${bullets}\n\nCite the sources listed above in your reporting. Use filters to narrow jurisdiction/agency.`;
}

export function handleSearch(req: SearchRequest): SearchResponse {
  const topK = req.topK ?? 5;
  const filtered = applyFilters(documents, req.filters);
  const scored = computeTfIdfScores(req.query, filtered);
  const answer = synthesizeAnswer(scored, Math.min(topK, 5), req.query);

  const hits: ScoredDoc[] = scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ doc, score, highlights }) => {
      const { text, ...meta } = doc;
      const snippet = bestSnippet(text, highlights);
      return {
        meta,
        score,
        snippet,
        highlights,
      };
    });

  return { answer, hits };
}
