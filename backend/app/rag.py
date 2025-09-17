"""
Internal RAG utilities used by the FastAPI service.

This module encapsulates ingestion, retrieval and generation logic for the
Maryland agricultural regulatory chatbot.  It is designed to be self‑
contained so that the FastAPI app can simply import a `RAGEngine` and call
`ingest()` at startup and `answer(question)` at runtime.

The ingestion pipeline reads all PDFs and plain‑text files from a data
directory, converts them into overlapping chunks, computes sentence
embeddings and builds both a dense FAISS index and a sparse TF–IDF
representation.  Retrieval uses hybrid search (vector + tf–idf) with
reciprocal rank fusion.  The answer generator optionally calls a
Llama API (OpenAI‑compatible) to produce a concise answer with
citations.  If no API key is configured, it falls back to returning
extractive snippets.
"""

from __future__ import annotations

import os
import re
import hashlib
import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

def normalize_whitespace(text: str) -> str:
    """Normalize runs of whitespace and newlines for cleaner embeddings."""
    text = text.replace("\r", "").replace("\t", " ")
    # collapse three or more newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces
    text = re.sub(r"[ \u00A0]{2,}", " ", text)
    # strip spaces at line boundaries
    text = re.sub(r"[ \u00A0]+\n", "\n", text)
    text = re.sub(r"\n[ \u00A0]+", "\n", text)
    return text.strip()


def chunk_pages(pages: List[str], chunk_chars: int = 1500, overlap_chars: int = 200) -> List[Dict[str, any]]:
    """
    Chunk a list of page texts into overlapping strings.  Each chunk will
    contain up to ``chunk_chars`` characters and will overlap by
    ``overlap_chars`` characters with the previous chunk.  Returns a list of
    dictionaries containing the chunk text and the page range it covers.
    """
    chunks: List[Dict[str, any]] = []
    buf = ""
    buf_pages: List[int] = []
    def flush():
        nonlocal buf, buf_pages
        if buf.strip():
            chunks.append({
                "text": buf.strip(),
                "page_start": buf_pages[0] + 1,
                "page_end": buf_pages[-1] + 1
            })
        buf = ""
        buf_pages = []
    for idx, page in enumerate(pages):
        remaining = page
        while remaining:
            space_left = chunk_chars - len(buf)
            if space_left <= 0:
                carry = buf[-overlap_chars:] if overlap_chars > 0 else ""
                flush()
                if carry:
                    buf = carry
                    buf_pages = [buf_pages[-1]] if buf_pages else [idx]
            take = remaining[:max(1, space_left)]
            buf += take
            if not buf_pages:
                buf_pages = [idx]
            remaining = remaining[len(take):]
        # add newline to separate pages
        buf += "\n"
        if idx not in buf_pages:
            buf_pages.append(idx)
        if len(buf) >= chunk_chars:
            carry = buf[-overlap_chars:] if overlap_chars > 0 else ""
            flush()
            if carry:
                buf = carry
                buf_pages = [idx]
    flush()
    return chunks


class RAGEngine:
    """Encapsulates ingestion, retrieval and answer generation."""

    def __init__(self, data_dir: str, chunk_chars: int = 1500, overlap_chars: int = 200):
        self.data_dir = data_dir
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        # placeholders for loaded structures
        self.df: Optional[pd.DataFrame] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None  # type: ignore
        self.embed_model: Optional[SentenceTransformer] = None
        # LLM settings from environment
        self.llama_api_key = os.environ.get("LLAMA_API_KEY")
        self.llama_base_url = os.environ.get("LLAMA_BASE_URL", "https://api.llama.com/compat/v1/")
        self.llama_model = os.environ.get("LLAMA_MODEL", "")

    def ingest(self) -> None:
        """
        Walk the data directory, load all PDFs and text files, chunk them,
        compute embeddings, build the FAISS index and the TF–IDF matrix.  This
        method populates instance attributes for retrieval.
        """
        records: List[Dict[str, any]] = []
        doc_counter = 0
        # iterate through files
        for root, _dirs, files in os.walk(self.data_dir):
            for fname in files:
                path = os.path.join(root, fname)
                # skip hidden files
                if fname.startswith('.'):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in {'.pdf', '.txt'}:
                    continue
                doc_counter += 1
                doc_id = f"doc{doc_counter:04d}"
                doc_title = fname
                if ext == '.pdf':
                    try:
                        with fitz.open(path) as pdf:
                            pages_text: List[str] = []
                            for page in pdf:
                                text = page.get_text("text")
                                text = normalize_whitespace(text)
                                pages_text.append(text)
                    except Exception:
                        continue
                else:
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    except Exception:
                        continue
                    text = normalize_whitespace(text)
                    # treat entire text as single page for chunking
                    pages_text = text.split("\f") if "\f" in text else [text]
                # chunk
                chunks = chunk_pages(pages_text, self.chunk_chars, self.overlap_chars)
                for idx, ch in enumerate(chunks):
                    records.append({
                        "chunk_id": f"{doc_id}_c{idx}",
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "text": ch['text'],
                        "page_start": ch['page_start'],
                        "page_end": ch['page_end'],
                        "section_path": [],
                        "source_url": "",  # unknown in offline mode
                    })
        if not records:
            # no documents found; still initialize empty structures
            self.df = pd.DataFrame(columns=["chunk_id","doc_id","doc_title","text","page_start","page_end","section_path","source_url"])
            self.faiss_index = faiss.IndexFlatIP(384)
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([""])
            self.embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            return
        # build dataframe
        self.df = pd.DataFrame.from_records(records)
        # load embedding model lazily
        self.embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        # compute document embeddings
        docs = ["Represent the document for retrieval: " + t for t in self.df['text'].tolist()]
        embeddings = self.embed_model.encode(docs, normalize_embeddings=True, show_progress_bar=False).astype('float32')
        # build FAISS index (inner product on normalized embeddings = cosine)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        # compute TF–IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1, max_df=0.95)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['text'].tolist())

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string for semantic search."""
        assert self.embed_model is not None
        q = "Represent the question for retrieving supporting documents: " + query
        vec = self.embed_model.encode([q], normalize_embeddings=True)[0]
        return vec.astype('float32')

    @staticmethod
    def _rrf(rank_lists: List[List[int]], k: int = 60) -> List[int]:
        """Simple reciprocal rank fusion for multiple rank lists."""
        from collections import defaultdict
        scores: Dict[int, float] = defaultdict(float)
        for rl in rank_lists:
            for rank, idx in enumerate(rl, start=1):
                scores[idx] += 1.0 / (k + rank)
        # sort by descending fused score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _score in fused]

    def search(self, query: str, k: int = 8) -> List[Dict[str, any]]:
        """
        Perform enhanced hybrid retrieval with better passage selection.
        Uses semantic search, TF-IDF search, and keyword matching for comprehensive results.
        """
        if self.df is None or self.faiss_index is None or self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("RAGEngine not ingested; call ingest() first")
        
        # Expand search to get more candidates
        search_k = min(k * 3, len(self.df))  # Get 3x more candidates for better selection
        
        # Semantic search
        qv = self._embed_query(query)
        D, I = self.faiss_index.search(np.array([qv]), search_k)
        vec_ids = [int(i) for i in I[0] if i >= 0]
        
        # TF-IDF search
        q_tfidf = self.tfidf_vectorizer.transform([query])
        sims = (q_tfidf @ self.tfidf_matrix.T).toarray()[0]
        tf_indices = np.argsort(-sims)[:search_k].tolist()
        
        # Keyword-based search for additional context
        query_words = set(query.lower().split())
        keyword_matches = []
        for idx, row in self.df.iterrows():
            text_words = set(row['text'].lower().split())
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                keyword_matches.append((idx, overlap))
        
        # Sort by keyword overlap and take top matches
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        keyword_ids = [idx for idx, _ in keyword_matches[:search_k]]
        
        # Fuse all three ranking methods
        fused = self._rrf([vec_ids, tf_indices, keyword_ids])
        
        # Select diverse passages - avoid duplicates from same document
        selected = []
        seen_docs = set()
        for idx in fused:
            if len(selected) >= k:
                break
            row = self.df.iloc[idx]
            doc_id = row['doc_id']
            # Allow multiple passages from same document if we don't have enough diversity
            if doc_id not in seen_docs or len(selected) < k // 2:
                selected.append(idx)
                if len(selected) < k // 2:  # Only track first half for diversity
                    seen_docs.add(doc_id)
        
        # Build results
        hits = []
        for idx in selected:
            row = self.df.iloc[idx]
            hits.append({
                "chunk_id": row['chunk_id'],
                "doc_id": row['doc_id'],
                "doc_title": row['doc_title'],
                "text": row['text'],
                "page_start": int(row['page_start']),
                "page_end": int(row['page_end']),
                "section_path": row['section_path'],
                "source_url": row['source_url'],
            })
        return hits

    def generate_answer(self, question: str, passages: List[Dict[str, any]]) -> str:
        """
        Use the Llama API to generate an answer given a question and a
        collection of passages.  The answer will cite sources as [number].
        If the API is not configured or an error occurs, an extractive
        summary is returned.
        """
        if not passages:
            return "I couldn't find relevant text in the indexed documents."
        # Build context string
        context_blocks = []
        for i, p in enumerate(passages, start=1):
            loc = f"Pages {p['page_start']}-{p['page_end']}"
            sect = " > ".join(p['section_path']) if p['section_path'] else ""
            header = f"[{i}] {p['doc_title']} ({loc}) {sect}".strip()
            snippet = p['text'][:2000]
            context_blocks.append(f"{header}\n\n{snippet}")
        context = "\n\n".join(context_blocks)
        system_prompt = (
            "You are an expert agriculture regulatory assistant for the Maryland Department of Agriculture. "
            "Your role is to provide comprehensive, accurate, and well-structured answers about Maryland agriculture regulations. "
            
            "INSTRUCTIONS:\n"
            "1. Use ONLY the provided source documents to answer questions\n"
            "2. Provide complete, detailed answers that fully address the question\n"
            "3. Structure your response with clear headings and bullet points\n"
            "4. Include all relevant details, requirements, and procedures\n"
            "5. Cite sources using [1], [2], etc. format\n"
            "6. If information is incomplete in sources, clearly state what's missing\n\n"
            
            "RESPONSE FORMAT:\n"
            "**Answer:** [Direct, complete answer to the question]\n\n"
            "**Key Requirements:**\n"
            "• [Requirement 1] [1]\n"
            "• [Requirement 2] [2]\n\n"
            "**Additional Details:**\n"
            "• [Important detail 1]\n"
            "• [Important detail 2]\n\n"
            "**Important Notes:**\n"
            "*[Any warnings, exceptions, or special considerations]*\n\n"
            "**Sources:** All information is from the Maryland Code of Regulations and available at dsd.maryland.gov\n\n"
            "If you cannot find complete information in the sources, clearly state what's missing and recommend contacting the Maryland Department of Agriculture for clarification."
        )
        if OpenAI is None or not self.llama_api_key or not self.llama_model:
            # Fallback: return structured extractive snippets with citations
            bullets = []
            bullets.append("**Answer:** Based on the Maryland Department of Agriculture regulations:")
            bullets.append("")
            bullets.append("**Key Requirements:**")
            
            for i, p in enumerate(passages, start=1):
                snippet = p['text'][:400].strip()
                # Clean up the snippet
                snippet = snippet.replace('\n', ' ').replace('\r', ' ')
                # Ensure it ends properly
                if not snippet.endswith('.') and not snippet.endswith(';'):
                    snippet += "..."
                bullets.append(f"• {snippet} [{i}]")
            
            bullets.append("")
            bullets.append("**Important Notes:**")
            bullets.append("*This information is from the Maryland Code of Regulations. For specific guidance, contact the Maryland Department of Agriculture.*")
            bullets.append("")
            bullets.append("**Sources:** Available at dsd.maryland.gov")
            
            return "\n".join(bullets)
        try:
            print(f"Using Llama API: {self.llama_base_url} with model: {self.llama_model}")
            client = OpenAI(api_key=self.llama_api_key, base_url=self.llama_base_url)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}\n\n" \
                                       "Write a short answer with bullet points and cite sources by [number]."}
            ]
            resp = client.chat.completions.create(model=self.llama_model, messages=messages, temperature=0.2)
            print("Llama API response successful")
            if resp and resp.choices and len(resp.choices) > 0:
                return resp.choices[0].message.content
            else:
                print("Empty response from Llama API")
                raise Exception("Empty response from Llama API")
        except Exception as e:
            # Log the error for debugging
            print(f"Error calling Llama API: {e}")
            # fallback on error - use structured format
            bullets = []
            bullets.append("**Answer:** Based on the Maryland Department of Agriculture regulations:")
            bullets.append("")
            bullets.append("**Key Requirements:**")
            
            for i, p in enumerate(passages, start=1):
                snippet = p['text'][:400].strip()
                # Clean up the snippet
                snippet = snippet.replace('\n', ' ').replace('\r', ' ')
                if not snippet.endswith('.') and not snippet.endswith(';'):
                    snippet += "..."
                bullets.append(f"• {snippet} [{i}]")
            
            bullets.append("")
            bullets.append("**Important Notes:**")
            bullets.append("*This information is from the Maryland Code of Regulations. For specific guidance, contact the Maryland Department of Agriculture.*")
            bullets.append("")
            bullets.append("**Sources:** Available at dsd.maryland.gov")
            
            return "\n".join(bullets)

    def answer(self, question: str, k: int = 5) -> Tuple[str, List[Dict[str, str]]]:
        """
        Full pipeline: retrieve passages and generate an answer.  Returns the
        answer string and a list of sources with labels, page ranges and
        (placeholder) URLs.
        """
        passages = self.search(question, k)
        answer_text = self.generate_answer(question, passages)
        # build sources list from passages in order
        sources = []
        for i, p in enumerate(passages, start=1):
            # Extract COMAR number from document title if available
            comar_number = "15.15.01.01"  # Default based on the PDF content
            if "Code of Maryland Regulations" in p['doc_title']:
                # Try to extract COMAR number from text content
                import re
                comar_match = re.search(r'(\d+\.\d+\.\d+\.\d+(?:-\d+)?)', p['text'])
                if comar_match:
                    comar_number = comar_match.group(1)
            
            # Build website URL based on COMAR number
            # Format: https://dsd.maryland.gov/regulations/Pages/title.subtitle.chapter.section.aspx
            base_url = "https://dsd.maryland.gov/regulations/Pages/"
            if "-" in comar_number:
                # Subsection format (e.g., 15.15.01.01-1)
                # For subsections, use the main section URL
                main_section = comar_number.split('-')[0]
                url = f"{base_url}{main_section}.aspx"
            else:
                # Section format (e.g., 15.15.01.01)
                url = f"{base_url}{comar_number}.aspx"
            
            # Parse COMAR number for display
            parts = comar_number.split('.')
            if len(parts) >= 4:
                title = parts[0]
                subtitle = parts[1] 
                chapter = parts[2]
                section = parts[3]
                comar_display = f"Title {title}, Subtitle {subtitle}, Chapter {chapter}, Section {section}"
            else:
                comar_display = f"COMAR {comar_number}"
            
            label = f"[{i}] {p['doc_title']} - {comar_display}"
            sect = " > ".join(p['section_path']) if p['section_path'] else ""
            if sect:
                label += f" — {sect}"
            
            sources.append({
                "label": label,
                "pages": f"{p['page_start']}-{p['page_end']}",
                "url": url,
                "comar_number": comar_number,
                "comar_display": comar_display
            })
        return answer_text, sources
