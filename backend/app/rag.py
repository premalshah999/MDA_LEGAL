from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote_plus

import psycopg
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievalPlan:
    """Structured instructions returned by the LLM query planner."""

    normalized_question: str
    search_queries: List[str]
    section_hints: List[str]
    in_scope: bool
    small_talk: bool


@dataclass
class RerankResult:
    ordered_docs: List[Document]
    needs_web: bool
    missing_elements: List[str]


# ---------------------------------------------------------------------------
# PGVector helpers
# ---------------------------------------------------------------------------


def _build_database_url() -> str:
    project_ref = os.getenv("SUPABASE_PROJECT_REF")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    region = os.getenv("SUPABASE_REGION", "us-east-1")
    if not project_ref or not password:
        raise ValueError("Supabase project credentials are not configured")
    encoded = quote_plus(password)
    user = f"postgres.{project_ref}"
    host = f"aws-1-{region}.pooler.supabase.com"
    return f"postgresql+psycopg://{user}:{encoded}@{host}:5432/postgres?sslmode=require"


def _section_to_url(section: str) -> str:
    if not section:
        return "https://dsd.maryland.gov/Pages/COMARSearch.aspx"
    parts = [p for p in re.split(r"[.\s]", section) if p]
    padded = []
    for part in parts[:4]:
        base, _, suffix = part.partition("-")
        if base.isdigit():
            base = f"{int(base):02d}"
        padded.append(base + (f"-{suffix}" if suffix else ""))
    slug = ".".join(padded)
    return f"https://dsd.maryland.gov/regulations/Pages/{slug}.aspx"


def _unique_stripped(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if not item:
            continue
        value = item.strip()
        if not value or value.lower() in seen:
            continue
        seen.add(value.lower())
        result.append(value)
    return result


def _doc_id(doc: Document, default: str) -> str:
    meta = doc.metadata or {}
    return str(
        meta.get("id")
        or meta.get("doc_id")
        or meta.get("docId")
        or meta.get("source_id")
        or default
    )


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------


class RAGEngine:
    """Production-grade RAG controller with LLM planning, re-ranking and fallback search."""

    def __init__(self) -> None:
        self.db_uri = _build_database_url()
        self._psycopg_url = self.db_uri.replace("postgresql+psycopg://", "postgresql://")
        self.collection_name = self._discover_collection(os.getenv("PGVECTOR_COLLECTION"))

        # Embeddings and vector store
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        try:
            self.vector_store = PGVector(
                connection_string=self.db_uri,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                create_extension=False,
            )
        except Exception as exc:  # pragma: no cover - only triggered when PGVector misconfigured
            logger.error("Failed to connect PGVector store: %s", exc)
            self.vector_store = None

        # LLM configuration
        api_key = os.getenv("LLAMA_API_KEY", "").strip()
        base_url = os.getenv("LLAMA_BASE_URL", "").strip()
        model = os.getenv("LLAMA_MODEL", "").strip()
        if not (api_key and base_url and model):
            raise ValueError("LLM configuration missing: set LLAMA_API_KEY, LLAMA_BASE_URL, LLAMA_MODEL")

        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            timeout=20,
            max_retries=2,
        )

        self.query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
You are a Maryland COMAR research planner. Assess if the user request belongs to COMAR Titles 15 or 26.
Return strict JSON with keys: "normalized_question" (string), "search_queries" (list of 2-4 strings), "section_hints" (list of COMAR section numbers), "in_scope" (true/false), "small_talk" (true/false).
If the request is small talk (hi, thanks, etc.) mark small_talk true. If it is outside Maryland agriculture/environment regulations mark in_scope false.
Ensure section hints reference Title 15 unless the question clearly cites Title 26.
""".strip()),
                ("human", """
Conversation summary: {history}
Question: {question}
""".strip()),
            ]
        )

        self.rerank_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
You re-rank Maryland regulatory excerpts.
Given a question and candidate snippets, assign higher priority to precise Title 15 (or relevant Title 26) matches with definitions, obligations, timelines or exceptions. Penalise vague or duplicate chunks.
Respond with JSON: {"ranked": [{"id": "d1", "score": number, "reason": string}, ...], "needs_web": true/false, "missing_elements": [strings]}.
""".strip()),
                ("human", """
Question: {question}
Candidates:
{candidates}
""".strip()),
            ]
        )

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
You are the Maryland Agriculture regulatory assistant.
Primary scope: COMAR Title 15. Include Title 26 only when explicitly referenced.
Use the supplied CONTEXT only. Do not invent sections. Temperature must remain 0.
Format the reply exactly with the following Markdown sections:
**Short answer** – 1-3 sentences.
**What the rule says** – bullet list of precise points.
**Steps / Requirements** – numbered checklist, or "None noted" if absent.
**Cross-references** – cite sections used (comma separated) or "None".
**Edge cases / Exceptions / Warnings** – bullet list or "None".
**Sources** – list each citation as - [section-id](URL). If context missing, add "Missing from corpus. Verify via DSD/LII." at the end.
""".strip()),
                ("human", """
Conversation summary: {history}
User notes: {memories}
Scope note: {scope_note}
Question: {question}
CONTEXT:
{context}
""".strip()),
            ]
        )

        self.web_search = DuckDuckGoSearchAPIWrapper()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self) -> None:
        """Startup hook – no ingestion required because data lives in Supabase."""
        logger.info("Supabase PGVector collection '%s' ready", self.collection_name)

    def answer(self, question: str, k: int = 5, **kwargs: Any) -> Tuple[str, List[Dict[str, Any]]]:
        result = self.answer_question(question, k=k, **kwargs)
        return result["answer"], result["sources"]

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        k: int = 5,
        conversation: Optional[Sequence[Dict[str, Any]]] = None,
        memories: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        history_text = self._summarise_history(conversation or [])

        plan = self._plan_query(question, history_text)
        if not plan.in_scope:
            return {
                "answer": "I'm here to help with Maryland COMAR regulations (Title 15; Title 26 when relevant). How can I help with a rule or procedure?",
                "sources": [],
            }
        if plan.small_talk:
            return {
                "answer": "Hello! Ask me about Maryland agriculture regulations and I'll cite the COMAR sections for you.",
                "sources": [],
            }

        search_terms = _unique_stripped([plan.normalized_question, *plan.search_queries])
        if not search_terms:
            search_terms = [question]

        candidates = self._retrieve_candidates(search_terms, top_k=max(k * 2, 12), filters=filters)
        cross_refs = self._expand_cross_references(question, candidates, plan.section_hints)
        if cross_refs:
            candidates.extend(self._fetch_sections(cross_refs))
        candidates = self._dedupe_documents(candidates)[:12]

        reranked = self._rerank(question, candidates)
        working_set = reranked.ordered_docs[: max(k, 6)]

        if reranked.needs_web or not self._contains_title_15_26(working_set):
            web_docs = self._web_fallback(plan.normalized_question or question)
            if web_docs:
                merged = self._dedupe_documents(working_set + web_docs)
                reranked = self._rerank(question, merged)
                working_set = reranked.ordered_docs[: max(k, 6)]

        answer_text = self._generate_answer(
            question=question,
            history_text=history_text,
            memories_text=self._summarise_memories(memories or []),
            docs=working_set,
            scope_note=self._scope_note(working_set, reranked.missing_elements),
        )

        sources = self._format_sources(working_set)
        if not sources:
            sources = self._infer_sources_from_answer(answer_text)

        return {"answer": answer_text, "sources": sources}

    # ------------------------------------------------------------------
    # Query planning & retrieval
    # ------------------------------------------------------------------

    def _plan_query(self, question: str, history_text: str) -> RetrievalPlan:
        try:
            prompt = self.query_prompt.format_prompt(question=question, history=history_text)
            response = self.llm.invoke(prompt.to_messages())
            payload = json.loads(response.content)
        except Exception as exc:  # pragma: no cover - defensive parsing guard
            logger.warning("Query planner failed: %s", exc)
            return RetrievalPlan(question, [question], [], True, False)

        return RetrievalPlan(
            normalized_question=str(payload.get("normalized_question") or question).strip(),
            search_queries=_unique_stripped(payload.get("search_queries", [])),
            section_hints=_unique_stripped(payload.get("section_hints", [])),
            in_scope=bool(payload.get("in_scope", True)),
            small_talk=bool(payload.get("small_talk", False)),
        )

    def _retrieve_candidates(
        self,
        search_terms: Sequence[str],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Document]:
        docs: List[Document] = []
        seen: Set[str] = set()
        for term in search_terms:
            results = self._vector_search(term, top_k, filters)
            for idx, doc in enumerate(results):
                key = _doc_id(doc, f"{term}-{idx}")
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
        return docs

    def _vector_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[Document]:
        if self.vector_store is None:
            return self._keyword_search(query, top_k)
        try:
            results = self.vector_store.similarity_search_with_score(query, k=min(top_k, 12), filter=filters)
        except Exception as exc:  # pragma: no cover - PGVector edge
            logger.warning("Vector search failed (%s), using keyword fallback", exc)
            return self._keyword_search(query, top_k)
        docs: List[Document] = []
        for doc, score in results:
            doc.metadata = {**(doc.metadata or {}), "score": float(score)}
            docs.append(doc)
        return docs

    def _keyword_search(self, query: str, top_k: int) -> List[Document]:
        tokens = re.findall(r"[a-zA-Z]{4,}", query.lower())
        stop = {"maryland", "regulations", "regulation", "title", "subtitle", "chapter", "comar"}
        keywords = [t for t in tokens if t not in stop][:4]
        if not keywords:
            keywords = tokens[:2]
        if not keywords:
            return []
        sql = """
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE collection_id = %s AND ({conditions})
            LIMIT %s
        """
        conds = " OR ".join(["document ILIKE %s" for _ in keywords])
        sql = sql.format(conditions=conds)
        params: Tuple[Any, ...] = (self._collection_uuid(), *[f"%{kw}%" for kw in keywords], top_k)
        docs: List[Document] = []
        try:
            with psycopg.connect(self._psycopg_url, connect_timeout=10, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    for text, meta in cur.fetchall() or []:
                        docs.append(Document(page_content=text or "", metadata=meta or {}))
        except Exception as exc:  # pragma: no cover - fallback guard
            logger.warning("Keyword search failed: %s", exc)
        return docs

    # ------------------------------------------------------------------
    # Cross references & fallbacks
    # ------------------------------------------------------------------

    def _expand_cross_references(
        self,
        question: str,
        docs: Sequence[Document],
        hints: Sequence[str],
    ) -> List[str]:
        targets: Set[str] = set(hints)
        abs_pat = re.compile(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
        rel_pat = re.compile(r"Regulation\s*\.?(\d{2})", re.IGNORECASE)
        text = question + "\n" + "\n".join(doc.page_content[:800] for doc in docs)
        targets.update(abs_pat.findall(text))
        for doc in docs:
            sec = str((doc.metadata or {}).get("section") or "").strip()
            if not sec:
                continue
            parts = sec.split(".")
            if len(parts) < 4:
                continue
            base = parts[:3]
            for match in rel_pat.findall(doc.page_content[:400]):
                try:
                    full = ".".join([*base, f"{int(match):02d}"])
                    targets.add(full)
                except ValueError:
                    continue
        return list(targets)

    def _fetch_sections(self, sections: Sequence[str]) -> List[Document]:
        if not sections:
            return []
        sec_list = [s.strip() for s in sections if s and s.strip()]
        if not sec_list:
            return []
        clauses = []
        params: List[Any] = [self._collection_uuid()]
        for sec in sec_list:
            clauses.append("(cmetadata->>'section' = %s OR cmetadata->>'comar_number' = %s OR cmetadata->>'comarNumber' = %s)")
            params.extend([sec, sec, sec])
        limit = min(len(sec_list) * 3, 24)
        params.append(limit)
        sql = f"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE collection_id = %s AND ({' OR '.join(clauses)})
            LIMIT %s
        """
        docs: List[Document] = []
        try:
            with psycopg.connect(self._psycopg_url, connect_timeout=10, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(params))
                    for text, meta in cur.fetchall() or []:
                        docs.append(Document(page_content=text or "", metadata=meta or {}))
        except Exception as exc:  # pragma: no cover - fallback guard
            logger.info("Cross-reference fetch failed: %s", exc)
        return docs

    def _web_fallback(self, query: str) -> List[Document]:
        try:
            dsd_q = f"site:dsd.maryland.gov {query} COMAR Title 15"
            lii_q = f"site:law.cornell.edu/regulations/maryland {query} COMAR"
            results = self.web_search.results(dsd_q, max_results=5) + self.web_search.results(lii_q, max_results=5)
        except Exception as exc:  # pragma: no cover - network guard
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []
        urls: List[str] = []
        for res in results:
            url = str(res.get("link") or "")
            if not url:
                continue
            if "dsd.maryland.gov" in url or "law.cornell.edu/regulations/maryland" in url:
                if url not in urls:
                    urls.append(url)
        if not urls:
            return []
        try:
            try:
                pages = WebBaseLoader(urls, header_template={"User-Agent": "MDA-Legal-RAG/1.0"}).load()
            except TypeError:  # older langchain versions
                pages = WebBaseLoader(urls, requests_kwargs={"headers": {"User-Agent": "MDA-Legal-RAG/1.0"}}).load()
        except Exception as exc:  # pragma: no cover - network guard
            logger.warning("Web loader failed: %s", exc)
            return []
        docs = self.splitter.split_documents(pages)
        sec_pat = re.compile(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
        for doc in docs:
            match = sec_pat.search(doc.page_content[:600])
            if match:
                doc.metadata["section"] = match.group(1)
        return docs[:6]

    # ------------------------------------------------------------------
    # LLM re-ranking & answer generation
    # ------------------------------------------------------------------

    def _rerank(self, question: str, docs: Sequence[Document]) -> RerankResult:
        if not docs:
            return RerankResult([], needs_web=True, missing_elements=["No context"])
        candidates = []
        for idx, doc in enumerate(docs, 1):
            md = doc.metadata or {}
            section = md.get("section") or md.get("comar_number") or md.get("comarNumber") or ""
            title = md.get("title") or md.get("doc_title") or md.get("docTitle") or ""
            text = re.sub(r"\s+", " ", doc.page_content.strip())[:600]
            candidates.append({
                "id": f"d{idx}",
                "section": section,
                "title": title,
                "text": text,
            })
        try:
            prompt = self.rerank_prompt.format_prompt(
                question=question,
                candidates=json.dumps(candidates, ensure_ascii=False),
            )
            response = self.llm.invoke(prompt.to_messages())
            payload = json.loads(response.content)
            order = payload.get("ranked", [])
            ordered_ids = [item.get("id") for item in order if isinstance(item, dict) and item.get("id")]
            lookup = {f"d{idx}": doc for idx, doc in enumerate(docs, 1)}
            ordered_docs = [lookup[i] for i in ordered_ids if i in lookup]
            leftovers = [doc for doc in docs if doc not in ordered_docs]
            needs_web = bool(payload.get("needs_web", False))
            missing = [str(m).strip() for m in payload.get("missing_elements", []) if str(m).strip()]
            return RerankResult(ordered_docs + leftovers, needs_web, missing)
        except Exception as exc:  # pragma: no cover - guard against parsing
            logger.warning("Re-ranker failed: %s", exc)
            return RerankResult(list(docs), needs_web=False, missing_elements=[])

    def _generate_answer(
        self,
        question: str,
        history_text: str,
        memories_text: str,
        docs: Sequence[Document],
        scope_note: str,
    ) -> str:
        context = self._format_docs_for_prompt(docs)
        if not context:
            return (
                "**Short answer**\nI do not have matching COMAR Title 15/26 context for this question.\n\n"
                "**What the rule says**\n- Missing from corpus\n\n"
                "**Steps / Requirements**\nNone noted\n\n"
                "**Cross-references**\nNone\n\n"
                "**Edge cases / Exceptions / Warnings**\nNone\n\n"
                "**Sources**\n- Missing from corpus. Verify via DSD/LII."
            )
        try:
            prompt = self.answer_prompt.format_prompt(
                question=question,
                history=history_text,
                memories=memories_text,
                context=context,
                scope_note=scope_note,
            )
            response = self.llm.invoke(prompt.to_messages())
            return response.content.strip()
        except Exception as exc:  # pragma: no cover - guard against generation failure
            logger.error("Answer generation failed: %s", exc)
            return (
                "**Short answer**\nI hit an issue while generating the response.\n\n"
                "**What the rule says**\n- Please try again in a moment.\n\n"
                "**Steps / Requirements**\nNone noted\n\n"
                "**Cross-references**\nNone\n\n"
                "**Edge cases / Exceptions / Warnings**\nNone\n\n"
                "**Sources**\n- Missing from corpus. Verify via DSD/LII."
            )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _summarise_history(self, conversation: Sequence[Dict[str, Any]]) -> str:
        if not conversation:
            return "(none)"
        entries = []
        for msg in conversation[-6:]:
            role = msg.get("role") or msg.get("author") or "user"
            text = (msg.get("content") or "").strip()
            if not text:
                continue
            entries.append(f"{role}: {text[:160]}")
        return " \n".join(entries) if entries else "(none)"

    def _summarise_memories(self, memories: Sequence[str]) -> str:
        items = [m.strip() for m in memories if isinstance(m, str) and m.strip()]
        if not items:
            return "(none)"
        return "; ".join(items[:4])

    def _format_docs_for_prompt(self, docs: Sequence[Document]) -> str:
        blocks = []
        sec_pat = re.compile(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
        for idx, doc in enumerate(docs[:8], 1):
            md = doc.metadata or {}
            section = md.get("section") or md.get("comar_number") or md.get("comarNumber")
            if not section:
                match = sec_pat.search(doc.page_content)
                if match:
                    section = match.group(1)
            header = f"[Source {idx}] {section or md.get('title') or 'Excerpt'}"
            text = re.sub(r"\s+", " ", doc.page_content.strip())
            blocks.append(f"{header}: {text[:1100]}")
        return "\n\n".join(blocks)

    def _format_sources(self, docs: Sequence[Document]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for doc in docs:
            md = doc.metadata or {}
            section = (
                md.get("section")
                or md.get("comar_number")
                or md.get("comarNumber")
                or self._extract_section(doc.page_content)
            )
            if not section:
                continue
            url = _section_to_url(section)
            key = f"{section}:{url}"
            if key in seen:
                continue
            seen.add(key)
            snippet = re.sub(r"\s+", " ", doc.page_content.strip())[:220]
            citations.append(
                {
                    "id": _doc_id(doc, section),
                    "label": f"COMAR {section}",
                    "pages": md.get("pages") or "",
                    "url": url,
                    "doc_id": md.get("id") or md.get("doc_id") or md.get("docId"),
                    "doc_title": md.get("title") or md.get("doc_title") or md.get("docTitle"),
                    "comar_number": section,
                    "comar_display": f"COMAR {section}",
                    "snippet": snippet,
                }
            )
        return citations[:6]

    def _infer_sources_from_answer(self, answer: str) -> List[Dict[str, Any]]:
        pattern = re.compile(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
        found = _unique_stripped(pattern.findall(answer))
        citations: List[Dict[str, Any]] = []
        for section in found[:4]:
            citations.append(
                {
                    "id": section,
                    "label": f"COMAR {section}",
                    "pages": "",
                    "url": _section_to_url(section),
                    "comar_number": section,
                    "comar_display": f"COMAR {section}",
                }
            )
        return citations

    def _scope_note(self, docs: Sequence[Document], missing: Sequence[str]) -> str:
        sections = []
        titles: Set[str] = set()
        for doc in docs:
            md = doc.metadata or {}
            section = (
                md.get("section")
                or md.get("comar_number")
                or md.get("comarNumber")
                or self._extract_section(doc.page_content)
            )
            if section:
                sections.append(section)
                if section.startswith("15"):
                    titles.add("15")
                if section.startswith("26"):
                    titles.add("26")
        sections = _unique_stripped(sections)[:6]
        title_note = f"Titles {', '.join(sorted(titles))}" if titles else "No Title 15/26 detected"
        missing_note = f" Missing: {', '.join(missing)}." if missing else ""
        if sections:
            return f"Primary context: {title_note}. Key sections: {', '.join(sections)}.{missing_note}"
        return f"Primary context: {title_note}.{missing_note}"

    def _contains_title_15_26(self, docs: Sequence[Document]) -> bool:
        for doc in docs:
            section = (
                (doc.metadata or {}).get("section")
                or (doc.metadata or {}).get("comar_number")
                or (doc.metadata or {}).get("comarNumber")
                or self._extract_section(doc.page_content)
            )
            if section and (section.startswith("15.") or section.startswith("26.")):
                return True
        return False

    def _extract_section(self, text: str) -> Optional[str]:
        match = re.search(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)", text or "")
        return match.group(1) if match else None

    def _dedupe_documents(self, docs: Sequence[Document]) -> List[Document]:
        seen: Set[str] = set()
        unique_docs: List[Document] = []
        for doc in docs:
            key = _doc_id(doc, str(len(unique_docs)))
            content_hash = hash(doc.page_content.strip())
            combined = f"{key}:{content_hash}"
            if combined in seen:
                continue
            seen.add(combined)
            unique_docs.append(doc)
        return unique_docs

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _discover_collection(self, configured: Optional[str]) -> str:
        query = """
            SELECT c.name, COUNT(e.uuid) AS count
            FROM langchain_pg_collection c
            LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
            GROUP BY c.name
            ORDER BY count DESC
        """
        try:
            with psycopg.connect(self._psycopg_url, connect_timeout=10, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall() or []
        except Exception as exc:  # pragma: no cover - guard for connectivity
            logger.error("Could not list PGVector collections: %s", exc)
            return configured or "default"
        if not rows:
            return configured or "default"
        names = {row[0]: int(row[1]) for row in rows}
        if configured and configured in names:
            return configured
        for name, count in names.items():
            if count > 0:
                return name
        return next(iter(names))

    @lru_cache(maxsize=1)
    def _collection_uuid(self) -> str:
        sql = "SELECT uuid FROM langchain_pg_collection WHERE name = %s LIMIT 1"
        with psycopg.connect(self._psycopg_url, connect_timeout=10, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (self.collection_name,))
                row = cur.fetchone()
        if not row:
            raise RuntimeError(f"Collection '{self.collection_name}' not found in Supabase")
        return str(row[0])


def create_rag_engine() -> RAGEngine:
    return RAGEngine()


if __name__ == "__main__":  # pragma: no cover
    engine = create_rag_engine()
    response = engine.answer_question("What are the Maryland pesticide reporting requirements?")
    print(response["answer"])
