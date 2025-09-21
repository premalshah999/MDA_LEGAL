# rag.py - Production-Ready RAG Pipeline for FastAPI Integration
"""
Production-ready RAG pipeline using Supabase PGVector and HuggingFace embeddings.
This module integrates seamlessly with your existing FastAPI backend.

Required environment variables:
- SUPABASE_PROJECT_REF: Your Supabase project reference
- SUPABASE_REGION: Your Supabase region (e.g., us-east-2)  
- SUPABASE_DB_PASSWORD: Your database password
- LLAMA_API_KEY: Your LLM API key
- LLAMA_BASE_URL: Your LLM base URL
- LLAMA_MODEL: Your LLM model name
- PGVECTOR_COLLECTION: Collection name (default: langchain_pg_embedding)
"""

from __future__ import annotations
import os
import re
import logging
import hashlib
import concurrent.futures as cf
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import quote_plus

import torch
import psycopg
import numpy as np
from langchain.schema import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain.schema import StrOutputParser

# Web search and document loading for fallback
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Try to import FAISS optionally (not required on macOS)
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    _HAS_FAISS = True
except Exception:
    FAISS = None  # type: ignore
    _HAS_FAISS = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================

def get_database_url() -> str:
    """Build PostgreSQL connection URL with proper encoding"""
    project_ref = os.getenv("SUPABASE_PROJECT_REF")
    region = os.getenv("SUPABASE_REGION", "us-east-2")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    
    if not project_ref or not password:
        raise ValueError("Missing SUPABASE_PROJECT_REF or SUPABASE_DB_PASSWORD environment variables")
    
    user = f"postgres.{project_ref}"
    host = f"aws-1-{region}.pooler.supabase.com"
    port = 5432
    database = "postgres"
    
    # URL encode password to handle special characters
    encoded_password = quote_plus(password)
    
    # Use SQLAlchemy URL for psycopg v3 driver
    return f"postgresql+psycopg://{user}:{encoded_password}@{host}:{port}/{database}?sslmode=require"

def detect_collection_name(conn_str: str, configured: Optional[str]) -> str:
    """Detect an existing collection name from langchain_pg_collection.
    Preference order:
    1) Configured collection, if it exists AND has >0 embeddings.
    2) The collection with the highest number of embeddings (>0), if any.
    3) First available collection name, if none have embeddings.
    4) Fallback to "default".
    """
    try:
        psy_url = conn_str.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(psy_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.name, c.uuid, COALESCE(COUNT(e.uuid), 0) AS cnt
                    FROM langchain_pg_collection c
                    LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
                    GROUP BY c.name, c.uuid
                    ORDER BY cnt DESC, c.name ASC
                    """
                )
                rows = cur.fetchall() or []
                if not rows:
                    # No collections found
                    return (configured.strip() if configured and configured.strip() else "default")
                # Build lookup
                counts = {str(name): int(cnt) for (name, _uuid, cnt) in rows}
                names = [str(name) for (name, _uuid, _cnt) in rows]
                # If configured is provided, prefer it when valid and non-empty
                if configured and configured.strip():
                    cfg = configured.strip()
                    cfg_cnt = counts.get(cfg)
                    if cfg_cnt is None:
                        logger.warning(f"Configured PGVector collection '{cfg}' not found. Available: {names}")
                    elif cfg_cnt > 0:
                        logger.info(f"Using configured PGVector collection '{cfg}' with {cfg_cnt} embeddings.")
                        return cfg
                    # Configured exists but empty, try best non-empty
                    best = next((n for n in names if counts.get(n, 0) > 0), None)
                    if best:
                        logger.warning(
                            f"Configured collection '{cfg}' has 0 embeddings; auto-selecting non-empty collection '{best}'"
                        )
                        return best
                    # Fall back to configured (empty) or first name
                    return cfg or names[0]
                # No configured value: pick best non-empty or first
                best = next((n for n in names if counts.get(n, 0) > 0), None)
                if best:
                    logger.info(f"Auto-detected PGVector collection: {best} ({counts.get(best, 0)} embeddings)")
                    return best
                return names[0]
    except Exception as e:
        logger.warning(f"Could not auto-detect collection name: {e}")
    # Sensible default collection identifier
    return "default"

# ================== UTILITY FUNCTIONS ==================

def detect_embedding_model_from_db(conn_str: str) -> Tuple[int, str]:
    """Auto-detect embedding dimension and select appropriate model"""
    try:
        # Convert SQLAlchemy URL to a psycopg native URL
        psycopg_url = conn_str.replace("postgresql+psycopg://", "postgresql://")
        conn = psycopg.connect(psycopg_url)
        
        with conn:
            with conn.cursor() as cur:
                # Try to get embedding dimension from existing data
                cur.execute(
                    """
                    SELECT vector_dims(embedding) 
                    FROM langchain_pg_embedding 
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row and row[0]:
                    dim = int(row[0])
                else:
                    dim = 384  # Default dimension
        
        conn.close()
        
        # Map dimensions to models
        model_mapping = {
            384: "sentence-transformers/all-MiniLM-L6-v2",
            768: "sentence-transformers/all-mpnet-base-v2",
            1024: "sentence-transformers/all-roberta-large-v1"
        }
        
        model = model_mapping.get(dim, "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Auto-detected embedding dimension: {dim}, using model: {model}")
        return dim, model
        
    except Exception as e:
        logger.warning(f"Could not detect embedding dimension: {e}. Using default.")
        return 384, "sentence-transformers/all-MiniLM-L6-v2"

def section_to_url(section: str) -> str:
    """Convert a COMAR section number to the official DSD URL.
    Accepts values like '15.20.08.04' or '15.20.08.04-1'.
    """
    try:
        s = section.strip()
        if not s:
            return "https://dsd.maryland.gov/Pages/COMARSearch.aspx"
        # Normalize: ensure 4 dot-separated parts; keep any suffix (e.g., '-1') as-is
        parts = s.split(".")
        # Zero-pad numeric parts to 2 digits where possible
        norm_parts = []
        for i, p in enumerate(parts):
            if not p:
                continue
            # Keep suffix like '04-1' intact; only left part zero-pad if numeric
            if i < 4:
                base = p
                if "-" in p:
                    base = p.split("-")[0]
                if base.isdigit():
                    base = f"{int(base):02d}"
                    # restore suffix, if any
                    if "-" in p:
                        suffix = p[p.index("-"):]
                        base = base + suffix
                norm_parts.append(base)
            else:
                norm_parts.append(p)
        # Only use the first 4 parts for URL path
        slug = ".".join(norm_parts[:4])
        return f"https://dsd.maryland.gov/regulations/Pages/{slug}.aspx"
    except Exception:
        return "https://dsd.maryland.gov/Pages/COMARSearch.aspx"

# ================== CROSS-REFERENCE LOGIC ==================

# Cross-reference detection patterns
_abs_pat = re.compile(r"(?:COMAR\s*)?(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
_rel_pat = re.compile(r"Regulation\s*\.?(\d{2})([A-Z])?")

def _nums(sec: str) -> List[int]:
    """Convert section string to normalized number list"""
    parts = [int(p) for p in sec.split(".") if p.isdigit()]
    while len(parts) < 4:
        parts.append(0)
    return parts[:4]

def _fmt(nums: List[int]) -> str:
    """Format number list back to section string"""
    return f"{nums[0]:02d}.{nums[1]:02d}.{nums[2]:02d}.{nums[3]:02d}"

def extract_target_sections(question: str, base_docs: List[Document]) -> Set[str]:
    """Extract cross-referenced sections from question and documents"""
    targets: Set[str] = set()
    
    # Combine question and document text for analysis
    text = question + "\n" + "\n".join([d.page_content[:800] for d in base_docs])
    
    # Find absolute references (e.g., "15.15.04.03")
    for match in _abs_pat.finditer(text):
        targets.add(match.group(1))
    
    # Find relative references (e.g., "Regulation .03")
    for doc in base_docs:
        base_sec = doc.metadata.get("section")
        if not base_sec:
            continue
            
        base_nums = _nums(base_sec)
        for match in _rel_pat.finditer(doc.page_content[:2000]):
            nn = int(match.group(1))
            rel_nums = base_nums.copy()
            rel_nums[3] = nn
            targets.add(_fmt(rel_nums))
    
    return targets

def in_title_15_26(docs: List[Document]) -> bool:
    """Check if documents contain Title 15 or 26 content"""
    return any(doc.metadata.get("section", "").startswith(("15.", "26.")) for doc in docs)

# ================== WEB FALLBACK LOGIC ==================

def web_fetch_title_15_26(query: str, max_urls: int = 6) -> List[Document]:
    """Fetch additional content from web for Title 15/26 topics"""
    try:
        ddg = DuckDuckGoSearchAPIWrapper()
        user_agent = os.getenv("USER_AGENT", "MDA-Legal-RAG/1.0")
        
        # Search official Maryland sites
        q1 = f"site:dsd.maryland.gov {query} COMAR Title 15 OR Title 26"
        q2 = f"site:law.cornell.edu/regulations/maryland {query} COMAR 15 OR 26"
        
        results = ddg.results(q1, max_results=max_urls) + ddg.results(q2, max_results=max_urls)
        
        urls = []
        for r in results:
            url = r.get("link", "")
            if ("dsd.maryland.gov" in url) or ("law.cornell.edu/regulations/maryland" in url):
                urls.append(url)
        
        # Remove duplicates and limit
        urls = list(dict.fromkeys(urls))[:max_urls]
        if not urls:
            return []
        
        # Load web pages
        try:
            pages = WebBaseLoader(urls, header_template={"User-Agent": user_agent}).load()
        except TypeError:
            # Fallback for different WebBaseLoader API
            pages = WebBaseLoader(urls, requests_kwargs={"headers": {"User-Agent": user_agent}}).load()
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        web_docs = splitter.split_documents(pages)
        
        # Extract section numbers from content
        sec_pat = re.compile(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
        for doc in web_docs:
            match = sec_pat.search(doc.page_content[:700])
            if match:
                doc.metadata["section"] = match.group(1)
        
        return web_docs
        
    except Exception as e:
        logger.warning(f"Web fallback failed: {e}")
        return []

def augment_with_web_if_needed(query: str, base_docs: List[Document], max_add: int = 12) -> List[Document]:
    """Augment results with web content if needed"""
    # Skip if already have Title 15/26 content
    if in_title_15_26(base_docs):
        return base_docs
    
    # Fetch web content
    web_docs = web_fetch_title_15_26(query, max_urls=6)
    if not web_docs:
        return base_docs
    
    # If FAISS is unavailable, fall back to a simple heuristic merge
    if not _HAS_FAISS:
        try:
            # Basic heuristic: prioritize docs that mention Title 15/26 or COMAR patterns
            pat = re.compile(r"\b(15\.\d+|26\.\d+|COMAR)\b", re.I)
            scored = []
            for d in web_docs:
                score = 0
                if pat.search(d.page_content[:1200]):
                    score += 1
                if pat.search((d.metadata.get("title") or "")):
                    score += 0.5
                scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_web = [d for _, d in scored[: min(6, len(scored))]]
            merged = base_docs + top_web
            # Deduplicate by content hash
            seen = set()
            unique = []
            for doc in merged:
                key = hashlib.sha1(doc.page_content.encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    unique.append(doc)
            return unique[:max_add]
        except Exception as e:
            logger.warning(f"Heuristic web augmentation failed: {e}")
            return base_docs
    
    try:
        # Create temporary FAISS index for web docs
        embedding_model = os.getenv("DETECTED_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        web_vs = FAISS.from_documents(web_docs, embeddings)  # type: ignore
        extra_docs = web_vs.similarity_search(query, k=min(6, len(web_docs)))
        
        # Merge and deduplicate
        seen = set()
        merged = []
        
        def doc_key(d):
            return (
                d.metadata.get("source") or d.metadata.get("url") or "",
                hashlib.sha1(d.page_content.encode()).hexdigest()
            )
        
        for doc in base_docs + extra_docs:
            key = doc_key(doc)
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        
        return merged[:max_add]
        
    except Exception as e:
        logger.warning(f"Web augmentation failed: {e}")
        return base_docs

# ================== MAIN RAG ENGINE ==================

class RAGEngine:
    """Production-ready RAG engine using Supabase PGVector"""
    
    def __init__(self):
        """Initialize RAG engine with Supabase connection"""
        # Database configuration
        self.db_uri = get_database_url()
        self._psycopg_url = self.db_uri.replace("postgresql+psycopg://", "postgresql://")
        configured_collection = os.getenv("PGVECTOR_COLLECTION")
        # Device detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        # LLM timeout for responsiveness
        try:
            self.llm_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "6"))
        except Exception:
            self.llm_timeout_seconds = 6.0

        # Auto-detect embedding settings
        self.embed_dim, self.embed_model_name = detect_embedding_model_from_db(self.db_uri)
        os.environ["DETECTED_EMBEDDING_MODEL"] = self.embed_model_name

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Detect or set collection name before vector store init
        self.collection_name = detect_collection_name(self.db_uri, configured_collection)
        self._collection_uuid_cache: Optional[str] = None

        # Initialize vector store (read-only)
        self.store = None
        self._sqlalchemy_store = False
        try:
            self.store = PGVector(
                connection_string=self.db_uri,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                create_extension=False,
            )
            self._sqlalchemy_store = True
            logger.info(f"Connected to PGVector collection: {self.collection_name} (SQLAlchemy)")
        except Exception as e:
            logger.warning(f"PGVector (SQLAlchemy) connection failed, will use psycopg fallback: {e}")
            self.store = None
            self._sqlalchemy_store = False

        # LLM configuration
        self.llm_api_key = os.getenv("LLAMA_API_KEY", "").strip()
        self.llm_base_url = os.getenv("LLAMA_BASE_URL", "https://api.llama.com/compat/v1/").strip()
        self.llm_model = os.getenv("LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8").strip()

        # Initialize LLM if configured
        if self.llm_api_key and self.llm_model and self.llm_base_url:
            try:
                self.llm = ChatOpenAI(
                    model=self.llm_model,
                    base_url=self.llm_base_url,
                    api_key=self.llm_api_key,
                    temperature=0.2,
                    max_tokens=1800,
                )
                self._llm_available = True
                logger.info(f"LLM initialized: {self.llm_model}")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}")
                self._llm_available = False
        else:
            logger.warning("LLM not configured - using extractive responses")
            self._llm_available = False

        # Updated system guidance for natural, structured answers without inline sources
        self.system_prompt = (
            "You are the Maryland Agriculture regulatory assistant. \n"
            "Primary sources: COMAR Title 15 (MDA) and Title 26 (Environment when relevant).\n\n"
            "Behavior:\n"
            "- Be accurate, concise, and helpful. Never be rude.\n"
            "- If the user goes off-topic, politely redirect to Maryland regulations.\n"
            "- Prefer the provided CONTEXT (RAG) for speed. If key facts are missing, consult the official sources for Titles 15/26.\n"
            "- If the correct answer is outside Titles 15/26, say so and still help with pointers.\n"
            "- Use regulatory reasoning: reconcile definitions, scope, and cross-references; summarize obligations and practical steps.\n"
            "- Critically evaluate the CONTEXT; do not invent facts not present there. If context is insufficient, say what is missing.\n\n"
            "Formatting:\n"
            "- Write a natural narrative with clear bold headings (e.g., **Overview**, **Requirements**, **Steps**, **Cross-references**, **Notes**).\n"
            "- Do NOT include a 'Sources' section in the answer.\n"
            "- Do NOT include URLs, web links, or citation identifiers in the answer. Citations will be shown separately.\n"
            "- Keep answers scoped to the question; do not over-hedge.\n\n"
            "Cross-references:\n"
            "- Resolve references like 'Regulation .03 of this chapter' or explicit COMAR sections (e.g., 15.15.04.03).\n\n"
            "History-aware clarity:\n"
            "- Consider the prior HISTORY when it helps keep continuity or avoid repetition.\n\n"
            "Follow-up:\n"
            "- When appropriate, end with a short **Follow-up** section containing 1–2 concise clarifying questions that help the user get a more precise answer (no links).\n"
            "- Keep follow-ups strictly within Titles 15/26 topics.\n\n"
            "Insufficient context:\n"
            "- If the CONTEXT does not clearly answer the QUESTION, ask for the specific COMAR section or program area, and avoid speculation.\n"
        )

        # Initialize retrieval chain components
        self._setup_retrieval_chain()
    
    def _setup_retrieval_chain(self):
        """Setup the retrieval and generation chain"""
        self.retriever = None
        if self.store is not None:
            try:
                self.retriever = self.store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 8, "fetch_k": 32, "lambda_mult": 0.5},
                )
            except Exception as e:
                logger.warning(f"Failed to initialize retriever from store, will use psycopg fallback: {e}")
                self.retriever = None
        
        if self._llm_available:
            # Setup prompt template
            self.prompt = ChatPromptTemplate.from_template(
                "{system}\n\nHISTORY:\n{history}\n\nUSER NOTES:\n{memories}\n\nSCOPE NOTE:\n{scope_note}\n\nCONTEXT (authoritative excerpts, do not invent beyond this):\n{context}\n\nQUESTION:\n{question}"
            )
            
            # Keep a simple chain that accepts precomputed context
            self.gen_chain = self.prompt | self.llm | StrOutputParser()

    # --- Conversation helpers ---
    def _format_history_for_prompt(self, conversation: List[Dict[str, Any]], max_items: int = 4) -> str:
        """Compact recent conversation into a short text block for the prompt."""
        try:
            if not conversation:
                return "(none)"
            # Take last N messages
            tail = conversation[-max_items:]
            parts: List[str] = []
            for m in tail:
                role = m.get("role") or m.get("author") or "user"
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if len(content) > 300:
                    content = content[:300] + "..."
                parts.append(f"{role}: {content}")
            return "\n".join(parts) if parts else "(none)"
        except Exception:
            return "(none)"

    def _format_memories_for_prompt(self, memories: List[str], max_items: int = 5) -> str:
        """Compact user memories into a short list for context."""
        try:
            if not memories:
                return "(none)"
            items = [str(m).strip() for m in memories if str(m).strip()][:max_items]
            if not items:
                return "(none)"
            return "\n".join(f"- {m[:240]}" + ("..." if len(m) > 240 else "") for m in items)
        except Exception:
            return "(none)"

    # --- Follow-up helpers ---
    def _suggest_followups(self, question: str, docs: List[Document], answer: str, max_items: int = 2) -> List[str]:
        """Heuristically suggest 1–2 useful follow-up questions within COMAR 15/26 scope."""
        q = (question or "").lower()
        followups: List[str] = []
        # Keyword-driven suggestions
        def add(x: str):
            if x not in followups and len(followups) < max_items:
                followups.append(x)
        if any(k in q for k in ["permit", "approval", "authorization"]):
            add("Do you want the specific COMAR section that sets permit application contents and timelines?")
        if any(k in q for k in ["record", "report", "reporting", "retain"]):
            add("Should I list the recordkeeping and reporting requirements with retention periods?")
        if any(k in q for k in ["penalty", "enforcement", "violation", "fine"]):
            add("Do you want enforcement and penalty provisions cross‑referenced for this topic?")
        if any(k in q for k in ["storage", "handling", "disposal", "transport"]):
            add("Do you need operational requirements (storage/handling) summarized as step‑by‑step checks?")
        if not followups:
            # Generic but helpful within scope
            add("Would you like me to pull the exact COMAR section text that governs this?")
            add("Should I check related definitions or exemptions that might change the outcome?")
        return followups[:max_items]

    def smart_retrieve(self, query: str, k: int = 8, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Enhanced retrieval with cross-reference expansion and web fallback"""
        # Step 1: Initial vector retrieval
        if self.retriever is not None:
            try:
                base_docs = self.retriever.get_relevant_documents(query)
            except Exception as e:
                logger.warning(f"Retriever failed, falling back to psycopg search: {e}")
                base_docs = self._psql_similarity_search_docs(query, k=k)
        else:
            base_docs = self._psql_similarity_search_docs(query, k=k)
        
        # Prefer Title 15/26 content by reordering
        def is_15_26(d: Document) -> int:
            sec = (d.metadata.get("section") or "").strip()
            return 1 if (sec.startswith("15.") or sec.startswith("26.")) else 0
        base_docs.sort(key=lambda d: is_15_26(d), reverse=True)

        # If few or weak hits, add a tiny keyword fallback
        if len(base_docs) < max(3, k // 2):
            kw_docs = self._psql_keyword_search_docs(query, k=min(6, k))
            base_docs.extend(kw_docs)

        # Step 2: Extract cross-referenced sections
        target_sections = extract_target_sections(query, base_docs)
        if filters and isinstance(filters.get("section"), str):
            target_sections.add(str(filters["section"]))
        
        # Step 3: Retrieve cross-referenced sections
        xref_docs = []
        for section in list(target_sections)[:6]:
            try:
                section_query = f"COMAR {section}"
                if self.retriever is not None:
                    additional_docs = []
                    try:
                        additional_docs = self.store.similarity_search(section_query, k=2, filter={"section": section})  # type: ignore
                    except Exception:
                        additional_docs = self.store.similarity_search(section_query, k=2)  # type: ignore
                else:
                    additional_docs = self._psql_similarity_search_docs(section_query, k=2)
                xref_docs.extend(additional_docs)
            except Exception as e:
                logger.warning(f"Failed to retrieve section {section}: {e}")
        
        # Step 4: Combine and deduplicate
        all_docs = base_docs + xref_docs
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hashlib.sha1(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Step 5: Web fallback if needed
        final_docs = augment_with_web_if_needed(query, unique_docs[: max(k, 8)], max_add=k)
        final_docs.sort(key=lambda d: is_15_26(d), reverse=True)
        return final_docs[:k]

    def _get_collection_uuid(self) -> Optional[str]:
        """Fetch and cache the UUID of the configured PGVector collection via psycopg."""
        if self._collection_uuid_cache:
            return self._collection_uuid_cache
        try:
            with psycopg.connect(
                self._psycopg_url,
                autocommit=True,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=15,
                keepalives_count=3,
            ) as conn:
                with conn.cursor() as cur:
                    try:
                        cur.execute("SET statement_timeout = 5000")
                    except Exception:
                        pass
                    cur.execute(
                        "SELECT uuid FROM langchain_pg_collection WHERE name = %s LIMIT 1",
                        (self.collection_name,),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        self._collection_uuid_cache = str(row[0])
                    else:
                        logger.warning(f"PGVector collection '{self.collection_name}' not found.")
                        self._collection_uuid_cache = None
        except Exception as e:
            logger.warning(f"Could not get collection UUID: {e}")
            self._collection_uuid_cache = None
        return self._collection_uuid_cache

    def _psql_similarity_search_docs(self, query: str, k: int = 8) -> List[Document]:
        try:
            import psycopg
            from pgvector.psycopg import register_vector  # type: ignore
        except Exception as e:
            logger.error(f"psycopg pgvector adapter not available: {e}")
            return []
        coll = self._get_collection_uuid()
        if not coll:
            return []
        # Compute embedding for the query
        try:
            vec = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []
        # Run similarity search against pgvector
        docs: List[Document] = []
        try:
            with psycopg.connect(
                self._psycopg_url,
                autocommit=True,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=15,
                keepalives_count=3,
            ) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    # Guard against long-running queries
                    try:
                        cur.execute("SET statement_timeout = 8000")
                    except Exception:
                        pass
                    cur.execute(
                        """
                        SELECT document, cmetadata, (embedding <-> %s) AS dist
                        FROM langchain_pg_embedding
                        WHERE collection_id = %s
                        ORDER BY embedding <-> %s
                        LIMIT %s
                        """,
                        (vec, coll, vec, k),
                    )
                    rows = cur.fetchall() or []
                    for (doc_text, meta, dist) in rows:
                        if not isinstance(meta, dict):
                            try:
                                meta = meta if isinstance(meta, dict) else {}
                            except Exception:
                                meta = {}
                        # Lower distance implies higher similarity; provide a simple score
                        try:
                            score = float(dist)
                        except Exception:
                            score = None
                        if score is not None:
                            meta = {**(meta or {}), "distance": score}
                        docs.append(Document(page_content=doc_text or "", metadata=meta or {}))
        except Exception as e:
            logger.warning(f"psycopg similarity search failed: {e}")
            return []
        return docs

    def _psql_keyword_search_docs(self, query: str, k: int = 5) -> List[Document]:
        """Very lightweight keyword fallback over raw text; limited and safe."""
        try:
            import psycopg
        except Exception:
            return []
        coll = self._get_collection_uuid()
        if not coll:
            return []
        # Extract a few meaningful tokens
        words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", query or "")]
        stop = {"maryland", "regulations", "title", "comar", "department", "state", "shall", "must"}
        tokens = [w for w in words if w not in stop][:4]
        if not tokens:
            tokens = [w for w in words][:2]
        if not tokens:
            return []
        like_params = [f"%{t}%" for t in tokens]
        conds = " OR ".join(["document ILIKE %s" for _ in like_params])
        sql = f"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE collection_id = %s AND ({conds})
            LIMIT %s
        """
        sql = sql.format(conds=conds)
        docs: List[Document] = []
        try:
            with psycopg.connect(
                self._psycopg_url,
                autocommit=True,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=15,
                keepalives_count=3,
            ) as conn:
                with conn.cursor() as cur:
                    try:
                        cur.execute("SET statement_timeout = 5000")
                    except Exception:
                        pass
                    cur.execute(sql, (coll, *like_params, k))
                    for (doc_text, meta) in cur.fetchall() or []:
                        if not isinstance(meta, dict):
                            meta = {}
                        docs.append(Document(page_content=doc_text or "", metadata=meta or {}))
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []
        return docs

    def _question_tokens(self, text: str) -> List[str]:
        w = re.findall(r"[a-zA-Z]{3,}", text.lower()) if text else []
        stop = {
            "the","and","for","you","your","with","from","this","that","what","when","where","who","how",
            "maryland","department","regulations","regulatory","state","comar","title","subtitle","chapter","section",
            "shall","must","may","notice","receive","received"
        }
        return [t for t in w if t not in stop][:8]

    def _grounding_score(self, docs: List[Document], question: str) -> float:
        if not docs:
            return 0.0
        toks = self._question_tokens(question)
        if not toks:
            toks = [t for t in re.findall(r"[a-zA-Z]{4,}", question.lower())][:4]
        hits = 0
        for d in docs[:6]:
            text = (d.page_content or "").lower()
            # Count token hits in text
            hits += sum(1 for t in toks if t in text)
        hit_rate = hits / max(1, len(toks) * min(6, len(docs)))
        # Title 15/26 boost
        t1526 = 1.0 if any((d.metadata.get("section") or "").startswith(("15.", "26.")) for d in docs) else 0.0
        # Distance (if available) — lower is better; map to 0..1
        dists = []
        for d in docs[:4]:
            try:
                dist = float(d.metadata.get("distance"))
                dists.append(dist)
            except Exception:
                pass
        dist_score = 0.0
        if dists:
            # Typical pgvector cosine or Euclidean distances: use a soft mapping
            m = sum(dists) / len(dists)
            # heuristic: <=0.35 strong, 0.35-0.7 medium, >0.7 weak
            if m <= 0.35:
                dist_score = 1.0
            elif m <= 0.7:
                dist_score = 0.6
            else:
                dist_score = 0.2
        # Combine heuristically
        base = 0.5 * hit_rate + 0.3 * t1526 + 0.2 * dist_score
        # Clamp
        return max(0.0, min(1.0, base))

    def _retrieve_docs(self, query: str, k: int = 8, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Compatibility wrapper: use smart_retrieve if available, otherwise a basic retriever/psycopg + optional web augment."""
        if hasattr(self, "smart_retrieve"):
            try:
                return self.smart_retrieve(query, k=k, filters=filters)
            except Exception as e:
                logger.warning(f"smart_retrieve failed: {e}")
        
        # Fallback: basic retrieval + web augment if no strong hits
        docs = self._psql_similarity_search_docs(query, k=k)
        if len(docs) < 2:
            docs += augment_with_web_if_needed(query, docs, max_add=k)
        return docs[:k]

    def answer_question(self, question: str, k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Main entry point for question answering. Combines fast RAG retrieval with
        LLM reasoning and graceful web fallback for Title 15/26 when needed.
        """
        try:
            # Fast path for smalltalk/greetings (guarded for older instances)
            try:
                if hasattr(self, "_is_smalltalk") and self._is_smalltalk(question):
                    if hasattr(self, "_smalltalk_reply"):
                        return self._smalltalk_reply()
                    return {"answer": "Hello! How can I help with COMAR Titles 15 and 26?", "sources": []}
            except Exception:
                pass
            
            conversation = kwargs.get("conversation") or []
            memories = kwargs.get("memories") or ""
            filters = kwargs.get("filters") or None

            # Use retrieval (smart when available)
            source_docs = self._retrieve_docs(question, k=k, filters=filters)

            # If grounding is weak, try a targeted keyword fallback merge
            gscore = self._grounding_score(source_docs, question)
            if gscore < 0.35:
                try:
                    extra = self._psql_keyword_search_docs(question + " enforcement compliance contact", k=min(6, max(3, k)))
                    # Merge and dedupe
                    seen = set(hashlib.sha1(d.page_content.encode()).hexdigest() for d in source_docs)
                    for d in extra:
                        key = hashlib.sha1(d.page_content.encode()).hexdigest()
                        if key not in seen:
                            source_docs.append(d)
                            seen.add(key)
                    # Recompute score after merge
                    gscore = self._grounding_score(source_docs, question)
                except Exception as e:
                    logger.info(f"Keyword merge failed: {e}")

            # Always prepare a fast extractive answer first
            fast_answer = self._generate_extractive_answer(question, source_docs)

            # Gate LLM paraphrase on grounding quality
            answer: str = fast_answer
            if self._llm_available and gscore >= 0.35 and len(source_docs) >= 2:
                prompt_vars = {
                    "system": self.system_prompt,
                    "history": self._format_history_for_prompt(conversation),
                    "memories": self._format_memories_for_prompt(memories if isinstance(memories, list) else [str(memories)]),
                    "scope_note": self._scope_note(source_docs),
                    "context": self._format_docs_for_prompt(source_docs),
                    "question": question,
                }
                # Run LLM paraphrasing in a background thread with timeout
                with cf.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(self.gen_chain.invoke, prompt_vars)
                    try:
                        llm_text = fut.result(timeout=self.llm_timeout_seconds)
                        answer = self._sanitize_answer(llm_text or fast_answer)
                    except cf.TimeoutError:
                        logger.info("LLM generation timed out; returning fast extractive answer")
                        answer = fast_answer
            else:
                # If still weakly grounded, be explicit and ask for the specific program/section
                if gscore < 0.35:
                    answer = (
                        "**Overview**\n\n"
                        "I don't have enough COMAR context to precisely answer who to contact. The right contact depends on the program and the specific regulation cited in your notice.\n\n"
                        "**Follow-up**\n\n"
                        "- Which program or division is named on the notice (e.g., Animal Health, Plant Protection)?\n"
                        "- What COMAR chapter/section is cited?"
                    )

            # Format sources for API response
            sources = self._format_sources(source_docs)
            if not sources:
                sources = self._fallback_sources_from_text(question)

            return {"answer": answer, "sources": sources}

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "sources": []
            }
    
    def _sanitize_answer(self, text: str) -> str:
        """Remove inline sources and unwanted boilerplate; trim leading labels."""
        if not text:
            return text
        # Remove any trailing Sources: section
        text = re.sub(r"\n+sources?:.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
        # Remove markdown links [label](url) while keeping the label
        text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
        # Remove bare URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove explicitly labeled prompt headings that might leak
        for heading in [
            r"^\s*short\s*answer\s*:\s*",
            r"^\s*what\s*the\s*rule\s*says\s*:\s*",
            r"^\s*steps?\s*/?\s*requirements\s*:\s*",
            r"^\s*cross-?references\s*:\s*",
            r"^\s*edge\s*cases?\s*/?\s*exceptions\s*/?\s*warnings\s*:\s*",
            r"^\s*missing\s*from\s*corpus\s*:\s*",
        ]:
            text = re.sub(heading, "", text, flags=re.IGNORECASE | re.MULTILINE)
        # Collapse extra whitespace created by removals
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    
    def _format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Format document sources for API response with COMAR URLs and display fields."""
        sources: List[Dict[str, Any]] = []
        sec_pat = re.compile(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
        
        for i, doc in enumerate(docs, 1):
            md = doc.metadata or {}
            # Determine COMAR section
            section = (
                md.get("section")
                or md.get("comar_number")
                or md.get("comarNumber")
                or (sec_pat.search(doc.page_content[:800]).group(1) if sec_pat.search(doc.page_content[:800]) else "")
            )
            section = str(section).strip()
            url = section_to_url(section) if section else (md.get("url") or md.get("source") or "")
            title = md.get("title") or md.get("doc_title") or md.get("docTitle") or ""
            
            label = f"COMAR {section}" if section else (title or f"Source {i}")
            snippet_src = doc.page_content.strip()
            snippet = (snippet_src[:240] + "...") if len(snippet_src) > 240 else snippet_src
            
            sources.append({
                "id": md.get("id") or f"src_{i}",
                "label": label,
                "pages": md.get("pages") or "",
                "url": url,
                "doc_id": md.get("doc_id") or md.get("docId"),
                "doc_title": title,
                "comar_number": section or None,
                "comar_display": (f"COMAR {section}" if section else None),
                "snippet": snippet,
            })
        
        return sources

    def _fallback_sources_from_text(self, text: str, max_items: int = 3) -> List[Dict[str, Any]]:
        """Heuristically extract COMAR sections from text and build clickable citations."""
        pat = re.compile(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
        seen: Set[str] = set()
        results: List[Dict[str, Any]] = []
        for m in pat.finditer(text or ""):
            sec = m.group(1)
            if sec in seen:
                continue
            seen.add(sec)
            url = section_to_url(sec)
            results.append({
                "id": f"inferred_{sec}",
                "label": f"COMAR {sec}",
                "pages": "",
                "url": url,
                "doc_id": None,
                "doc_title": None,
                "comar_number": sec,
                "comar_display": f"COMAR {sec}",
                "snippet": None,
            })
            if len(results) >= max_items:
                break
        return results

    def _scope_note(self, docs: List[Document]) -> str:
        """Summarize the scope for the LLM: Titles present and key sections."""
        if not docs:
            return (
                "No authoritative context retrieved. If relevant, focus on COMAR Titles 15 or 26."
            )
        sections: List[str] = []
        titles_present: set[str] = set()
        sec_pat = re.compile(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
        for d in docs[:8]:
            md = d.metadata or {}
            sec = (md.get("section") or md.get("comar_number") or md.get("comarNumber") or "").strip()
            if not sec:
                m = sec_pat.search(d.page_content[:800])
                if m:
                    sec = m.group(1)
            if sec:
                sections.append(sec)
                if sec.startswith("15"):
                    titles_present.add("15")
                if sec.startswith("26"):
                    titles_present.add("26")
        sections = list(dict.fromkeys(sections))[:6]
        title_note = (
            "Titles " + ", ".join(sorted(titles_present)) if titles_present else "No Title 15/26 detected"
        )
        if sections:
            return f"Primary context: {title_note}. Key sections: {', '.join(sections)}."
        return f"Primary context: {title_note}."

    def _format_docs_for_prompt(self, docs: List[Document]) -> str:
        """Format retrieved docs into a compact, URL-free context for the LLM."""
        if not docs:
            return ""
        parts: List[str] = []
        sec_pat = re.compile(r"(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)")
        for i, d in enumerate(docs[:6], 1):
            md = d.metadata or {}
            sec = (md.get("section") or md.get("comar_number") or md.get("comarNumber") or "").strip()
            if not sec:
                m = sec_pat.search(d.page_content[:800])
                if m:
                    sec = m.group(1)
            title = (md.get("title") or md.get("doc_title") or md.get("docTitle") or "").strip()
            header = f"[Source {i}] COMAR {sec}" if sec else f"[Source {i}] {title or 'Unlabeled source'}"
            # Clean and trim content, remove URLs
            text = d.page_content.strip()
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"https?://\S+", "", text)
            if len(text) > 1200:
                text = text[:1200] + "..."
            parts.append(f"{header}\n{text}")
        return "\n\n---\n\n".join(parts)

    def _generate_extractive_answer(self, question: str, docs: List[Document]) -> str:
        """Simple extractive-style answer using retrieved docs without URLs or inline sources."""
        if not docs:
            return (
                "**Overview**\n\n"
                "I don't have enough context to answer precisely. However, this likely concerns Maryland COMAR requirements. "
                "If you can share more detail (title, subtitle, or section), I can provide a targeted summary."
            )
        # Take top few docs and synthesize a concise summary
        parts: List[str] = []
        for d in docs[:3]:
            chunk = d.page_content.strip()
            # Trim long chunks
            if len(chunk) > 500:
                chunk = chunk[:500] + "..."
            parts.append(chunk)
        body = "\n\n".join(parts)
        return (
            "**Overview**\n\n"
            "Here is a concise summary based on the relevant Maryland regulatory text.\n\n"
            f"{body}\n\n"
            "If you need specifics, ask about a particular COMAR section (for example, give the numeric section)."
        )

    def ingest(self) -> None:
        """Compatibility no-op for Supabase-backed engine.
        Ensures startup succeeds without local ingestion.
        """
        try:
            # Touch the vector store to validate connectivity
            _ = getattr(self, "collection_name", None)
            # Validate psycopg fallback can see the collection
            _ = self._get_collection_uuid()
            logger.info("Ingest skipped (using existing Supabase PGVector collection).")
        except Exception as e:
            logger.warning(f"Ingest check failed: {e}")

    def answer(self, question: str, k: int = 5, **kwargs):
        """Compatibility wrapper returning tuple (answer_text, sources)."""
        result = self.answer_question(question, k=k, **kwargs)
        return result.get("answer", ""), result.get("sources", [])

# ================== CONVENIENCE FUNCTIONS ==================

def create_rag_engine() -> RAGEngine:
    """Factory function to create a RAG engine instance"""
    return RAGEngine()

# Test the configuration on import
if __name__ == "__main__":
    try:
        engine = create_rag_engine()
        logger.info("RAG engine created successfully")
        
        # Test query
        result = engine.answer_question("What are the requirements for organic certification?")
        print("Test query result:", result["answer"][:100] + "...")
        
    except Exception as e:
        logger.error(f"Failed to create RAG engine: {e}")
        raise