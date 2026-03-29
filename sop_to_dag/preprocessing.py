"""Preprocessing: Chunking + FAISS indexing + RAG enrichment + entity resolution.

LangGraph pipeline (4 nodes):
  START -> agentic_chunk -> build_faiss_index -> enrich_chunks -> resolve_entities -> END

Always runs before conversion. Even short docs benefit from entity resolution.

Includes content-based caching: preprocessing results are keyed by SHA-256 hash
of the SOP document. The FAISS vector store is not serializable, so it is rebuilt
from cached chunks on cache hit.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from sop_to_dag.models import get_embeddings, get_model, safe_invoke
from sop_to_dag.schemas import (
    DependencyQueries,
    DependencyReview,
    DocumentChunks,
    EntityMap,
    EnrichedChunk,
    RAGPrepState,
)

# ---------------------------------------------------------------------------
# Inline prompts
# ---------------------------------------------------------------------------

_CHUNKING_SYSTEM = """\
You are a Document Segmentation Specialist for Standard Operating Procedures.

Split the SOP into logical semantic chunks at process boundaries. Each chunk
should represent a coherent section, phase, or sub-procedure.

Rules:
1. Do NOT summarize — retain 100% of the original text in the chunks.
2. Split at handoffs between departments, phases, or decision branches.
3. Do NOT split mid-sentence or mid-paragraph.
4. Each chunk must have a descriptive title.
5. Chunk IDs should be sequential integers starting from 0.
"""

_CHUNKING_HUMAN = """\
Split this SOP into semantic chunks:

---
{document}
---

Return a DocumentChunks object with the list of chunks.
"""

_ENRICHMENT_QUERY_SYSTEM = """\
You are a Dependency Analyst. For the given SOP chunk, identify any dangling
references — mentions of other sections, phases, teams, or processes that are
not fully defined within this chunk.

Generate search queries that would retrieve the missing context from other
chunks of the same SOP.

If there are no dangling references, return an empty list.
"""

_ENRICHMENT_QUERY_HUMAN = """\
## Chunk (ID: {chunk_id})
{chunk_text}

Generate search queries for any dangling references in this chunk.
"""

_ENRICHMENT_GRADE_SYSTEM = """\
You are a Relevance Grader. For each retrieved document, determine whether it
actually provides relevant context for the original query.

Be strict — only mark as relevant if the retrieved text directly addresses
the reference or dependency in question.
"""

_ENRICHMENT_GRADE_HUMAN = """\
## Original Query
{query}

## Retrieved Documents
{retrieved_docs}

Grade each retrieval. Return a DependencyReview with grades for each document.
"""

_CONDENSATION_SYSTEM = """\
You are a Context Condensation Specialist. A workflow-graph builder will use your
output to understand how this SOP chunk connects to the rest of the document.

You receive:
- A chunk of SOP text
- Context snippets retrieved from OTHER parts of the same SOP

Your job: produce ONE short, factual context note that tells the graph builder
what cross-chunk dependencies, upstream triggers, downstream handoffs, or shared
entities this chunk relies on.

Rules:
1. Be brutally concise — every sentence must carry a specific fact (a team name,
   system, threshold, condition, or handoff). Cut anything vague or redundant.
2. Drop any retrieved snippet that adds nothing useful to understanding this chunk.
3. Do NOT repeat information already present in the chunk itself.
4. Do NOT add information that is not in the retrieved text.
5. Merge overlapping facts — no duplicates.
6. Target 2-4 sentences. Use fewer if less context is relevant.
"""

_CONDENSATION_HUMAN = """\
## Chunk
{chunk_text}

## Retrieved Context from Other Sections
{retrieved_sections}

Write the context note. Return ONLY the note — no labels, no preamble.
"""

_ENTITY_RESOLUTION_SYSTEM = """\
You are a Terminology Standardization Specialist. Analyze the provided SOP
chunks and identify all entity mentions (teams, systems, processes, roles,
acronyms) that refer to the same thing using different names.

Group synonymous terms under a single canonical name. The canonical name should
be the most formal/complete version.

Examples:
- canonical: "Credit Bureau Reporting Disputes team"
  aliases: ["CBRD team", "CBRD group", "the CBRD"]
- canonical: "Fraud Resolution Guide"
  aliases: ["FRG", "the fraud guide", "Fraud Guide"]

Only include groups with 2+ names. Single-use terms don't need mapping.
"""

_ENTITY_RESOLUTION_HUMAN = """\
Analyze these SOP chunks for inconsistent terminology:

{chunks_text}

Return an EntityMap with all synonym groups found.
"""


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------


def agentic_chunk(state: RAGPrepState) -> RAGPrepState:
    """Node 1: LLM splits SOP at logical process boundaries."""
    logger.info("[PREPROCESSING 1/4] Agentic chunking — splitting SOP into semantic chunks...")
    doc_len = len(state["document"])
    logger.info("  Input document: %d chars", doc_len)

    llm = get_model("chunking")
    structured_llm = llm.with_structured_output(DocumentChunks)

    messages = [
        SystemMessage(content=_CHUNKING_SYSTEM),
        HumanMessage(content=_CHUNKING_HUMAN.format(document=state["document"])),
    ]

    result = safe_invoke(structured_llm, messages, context="chunking")
    state["chunks"] = [c.model_dump() for c in result.chunks]
    logger.info("  Produced %d chunks: %s",
                 len(state["chunks"]),
                 [c["title"] for c in state["chunks"]])

    return state


def build_faiss_index(state: RAGPrepState) -> RAGPrepState:
    """Node 2: Embed all chunks into FAISS using HuggingFace bge-base-en."""
    logger.info("[PREPROCESSING 2/4] Building FAISS index...")
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()
    texts = [c["text"] for c in state["chunks"]]
    metadatas = [{"chunk_id": c["chunk_id"], "title": c["title"]} for c in state["chunks"]]

    if texts:
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        state["vector_store"] = vector_store
        logger.info("  FAISS index built with %d documents", len(texts))
    else:
        state["vector_store"] = None
        logger.warning("  No texts to index — FAISS store is None")

    return state


def enrich_chunks(state: RAGPrepState) -> RAGPrepState:
    """Node 3: For each chunk, generate queries, retrieve, grade, condense."""
    total_chunks = len(state["chunks"])
    logger.info("[PREPROCESSING 3/4] RAG enrichment — processing %d chunks...", total_chunks)
    llm_query = get_model("enrichment")
    llm_grade = get_model("enrichment")
    llm_condense = get_model("enrichment")
    vector_store = state["vector_store"]
    enriched: List[dict] = []

    for chunk in state["chunks"]:
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]
        logger.info("  Chunk %d/%d: generating dependency queries...", chunk_id + 1, total_chunks)

        # Step 1: Generate dependency queries
        queries = _generate_queries(llm_query, chunk_id, chunk_text)
        logger.info("    %d queries generated", len(queries))

        # Step 2 + 3: Retrieve and grade per query, collect accepted text
        query_texts: List[str] = []
        accepted_sections: List[str] = []  # "Query: ...\n<accepted text>"

        for dep in queries:
            query_texts.append(dep.query)

            if vector_store is None:
                continue

            # Retrieve from FAISS (k=2)
            docs = vector_store.similarity_search(dep.query, k=2)
            # Filter out the chunk itself
            docs = [d for d in docs if d.metadata.get("chunk_id") != chunk_id]

            if not docs:
                continue

            # Grade retrievals
            graded = _grade_retrievals(llm_grade, dep.query, docs)
            accepted_docs = [doc for doc, grade in zip(docs, graded) if grade]
            logger.info("    Query '%s': %d/%d retrievals accepted",
                         dep.query[:60], len(accepted_docs), len(docs))

            if accepted_docs:
                combined = "\n\n".join(doc.page_content for doc in accepted_docs)
                accepted_sections.append(f"Query: {dep.query}\n{combined}")

        # Step 4: One condensation call per chunk (all queries combined)
        condensed_context = ""
        if accepted_sections:
            total_chars = sum(len(s) for s in accepted_sections)
            condensed_context = _condense_context(
                llm_condense, chunk_text, "\n\n---\n\n".join(accepted_sections)
            )
            logger.info("    Condensed %d chars of retrievals -> %d char note",
                         total_chars, len(condensed_context))

        enriched_chunk = EnrichedChunk(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            retrieved_context=condensed_context,
            generated_queries=query_texts,
        )
        enriched.append(enriched_chunk.model_dump())
        logger.info("    Chunk %d enriched: %d chars of condensed context",
                     chunk_id, len(condensed_context))

    state["enriched_chunks"] = enriched
    logger.info("  Enrichment complete: %d chunks processed", len(enriched))
    return state


def resolve_entities(state: RAGPrepState) -> RAGPrepState:
    """Node 4: Collect entity mentions, LLM groups synonyms, replace aliases."""
    logger.info("[PREPROCESSING 4/4] Entity resolution — standardizing terminology...")
    llm = get_model("entity_resolution")
    structured_llm = llm.with_structured_output(EntityMap)

    # Build combined chunks text for analysis
    chunks_text = "\n\n---\n\n".join(
        f"## Chunk {c['chunk_id']}: {c.get('title', '')}\n{c['text']}"
        for c in state["chunks"]
    )

    messages = [
        SystemMessage(content=_ENTITY_RESOLUTION_SYSTEM),
        HumanMessage(content=_ENTITY_RESOLUTION_HUMAN.format(chunks_text=chunks_text)),
    ]

    entity_map = safe_invoke(structured_llm, messages, context="entity_resolution")
    mappings = [m.model_dump() for m in entity_map.mappings]

    logger.info("  Found %d entity mappings", len(mappings))
    for m in mappings:
        logger.info("    '%s' <- %s", m["canonical"], m["aliases"])

    # Replace aliases with canonical forms in enriched chunks
    if mappings:
        for ec in state["enriched_chunks"]:
            ec["chunk_text"] = _apply_entity_map(ec["chunk_text"], mappings)
            if ec.get("retrieved_context"):
                ec["retrieved_context"] = _apply_entity_map(
                    ec["retrieved_context"], mappings
                )
        logger.info("  Applied entity map to %d enriched chunks", len(state["enriched_chunks"]))

    state["entity_map"] = mappings
    logger.info("[PREPROCESSING] Complete.")
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_queries(llm, chunk_id: int, chunk_text: str) -> List:
    """Generate dependency queries for a chunk."""
    structured_llm = llm.with_structured_output(DependencyQueries)
    messages = [
        SystemMessage(content=_ENRICHMENT_QUERY_SYSTEM),
        HumanMessage(
            content=_ENRICHMENT_QUERY_HUMAN.format(
                chunk_id=chunk_id, chunk_text=chunk_text
            )
        ),
    ]
    result = safe_invoke(structured_llm, messages, context=f"query_gen/chunk_{chunk_id}")
    return result.queries


def _grade_retrievals(llm, query: str, docs) -> List[bool]:
    """Grade each retrieval for relevance. Returns list of booleans."""
    structured_llm = llm.with_structured_output(DependencyReview)
    retrieved_text = "\n\n".join(
        f"[Doc {i}]: {d.page_content}" for i, d in enumerate(docs)
    )
    messages = [
        SystemMessage(content=_ENRICHMENT_GRADE_SYSTEM),
        HumanMessage(
            content=_ENRICHMENT_GRADE_HUMAN.format(
                query=query, retrieved_docs=retrieved_text
            )
        ),
    ]
    result = safe_invoke(structured_llm, messages, context="retrieval_grading")
    return [g.is_relevant for g in result.grades]


def _condense_context(llm, chunk_text: str, retrieved_sections: str) -> str:
    """Condense all retrieved context for a chunk into one unified note."""
    messages = [
        SystemMessage(content=_CONDENSATION_SYSTEM),
        HumanMessage(
            content=_CONDENSATION_HUMAN.format(
                chunk_text=chunk_text, retrieved_sections=retrieved_sections
            )
        ),
    ]
    response = safe_invoke(llm, messages, context="condensation")
    return response.content.strip()


def _apply_entity_map(text: str, mappings: List[dict]) -> str:
    """Replace all alias occurrences with canonical forms."""
    for mapping in mappings:
        canonical = mapping["canonical"]
        for alias in mapping["aliases"]:
            if alias != canonical and alias in text:
                text = text.replace(alias, canonical)
    return text


# ---------------------------------------------------------------------------
# Pipeline builder + runner
# ---------------------------------------------------------------------------


def _build_preprocessing_graph() -> StateGraph:
    """Build the 4-node LangGraph for preprocessing."""
    graph = StateGraph(RAGPrepState)

    graph.add_node("agentic_chunk", agentic_chunk)
    graph.add_node("build_faiss_index", build_faiss_index)
    graph.add_node("enrich_chunks", enrich_chunks)
    graph.add_node("resolve_entities", resolve_entities)

    graph.set_entry_point("agentic_chunk")
    graph.add_edge("agentic_chunk", "build_faiss_index")
    graph.add_edge("build_faiss_index", "enrich_chunks")
    graph.add_edge("enrich_chunks", "resolve_entities")
    graph.add_edge("resolve_entities", END)

    return graph


def run_preprocessing(document: str) -> RAGPrepState:
    """Run the full preprocessing pipeline on a raw SOP document.

    Returns RAGPrepState with enriched_chunks, vector_store, and entity_map.
    """
    initial_state = RAGPrepState(
        document=document,
        chunks=[],
        vector_store=None,
        enriched_chunks=[],
        entity_map=[],
    )

    graph = _build_preprocessing_graph()
    app = graph.compile()
    return app.invoke(initial_state)


# ---------------------------------------------------------------------------
# Content-based caching
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path("output/preprocessing_cache")


def _content_hash(text: str) -> str:
    """SHA-256 hash of document content (first 16 hex chars for filename)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: Path, content_hash: str) -> Path:
    return cache_dir / f"{content_hash}.json"


def save_to_cache(
    prep_state: RAGPrepState,
    source_text: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Persist preprocessing results to the content-based cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = _content_hash(source_text)
    path = _cache_path(cache_dir, h)

    data = {
        "content_hash": h,
        "chunks": prep_state["chunks"],
        "enriched_chunks": prep_state["enriched_chunks"],
        "entity_map": prep_state["entity_map"],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("[PREP CACHE] Saved: %s", path)
    return path


def load_from_cache(
    source_text: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Optional[dict]:
    """Load cached preprocessing results if they exist.

    Returns a dict with chunks, enriched_chunks, entity_map, and
    vector_store=None (caller must rebuild if needed). Returns None on miss.
    """
    h = _content_hash(source_text)
    path = _cache_path(cache_dir, h)

    if not path.exists():
        return None

    data = json.loads(path.read_text())

    # Validate hash matches (guard against manual file edits)
    if data.get("content_hash") != h:
        logger.warning("[PREP CACHE] Hash mismatch in %s — ignoring", path)
        return None

    logger.info("[PREP CACHE] Hit: %s", path)
    return {
        "chunks": data["chunks"],
        "enriched_chunks": data["enriched_chunks"],
        "entity_map": data["entity_map"],
        "vector_store": None,
    }


def rebuild_vector_store(prep_state: dict) -> None:
    """Rebuild the FAISS vector store from cached chunks (in-place).

    Modifies prep_state["vector_store"] directly.
    """
    from langchain_community.vectorstores import FAISS

    chunks = prep_state.get("chunks", [])
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    metadatas = [{"chunk_id": c["chunk_id"], "title": c["title"]} for c in chunks]
    embeddings = get_embeddings()
    prep_state["vector_store"] = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    logger.info("[PREP CACHE] Rebuilt FAISS vector store from %d chunks", len(chunks))


def cached_preprocessing(
    source_text: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    rebuild_faiss: bool = True,
) -> RAGPrepState:
    """Run preprocessing with content-based caching.

    Args:
        source_text: Raw SOP document text.
        cache_dir: Directory for cache files.
        force: If True, bypass cache and re-run preprocessing.
        rebuild_faiss: If True, rebuild FAISS vector store on cache hit.

    Returns:
        RAGPrepState with all fields populated.
    """
    if not force:
        cached = load_from_cache(source_text, cache_dir)
        if cached is not None:
            logger.info("[PREP CACHE] Using cached preprocessing")
            if rebuild_faiss:
                rebuild_vector_store(cached)
            return cached

    logger.info("[PREP CACHE] Cache miss — running preprocessing")
    prep_state = run_preprocessing(source_text)
    save_to_cache(prep_state, source_text, cache_dir)
    return prep_state
