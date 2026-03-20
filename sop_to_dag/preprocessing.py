"""Preprocessing: Chunking + FAISS indexing + RAG enrichment + entity resolution.

LangGraph pipeline (4 nodes):
  START -> agentic_chunk -> build_faiss_index -> enrich_chunks -> resolve_entities -> END

Always runs before conversion. Even short docs benefit from entity resolution.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from sop_to_dag.models import get_embeddings, get_model
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
    llm = get_model("chunking")
    structured_llm = llm.with_structured_output(DocumentChunks)

    messages = [
        SystemMessage(content=_CHUNKING_SYSTEM),
        HumanMessage(content=_CHUNKING_HUMAN.format(document=state["document"])),
    ]

    try:
        result = structured_llm.invoke(messages)
        state["chunks"] = [c.model_dump() for c in result.chunks]
    except Exception as e:
        logger.warning("Chunking failed, falling back to single chunk: %s", e)
        state["chunks"] = [
            {"chunk_id": 0, "title": "Full Document", "text": state["document"]}
        ]

    return state


def build_faiss_index(state: RAGPrepState) -> RAGPrepState:
    """Node 2: Embed all chunks into FAISS using HuggingFace bge-base-en."""
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()
    texts = [c["text"] for c in state["chunks"]]
    metadatas = [{"chunk_id": c["chunk_id"], "title": c["title"]} for c in state["chunks"]]

    if texts:
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        state["vector_store"] = vector_store
    else:
        state["vector_store"] = None

    return state


def enrich_chunks(state: RAGPrepState) -> RAGPrepState:
    """Node 3: For each chunk, generate queries, retrieve, grade, synthesize."""
    llm_query = get_model("enrichment")
    llm_grade = get_model("enrichment")
    vector_store = state["vector_store"]
    enriched: List[dict] = []

    for chunk in state["chunks"]:
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]

        # Step 1: Generate dependency queries
        queries = _generate_queries(llm_query, chunk_id, chunk_text)

        # Step 2 + 3: Retrieve and grade
        valid_context_parts: List[str] = []
        query_texts: List[str] = []

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
            for doc, grade in zip(docs, graded):
                if grade:
                    valid_context_parts.append(doc.page_content)

        enriched_chunk = EnrichedChunk(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            retrieved_context="\n\n".join(valid_context_parts),
            generated_queries=query_texts,
        )
        enriched.append(enriched_chunk.model_dump())

    state["enriched_chunks"] = enriched
    return state


def resolve_entities(state: RAGPrepState) -> RAGPrepState:
    """Node 4: Collect entity mentions, LLM groups synonyms, replace aliases."""
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

    try:
        entity_map = structured_llm.invoke(messages)
        mappings = [m.model_dump() for m in entity_map.mappings]
    except Exception as e:
        logger.warning("Entity resolution failed: %s", e)
        mappings = []

    # Replace aliases with canonical forms in enriched chunks
    if mappings:
        for ec in state["enriched_chunks"]:
            ec["chunk_text"] = _apply_entity_map(ec["chunk_text"], mappings)
            if ec.get("retrieved_context"):
                ec["retrieved_context"] = _apply_entity_map(
                    ec["retrieved_context"], mappings
                )

    state["entity_map"] = mappings
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
    try:
        result = structured_llm.invoke(messages)
        return result.queries
    except Exception as e:
        logger.warning("Query generation failed for chunk %d: %s", chunk_id, e)
        return []


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
    try:
        result = structured_llm.invoke(messages)
        return [g.is_relevant for g in result.grades]
    except Exception as e:
        logger.warning("Retrieval grading failed for query '%s': %s", query, e)
        return [False] * len(docs)


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
