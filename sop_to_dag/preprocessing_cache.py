"""Content-based preprocessing cache.

Caches preprocessing results (chunks, enriched_chunks, entity_map) keyed by
SHA-256 hash of the SOP document content. The FAISS vector store is not
serializable, so it is rebuilt from cached chunks on cache hit.

Usage:
    from sop_to_dag.preprocessing_cache import cached_preprocessing

    prep_state = cached_preprocessing(source_text)
    # Returns RAGPrepState — runs preprocessing only on cache miss.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from sop_to_dag.preprocessing import run_preprocessing
from sop_to_dag.schemas import RAGPrepState

logger = logging.getLogger(__name__)

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

    from sop_to_dag.models import get_embeddings

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
