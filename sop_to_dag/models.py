"""LLM + embedding model factories.

Temperature presets per stage category. Embedding model uses HuggingFace
bge-base-en for local FAISS indexing (no API cost).
"""

from langchain_openai import ChatOpenAI

_TEMPERATURE_MAP = {
    # Converter stages
    "top_down": 0.2,
    "code_based": 0.2,
    "graph_based": 0.2,
    "bottom_up": 0.2,
    "edge_vertex": 0.2,
    # Analyser checks
    "completeness": 0.0,
    "context": 0.0,
    # Refiner
    "triplet": 0.0,
    "resolver": 0.1,
    # Preprocessing
    "chunking": 0.1,
    "enrichment": 0.0,
    "entity_resolution": 0.0,
}

MODEL_NAME = "gpt-oss-120b"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def get_model(stage: str = "default") -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for the given stage."""
    temperature = _TEMPERATURE_MAP.get(stage, 0.1)
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature)


def get_embeddings():
    """Return a HuggingFace embeddings model for FAISS indexing.

    Uses bge-base-en-v1.5 (local, no API cost).
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
