"""LLM + embedding model factories.

Temperature presets per stage category. Embedding model uses HuggingFace
bge-base-en for local FAISS indexing (no API cost).

Provides `safe_invoke()` — a wrapper around LLM `.invoke()` that halts the
pipeline on non-200 responses (rate limits, server errors) so the checkpoint
system can handle recovery.
"""

import logging

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

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
    # Graph-first converter
    "graph_gen": 0.2,
    "graph_refine": 0.1,
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


class LLMStopError(Exception):
    """Raised when an LLM call gets a non-200 response that should halt the pipeline.

    The pipeline should NOT retry — instead it should stop and let the user
    resume from the last checkpoint once the issue (rate limit, quota, outage)
    is resolved.
    """

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


def safe_invoke(llm_or_structured, messages, *, context: str = ""):
    """Call llm.invoke(messages) and halt on non-200 API responses.

    Args:
        llm_or_structured: A ChatOpenAI or structured-output LLM instance.
        messages: List of LangChain messages to send.
        context: Short label for logging (e.g. "graph_gen step 1").

    Returns:
        The LLM response (AIMessage or Pydantic model for structured output).

    Raises:
        LLMStopError: On rate-limit (429), server error (5xx), or auth error (401/403).
            The pipeline should stop and resume from checkpoint.
    """
    try:
        return llm_or_structured.invoke(messages)
    except Exception as e:
        status = _extract_status_code(e)
        if status and status != 200:
            label = f" [{context}]" if context else ""
            logger.error(
                "LLM call failed%s with HTTP %d: %s. "
                "Pipeline stopped — resume from last checkpoint once resolved.",
                label, status, e,
            )
            raise LLMStopError(status, str(e)) from e
        # Non-HTTP errors (parsing, network timeout, etc.) — re-raise as-is
        raise


def _extract_status_code(exc: Exception) -> int | None:
    """Try to pull an HTTP status code from a LangChain/OpenAI exception."""
    # openai.RateLimitError, openai.APIStatusError, etc.
    if hasattr(exc, "status_code"):
        return exc.status_code
    # Some wrappers nest the original error
    if hasattr(exc, "__cause__") and hasattr(exc.__cause__, "status_code"):
        return exc.__cause__.status_code
    # Check string representation as last resort
    err_str = str(exc)
    for code in (429, 500, 502, 503, 504, 401, 403):
        if f"{code}" in err_str:
            return code
    return None


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
