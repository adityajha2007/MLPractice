"""SOP-to-JSON-DAG conversion system.

Flat package: schemas, models, storage, preprocessing, converter,
alternatives, analyser, refiner, loop, evaluation.
"""

import warnings

# Suppress LangChain's internal PydanticSerializationUnexpectedValue warnings.
# These fire on every `with_structured_output()` call because LangChain's
# internal wrapper uses Optional[T] for the `parsed` field, and Pydantic warns
# when a non-None value is serialized into an Optional slot. This is not
# actionable — the structured output works correctly.
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic serializer warnings.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
