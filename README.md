# SOP-to-JSON-DAG Conversion System

Converts Standard Operating Procedures (SOPs) into structured JSON Directed Acyclic Graphs (DAGs) using a multi-stage LLM pipeline with RAG-based preprocessing, entity resolution, and self-refinement.

## Architecture

```
Raw SOP --> preprocessing --> [enriched chunks + entity map + FAISS index]
                                      |
                    +-----------------+------------------+
                    |                                    |
             MAIN PATH                        COMPARISON ONLY
             PipelineConverter                BottomUp / EdgeVertex
             (3-stage: LLM+deterministic)    (produce graph for eval)
                    |
             Refinement Loop
             (Analyser <-> Refiner)
                    |
             FINAL JSON DAG
```

## Project Structure

```
sop_to_dag/
  schemas.py          # All Pydantic models + TypedDicts
  models.py           # LLM + embedding model factories
  storage.py          # GraphStore (file-based JSON persistence)
  preprocessing.py    # Chunking + FAISS indexing + RAG enrichment + entity resolution
  converter.py        # Main 3-stage pipeline (Outline -> Detail Pass -> Direct Text-to-Graph)
  alternatives.py     # Research paper comparisons: BottomUp, EdgeVertex
  analyser.py         # Topological + completeness + context checks
  refiner.py          # Triplet verification + error resolver + schema validator
  loop.py             # LangGraph StateGraph: Analyser <-> Refiner loop
  evaluation.py       # Metrics (node count, edge coverage, structural similarity)
  scripts/
    run_converter.py      # SOP -> preprocessed -> JSON DAG
    run_refinement.py     # Load stored graph -> refine
    run_full_pipeline.py  # End-to-end: preprocess -> convert -> refine
    compare_converters.py # Run all 3 converters, print metrics
  tests/
    fixtures/
      sample_sop_fraud.md
      sample_sop_onboarding.md
    test_schemas.py
    test_topological.py
    test_storage.py
    test_validator.py
    test_metrics.py
    test_graph_builder.py
    test_outline_parser.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pydantic langchain-core langchain-openai langgraph \
            langchain-community langchain-huggingface faiss-cpu
```

For running tests:

```bash
pip install pytest
```

## Usage

### Full pipeline (preprocess + convert + refine)

```bash
python -m sop_to_dag.scripts.run_full_pipeline sop_to_dag/tests/fixtures/sample_sop_fraud.md
```

### Convert only (preprocess + convert, no refinement)

```bash
python -m sop_to_dag.scripts.run_converter sop_to_dag/tests/fixtures/sample_sop_fraud.md
```

### Refine a stored graph

```bash
python -m sop_to_dag.scripts.run_refinement output/graphs/<graph_file>.json
```

### Compare all three converters

```bash
python -m sop_to_dag.scripts.compare_converters sop_to_dag/tests/fixtures/sample_sop_fraud.md
```

### Run tests

```bash
python -m pytest sop_to_dag/tests/ -v
```

## Pipeline Stages

### 1. Preprocessing (`preprocessing.py`)

A 4-node LangGraph pipeline that always runs before conversion:

- **Agentic chunking** -- LLM splits SOP at logical process boundaries (not mid-sentence)
- **FAISS indexing** -- Embeds chunks using HuggingFace bge-base-en-v1.5 (local, no API cost)
- **RAG enrichment** -- For each chunk, generates queries for dangling references, retrieves from FAISS, grades relevance, keeps valid context
- **Entity resolution** -- Groups synonymous terms (e.g., "CBRD team" / "Credit Bureau Reporting Disputes team") into canonical forms and normalizes all chunk text

### 2. Conversion (`converter.py`)

3-stage pipeline — LLM produces plain text, then deterministic compilation:

- **Step 1 (Outline)** -- LLM converts the full enriched SOP into a plain-text numbered outline with `DECISION:` prefixed decision points, indented `YES:`/`NO:` branches, and self-explanatory step text
- **Step 2 (Detail Pass)** -- For each enriched chunk, LLM verifies the outline captures every detail from that section, adding any missing actions, decisions, references, or specific values (codes, team names, thresholds)
- **Step 3 (Direct Text-to-Graph)** -- Pure Python parser reads the outline and emits graph nodes directly — `numbered step -> instruction`, `DECISION: -> question`, terminal keywords -> `terminal`. No intermediate Pydantic models, no LLM — zero hallucination risk

### 3. Refinement Loop (`loop.py`)

LangGraph StateGraph cycling between analysis and repair:

- **Analyser** -- Topological checks (pure Python), completeness check (LLM), context check (LLM)
- **Refiner** -- Triplet verification (prioritizes low-confidence edges), error resolution (2-hop neighborhood + optional FAISS RAG), schema validation (deterministic fixes)
- Loops until all checks pass or max iterations reached

### 4. Alternative Converters (`alternatives.py`)

For benchmarking only (no refinement loop):

- **BottomUpConverter** -- PADME-inspired: chunk -> process with context carryover -> merge
- **EdgeVertexConverter** -- Agent-S-inspired: extract vertices first -> map edges second

## Key Features

- **Confidence labels** -- Each node has `confidence: high|medium|low` indicating how directly its outgoing edges are supported by the SOP text. Low-confidence edges are verified first during refinement.
- **Entity resolution** -- Prevents duplicate nodes from inconsistent terminology across SOP sections.
- **RAG-enhanced refinement** -- The error resolver can use the FAISS vector store to retrieve relevant SOP sections for surgical fixes.
- **Inline prompts** -- Each module contains its own prompt constants co-located with the code that uses them.

## Configuration

The LLM backend is configured in `models.py`:

- **Model**: `gpt-oss-120b` (configurable via `MODEL_NAME`)
- **Embeddings**: `BAAI/bge-base-en-v1.5` (local HuggingFace model)
- **Temperature**: Varies by stage (0.2 for converters, 0.0 for analysers, 0.1 for resolvers)
