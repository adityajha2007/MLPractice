# SOP-to-JSON-DAG Conversion System

Converts Standard Operating Procedures (SOPs) into structured JSON Directed Acyclic Graphs (DAGs) using a multi-stage LLM pipeline with RAG-based preprocessing, entity resolution, and self-refinement.

## Architecture

```
Raw SOP --> preprocessing --> [enriched chunks + entity map + FAISS index]
                                      |
                    +-----------------+------------------+
                    |                                    |
             MAIN PATH                        COMPARISON ONLY
             PipelineConverter (v4)          BottomUp / EdgeVertex
             (graph-first: LLM structured    (produce graph for eval)
              output + chunk patching)
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
  preprocessing.py    # Chunking + FAISS + RAG enrichment + entity resolution + caching
  converter.py        # Graph-first pipeline (Full SOP -> graph JSON -> chunk-by-chunk patching)
  graph_ops.py        # Analysis + refinement + LangGraph self-refinement loop
  alternatives.py     # Research paper comparisons: BottomUp, EdgeVertex
  evaluation.py       # Metrics (node count, edge coverage, structural similarity)
  CONVERTER_EXPLAINED.md  # Detailed walkthrough of converter.py internals
  scripts/
    run_converter.py      # SOP -> preprocessed -> JSON DAG
    run_refinement.py     # Load stored graph -> refine
    run_full_pipeline.py  # End-to-end: preprocess -> convert -> refine
    compare_converters.py # Run all 3 converters, print metrics
  tests/
    fixtures/
      sample_sop_onboarding.md
    test_schemas.py
    test_topological.py
    test_storage.py
    test_validator.py
    test_metrics.py
    test_patch_application.py
    test_preprocessing_cache.py
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

## Caching and Resuming

Every stage saves checkpoints so you never lose progress on failure (e.g., token limit errors).

### Preprocessing cache

Preprocessing results (chunking, RAG enrichment, entity resolution) are automatically cached by SHA-256 of the SOP content. Subsequent runs for the same document skip preprocessing entirely.

```bash
# First run — preprocessing runs and is cached automatically
python -m sop_to_dag.scripts.run_full_pipeline sop_to_dag/tests/fixtures/sample_sop_fraud.md

# Second run (same SOP) — preprocessing is skipped, straight to conversion
python -m sop_to_dag.scripts.run_full_pipeline sop_to_dag/tests/fixtures/sample_sop_fraud.md

# Force re-run preprocessing (e.g., after changing the chunking prompt)
python -m sop_to_dag.scripts.run_full_pipeline sop_to_dag/tests/fixtures/sample_sop_fraud.md \
  --force-preprocess
```

### Converter resume

The converter checkpoints after every stage. Use `--resume` to pick up from where it left off:

```bash
# Resume a crashed run — skips completed stages, resumes mid-chunk if needed
python -m sop_to_dag.scripts.run_converter sop_to_dag/tests/fixtures/sample_sop_fraud.md \
  --resume output/stage_dumps/sample_sop_fraud_20260326_150000

python -m sop_to_dag.scripts.run_full_pipeline sop_to_dag/tests/fixtures/sample_sop_fraud.md \
  --resume output/stage_dumps/sample_sop_fraud_20260326_150000
```

### Refinement loop resume

The refinement loop dumps graph state after every iteration. On resume, it loads the latest iteration checkpoint and continues from there.

### Checkpoint summary

| Stage | Checkpoints saved | Resume behavior |
|-------|-------------------|-----------------|
| Preprocessing | `output/preprocessing_cache/{sha}.json` | Automatic — skips on cache hit |
| Converter Step 1 (graph gen) | `initial_graph.json` | Loads from cache, skips LLM call |
| Converter Step 2 (chunk patching) | `graph_p{pass}_c{chunk}.json` per chunk, `graph_after_pass_{N}.json` per pass | Finds latest per-chunk checkpoint, resumes from there |
| Refinement loop | `refine_iter{N}_graph.json` per iteration | Loads latest iteration, continues loop |

## Pipeline Stages

### 1. Preprocessing (`preprocessing.py`)

A 4-node LangGraph pipeline that always runs before conversion:

- **Agentic chunking** -- LLM splits SOP at logical process boundaries (not mid-sentence)
- **FAISS indexing** -- Embeds chunks using HuggingFace bge-base-en-v1.5 (local, no API cost)
- **RAG enrichment** -- For each chunk, generates queries for dangling references, retrieves from FAISS, grades relevance, condenses accepted retrievals into brief context notes (not full chunks)
- **Entity resolution** -- Groups synonymous terms (e.g., "CBRD team" / "Credit Bureau Reporting Disputes team") into canonical forms and normalizes all chunk text

### 2. Conversion (`converter.py`)

Graph-first pipeline — LLM produces structured graph JSON directly:

- **Step 1 (Graph Gen)** -- LLM converts the full enriched SOP into a workflow graph via structured output (`InitialGraph` Pydantic model). Produces `WorkflowNode` JSON directly — no lossy text-outline intermediate. Schema validation and topological checks run immediately after.
- **Step 2 (Graph Refine)** -- For each enriched chunk (2 passes), LLM produces a `GraphPatch` (add/modify/remove operations) comparing the chunk against the current graph. Patches are applied with rollback safety (70% size threshold, start node protection). Pass 2 catches cross-chunk temporal dependencies that Pass 1 missed.

See `CONVERTER_EXPLAINED.md` for a detailed walkthrough of every function and class.

### 3. Refinement Loop (`graph_ops.py`)

LangGraph StateGraph cycling between analysis and repair:

- **Analyser** -- Topological checks (pure Python), completeness check (LLM), context check (LLM), granularity check (LLM — flags coarse nodes that collapse multiple actions)
- **Refiner** -- Triplet verification (prioritizes low-confidence edges), granularity expansion (breaks coarse nodes into sub-steps), error resolution (2-hop neighborhood + optional FAISS RAG), schema validation (deterministic fixes)
- Loops until all checks pass or max iterations reached

### 4. Alternative Converters (`alternatives.py`)

For benchmarking only (no refinement loop):

- **BottomUpConverter** -- PADME-inspired: chunk -> process with context carryover -> merge
- **EdgeVertexConverter** -- Agent-S-inspired: extract vertices first -> map edges second

## Key Features

- **Graph-first conversion** -- LLM produces structured graph JSON from the start, preserving temporal dependencies, decision scope, and branch history that text-outline intermediates lose.
- **Patch-based refinement** -- Each chunk produces a surgical `GraphPatch` (add/modify/remove) rather than regenerating the whole graph, preserving earlier correct work.
- **Multi-pass chunk refinement** -- Pass 2 sees nodes added by Pass 1 from other chunks, enabling cross-chunk dependency wiring.
- **Rollback safety** -- Every patch is applied against a snapshot. Catastrophic patches (>30% node loss or start node removal) are automatically rolled back.
- **Confidence labels** -- Each node has `confidence: high|medium|low` indicating how directly its outgoing edges are supported by the SOP text. Low-confidence edges are verified first during refinement.
- **Context condensation** -- RAG enrichment condenses retrieved chunks into brief relevance notes instead of dumping full chunk text, reducing noise and token usage.
- **Entity resolution** -- Prevents duplicate nodes from inconsistent terminology across SOP sections.
- **Checkpoint-based resume** -- Every stage saves checkpoints; crashed runs resume from the last successful point without re-running completed work.
- **Inline prompts** -- Each module contains its own prompt constants co-located with the code that uses them.

## Configuration

The LLM backend is configured in `models.py`:

- **Model**: `gpt-oss-120b` (configurable via `MODEL_NAME`)
- **Embeddings**: `BAAI/bge-base-en-v1.5` (local HuggingFace model)
- **Temperature**: Varies by stage (0.2 for graph gen, 0.1 for graph refine/resolvers, 0.0 for analysers)
