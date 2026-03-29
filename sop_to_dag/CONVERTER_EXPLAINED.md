# How the SOP-to-DAG System Works

This document explains every stage of the pipeline — from raw SOP text to a fully-validated workflow graph. Each function, class, and design decision is covered.

---

## End-to-End Flow

```
Raw SOP Text (Markdown)
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│  PREPROCESSING  (preprocessing.py)                             │
│  LangGraph pipeline — 4 nodes, runs once per unique SOP       │
│                                                                │
│  1. Agentic Chunking      LLM splits SOP at logical boundaries│
│  2. FAISS Indexing         Embed chunks → vector store         │
│  3. RAG Enrichment         Per-chunk: query → retrieve → grade │
│                            → condense into ONE context note    │
│  4. Entity Resolution      Group synonyms → normalize all text │
│                                                                │
│  Results cached by SHA-256 of SOP content.                     │
│  Subsequent runs skip preprocessing entirely.                  │
└───────────────────────────┬────────────────────────────────────┘
                            │
               enriched_chunks + entity_map
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  CONVERTER  (converter.py) — Graph-First Pipeline v4           │
│                                                                │
│  Step 1:  Full enriched SOP → graph JSON (one LLM call)       │
│  Step 2:  Chunk-by-chunk graph patching (2 passes)            │
│                                                                │
│  Checkpoints after every chunk. Resume with --resume.          │
└───────────────────────────┬────────────────────────────────────┘
                            │
                       JSON DAG (nodes dict)
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  REFINEMENT LOOP  (graph_ops.py)                               │
│  LangGraph StateGraph:  Analyser ↔ Refiner                     │
│                                                                │
│  Analyser:  4 checks (topological, completeness, context,     │
│             granularity)                                       │
│  Refiner:   Per-category patching + triplet verification      │
│                                                                │
│  Cycles until all checks pass or max iterations reached.       │
│  Checkpoints after every iteration.                            │
└────────────────────────────────────────────────────────────────┘
```

---

## Why Graph-First (v4 vs v3)

The previous v3 converter used a 3-step pipeline:

```
v3: SOP → [LLM] → text outline → [LLM] → refined text → [parser] → graph
v4: SOP → [LLM] → graph JSON   → [LLM per chunk] → patched graph
```

The text outline was a **lossy intermediate** — it encoded topology via indentation but lost:
- Cross-chunk decision scope (a decision in section 2 referencing a branch target in section 5)
- Branch history (which branches converge and where)
- Nested convergence semantics (multiple levels of if/else merging back)

v4 eliminates the text intermediate. The LLM produces graph JSON directly, and refinement operates on the graph structure itself.

---

# Stage 1: Preprocessing (`preprocessing.py`)

Preprocessing runs as a 4-node LangGraph pipeline. Results are cached by SHA-256 of the SOP content — the second run with the same document skips straight to conversion.

## 1.1 Agentic Chunking — `agentic_chunk(state)`

**What it does:** LLM splits the raw SOP into semantic chunks at logical process boundaries (not mid-sentence or mid-step).

**How it works:**
1. Takes the full SOP document from `state["document"]`
2. Sends it to the LLM with `_CHUNKING_SYSTEM` prompt via `safe_invoke()`
3. LLM returns a `DocumentChunks` Pydantic model — a list of `Chunk` objects, each with `chunk_id`, `title`, and `text`
4. Stores the chunks as dicts in `state["chunks"]`

**Why a single LLM call instead of regex/heuristic splitting:** SOPs have inconsistent formatting — numbered lists, nested bullets, tables, free-text paragraphs. A rule-based splitter either over-splits (breaking a multi-step process) or under-splits (merging unrelated sections). The LLM understands process boundaries semantically.

**Prompt (`_CHUNKING_SYSTEM`):** Instructs the LLM to split at logical phase/section boundaries, keep each chunk self-contained, and preserve all original text verbatim.

## 1.2 FAISS Indexing — `build_faiss_index(state)`

**What it does:** Embeds all chunks into a FAISS vector store for similarity search.

**How it works:**
1. Creates LangChain `Document` objects from each chunk, with `chunk_id` in metadata
2. Calls `get_embeddings()` to get the HuggingFace `bge-base-en-v1.5` model (runs locally, no API cost)
3. Builds a FAISS index via `FAISS.from_documents()`
4. Stores the vector store in `state["vector_store"]`

**Why local embeddings:** Embedding calls are cheap but numerous (one per chunk). Using a local model avoids API round-trips and rate limits. `bge-base-en-v1.5` is a strong general-purpose embedding model that runs on CPU.

## 1.3 RAG Enrichment — `enrich_chunks(state)`

**What it does:** For each chunk, identifies dangling references (mentions of other sections, teams, systems) and retrieves relevant context from other chunks. Then condenses ALL retrieved context into **one crisp note per chunk**.

**How it works (per chunk):**

1. **Generate queries** — `_generate_queries(llm, chunk_id, chunk_text)`:
   - LLM reads the chunk and produces a `DependencyQueries` model — a list of `Dependency` objects, each with a `query` string and `reference_text` (the snippet that triggered the query)
   - Example: chunk mentions "as per the CBRD team's escalation process" → query = "CBRD team escalation process steps"

2. **Retrieve + grade (per query):**
   - FAISS similarity search retrieves top-2 chunks for each query (excluding the chunk itself)
   - `_grade_retrievals(llm, query, docs)` sends each retrieval to the LLM for relevance grading → returns `List[bool]`
   - Accepted (relevant) retrievals are collected along with their query text

3. **Condense (per chunk, NOT per query)** — `_condense_context(llm, chunk_text, retrieved_sections)`:
   - All accepted retrievals across ALL queries for this chunk are combined into one input
   - One LLM call produces a **single unified context note** (2-4 sentences)
   - The prompt tells the LLM to: be brutally concise, drop irrelevant snippets, not repeat what's already in the chunk, merge overlapping facts, and focus on cross-chunk dependencies/triggers/handoffs

4. **Output:** An `EnrichedChunk` with the original text + the condensed context note + the list of queries generated

**Why condense at chunk level (not query level):** Per-query condensation produced N separate notes per chunk that often overlapped. A single chunk-level call deduplicates naturally, produces a more coherent note, and costs fewer LLM calls.

**Prompts:**
- `_ENRICHMENT_QUERY_SYSTEM` / `_ENRICHMENT_QUERY_HUMAN` — query generation
- `_ENRICHMENT_GRADE_SYSTEM` / `_ENRICHMENT_GRADE_HUMAN` — retrieval relevance grading
- `_CONDENSATION_SYSTEM` / `_CONDENSATION_HUMAN` — chunk-level context condensation

## 1.4 Entity Resolution — `resolve_entities(state)`

**What it does:** Groups synonymous terms across the entire SOP (e.g., "CBRD team" / "Credit Bureau Reporting Disputes team") and normalizes all chunk text to use canonical forms.

**How it works:**
1. Concatenates all chunks into one text block
2. LLM returns an `EntityMap` — a list of `EntityMapping` objects, each mapping a `canonical` name to its `aliases`
3. `_apply_entity_map(text, mappings)` does string replacement across all enriched chunks — every alias occurrence is replaced with the canonical form

**Why this matters:** Without entity resolution, the converter creates duplicate nodes for the same concept under different names. "Notify CBRD" and "Notify Credit Bureau Reporting Disputes team" would become two separate instruction nodes.

## Preprocessing Cache

### `_content_hash(text) -> str`
SHA-256 hash of the SOP text. Used as the cache key.

### `save_to_cache(state, source_text, cache_dir)`
Serializes the preprocessing result to `{cache_dir}/{hash}.json`. The FAISS vector store is NOT serialized (it's not JSON-serializable) — only chunks, enriched chunks, and entity map are saved.

### `load_from_cache(source_text, cache_dir) -> Optional[dict]`
Loads the cached result if it exists. Returns `None` on cache miss.

### `rebuild_vector_store(state)`
On cache hit, rebuilds the FAISS index from the cached chunks. Called by `cached_preprocessing()` when loading from cache.

### `cached_preprocessing(source_text, cache_dir, force, rebuild_faiss) -> RAGPrepState`
The main entry point for scripts. Checks the cache first, runs the full pipeline on miss, and saves to cache. The `force` flag bypasses the cache (useful after changing preprocessing prompts).

---

# Stage 2: Converter (`converter.py`)

## LLM Call Helpers

### `_llm_call(stage, system, human, **format_kwargs) -> str`

Plain-text LLM call. Sends a system + human message pair via `safe_invoke()` and returns the raw text response. Uses `get_model(stage)` to get a `ChatOpenAI` instance with the appropriate temperature for the given stage.

### `_structured_llm_call(stage, schema, system, human, **format_kwargs)`

The workhorse LLM call for the v4 pipeline. Returns a **Pydantic model instance** parsed from the LLM's structured output.

- Uses `llm.with_structured_output(schema)` to constrain the LLM to produce valid JSON matching the Pydantic model
- All calls go through `safe_invoke()` — if the API returns a non-200 response (429 rate limit, 5xx server error, 401/403 auth error), the pipeline **halts immediately** with an `LLMStopError` instead of silently retrying. The checkpoint system handles recovery.

### `safe_invoke(llm_or_structured, messages, context)` (in `models.py`)

Centralized wrapper around every `.invoke()` call in the system. Catches HTTP errors from the LLM API and raises `LLMStopError` for:
- **429** — rate limit (the error you see in logs as "Token limit reached")
- **5xx** — server errors
- **401/403** — authentication failures

Non-HTTP errors (parsing failures, network timeouts) are re-raised as-is.

The `context` parameter is a short label (e.g., `"structured/graph_gen"`) included in error logs so you know exactly which stage failed.

## Utility Functions

### `_to_snake_case(text) -> str`

Converts text to a snake_case node ID — strips non-alphanumeric characters, lowercases, takes first 4 words, joins with underscores. Falls back to `"node"` if empty.

### `_reassemble_enriched_sop(source_text, enriched_chunks) -> str`

Reassembles enriched chunks into one continuous document for Step 1. For each chunk, includes the chunk text and appends any condensed cross-reference context in `[Cross-reference context: ...]` brackets. If no enriched chunks exist, returns raw `source_text`.

### `_nodes_list_to_dict(nodes: List[WorkflowNode]) -> Dict[str, Dict[str, Any]]`

Converts the `List[WorkflowNode]` from the LLM's `InitialGraph` output into the standard `{node_id: node_data_dict}` format. Calls `model_dump()` on each Pydantic model.

### `_ensure_start_node(nodes) -> None`

Ensures the first node has `id="start"`. If it doesn't:

1. Pops the first node from the dict
2. Renames its `id` to `"start"`
3. Rebuilds the dict with `"start"` first (preserves insertion order)
4. Scans ALL nodes and updates every reference to the old ID — `next` fields and `options` values

This is necessary because the LLM might name the first node `"begin_processing"` despite being told to use `"start"`.

## PipelineConverter Class

### `converter_id = "pipeline_v4"`

Identifies this converter version in logs and stored graph metadata.

### `convert(source_text, enriched_chunks, dump_dir, resume) -> Dict[str, Any]`

The main entry point. Runs the full 2-step pipeline.

#### Setup

- Creates the dump directory if `dump_dir` is provided
- Calls `_reassemble_enriched_sop()` to combine chunks into one document
- Dumps the enriched SOP text to disk
- Instantiates a `SchemaValidator` (from `graph_ops.py`)

#### Step 1/2 — Full SOP → Graph

One LLM call generates the entire initial graph.

1. **Cache check:** if `resume=True` and `initial_graph.json` exists in `dump_dir`, loads from cache and skips the LLM call
2. **LLM call:** `_structured_llm_call(stage="graph_gen", schema=InitialGraph)` — the LLM produces an `InitialGraph` with:
   - `reasoning` — detailed analysis of decision points, branches, convergence, cross-section dependencies (acts as chain-of-thought)
   - `nodes` — the complete graph as a list of `WorkflowNode` objects
3. **Post-processing:**
   - `_nodes_list_to_dict()` converts list → dict
   - `_ensure_start_node()` renames first node if needed
   - `SchemaValidator.validate_and_fix()` applies deterministic fixes
   - `get_graph_issues()` checks for orphans, broken links, dead ends
4. **Dump:** saves `initial_graph.json`

**Temperature:** 0.2 (`graph_gen` stage) — slightly creative to produce diverse node structures, but not so high that it hallucinates edges.

#### Step 2/2 — Chunk-by-Chunk Graph Refinement

Iterates over enriched chunks to add missing details. Runs **2 passes** — the second pass sees nodes added by the first from other chunks.

**Skip condition:** if ≤1 enriched chunks, this step is skipped.

**Resume logic (per pass):**
1. Check for a per-pass checkpoint (`graph_after_pass_{N}.json`) — if found, load and skip the entire pass
2. Check for per-chunk checkpoints (`graph_p{pass}_c{chunk}.json`) — find the latest one within this pass and resume from the next chunk

**Per chunk (within a pass):**

1. **Chunk preparation:** assembles chunk text + condensed cross-reference context
2. **Snapshot:** deep-copies current nodes dict for rollback
3. **LLM call:** `_structured_llm_call(stage="graph_refine", schema=GraphPatch)` receives:
   - `adjacency_map` — simplified text view of connections (from `generate_adjacency_map()`)
   - `nodes_json` — full node details as JSON
   - `chunk_text` — the SOP section to verify
4. **No-op check:** if all operation lists are empty, skip to next chunk
5. **Reasoning dump:** saves `patch_p{pass}_c{chunk}_reasoning.txt` for debugging
6. **Apply patch:** `apply_patch(nodes, patch)` executes add → modify → remove
7. **Schema validation:** `SchemaValidator.validate_and_fix()` cleans up issues
8. **Topological validation:** `get_graph_issues()` checks integrity
9. **Rollback checks** — two conditions trigger rollback:
   - **Size sanity:** graph shrank to <70% of pre-patch size
   - **Start node protection:** `"start"` node was removed
10. **Error handling:** any exception → rollback to snapshot, continue
11. **Checkpoint:** dumps `graph_p{pass}_c{chunk}.json` after each chunk

**Early termination:** if pass 2 makes zero changes, the graph is stable — no further passes.

**Temperature:** 0.1 (`graph_refine` stage) — more conservative than graph gen since patches should be precise.

After all passes, dumps `final_graph.json` and logs type distribution.

### `_load_cache(dump_dir, name) -> Optional[str]`

Checks if a cached stage output exists — tries both `.txt` and `.json` extensions. Used when `resume=True`.

### `_dump_stage(dump_dir, name, content) -> None`

Writes stage output to disk. Auto-picks extension: `.json` if content starts with `{` or `[`, otherwise `.txt`.

## Prompts

### `_GRAPH_SYSTEM` — Step 1 System Prompt

Instructs the LLM (as "Process Logic Engineer") to produce a list of `WorkflowNode` JSON objects. Contains:

- **Exact node schema** with all 9 fields and their constraints
- **Node type rules** for each type (instruction, question, terminal, reference)
- **Structure rules:** first node `id="start"`, must have terminal `"end"` node, all referenced IDs must exist, convergence semantics
- **Content rules:** capture MAXIMUM detail (every click, check, data entry), include role/system metadata, confidence levels
- **Two few-shot examples:** a decision flow and a reference node

### `_GRAPH_HUMAN` — Step 1 Human Prompt

Simple template: "Convert this SOP into a workflow graph" + `{enriched_sop}`.

### `_PATCH_SYSTEM` — Step 2 System Prompt

Instructs the LLM (as "Graph Refinement Engineer") to produce a `GraphPatch`. Contains:

- **What the LLM receives:** adjacency map, full nodes JSON, one SOP chunk
- **Patch operation rules:** how to add (wire into graph), modify (include ALL fields), remove (update references first)
- **Insertion patterns:** inserting between A→B, inserting a decision that splits a chain
- **No-op rule:** if chunk is already captured, return empty lists
- **Node schema reminder**

### `_PATCH_HUMAN` — Step 2 Human Prompt

Template with `{adjacency_map}`, `{nodes_json}`, `{chunk_text}`.

---

# Stage 3: Refinement Loop (`graph_ops.py`)

A LangGraph StateGraph that cycles between analysis and repair until all quality checks pass or max iterations (10) are reached.

## Topological Checks (Pure Python, Free)

### `get_graph_issues(nodes) -> str`

Scans the graph for structural problems. Returns `"Topology Valid."` or a report listing:
- **Orphan nodes** — no incoming edges (except `start`)
- **Dead-end nodes** — non-terminal nodes with no outgoing edges
- **Broken links** — `next` or `options` pointing to non-existent node IDs

No LLM call — pure dict traversal.

### `generate_adjacency_map(nodes) -> str`

Generates a human-readable text representation of the graph's connections. Used as context for LLM calls throughout the system. Example output:
```
start -> check_fraud_score
check_fraud_score -> [Yes: escalate_to_supervisor, No: close_case]
```

### `get_all_issues_structured(nodes) -> dict`

Returns structured issue data (lists of orphans, dead-ends, broken links) rather than a text report. Used programmatically.

## LLM-Based Quality Checks

### `check_completeness(nodes, source_text) -> RefineFeedback`

**Prompt:** `_COMPLETENESS_SYSTEM` / `_COMPLETENESS_HUMAN`

Compares the graph against the original SOP text. Returns a `RefineFeedback` model:
- `is_complete: bool` — whether the graph captures all SOP content
- `missing_branches: List[str]` — specific "If X then Y" rules or steps missing

The LLM receives the adjacency map, full nodes JSON, and original SOP.

**Temperature:** 0.0 — deterministic analysis, no creativity needed.

### `check_context(nodes, source_text) -> Dict`

**Prompt:** `_CONTEXT_SYSTEM` / `_CONTEXT_HUMAN`

Checks whether connected nodes are logically adjacent — does it make sense for node A to flow into node B? Returns `{"is_valid": bool, "issues": List[str]}`.

Catches issues like: an "Approve claim" instruction flowing directly into "Deny claim" with no decision point between them.

**Temperature:** 0.0.

### `check_granularity(nodes, source_text) -> GranularityFeedback`

**Prompt:** `_GRANULARITY_SYSTEM` / `_GRANULARITY_HUMAN`

Flags nodes that collapse multiple distinct actions into one. Returns:
- `is_granular: bool`
- `coarse_nodes: List[CoarseNode]` — each with `node_id`, `reason`, and `suggested_split` (how many sub-steps)

Example: a node "Process the dispute, update the system, and notify the customer" should be 3 separate nodes.

**Temperature:** 0.0.

## Analysis Orchestrator — `analyse(state) -> GraphState`

Runs all 4 checks in sequence and aggregates results.

1. **Topological check** — `get_graph_issues()` (free, always first)
2. **Completeness check** — `check_completeness()` (LLM)
3. **Context check** — `check_context()` (LLM)
4. **Granularity check** — `check_granularity()` (LLM)

**Output:**
- `state["is_complete"]` — `True` only if ALL 4 checks pass
- `state["feedback"]` — flat string of all issues (for logging)
- `state["categorized_feedback"]` — `Dict[str, str]` mapping each category name to its feedback text. Only categories with issues are included. Example:
  ```python
  {
      "topological": "Fix these topological issues: ...",
      "completeness": "Add the following missing branches: ...",
      "granularity": "Expand these coarse nodes: 'process_dispute' (..., split into ~3 steps)"
  }
  ```
- `state["analysis_report"]` — detailed report for logging/debugging

The categorized feedback is what drives **per-category patching** in the refiner.

## Triplet Verification — `TripletVerifier`

### `verify(nodes, source_text) -> List[dict]`

Verifies that each edge in the graph is supported by the SOP text.

1. **Build triplets** — `_build_triplets(nodes)` extracts every edge as `(source_id, edge_label, target_id)`. For question nodes, the edge label is the option key (e.g., "Yes", "No"). For instruction nodes, it's "next".
2. **Sort by confidence** — low-confidence edges are checked first (most likely to be wrong)
3. **Batch verification** — sends triplets to the LLM in batches. The LLM returns a `_TripletVerification` model with `is_valid` and `explanation` for each triplet.

**Prompt:** `_TRIPLET_SYSTEM` / `_TRIPLET_HUMAN`

**Temperature:** 0.0 — deterministic true/false judgment.

Returns a list of invalid triplet dicts, each with `source_id`, `edge_label`, `target_id`, and `explanation`.

## Patch Application — `apply_patch(nodes, patch)`

Shared by both `converter.py` (Step 2) and `graph_ops.py` (refinement). Applies a `GraphPatch` in this order:

1. **Add nodes** — insert new nodes. Skip if ID already exists (prevents overwrites).
2. **Modify nodes** — replace existing nodes by ID. The entire node dict is replaced. Skip if ID not found.
3. **Remove nodes** — delete by ID. Before deletion, logs warnings for any remaining nodes whose `next` or `options` still reference the removed ID.

Returns the mutated nodes dict.

## Graph Patch Resolver — `GraphPatchResolver`

### `resolve(nodes, feedback, source_text) -> Tuple[Dict, GraphPatch]`

The LLM-based repair tool. Sees the **full graph** + **specific feedback** + **source SOP** and produces a `GraphPatch`.

**How it works:**
1. Generates adjacency map and nodes JSON
2. Sends to LLM via `safe_invoke()` with `_PATCH_RESOLVER_SYSTEM` / `_PATCH_RESOLVER_HUMAN` prompts
3. If no changes needed, returns immediately
4. Takes a **snapshot** before applying the patch
5. Applies the patch via `apply_patch()`
6. **Rollback safety:**
   - If graph shrank to <70% of pre-patch size → rollback
   - If `"start"` node was removed → rollback

**Prompt:** `_PATCH_RESOLVER_SYSTEM` instructs the LLM to fix ONLY the specific issues described in the feedback. Includes the same node schema rules and patch operation patterns as the converter's `_PATCH_SYSTEM`.

**Temperature:** 0.1 (`resolver` stage).

## Schema Validator — `SchemaValidator`

### `validate_and_fix(nodes) -> Tuple[Dict, List[str]]`

Deterministic (no LLM) fixes applied after every patch:

- Terminal nodes: clear `next` and `options` to `null`
- Question nodes without `options`: downgrade to `instruction` type
- Instruction nodes without `next`: flag for manual resolution
- Question nodes with `next`: clear `next` (questions use `options` for routing)

Attempts to construct a `WorkflowNode` Pydantic model for each node — if validation fails, logs the error and keeps the raw data.

Returns `(fixed_nodes_dict, list_of_fix_descriptions)`.

## Refinement Orchestrator — `refine(state) -> GraphState`

Runs per-category patching — each issue category gets its own `GraphPatchResolver` call with a narrow mandate.

**Step 1: Triplet verification**
- `TripletVerifier.verify()` finds invalid edges
- If any are found, adds a `"triplets"` category to `categorized_feedback`

**Step 2: Per-category patching**

Iterates over categories in order: `topological → completeness → context → granularity → triplets`

For each category with feedback:
1. Calls `GraphPatchResolver.resolve(nodes, category_feedback, source_text)` with ONLY that category's feedback
2. The resolver sees the full graph but has a narrow mandate — "fix these topological issues" or "expand these coarse nodes"
3. Each category has independent rollback via the resolver's built-in safety checks

**Why per-category (not one big patch):** A single "fix everything" call risks collateral damage — the LLM might restructure the graph while fixing a granularity issue. Per-category calls give each fix a narrow scope. If one category's patch is catastrophic, it rolls back independently without losing fixes from other categories.

**Step 3: Schema validation**
- `SchemaValidator.validate_and_fix()` cleans up any issues introduced by patches

Updates `state["nodes"]`, increments `state["iteration"]`.

## LangGraph Loop — `_build_graph()` and `run_refinement()`

### `_build_graph(store, max_iterations, dump_dir) -> StateGraph`

Builds the LangGraph StateGraph:

```
START → analyse → should_continue → refine → analyse → ...
                                  ↘ END
```

- `analyse_node` — calls `analyse()`, dumps iteration checkpoint
- `refine_node` — calls `refine()`, saves to GraphStore if provided
- `should_continue` — routes to `"refine"` if `is_complete=False` and `iteration < max_iterations`, otherwise `"end"`

### `run_refinement(graph_state, max_iterations, store, dump_dir, resume) -> GraphState`

Entry point for the refinement loop.

1. Sets up dump directory, optionally loads latest iteration checkpoint on resume
2. Compiles the StateGraph
3. Invokes with the graph state
4. Returns the final state

**Resume logic:** scans `dump_dir` for `refine_iter{N}_graph.json` files, finds the highest N, loads that graph state, and sets `iteration=N` so the loop continues from where it left off.

**Checkpoints per iteration:** after each analyse step, dumps `refine_iter{N}_graph.json` and `refine_iter{N}_report.txt`.

---

# Shared Infrastructure

## `models.py` — LLM Configuration

### `get_model(stage) -> ChatOpenAI`

Returns a ChatOpenAI instance with temperature set by stage:

| Stage | Temperature | Purpose |
|-------|-------------|---------|
| `graph_gen` | 0.2 | Initial graph generation (slightly creative) |
| `graph_refine` | 0.1 | Chunk patches (precise) |
| `completeness`, `context` | 0.0 | Analyser checks (deterministic) |
| `triplet` | 0.0 | Edge verification (deterministic) |
| `resolver` | 0.1 | Patch resolution (precise) |
| `chunking` | 0.1 | SOP chunking |
| `enrichment` | 0.0 | RAG query/grading |
| `entity_resolution` | 0.0 | Synonym grouping |

### `safe_invoke(llm_or_structured, messages, context) -> response`

Centralized error handling for ALL LLM calls across the system. On non-200 HTTP response:
- Logs the error with the stage context label
- Raises `LLMStopError` — halts the pipeline
- The checkpoint system handles recovery (re-run with `--resume`)

### `LLMStopError`

Custom exception for non-recoverable API errors. Carries the HTTP status code. The pipeline should NOT retry — it should stop and let the user resume once the issue (rate limit, quota, outage) is resolved.

## `schemas.py` — Pydantic Models

### `WorkflowNode`

The core node model. Validators enforce:
- Question nodes must have `options`
- Instruction nodes must have `next`
- Terminal nodes get `next` and `options` cleared

### `InitialGraph`

Step 1 output: `reasoning` (chain-of-thought) + `nodes` (list of `WorkflowNode`).

### `GraphPatch`

Step 2 / refinement output: `reasoning` + `add_nodes` + `modify_nodes` + `remove_nodes`.

### `GraphState`

LangGraph TypedDict flowing through the refinement loop. Key fields:
- `nodes` — the current graph
- `feedback` — flat feedback string
- `categorized_feedback` — `Dict[str, str]` mapping category → feedback text
- `iteration` — current loop iteration
- `is_complete` — whether all checks passed

## `storage.py` — GraphStore

File-based JSON persistence for graphs. Wraps graphs in a metadata envelope (timestamp, converter_id, status) and provides save/load/list operations.

---

# Checkpoint and Resume Summary

| Stage | What's checkpointed | File pattern | Resume behavior |
|-------|---------------------|--------------|-----------------|
| Preprocessing | Full result (chunks, enriched chunks, entity map) | `output/preprocessing_cache/{sha256}.json` | Automatic — cache hit skips all preprocessing |
| Converter Step 1 | Initial graph | `initial_graph.json` | Loads from cache, skips LLM call |
| Converter Step 2 | Graph after each chunk | `graph_p{pass}_c{chunk}.json` | Finds latest per-chunk checkpoint, resumes from next chunk |
| Converter Step 2 | Graph after each pass | `graph_after_pass_{N}.json` | Loads pass checkpoint, skips entire pass |
| Refinement loop | Graph + report per iteration | `refine_iter{N}_graph.json` | Loads latest iteration, continues loop |

---

# Output Format

The final output is a Python dict mapping node IDs to node data:

```json
{
  "start": {
    "id": "start",
    "type": "instruction",
    "text": "Receive dispute notification",
    "next": "is_code_183_question",
    "options": null,
    "external_ref": null,
    "role": "Analyst",
    "system": "Case Management System",
    "confidence": "high"
  },
  "is_code_183_question": {
    "id": "is_code_183_question",
    "type": "question",
    "text": "Is dispute code 183 or 186?",
    "next": null,
    "options": {"Yes": "route_to_fraud", "No": "route_to_standard"},
    "external_ref": null,
    "role": "Analyst",
    "system": "Mainframe",
    "confidence": "high"
  }
}
```

### Node Types

| Type | `next` | `options` | Description |
|------|--------|-----------|-------------|
| `instruction` | Required (node_id) | null | A step to execute |
| `question` | null | Required (`{"Yes": id, "No": id}`) | A decision point |
| `terminal` | null | null | End of a process path |
| `reference` | Required (node_id) | null | Links to external document via `external_ref` |
