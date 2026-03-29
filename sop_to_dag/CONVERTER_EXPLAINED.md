# How the Converter Works

This document explains the full SOP-to-JSON-DAG conversion pipeline — from raw text to a fully-connected workflow graph.

## The Big Picture

```
Raw SOP Text
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  PREPROCESSING (preprocessing.py)                       │
│  LangGraph pipeline — runs before conversion            │
│                                                         │
│  1. Agentic Chunking    → semantic chunks               │
│  2. FAISS Indexing       → vector store                  │
│  3. RAG Enrichment       → chunks + cross-ref context    │
│  4. Entity Resolution    → canonical term mapping        │
└──────────────────────────┬──────────────────────────────┘
                           │
              enriched_chunks + entity_map
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  CONVERTER (converter.py) — Graph-First Pipeline (v4)   │
│                                                         │
│  Step 1  [LLM]  Full enriched SOP → graph JSON directly │
│  Step 2  [LLM]  Chunk-by-chunk graph patching (2 passes)│
│                                                         │
│  Both steps use structured output (Pydantic models).    │
│  No lossy text-outline intermediate.                    │
└──────────────────────────┬──────────────────────────────┘
                           │
                      JSON DAG (nodes dict)
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  REFINEMENT LOOP (graph_ops.py)                         │
│  LangGraph StateGraph: Analyser ↔ Refiner               │
│  Cycles until all checks pass or max iterations          │
└─────────────────────────────────────────────────────────┘
```

---

## Why Graph-First (v4 vs v3)

The previous v3 converter used a 3-step pipeline: LLM generates **text outline** → LLM refines **text** chunk-by-chunk → deterministic parser converts text to graph. The text outline was a lossy intermediate — it captured topology via indentation but lost longer temporal dependencies (cross-chunk decision scope, branch history, nested convergence semantics).

v4 eliminates the text intermediate entirely: the LLM produces the **graph JSON directly** in Step 1, then each chunk refines the **graph** (not text) via structured patches in Step 2. This preserves richer structural information from the start and allows refinement to reason about graph topology directly.

```
OLD (v3): SOP → [LLM] → text outline → [LLM] → refined text → [parser] → graph
NEW (v4): SOP → [LLM] → graph JSON   → [LLM per chunk] → patched graph
```

---

## Prompts

### `_GRAPH_SYSTEM` — Step 1 System Prompt

Instructs the LLM (acting as a "Process Logic Engineer") to convert an SOP directly into a list of `WorkflowNode` JSON objects. Contains:

- **Exact node schema** with all fields (`id`, `type`, `text`, `next`, `options`, `external_ref`, `role`, `system`, `confidence`)
- **Node type rules** for each of the 4 types (`instruction`, `question`, `terminal`, `reference`) — what fields are required, what must be null
- **Structure rules** — first node must be `id="start"`, must have a terminal `"end"` node, all referenced IDs must exist, branch convergence semantics
- **Content rules** — capture MAXIMUM detail, include role/system metadata, use confidence levels (`high`=explicit, `medium`=inferred, `low`=guess)
- **Two few-shot examples** — a simple decision flow and a reference node

### `_GRAPH_HUMAN` — Step 1 Human Prompt

Simple template: tells the LLM to convert the SOP and passes the full enriched SOP text via `{enriched_sop}`.

### `_PATCH_SYSTEM` — Step 2 System Prompt

Instructs the LLM (acting as a "Graph Refinement Engineer") to compare one SOP chunk against the existing graph and produce a `GraphPatch`. Contains:

- **What the LLM receives** — adjacency map, full nodes JSON, one SOP chunk
- **Patch operation rules** — how to add nodes (wire into graph), modify nodes (include ALL fields), remove nodes (update references first)
- **Insertion patterns** — how to insert a node between A→B, how to insert a decision that splits a chain
- **No-op rule** — if the chunk is already captured, return empty lists
- **Node schema reminder** — so the LLM doesn't produce malformed nodes

### `_PATCH_HUMAN` — Step 2 Human Prompt

Template providing: `{adjacency_map}` (simplified text view of connections), `{nodes_json}` (full node details), `{chunk_text}` (the SOP section to verify).

---

## Helper Functions

### `_llm_call(stage, system, human, **format_kwargs) -> str`

Plain-text LLM call — sends a system + human message pair and returns the raw text response. Uses `get_model(stage)` to get a `ChatOpenAI` instance with the appropriate temperature for the given stage. Not currently used in the v4 pipeline (retained for potential future use) but follows the same call pattern as the structured variant.

### `_structured_llm_call(stage, schema, system, human, retries=2, **format_kwargs)`

The workhorse LLM call for the v4 pipeline. Sends a system + human message pair and returns a **Pydantic model instance** parsed from the LLM's structured output.

- Uses `llm.with_structured_output(schema)` to constrain the LLM to produce valid JSON matching the Pydantic model
- **Retry logic**: if structured output parsing fails (e.g., malformed JSON from the LLM), retries up to `retries` times before raising
- Used with `InitialGraph` in Step 1 and `GraphPatch` in Step 2

### `_to_snake_case(text) -> str`

Converts a text description to a snake_case node ID. Strips non-alphanumeric characters, lowercases, takes the first 4 words, joins with underscores. Falls back to `"node"` if the result is empty. Used for programmatic ID generation when needed.

### `_reassemble_enriched_sop(source_text, enriched_chunks) -> str`

Reassembles enriched chunks into one continuous document for Step 1. For each chunk, includes the chunk text and appends any RAG-retrieved cross-reference context in `[Cross-reference context: ...]` brackets. If no enriched chunks exist, returns the raw `source_text` as-is.

### `_nodes_list_to_dict(nodes: List[WorkflowNode]) -> Dict[str, Dict[str, Any]]`

Converts the `List[WorkflowNode]` returned by the LLM's `InitialGraph` into the standard nodes dict format (`{node_id: node_data_dict}`). Calls `model_dump()` on each Pydantic model and keys by the node's `id` field.

### `_ensure_start_node(nodes) -> None`

Ensures the first node in the graph has `id="start"`. If a `"start"` key already exists, does nothing. Otherwise:

1. Pops the first node from the dict
2. Renames its `id` to `"start"`
3. Rebuilds the dict with `"start"` first to preserve insertion order
4. Updates **all references** to the old ID across the entire graph — scans every node's `next` field and every value in `options` dicts, replacing the old ID with `"start"`

This is necessary because the LLM might name the first node something like `"begin_processing"` despite being told to use `"start"`.

---

## Patch Application

### `_apply_patch(nodes, patch: GraphPatch) -> Dict`

Applies a `GraphPatch` to the existing nodes dict. The patch contains three operation lists, applied in this order:

1. **Add nodes** (`patch.add_nodes`) — inserts new nodes into the graph. If a node ID already exists, the add is **skipped** with a warning (prevents accidental overwrites).

2. **Modify nodes** (`patch.modify_nodes`) — replaces existing nodes by ID. The entire node dict is replaced (not merged), so the LLM must include ALL fields. If the ID doesn't exist in the graph, the modify is **skipped** with a warning.

3. **Remove nodes** (`patch.remove_nodes`) — deletes nodes by ID. Before deletion, checks for **dangling references** — if any remaining node's `next` or `options` still points to the removed ID, a warning is logged. The removal still proceeds (the modify_nodes list should have updated those references).

Returns the mutated nodes dict.

---

## PipelineConverter Class

### Overview

The main converter class. Exposes a single `convert()` method that takes raw SOP text + optional enriched chunks and returns a fully-connected JSON DAG.

- **`converter_id = "pipeline_v4"`** — identifies this converter version in logs and stored graph metadata.

### `convert(source_text, enriched_chunks, dump_dir, resume) -> Dict[str, Any]`

The main entry point. Runs the full 2-step pipeline:

#### Setup

- Creates the dump directory if `dump_dir` is provided (for debugging/caching)
- Calls `_reassemble_enriched_sop()` to combine chunks into one document
- Dumps the enriched SOP to disk
- Instantiates a `SchemaValidator` (from `graph_ops.py`) for deterministic fixes

#### Step 1/2 — Full SOP → Graph

Generates the initial graph from the full enriched SOP in one LLM call.

1. **Cache check**: if `resume=True` and an `initial_graph` file exists in `dump_dir`, loads from cache and skips the LLM call
2. **LLM call**: `_structured_llm_call(stage="graph_gen", schema=InitialGraph, ...)` — the LLM produces an `InitialGraph` with a `reasoning` field (detailed analysis) and a `nodes` list
3. **Post-processing**:
   - `_nodes_list_to_dict()` converts the list to the standard dict format
   - `_ensure_start_node()` renames the first node to `"start"` if needed
   - `SchemaValidator.validate_and_fix()` applies deterministic fixes (e.g., terminal nodes get `next` cleared, questions without options become instructions)
   - `get_graph_issues()` checks for orphans, broken links, and dead ends — logs warnings if found
4. **Dump**: saves `initial_graph.json` to disk

#### Step 2/2 — Chunk-by-Chunk Graph Refinement (Multi-Pass)

Iterates over enriched chunks to add missing granular details to the graph. Runs **2 passes** over all chunks — the second pass sees nodes added by the first, catching cross-chunk temporal dependencies.

**Skip condition**: if there are 0–1 enriched chunks, this step is skipped entirely.

For each pass, for each chunk:

1. **Chunk preparation**: assembles chunk text + any RAG cross-reference context
2. **Snapshot**: deep-copies the current nodes dict for rollback safety
3. **LLM call**: `_structured_llm_call(stage="graph_refine", schema=GraphPatch, ...)` — receives the adjacency map, full nodes JSON, and chunk text. Returns a `GraphPatch` with add/modify/remove operations
4. **No-op check**: if all three operation lists are empty, skips to next chunk
5. **Patch reasoning dump**: if `dump_dir` is set, saves the LLM's reasoning to `patch_p{pass}_c{chunk}_reasoning.txt` for debugging
6. **Apply patch**: calls `_apply_patch()` to mutate the graph
7. **Schema validation**: `SchemaValidator.validate_and_fix()` cleans up any schema issues introduced by the patch
8. **Topological validation**: `get_graph_issues()` checks graph integrity after the patch
9. **Rollback checks** — two safety conditions trigger a full rollback to the pre-patch snapshot:
   - **Size sanity**: if the graph shrank to less than 70% of its pre-patch size (prevents a bad patch from wiping the graph)
   - **Start node protection**: if the `"start"` node was removed
10. **Error handling**: if the LLM call or patch application throws, rolls back to the pre-patch snapshot and continues

**Early termination**: if pass 2 makes zero changes across all chunks, the graph is considered stable and further passes are skipped.

After all passes, dumps `final_graph.json` and logs a type distribution summary (how many instruction/question/terminal/reference nodes).

### `_load_cache(dump_dir, name) -> Optional[str]`

Static method. Checks if a cached stage output exists in `dump_dir` — tries both `.txt` and `.json` extensions. Returns the file content if found, `None` otherwise. Used when `resume=True` to skip expensive LLM calls for stages that already completed.

### `_dump_stage(dump_dir, name, content) -> None`

Static method. Writes a stage output to disk. Automatically picks the file extension — `.json` if the content starts with `{` or `[`, otherwise `.txt`. Used for debugging (inspect intermediate outputs) and caching (resume from a failed run).

---

## Output Format

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
    "confidence": "high",
    "role": "Analyst",
    "system": "Case Management System"
  },
  "is_code_183_question": {
    "id": "is_code_183_question",
    "type": "question",
    "text": "Is dispute code 183 or 186?",
    "next": null,
    "options": {"Yes": "route_to_fraud", "No": "route_to_standard"},
    "external_ref": null,
    "confidence": "high",
    "role": "Analyst",
    "system": "Mainframe"
  }
}
```

### Node Types

| Type | `next` | `options` | `role` | `system` | Description |
|---|---|---|---|---|---|
| `instruction` | Required (node_id) | null | Optional | Optional | A step to execute |
| `question` | null | Required (`{"Yes": id, "No": id}`) | Optional | Optional | A decision point |
| `terminal` | null | null | null | null | End state |
| `reference` | Required (node_id) | null | Optional | Optional | Like instruction, but references external doc |

---

## Pydantic Models (from schemas.py)

### `InitialGraph`

The structured output schema for Step 1. Fields:
- **`reasoning: str`** — detailed analysis of the SOP structure, decision points, branches, convergence points, and cross-section dependencies. The LLM writes this before producing nodes, acting as a chain-of-thought.
- **`nodes: List[WorkflowNode]`** — the complete workflow graph as a list of nodes.

### `GraphPatch`

The structured output schema for Step 2. Fields:
- **`reasoning: str`** — what this chunk adds/changes and why.
- **`add_nodes: List[WorkflowNode]`** — new nodes to insert (default: empty list).
- **`modify_nodes: List[WorkflowNode]`** — existing nodes to replace by ID (default: empty list).
- **`remove_nodes: List[str]`** — IDs of nodes to delete (default: empty list).

### `WorkflowNode`

The core node model with validation:
- Terminal nodes get `next` and `options` force-cleared to `null`
- Question nodes must have `options` (validation error otherwise)
- Instruction nodes must have `next` (validation error otherwise)
- Question text should end with `?`

---

## What Makes This Design Work

1. **Graph from the start.** The LLM produces structured graph JSON directly — no lossy text-outline intermediate that loses temporal dependencies, decision scope, or branch history.

2. **Structured output via Pydantic.** `with_structured_output()` constrains the LLM to produce valid JSON matching the schema. Retry logic handles the occasional parsing failure.

3. **Patch-based refinement preserves existing work.** Each chunk produces a surgical `GraphPatch` (add/modify/remove) rather than regenerating the whole graph. This prevents earlier correct work from being lost during refinement.

4. **Multi-pass catches cross-chunk dependencies.** Pass 2 sees nodes that Pass 1 added from other chunks. A decision in chunk 3 that references a branch target added from chunk 5 in Pass 1 can be wired correctly in Pass 2.

5. **Rollback safety prevents catastrophic patches.** Every patch is applied against a snapshot. If the graph shrinks by >30% or loses the start node, the entire patch is rolled back and the previous graph is preserved.

6. **Topological validation after every patch.** `get_graph_issues()` catches orphans, broken links, and dead ends immediately after each patch — issues surface early rather than accumulating.

7. **Confidence labels drive refinement priority.** Downstream in the refinement loop, the `TripletVerifier` checks low-confidence edges first — the ones most likely to be incorrect.

8. **Full graph context for patches.** The LLM sees the complete adjacency map and full nodes JSON for every patch, not just a local window. This enables coordinated multi-node changes (e.g., inserting a decision that splits a chain requires modifying the upstream node AND adding the new nodes).
