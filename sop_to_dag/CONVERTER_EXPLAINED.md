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
│  CONVERTER (converter.py) — Plain-Text Pipeline         │
│                                                         │
│  Step 1  [LLM]  Full enriched SOP → plain-text outline  │
│  Step 2  [LLM]  Chunk-by-chunk detail pass (gap-fill)   │
│  Step 3  [CODE] Direct text-to-graph compile → JSON DAG │
│                                                         │
│  LLM's job ends after Step 2. Step 3 is pure code.      │
└──────────────────────────┬──────────────────────────────┘
                           │
                      JSON DAG (nodes dict)
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  REFINEMENT LOOP (loop.py)                              │
│  LangGraph StateGraph: Analyser ↔ Refiner               │
│  Cycles until all checks pass or max iterations          │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Outline — LLM Generates a Plain-Text Numbered Outline

**Input:** Full enriched SOP (reassembled from enriched chunks with cross-references inlined)
**Output:** A plain-text numbered outline
**Method:** LLM with plain-text output (no structured output wrestling)

The LLM reads the full SOP and produces a numbered outline that captures the complete workflow logic:

```
1. Begin processing indirect dispute
2. DECISION: Is the dispute code 183 or 186?
  YES:
    3. Check if borrower mentions fraud indicators
  NO:
    4. Check force memo for fraud keywords
5. Update order status
```

**Format rules:**
- Sequential actions: numbered lines (`1. Do something`)
- Decision points: `DECISION:` prefix, phrased as YES/NO questions
- Branches: indented `YES:` / `NO:` blocks under decisions
- Convergence: un-indent back to show where branches rejoin
- External references: inline `Refer to: <document name>` in step text

**Why plain text instead of structured output:** The LLM produces better outlines when it can write freely in a simple numbered format. No schema wrestling, no field-level hallucination. The deterministic parser in Step 3 handles all the structuring.

**Prompt role:** `Process Logic Engineer` — focuses on capturing the COMPLETE workflow with self-explanatory step text, all decision paths covered, and cross-section references inlined.

---

## Step 2: Detail Pass — Chunk-by-Chunk Gap Filling

**Input per call:** Current outline + one enriched SOP chunk
**Output:** Updated outline with any missing details added
**Method:** LLM with plain-text output, called sequentially for each chunk

Each enriched chunk is compared against the current outline. The LLM checks if every action, decision, reference, and detail from that section is captured, and adds anything missing in the correct location.

**Rules:**
1. Do NOT remove or restructure existing steps — only ADD
2. If the section references external documents not in the outline — add them
3. Return the COMPLETE updated outline (all existing steps + additions)
4. If nothing is missing, return the outline unchanged

**Why this step exists:** The single-shot outline in Step 1 captures the overall structure well, but may miss specific codes, team names, threshold values, or cross-references that appear in individual SOP sections. The detail pass is a "diff and patch" that ensures no granular detail is lost.

**Skip condition:** If there are 0–1 enriched chunks (single chunk or no preprocessing), this step is skipped.

**LLM call count:** 1 (outline) + N (detail passes) = N+1 calls. For typical 3–6 chunk SOPs: 4–7 calls.

---

## Step 3: Direct Text-to-Graph Build — Pure Compilation

**Input:** Final refined plain-text outline
**Output:** `Dict[str, node_data]` — the JSON DAG
**Method:** Pure Python parser + graph builder in one pass. No LLM. Zero hallucination risk.

This is `parse_outline_to_graph()`. It reads the text outline and emits graph nodes directly using `_GraphBuilder` — no intermediate Pydantic model (like `PseudocodeBlock`) is constructed. The parser calls the builder's emit methods as it encounters each line.

### How Parsing Works

The parser (`_parse_and_emit`) walks lines recursively:

| Outline pattern | What happens |
|---|---|
| `3. Do something` | Calls `builder._emit_action("Do something")` |
| `4. DECISION: Is it valid?` | Parses YES/NO branches recursively, then calls `builder._emit_conditional(...)` |
| `YES:` / `NO:` | Branch markers — triggers recursive sub-parse at deeper indent |
| `5. Update status` (after un-indent) | Convergence — chained after the decision's branch tails |

### Mapping Rules

| Outline element | Graph node type | Description |
|---|---|---|
| Regular numbered step | `instruction` | A step to perform. Has `next` pointing to the following node. |
| Step with "Refer to..." | `reference` | Same as instruction but includes `external_ref` field. |
| Step with terminal keyword | `terminal` | End state. No outgoing edges. |
| `DECISION:` step | `question` | Decision node. Has `options: {"Yes": node_id, "No": node_id}`. |
| Empty branch / dangling end | `terminal` | Auto-generated end node. |

### The Head/Tails Convergence Pattern

This is the key algorithm that makes the graph fully connected. Every emit function returns:

```
(head_id, tail_ids)
```

- **head_id**: The entry point of the emitted subgraph (the first node)
- **tail_ids**: Nodes whose `next` is still `None` — they need to be wired to whatever comes after

#### How it works for each type:

**Action → instruction:**
```
Returns (node_id, [node_id])
         ↑ entry    ↑ this node's `next` is None, caller will wire it
```

**Action → terminal:**
```
Returns (terminal_id, [])
         ↑ entry        ↑ empty — nothing comes after a terminal
```

**Decision → question:**
```
                    ┌── Yes branch ──→ [steps...] → tail_A (next=None)
  question_node ──┤
                    └── No branch  ──→ [steps...] → tail_B (next=None)

Returns (question_id, [tail_A, tail_B])
         ↑ entry       ↑ BOTH branches' exits need wiring
```

#### Chaining a step list (`_chain_results`):

When processing `[step1, step2, step3]`:

1. Emit step1 → get `(head1, tails1)`
2. Emit step2 → get `(head2, tails2)`
3. Emit step3 → get `(head3, tails3)`
4. Wire: `tails1.next → head2`, `tails2.next → head3`
5. Return: `(head1, tails3)` — first node is entry, last step's tails are exits

This is what makes **branch convergence** work:

```
Steps: [DECISION: Is it valid?, "Continue processing"]

1. Emit Decision:
   question → Yes → handle_valid (tail)
            → No  → handle_invalid (tail)
   Returns: (question, [handle_valid, handle_invalid])

2. Emit "Continue processing":
   Returns: (continue, [continue])

3. Chain: handle_valid.next → continue
          handle_invalid.next → continue    ← CONVERGENCE!

4. Return: (question, [continue])
```

Both branches merge into the same downstream node. This works recursively for nested conditionals at any depth.

### Walk-through Example

Given this outline:
```
1. Receive dispute
2. DECISION: Is code 183?
  YES:
    3. Route to fraud
  NO:
    4. Route to standard
5. Send confirmation email
6. End processing
```

The parser+builder produces:

```
start ──→ is_code_183_question
(Receive       │
 dispute)  ┌─Yes─┴──No──┐
           ▼             ▼
     route_to_fraud  route_to_standard
           │             │
           └──────┬──────┘
                  ▼
        send_confirmation_email
                  │
                  ▼
              end ← TERMINAL (keyword detected)
```

### Node Detection

**Terminal detection** — if the step text (lowered) contains any of these keywords, it routes to the shared terminal:
- "end processing", "end of process", "end of procedure"
- "process complete", "no further action", "workflow complete"
- "close the case", "mark as complete", "mark as done"
- (and more — 16 keywords total)

**Reference detection** — regex match on `Refer to: <name>` or `Refer <name>`:
- Sets `node_type = "reference"` and populates `external_ref`

### Confidence Labels

| Confidence | Meaning | When assigned |
|---|---|---|
| `high` | Directly stated in SOP | All nodes from explicit outline steps |
| `low` | Inferred for connectivity | Auto-generated terminals from `_terminate_tails` or `_ensure_terminals` |

Low-confidence nodes are prioritized during the refinement loop — the `TripletVerifier` checks these edges first.

### Safety Nets

After the main parse, two safety passes run:

1. **`_terminate_tails(tails)`** — Any tail nodes still dangling at the top level get wired to a shared terminal node (marked `confidence: "low"`)

2. **`_ensure_terminals()`** — Final sweep: any `instruction` or `reference` node with `next=None` gets a terminal. This catches edge cases the parser might miss

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
    "confidence": "high"
  },
  "is_code_183_question": {
    "id": "is_code_183_question",
    "type": "question",
    "text": "Is dispute code 183 or 186?",
    "next": null,
    "options": {"Yes": "route_to_fraud", "No": "route_to_standard"},
    "external_ref": null,
    "confidence": "high"
  }
}
```

### Node Types

| Type | `next` | `options` | Description |
|---|---|---|---|
| `instruction` | Required (node_id) | null | A step to execute |
| `question` | null | Required (`{"Yes": id, "No": id}`) | A decision point |
| `terminal` | null | null | End state |
| `reference` | Required (node_id) | null | Like instruction, but references external doc |

---

## Backward Compatibility

The old path through `PseudocodeBlock` still works:
- `parse_outline(text)` returns a `PseudocodeBlock` (Pydantic model tree)
- `_GraphBuilder.build(pseudocode)` accepts a `PseudocodeBlock` and returns nodes

These are retained for tests and backward compatibility but are no longer used in the production pipeline. The production path is `parse_outline_to_graph(text)` which goes directly from text to graph nodes.

---

## What Makes This Design Work

1. **Plain-text outline as the LLM's only job.** The LLM writes a simple numbered list — no schema wrestling, no field-level hallucination. The format is natural for the model to produce and trivial for deterministic code to parse.

2. **Detail pass ensures granularity.** The chunk-by-chunk verification catches specific codes, team names, and threshold values the single-shot outline may have glossed over. The outline grows more detailed with each pass.

3. **Direct text-to-graph with no intermediate models.** `parse_outline_to_graph` reads text and emits graph nodes in one pass. No `PseudocodeBlock` construction means no ~50+ Pydantic model instances allocated and immediately discarded.

4. **Head/tails pattern prevents orphaned nodes.** Every emit function returns (head_id, tail_ids). Both branches' exit points bubble up to the parent, which wires them to the next step. Convergence is automatic at any nesting depth.

5. **Confidence labels drive refinement priority.** The refinement loop knows which edges to verify first — low-confidence auto-terminals are checked before high-confidence SOP-stated edges.

6. **LLM narrows ambiguity, code compiles structure.** Steps 1–2 use the LLM for what it's good at (understanding natural language, identifying decision logic). Step 3 uses deterministic code for what it's good at (building a correct, connected graph with no hallucination).
