# Graph Comparison Tool — Metrics & Calculations

## What This Tool Does

Compares an **auto-generated graph** against a **human-created graph** for the same SOP, using the **SOP document** as the ultimate source of truth. It answers:

1. How much of the human graph did our auto graph capture? (Recall)
2. How much of our auto graph is real vs hallucinated? (Precision)
3. Where is our auto graph *better* than the human graph? (Auto Advantages)
4. Where are we missing real content? (True Gaps)

---

## Pipeline Overview

```text
Auto Graph ──┐
             ├─→ Normalize ─→ Embed ─→ LLM Match ─→ Edge Compare ─→ Metrics
Human Graph ─┘                                        │         │
                                              deterministic  LLM (decisions)
                                                            ▼
SOP Document ───────────────────────────→ LLM Grounding ─→ Verdicts
                                        (unmatched nodes)
```

**Five steps:**

| Step | What | Method |
|------|------|--------|
| 0. Normalize | Convert both graphs to `{id, type, text, next, options}` | Deterministic mapping |
| 1. Align Nodes | Match auto nodes ↔ human nodes | Embeddings (top-10 candidates) + LLM (final decision) |
| 2. Compare Edges | Check if connections match between aligned pairs | Deterministic path search + LLM for decision branches |
| 3. SOP Grounding | For unmatched nodes, check if content is in the SOP | Embeddings (top-3 SOP chunks) + LLM (grounded/not) |
| 4. Compute Metrics | Aggregate everything into numbers | Formulas below |

---

## Step 0: Normalization

Both graphs are converted to a common format before any comparison.

**Human graph format:**
```
node_id: { type: activity|decision|start|terminal, action: "text", next: "id" | {"next": "id"} | {"yes": "id", "no": "id"} }
```

**Auto graph format (WorkflowNode):**
```
node_id: { type: instruction|question|terminal|reference, text: "...", next: "id", options: {"label": "id"} }
```

**Type mapping:** `activity → instruction`, `decision → question`, `start → instruction`

**Next field handling:**
- `next: "some_id"` → kept as `next`
- `next: {"next": "some_id"}` → unwrapped to `next: "some_id"` (human format quirk)
- `next: {"yes": "a", "no": "b"}` → becomes `options: {"yes": "a", "no": "b"}`

---

## Step 1: Hybrid Node Alignment

**Goal:** Match nodes between the two graphs by meaning, not by ID or exact text.

### Why hybrid (not just embeddings)?

Embedding similarity alone produces false matches — "Check fraud score" and "Check dispute code" might score 0.85+ because they share structure, but they're completely different steps. LLM matching understands the *meaning*.

### Process

**Phase 1 — Embedding candidate selection:**
1. Embed all node texts from both graphs using `BAAI/bge-base-en-v1.5` (local, no API cost)
2. Compute full cosine similarity matrix: `auto_nodes × human_nodes`
3. For each human node, pick the **top 10** auto nodes by similarity as candidates

**Phase 2 — LLM matching (batched):**
1. Send batches of 10 human nodes to the LLM, each with their 10 auto candidates
2. LLM decides which candidates genuinely match (by meaning, not wording)
3. LLM can match **multiple auto nodes to one human node** (many-to-one)

**Phase 3 — Build alignment maps:**
- `human_to_auto`: For each human node → list of matched auto nodes + their embedding similarities
- `auto_to_human`: For each auto node → its best human match
- `covered_human`: Human nodes with at least one auto match
- `grounded_auto`: Auto nodes that map to at least one human node
- `unmatched_human`: Human nodes with no auto match
- `unmatched_auto`: Auto nodes with no human match

### Many-to-One Matching

A single broad human node like *"Process the dispute"* may correspond to 3 granular auto nodes: *"Open the form"*, *"Enter dispute code"*, *"Submit for review"*. The LLM is instructed to allow this, so our more granular auto graph isn't penalized.

---

## Step 2: Hybrid Edge Comparison

**Goal:** Check if the connections (edges) between matched nodes are preserved — with special attention to decision branches.

### Phase 1: Deterministic path check (free)

A human edge `h1 → h2` is "matched" if there exists an auto path connecting any auto node aligned to `h1` to any auto node aligned to `h2`:

```text
Human:  h1 ────────────→ h2
         ↕ (aligned)       ↕ (aligned)
Auto:   a1 → a_mid → a2   (path ≤ 3 hops)
```

**Why allow indirect paths (up to 3 hops)?**
Because our auto graph is more granular — one human edge `h1 → h2` might correspond to `a1 → a_intermediate → a2` in our graph.

This catches most matches and costs nothing.

### Phase 2: LLM validation (decision edges only)

Edges that fail Phase 1 are split into two categories:

- **Decision edges** (source is a question/decision node) — sent to the LLM for semantic validation
- **Sequential edges** (source is an instruction/activity) — go straight to mismatches, no LLM needed

**Why only decision edges?** Decision branches carry the actual logic — "If fraud, go to escalation" vs "If not fraud, close case". A structural mismatch here means the auto graph might have different branching logic. Sequential instruction→instruction edges are trivial flow connections where structural mismatch usually just means different granularity.

**What the LLM sees for each unmatched decision edge:**

1. The human edge: source node text, target node text, branch label
2. Which auto nodes are aligned to the source and target
3. The **2-hop neighborhood subgraph** around those auto nodes — so the LLM can see the local auto graph structure and determine if the flow is preserved through a different path

**What the LLM decides:**

- `preserved: true` — the auto graph handles this flow, just structured differently (e.g., splits the decision into two sub-decisions that reach the same outcome)
- `preserved: false` + `reason` — the branching logic is genuinely different, or the path doesn't exist

### What's tracked

- `matched_edges`: Human edges matched deterministically + LLM-recovered decision edges
- `total_auto_edges` / `total_human_edges`: Total edge counts
- `mismatches`: Edges that failed both phases, split into:
  - Decision edge mismatches (with LLM reasoning)
  - Sequential edge mismatches (structural only)
- `type_matches`: How many aligned node pairs agree on type (question vs instruction etc.)

---

## Step 3: LLM-Validated SOP Grounding

**Goal:** For nodes that exist in only one graph, determine if they represent real SOP content.

### Why this matters

- An auto-only node might be **real content the human missed** (auto advantage) or **hallucinated**
- A human-only node might be a **real gap in our graph** or **something the human added beyond the SOP**

The SOP document is the ground truth — not the human graph.

### Process

**Phase 1 — Retrieve relevant SOP excerpts:**
1. Split SOP into overlapping chunks (~500 chars each, 20% overlap)
2. Embed all SOP chunks + unmatched node texts
3. For each unmatched node, find top-3 most similar SOP chunks by cosine similarity

**Phase 2 — LLM validation (batched):**
1. Send batches of 15 nodes to LLM with the full SOP document
2. LLM determines for each node: `grounded` (true/false) + `reason` (one sentence)
3. "Grounded" = the SOP explicitly describes this step/action/decision
4. "Not grounded" = the SOP doesn't mention this, even if it's a reasonable step

### Four Verdicts

| Node Source | SOP Grounded? | Verdict | Meaning |
|-------------|---------------|---------|---------|
| Auto-only | Yes | **Auto Advantage** | We caught SOP content the human missed |
| Auto-only | No | **Hallucinated** | Our graph invented a step not in the SOP |
| Human-only | Yes | **True Gap** | We missed real SOP content |
| Human-only | No | **Human Extrapolation** | Human added context beyond the SOP |

---

## Step 4: Metrics

### Core Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Node Recall** | `covered_human / total_human` | What fraction of human nodes our auto graph captures |
| **Node Precision** | `grounded_auto / total_auto` | What fraction of our auto nodes are real (not extra) |
| **Node F1** | `2 * P * R / (P + R)` | Harmonic mean — balances precision and recall |

**Example:** If human has 50 nodes and we match 40 → recall = 80%. If we have 100 nodes and 60 are grounded → precision = 60%. F1 = 68.6%.

### Edge Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Edge Recall** | `matched_edges / total_human_edges` | How many human connections we preserve |
| **Edge Precision** | `matched_edges / total_auto_edges` | How many of our connections match human ones |
| **Edge F1** | `2 * P * R / (P + R)` | Combined edge coverage |

Edge recall is often lower than node recall — we might capture the right steps but connect them differently.

### Type Accuracy

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Type Accuracy** | `type_matches / grounded_auto` | Agreement on node types for matched pairs |

Common disagreements: `instruction vs question` (we model a step as a decision where the human just has an action, or vice versa).

### Structural Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Structural Score** | `(node_F1 + edge_F1) / 2` | Overall structural similarity |
| **Granularity Ratio** | `total_auto / covered_human` | Auto nodes per matched human node |
| **Avg Fan-Out** | `mean(auto matches per human node)` | Average many-to-one ratio |

**Granularity interpretation:**
- `> 1.3x` → Auto graph is more granular (breaks steps into sub-steps)
- `0.9 - 1.1x` → Similar granularity
- `< 0.8x` → Human graph is more granular

### Similarity Distribution

Buckets the embedding similarity scores of matched pairs:

| Range | Quality |
|-------|---------|
| 0.90 - 1.00 | Near-identical wording |
| 0.80 - 0.90 | Same step, different wording |
| 0.70 - 0.80 | Likely same step, loosely worded |

Note: These are embedding similarities for reporting only. The actual matching decision is made by the LLM, not by these thresholds.

### SOP Grounding Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Auto Advantages** | Count of grounded auto-only nodes | SOP content we caught that human missed |
| **Hallucinated** | Count of not-grounded auto-only nodes | Steps we invented |
| **Auto Advantage Rate** | `auto_advantages / total_auto_only` | Quality of our extra nodes |
| **True Gaps** | Count of grounded human-only nodes | SOP content we missed |
| **Human Extrapolations** | Count of not-grounded human-only nodes | Stuff human added beyond SOP |
| **True Gap Rate** | `true_gaps / total_human_only` | How many misses are real |

---

## Output: Similarity Matrix

The report includes the full `auto × human` cosine similarity matrix. Cells are formatted as:

- **Bold** = LLM-confirmed match
- *Italic* = Above 0.70 embedding similarity but NOT matched by LLM (false positive caught)
- Plain = Below threshold

This lets you visually inspect where embeddings and LLM disagree.

---

## Output: Markdown Report Sections

| Section | Content |
|---------|---------|
| 1. Overview | Node/edge counts, orphans, dead ends, type distribution for both graphs |
| 2. Core Metrics | Table with all metrics + human-readable interpretations |
| 3. Alignment Quality | Similarity stats, granularity ratio, fan-out |
| 4. Type Disagreements | Where matched pairs disagree on node type |
| 5. Matched Pairs | Full table sorted by similarity (weakest first = most likely wrong) |
| 6. Many-to-One | Human nodes broken into multiple auto sub-steps |
| 7. SOP Grounding | Auto advantages, hallucinations, true gaps, human extrapolations — each with LLM reasoning |
| 8. Edge Mismatches | Decision mismatches (with LLM reasoning) + sequential mismatches (table) |
| 9. Similarity Matrix | Full auto × human cosine similarity grid |
| 10. Key Takeaways | Auto-generated insights based on metric values |

---

## How to Interpret Results

**Best case for our auto graph:**
- High node recall (>80%) — we capture most human steps
- Many auto advantages — we catch SOP content the human missed
- Few hallucinations — our extra nodes are real
- Few true gaps — we're not missing much real content
- Granularity ratio >1 — we're more detailed, not less

**Warning signs:**
- Low node recall (<60%) — fundamental coverage problem
- Many hallucinations — the LLM is inventing steps
- Many true gaps — real SOP content is being skipped
- Edge recall << node recall — right steps, wrong connections

---

## Dependencies

All self-contained in `graph_compare.py`. No imports from `sop_to_dag`.

| Dependency | Purpose |
|-----------|---------|
| `BAAI/bge-base-en-v1.5` via `langchain_huggingface` | Local text embeddings (no API cost) |
| `gpt-oss-120b` via `langchain_openai` | LLM for matching + SOP grounding |
| `numpy` | Cosine similarity matrix computation |

---

## Usage

```bash
# Full comparison with SOP grounding
python -m sop_to_dag.graph_compare \
    --auto output/auto_graph.json \
    --human output/human_graph.json \
    --sop input/sop.md \
    --md report.md

# Without SOP (skip grounding, just node/edge comparison)
python -m sop_to_dag.graph_compare \
    --auto output/auto_graph.json \
    --human output/human_graph.json

# Save full JSON report too
python -m sop_to_dag.graph_compare \
    --auto auto.json --human human.json --sop sop.md \
    --output report.json --md report.md
```
