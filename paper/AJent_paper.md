# AJent: Scalable Conversion of Long-Horizon Standard Operating Procedures into Executable Workflow Graphs

**Aditya Jha**

---

## Abstract

Standard Operating Procedures (SOPs) encode critical institutional knowledge as free-form natural language, yet their length, inconsistent terminology, and complex branching logic make them resistant to automated parsing by large language model (LLM) agents. We introduce **AJent**, a modular framework that converts arbitrarily long SOP documents into structured, executable workflow graphs represented as typed JSON DAGs. Unlike prior approaches that rely on bottom-up chunk-and-merge strategies or manual graph construction, AJent employs a **graph-first** conversion pipeline: the LLM produces a complete workflow graph directly from the enriched SOP text, followed by chunk-level patch-based refinement that preserves global coherence. Central to AJent is a four-stage **RAG-augmented preprocessing** pipeline --- semantic chunking, FAISS-based cross-chunk retrieval, graded context condensation, and entity resolution --- that standardizes terminology and resolves cross-section dependencies before conversion. Post-conversion, a **self-refinement loop** iterates through topological validation, LLM-based completeness and context audits, decision-edge triplet verification, and deterministic schema enforcement until the graph stabilizes. We further contribute a **hybrid graph comparison** framework that combines embedding-based candidate selection with LLM-verified semantic matching to evaluate auto-generated graphs against human-curated baselines with SOP-grounded attribution. AJent's architecture is designed for production-length SOPs (10,000+ words) where prior methods degrade, and includes checkpoint-based fault tolerance for reliable processing under API rate limits. We propose evaluation on business process, onboarding, and compliance SOPs, comparing against bottom-up (PADME-style), edge-vertex (Agent-S-style), and direct prompting baselines across node coverage, edge fidelity, structural similarity, and SOP grounding metrics.

**Keywords:** Standard Operating Procedures, LLMs, Workflow Graphs, DAG Generation, RAG, Self-Refinement.

---

## 1. Introduction

Organizations encode procedural knowledge in Standard Operating Procedures (SOPs) --- step-by-step documents that prescribe how to execute business processes, onboard employees, handle compliance cases, or respond to incidents. These documents are designed for human readers, featuring free-form prose, nested conditionals, cross-section references, inconsistent terminology, and implicit temporal dependencies. Converting SOPs into machine-readable representations that preserve their full logical structure remains an open challenge, particularly for long-horizon procedures spanning dozens of pages and hundreds of steps.

Recent work has demonstrated that LLM-based agents can transform procedural text into graph representations. PADME [1] introduces a two-phase Teach-Execute framework that iteratively chunks documents, extracts information bottom-up, and aggregates sub-graphs. SOPRAG [2] uses a "Procedure Card" approach to extract macro-level skeletons before grafting micro-steps. Agent-S [3] employs a two-stage prompting strategy --- first extracting isolated entities, then mapping relationships. These methods advance the state of the art, but they share a critical limitation: **the intermediate representations they construct during parsing are lossy**. Bottom-up chunking loses cross-section dependencies. Skeleton-then-detail approaches lose temporal ordering within sections. Entity-then-edge pipelines lose the contextual flow that determines edge semantics.

Moreover, existing approaches have not addressed the compound challenges that arise at scale. Long SOPs (10,000--50,000 words) exhibit phenomena that short procedures do not: inconsistent terminology across sections written by different authors, dangling cross-references to other sections or external guides, and decision logic that spans multiple pages. No prior work provides an integrated solution for terminology standardization, cross-reference resolution, and multi-pass quality assurance in a single pipeline.

We introduce **AJent** (Aditya Jha's Agent), a framework that addresses these gaps through three core design decisions:

1. **Graph-first conversion.** Rather than constructing intermediate text outlines or entity lists, AJent prompts the LLM to produce a typed JSON workflow graph *directly* from the full enriched SOP. This eliminates lossy intermediate representations and preserves temporal dependencies, decision scope, and branch topology from the outset. The initial graph is then refined through chunk-level patch operations that add missing detail without disrupting global structure.

2. **RAG-augmented preprocessing.** Before any conversion, a four-stage LangGraph pipeline segments the SOP, builds a FAISS index over chunks, retrieves and grades cross-section context for each chunk, and resolves entity aliases to canonical forms. This ensures the converter operates on text where terminology is consistent and cross-references are explicit.

3. **Self-refinement with multi-signal analysis.** A LangGraph-orchestrated loop alternates between analysis (topological checks, LLM-based completeness audits, context adjacency verification, decision-edge triplet validation) and patch-based refinement until all quality signals converge, with deterministic schema validation as a final invariant.

We further contribute a **hybrid graph comparison** framework for evaluation, combining embedding-based candidate retrieval with LLM-verified semantic matching to align nodes between auto-generated and human-curated graphs, followed by structural edge comparison and SOP-grounded attribution of unmatched nodes.

The remainder of this paper is organized as follows: Section 2 details the AJent methodology, Section 3 describes the experimental setup and proposed evaluation, Section 4 discusses related work, Section 5 provides a detailed comparative analysis with contemporary systems, Section 6 analyzes AJent's self-reflection architecture and its theoretical grounding, Section 7 presents the multi-layer hallucination mitigation strategy, Section 8 discusses token efficiency, Section 9 covers key design decisions, and Sections 10--11 conclude with limitations and future directions.

---

## 2. AJent Methodology

AJent transforms unstructured SOP documents into executable workflow graphs through a three-phase pipeline: **Preprocessing** (Section 2.1), **Conversion** (Section 2.2), and **Refinement** (Section 2.3). Figure 1 illustrates the end-to-end architecture.

```
SOP Document
     |
     v
[Phase 1: RAG-Augmented Preprocessing]
  Semantic Chunking --> FAISS Indexing --> RAG Enrichment --> Entity Resolution
     |
     v
  Enriched, Entity-Resolved Chunks
     |
     v
[Phase 2: Graph-First Conversion]
  Step 1: Full SOP --> Initial Graph (structured JSON output)
  Step 2: Chunk-by-Chunk Patch Refinement (multi-pass)
     |
     v
  Draft Workflow Graph
     |
     v
[Phase 3: Self-Refinement Loop]
  Analyse (Topological + Completeness + Context + Triplets)
     |  ^
     v  |
  Refine (Patch-based resolution per issue)
     |
     v
  Final Validated Graph
```
*Figure 1: AJent end-to-end pipeline.*

### 2.1 Graph Representation

AJent represents workflow graphs as directed graphs $G = (V, E)$ where each node $v \in V$ is a typed operator with a strict JSON schema. We define four node types:

- **Instruction**: A linear action step. Must specify exactly one successor via the `next` field.
- **Question**: A binary decision point. Must specify exactly two successors via `options: {"Yes": id, "No": id}`. Multi-way decisions in the SOP (3+ branches) are decomposed into chains of binary questions, ensuring a uniform branching structure amenable to deterministic validation.
- **Terminal**: An end-of-flow marker. Must have no successors.
- **Reference**: A static data lookup pointing to an external document, with a mandatory `external_ref` field and a `next` successor.

Each node carries metadata including `role` (who performs the action), `system` (software or tool used), and a **confidence label** $c \in \{\text{high}, \text{medium}, \text{low}\}$ indicating whether the node's outgoing edges are explicitly stated in the SOP ($\text{high}$), inferred from context ($\text{medium}$), or guessed to maintain connectivity ($\text{low}$). This confidence annotation supports downstream human-in-the-loop review by directing attention to uncertain edges.

All node schemas are enforced via Pydantic model validators at every pipeline stage, providing compile-time-like guarantees on graph well-formedness:

- Question nodes without `options` are automatically downgraded to instructions.
- Terminal nodes have `next` and `options` forcibly nullified.
- Instruction nodes without `next` are flagged for manual resolution.
- Question nodes with spurious `next` fields have them removed.

Unlike XML-based representations (BPMN [4]) or free-form text outlines, this strict JSON schema enables both deterministic validation and LLM-based structured output via `with_structured_output()`, ensuring that every LLM call produces schema-compliant nodes.

### 2.2 Phase 1: RAG-Augmented Preprocessing

Long SOPs suffer from three problems that must be resolved *before* conversion: (a) cross-section dependencies are implicit, (b) terminology varies across sections, and (c) the full document may exceed the LLM's effective context window. AJent addresses these through a four-node LangGraph pipeline.

#### 2.2.1 Semantic Chunking

An LLM segments the raw SOP into semantically coherent chunks at natural process boundaries --- phase transitions, department handoffs, or decision-branch entry points. Unlike fixed-length chunking, this preserves logical units intact: a multi-step sub-procedure stays in one chunk, and closely related decision branches are not split mid-logic.

#### 2.2.2 FAISS Indexing

All chunks are embedded using a local embedding model (BAAI/bge-base-en-v1.5) and indexed in a FAISS vector store. This enables efficient cross-chunk retrieval without API cost, as the embedding model runs entirely on CPU.

#### 2.2.3 RAG Enrichment

For each chunk, a Dependency Analyst agent identifies dangling references --- mentions of other sections, teams, systems, or processes not fully defined within the chunk. For each reference, it generates a search query, retrieves the top-$k$ most similar chunks from the FAISS index (excluding the chunk itself), and a Relevance Grader agent filters retrievals for strict relevance. Accepted retrievals are condensed by a Context Condensation agent into a concise factual note (2--4 sentences) capturing only cross-chunk dependencies, upstream triggers, downstream handoffs, and shared entities. This condensed context is appended to the chunk for the converter's consumption.

This three-agent retrieval pipeline (generate queries $\rightarrow$ retrieve and grade $\rightarrow$ condense) avoids the common failure mode of RAG systems where noisy retrievals degrade downstream quality. The grading step rejects irrelevant retrievals, and the condensation step prevents context bloat.

#### 2.2.4 Entity Resolution

A Terminology Standardization agent analyzes all chunks simultaneously, identifies groups of synonymous terms (e.g., "CBRD team", "Credit Bureau Reporting Disputes team", "the CBRD"), and maps them to canonical forms. Aliases are then replaced with canonical names in both the chunk text and any retrieved context. This ensures the converter sees consistent terminology, preventing duplicate nodes for the same entity described with different names --- a prevalent issue in production SOPs authored by multiple writers.

**Content-Based Caching.** Preprocessing results are keyed by the SHA-256 hash of the input document and persisted to disk. On subsequent runs with the same document, the cached chunks, enriched chunks, and entity maps are loaded directly, with only the FAISS vector store rebuilt from cached chunks (as FAISS objects are not serializable). This eliminates redundant LLM calls during iterative development.

### 2.3 Phase 2: Graph-First Conversion

The core insight of AJent's conversion strategy is that **producing the graph directly avoids the lossy intermediates** of prior approaches. Text outlines lose temporal dependencies. Entity lists lose contextual flow. Procedure cards lose decision scope. By prompting the LLM to output structured JSON graph nodes from the start, AJent preserves all structural information.

#### 2.3.1 Step 1: Full SOP to Initial Graph

The enriched, entity-resolved chunks are reassembled into a single document (without cross-reference context notes, which would be redundant noise at this global scale). A Process Architect agent receives the full enriched SOP along with a comprehensive pattern guide containing eight modeling patterns:

| Pattern | Description |
|---------|-------------|
| P1. Multi-option chains | Multi-way decisions $\rightarrow$ chains of binary Yes/No questions |
| P2. Nested conditionals | If/else trees $\rightarrow$ chained question nodes |
| P3. Cross-system data flows | Copy from System A, paste to System B $\rightarrow$ separate nodes per system |
| P4. Auto-populate fallbacks | Action $\rightarrow$ check $\rightarrow$ fallback path $\rightarrow$ converge |
| P5. Field validation lists | Multiple fields $\rightarrow$ single node listing all fields |
| P6. Email/escalation templates | Capture full template (recipients, subject, body) |
| P7. Retry loops | Failed action $\rightarrow$ wait $\rightarrow$ back-edge to retry |
| P8. Simple decision flows | Basic if/else branching patterns |

*Table 1: AJent pattern guide for graph-first conversion.*

The LLM produces a structured `InitialGraph` output containing a reasoning chain and a list of `WorkflowNode` objects. The graph undergoes immediate schema validation and topological checking (orphan detection, dead-end detection, broken link detection) before proceeding.

#### 2.3.2 Step 2: Chunk-by-Chunk Patch Refinement

While the initial graph captures the global structure, it may miss fine-grained details within individual sections. AJent addresses this through a multi-pass patch refinement:

For each enriched chunk (now including its cross-reference context), a Graph Refinement Engineer agent performs a line-by-line audit:

1. **Extract**: Read the SOP chunk sentence by sentence, listing every discrete detail (actions, decisions, systems, roles, field names, thresholds, temporal dependencies, exception paths).
2. **Match**: For each extracted detail, check whether the current graph covers it (COVERED), partially covers it (PARTIAL), or misses it (MISSING).
3. **Patch**: For every MISSING or PARTIAL detail, produce add/modify/remove operations as a structured `GraphPatch`.

Each patch is validated for schema compliance, topological integrity, and sanity (rejecting patches that shrink the graph by $>30\%$ or remove the start node). Two passes over all chunks are performed; if the second pass produces zero changes, the graph is declared stable.

**Checkpoint-Based Fault Tolerance.** After every chunk, the current graph state is checkpointed to disk. If the pipeline is interrupted (API rate limit, server error, network failure), it can resume from the last checkpoint via the `--resume` flag. The `safe_invoke()` wrapper implements automatic retry with exponential backoff (up to 2 retries with 500-second wait) for rate-limit and server errors, and halts immediately on authentication errors. This makes AJent suitable for processing long SOPs that require dozens of LLM calls over extended periods.

### 2.4 Phase 3: Self-Refinement Loop

After conversion, AJent enters a LangGraph-orchestrated analyse-refine loop that iterates until the graph passes all quality checks or reaches a maximum iteration count.

#### 2.4.1 Analysis (Multi-Signal Quality Assessment)

Each analysis iteration evaluates the graph through four complementary signals:

1. **Topological Check** (deterministic, zero-cost): Pure Python scan for orphan nodes (unreferenced non-start nodes), dead ends (non-terminal nodes with no outgoing edges), and broken links (references to undefined node IDs).

2. **Completeness Check** (LLM-based): A Process Quality Auditor breaks the SOP into its natural sections and, for each section, finds the graph nodes that cover it. Every section is marked as `covered` or `missing`. The check explicitly avoids false positives for correct modeling patterns (multi-way decisions as binary chains, loops as back-edges, field lists as single nodes).

3. **Context Adjacency Check** (LLM-based): A Process Flow Analyst verifies that connected nodes are logically adjacent --- that transitions make sense given the SOP, no intermediate steps are skipped, and edge directions are correct. Back-edges for loops and cross-system multi-node sequences are recognized as valid.

4. **Triplet Verification** (LLM-based): All decision-node triplets $(source, edge\_label, target)$ are extracted and verified against the SOP. Invalid triplets (wrong branch target, incorrect condition, mismatched answer) are flagged. Previously verified triplets are cached to avoid redundant LLM calls across iterations.

The graph is declared complete only if all four signals pass simultaneously.

#### 2.4.2 Refinement (Issue-by-Issue Patch Resolution)

Each flagged issue is resolved individually through a Graph Patch Resolver that sees the **full graph** plus the **full SOP** and produces a coordinated `GraphPatch`. This full-context approach, as opposed to local 2-hop-window repairs used in prior work, enables the resolver to:

- Insert multi-node decision branches that span many hops
- Restructure sections while maintaining global connectivity
- Expand coarse nodes while correctly rewiring surrounding edges
- Fix invalid triplets with coordinated option and target changes

The refinement follows a strict ordering to prevent cascading issues:

1. Resolve completeness issues (add missing SOP content)
2. Resolve context issues (fix logical adjacency problems)
3. Run triplet verification on *all* decision edges (catches bad edges from both the original graph and patches applied in steps 1--2)
4. Resolve invalid triplets
5. Fresh topological scan (catches structural issues introduced by LLM patches)
6. Deterministic schema validation

After refinement, a **post-loop merge** pass collapses overly granular sequential instruction chains where adjacent instruction nodes share the same role and system and the second node has only one incoming edge.

### 2.5 Graph Comparison Framework

To evaluate auto-generated graphs against human-curated baselines, AJent includes a **hybrid comparison framework** that combines embedding retrieval with LLM-verified semantic matching.

#### 2.5.1 Node Alignment (Hybrid: Embeddings + LLM)

1. **Phase 1 --- Embedding Retrieval**: All node texts from both graphs are embedded using bge-base-en-v1.5. A cosine similarity matrix is computed, and for each human node, the top-$K$ most similar auto nodes are selected as candidates.

2. **Phase 2 --- LLM Matching**: Candidates are sent to an LLM in batches. The LLM performs semantic matching, allowing one-to-many alignment (a single human node may map to multiple granular auto nodes). Matching is based on meaning, not wording, with type-awareness (decisions rarely match instructions).

#### 2.5.2 Edge Comparison (Deterministic + LLM)

1. **Phase 1 --- Deterministic Path Check**: For each human edge $(h_{src}, h_{tgt})$, the system checks whether the aligned auto nodes have a direct edge or a short path (up to 3 hops) between them.

2. **Phase 2 --- LLM Validation**: Unmatched *decision* edges are sent to an LLM with subgraph context from both graphs. The LLM determines whether the auto graph semantically preserves the flow, even if the structure differs. Non-decision edges are classified structurally only, as they are typically trivial.

#### 2.5.3 SOP Grounding

For nodes that exist in only one graph, a grounding check determines whether the content is actually present in the SOP document:

- **Auto-only nodes** grounded in the SOP = **Auto Advantages** (content the human missed)
- **Auto-only nodes** not grounded = **Hallucinations**
- **Human-only nodes** grounded in the SOP = **True Gaps** (content AJent missed)
- **Human-only nodes** not grounded = **Human Extrapolations** (added beyond the SOP)

This four-way attribution provides nuanced evaluation beyond simple precision/recall.

#### 2.5.4 Metrics

We report the following metrics:

| Metric | Definition |
|--------|-----------|
| Node Recall | Fraction of human nodes covered by at least one aligned auto node |
| Node Precision | Fraction of auto nodes aligned to at least one human node |
| Node F1 | Harmonic mean of node precision and recall |
| Edge Recall | Fraction of human edges matched (deterministic + LLM) |
| Edge Precision | Fraction of auto edges matched |
| Edge F1 | Harmonic mean of edge precision and recall |
| Type Accuracy | Fraction of aligned pairs where node types agree |
| Structural Score | Average of node F1 and edge F1 |
| Granularity Ratio | Auto nodes per covered human node (measures over/under-decomposition) |
| SOP Grounding Rate | Fraction of unmatched auto nodes that are SOP-grounded (auto advantages vs hallucinations) |

*Table 2: AJent evaluation metrics.*

---

## 3. Experimental Setup and Proposed Evaluation

### 3.1 Baselines

We compare AJent against four conversion approaches, all using the same underlying LLM and preprocessing:

1. **Bottom-Up (PADME-style)** [1]: The SOP is chunked, each chunk is independently converted to nodes with context carryover from prior chunks, and the resulting sub-graphs are merged via a dedicated merge agent. This mirrors the iterative chunking and sequential extraction approach of PADME's Teach phase.

2. **Edge-Vertex (Agent-S-style)** [3]: A two-stage pipeline that first extracts all workflow entities (nodes) without connections, then maps edges in a second pass. This separates structural decomposition from relationship mapping.

3. **Direct Prompting (Zero-Shot)**: The full SOP is sent to the LLM with a graph-generation prompt but without preprocessing, pattern guides, or refinement. This represents the naive baseline.

4. **ReAct-Style** [5]: An interleaving reasoning-and-acting approach where the LLM generates nodes incrementally, reflecting on the partial graph before adding each new node.

### 3.2 Datasets

We propose evaluation on three categories of SOP documents:

| Dataset | Description | Avg. Length (words) | Complexity |
|---------|-------------|--------------------:|-----------|
| Business Process SOPs | Financial operations, claims processing, dispute resolution | 5,000--15,000 | High branching, cross-system flows |
| Onboarding SOPs | Employee/vendor onboarding procedures | 1,000--5,000 | Moderate branching, role-based |
| Compliance SOPs | Regulatory compliance, audit procedures | 3,000--10,000 | Heavy conditionals, external references |

*Table 3: Proposed evaluation datasets.*

For each SOP, a human domain expert creates a gold-standard workflow graph. Inter-annotator agreement is measured on a subset by having two experts independently produce graphs and computing structural similarity.

### 3.3 Proposed Evaluation Dimensions

#### 3.3.1 Graph Quality (Auto vs Human)

Using the hybrid comparison framework (Section 2.5), we measure:

| Metric | AJent | Bottom-Up | Edge-Vertex | Direct | ReAct |
|--------|:-----:|:---------:|:-----------:|:------:|:-----:|
| Node F1 | | | | | |
| Edge F1 | | | | | |
| Type Accuracy | | | | | |
| Structural Score | | | | | |
| Granularity Ratio | | | | | |

*Table 4: Graph quality comparison (to be filled with experimental results).*

#### 3.3.2 Long-Document Scalability

A key hypothesis is that AJent degrades more gracefully on long SOPs than baselines. We propose measuring quality as a function of document length:

| Document Length (words) | AJent Node F1 | Bottom-Up Node F1 | Edge-Vertex Node F1 | Direct Node F1 |
|------------------------:|:-------------:|:------------------:|:--------------------:|:--------------:|
| 1,000--2,000 | | | | |
| 2,000--5,000 | | | | |
| 5,000--10,000 | | | | |
| 10,000--20,000 | | | | |
| 20,000+ | | | | |

*Table 5: Scalability analysis --- Node F1 vs document length (to be filled).*

We hypothesize that:
- Direct prompting degrades sharply beyond 5,000 words as the LLM loses coherence over long contexts.
- Bottom-up chunking degrades on cross-section dependencies as document length increases.
- Edge-vertex loses edge accuracy at scale because the edge-mapping stage receives too many nodes.
- AJent's graph-first + patch refinement architecture maintains quality through explicit cross-chunk context retrieval and multi-pass refinement.

#### 3.3.3 SOP Grounding Analysis

For each method, we report the breakdown of unmatched nodes:

| Method | Auto Advantages | Hallucinations | True Gaps | Hallucination Rate |
|--------|:--------------:|:--------------:|:---------:|:-----------------:|
| AJent | | | | |
| Bottom-Up | | | | |
| Edge-Vertex | | | | |
| Direct | | | | |

*Table 6: SOP grounding analysis (to be filled).*

This table reveals not just *how many* nodes each method gets wrong, but *how* it fails --- whether by hallucinating content not in the SOP or by missing content that is.

#### 3.3.4 Component Ablation

To isolate the contribution of each AJent component, we propose an ablation study:

| Configuration | Node F1 | Edge F1 | Structural Score |
|---------------|:-------:|:-------:|:----------------:|
| Full AJent | | | |
| -- w/o Entity Resolution | | | |
| -- w/o RAG Enrichment | | | |
| -- w/o Patch Refinement (Step 2) | | | |
| -- w/o Self-Refinement Loop | | | |
| -- w/o Pattern Guide | | | |
| -- w/o Confidence Labels | | | |

*Table 7: Ablation study (to be filled).*

#### 3.3.5 Efficiency Analysis

| Method | LLM Calls | Total Tokens | Wall-Clock Time (min) |
|--------|:---------:|:------------:|:---------------------:|
| AJent | | | |
| Bottom-Up | | | |
| Edge-Vertex | | | |
| Direct | | | |

*Table 8: Computational efficiency (to be filled).*

---

## 4. Related Work

### 4.1 LLM-Based Procedure Structuring

Learning to autonomously structure long-horizon procedures from natural language has attracted significant recent attention. **PADME** [1] introduces a two-phase Teach-Execute framework where a structuring agent converts free-form SOPs into decision graphs through iterative document chunking and sequential node extraction with context carryover. While effective, PADME's bottom-up aggregation can lose cross-section dependencies and requires a separate merge step that may introduce inconsistencies. AJent's graph-first approach avoids this by producing the global structure in a single LLM call, then refining locally.

**SOPRAG** [2] extracts a "Procedure Card" --- a macro-level skeleton of overarching goals and major decision points --- before grafting micro-steps. This top-down strategy captures high-level structure well but may lose fine-grained temporal ordering within sections. AJent's pattern-guided conversion captures both macro structure and micro detail in a single pass.

**Agent-S** [3] employs a universal prompting strategy with two stages: first extracting isolated entities (actions and decisions), then mapping relationships between them. This edge-vertex decomposition is clean conceptually but loses contextual flow during the vertex extraction stage, as the LLM sees actions in isolation rather than in their procedural context. AJent preserves contextual flow by producing graph edges simultaneously with nodes.

### 4.2 Business Process Modeling with LLMs

Recent work has applied LLMs to business process modeling notation (BPMN). The BPMN Assistant [4] introduces a specialized JSON intermediate representation for comparing against XML-based BPMN models and uses LLM function calling for incremental process modeling. Planetarium and related work from JP Morgan [6] explore translating natural language procedures into strict logical formats (PDDL, Python pseudocode) to accurately capture decision trees and control flows.

AJent differs by using a code-based node format (typed JSON with Pydantic validation) rather than BPMN XML, enabling deterministic schema enforcement at every pipeline stage. The strict type system (instruction, question, terminal, reference) with mandatory field constraints catches structural errors immediately rather than during downstream execution.

### 4.3 Graph-Based LLM Reasoning

Graph-based representations have been used to structure LLM reasoning beyond procedure modeling. **AgentKit** [7] requires users to manually decompose tasks into LEGO-like pieces and build a dependency graph. **SPRING** [8] parses game documentation into a fixed DAG of interdependent questions traversed in topological order. Both require manual graph construction, limiting scalability. AJent automatically constructs graphs from free-form text, eliminating manual engineering.

### 4.4 RAG for Document Understanding

Retrieval-Augmented Generation (RAG) has been widely applied to question answering and document understanding [9]. AJent adapts RAG for a novel purpose: resolving *intra-document* cross-references in long SOPs. Unlike typical RAG pipelines that retrieve from external knowledge bases, AJent's FAISS index contains chunks of the *same document*, enabling cross-section context enrichment. The three-stage retrieval pipeline (query generation, graded retrieval, context condensation) is specifically designed to prevent noisy retrievals from degrading the converter's output.

### 4.5 Self-Refinement in LLM Systems

Self-refinement loops where LLMs critique and improve their own output have been explored in code generation [10] and reasoning tasks [11]. AJent's refinement loop is distinguished by its **multi-signal analysis** (four complementary quality checks) and **issue-by-issue resolution** (each issue gets its own patch call with full graph context), as opposed to single-critique-single-revision approaches. The combination of deterministic checks (topological, schema) with LLM-based checks (completeness, context, triplets) provides both guaranteed structural invariants and semantic quality assessment.

---

## 5. Comparative Analysis with Contemporary Approaches

This section provides a detailed, dimension-by-dimension comparison between AJent and the five contemporary systems whose ideas informed its design (see Figure in image). We organize the comparison across four axes: SOP parsing strategy, node format and schema, hallucination mitigation, and scalability to long documents.

### 5.1 SOP Parsing Strategies: A Taxonomy

The fundamental architectural choice in any SOP-to-graph system is *how* the procedural text is decomposed and restructured. We identify three distinct paradigms in the current literature, each with characteristic strengths and failure modes, and position AJent's graph-first approach as a fourth.

#### 5.1.1 Bottom-Up Parsing (PADME [1])

PADME introduces the concept of **iteratively chunking documents and sequentially extracting information**, retaining context from previously processed chunks to build a continuous understanding. The structuring agent segments the procedure into units $S_k$ ($k = 1, \ldots, m$), converts each segment into a local decision subgraph, and merges subgraphs into a global DAG $G = (V, E)$.

*Strengths:* The bottom-up approach is naturally suited to procedures with clear phase boundaries. Context carryover from prior chunks provides incremental awareness. PADME achieves state-of-the-art results on short-to-medium-length procedures (ALFWorld tasks of 5--15 steps, ScienceWorld tasks of 20--50 steps).

*Limitations at scale:* The sequential chunk-then-merge pipeline faces compounding problems as document length grows:
1. **Cross-section dependency loss.** Each chunk is processed with only a summary of prior chunks' node IDs as context. In a 30-chunk SOP where chunk 25 references a decision made in chunk 3, the carryover context has been compressed through 22 intermediate summaries, losing specifics.
2. **Merge-stage fragility.** The final merge agent must deduplicate, fix cross-chunk edges, and ensure global connectivity across all independently-produced subgraphs. This is itself a complex reasoning task that scales with the number of chunks.
3. **Duplicate node creation.** Without entity resolution, sections written by different authors using variant terminology (e.g., "CBRD team" vs "Credit Bureau Reporting Disputes") produce duplicate sub-procedures.

AJent addresses these by (a) producing the global graph structure in a single call before chunk-level refinement, (b) performing entity resolution *before* conversion to prevent duplicates, and (c) using intra-document RAG to make cross-section dependencies explicit.

#### 5.1.2 Top-Down Parsing (SOPRAG [2])

SOPRAG introduces a **Procedure Card** layer that functions as a search-space pruner, extracting the overarching goals and major decision points first before processing micro-steps. The framework uses specialized Entity, Causal, and Flow graph experts in a Mixture-of-Experts architecture, replacing flat chunking with structured multi-view retrieval.

*Strengths:* The top-down approach captures the global procedure skeleton before filling in detail, ensuring macro-level correctness. The Procedure Card eliminates computational noise by constraining the scope of subsequent processing.

*Limitations:* The Procedure Card is optimized for *retrieval* (selecting the right SOP from a corpus) rather than *conversion* (transforming an SOP into an executable graph). The macro-skeleton may impose a rigid structure that doesn't accommodate the fine-grained conditional logic within sections. Additionally, the multi-agent expert architecture introduces coordination complexity.

AJent incorporates the top-down insight differently: the graph-first conversion step implicitly produces a macro-skeleton (the initial graph's decision topology), which is then refined with chunk-level detail in the patch stage. This achieves the top-down benefit without a separate skeleton extraction step.

#### 5.1.3 Edge-Vertex Parsing (Agent-S [3], Universal Prompting [12])

Agent-S and the Universal Prompting Strategy for process extraction [12] employ a **two-stage decomposition**: first extract all entities/activities, then map relationships/edges. Neuberger et al. [12] demonstrate that carefully engineered prompts with quantity of examples, definitional specificity, and format rigor can achieve 8% F1 improvement over ML baselines for extracting activities, actors, and relations from process descriptions.

*Strengths:* Separating entity identification from relationship mapping reduces the complexity of each LLM call. The Universal Prompting work [12] shows this approach is model-agnostic and universally applicable across different LLMs.

*Limitations:*
1. **Contextual flow loss.** When extracting entities in isolation, the LLM loses the procedural context that determines edge semantics. "Verify account status" and "Update account status" may be identified as entities, but their ordering and dependency are ambiguous without contextual flow.
2. **Combinatorial edge mapping.** In the second stage, the LLM must consider all possible edges among $N$ entities --- an $O(N^2)$ reasoning task. For long SOPs producing 100+ entities, this exceeds practical LLM reasoning capacity.
3. **No iterative refinement.** Both Agent-S and the Universal Prompting approach produce the graph in a single pass with no quality feedback loop.

AJent produces nodes and edges simultaneously in a single structured output, preserving contextual flow. The pattern guide (Table 1) resolves common edge-mapping ambiguities (retry loops, conditional branches, cross-system flows) at generation time rather than post-hoc.

#### 5.1.4 Graph-First Parsing (AJent)

AJent's approach is a **synthesis** of the above paradigms: it captures global structure like top-down methods, processes detail at the chunk level like bottom-up methods, and uses structured output like edge-vertex methods --- but without the lossy intermediates of any individual approach. The key insight is that modern LLMs with structured output capabilities can produce well-formed JSON graphs of 50--200 nodes in a single call, making the graph itself the natural "first output" rather than an intermediate text representation.

### 5.2 Comparison Table: AJent vs Contemporary Systems

| Dimension | PADME [1] | SOPRAG [2] | Agent-S [3] | BPMN Asst. [4] | Universal Prompting [12] | **AJent** |
|-----------|:---------:|:----------:|:-----------:|:---------:|:----------------:|:---------:|
| **Parsing strategy** | Bottom-up chunk-merge | Top-down Procedure Card | Edge-vertex two-stage | LLM function calling | Two-stage entity-relation | Graph-first + patch |
| **Intermediate repr.** | Subgraph per chunk | Macro skeleton | Entity list (no edges) | JSON edit operations | Entity-relation pairs | Direct JSON graph |
| **Node format** | Decision graph (5 types) | Multi-view graph | Action/Decision | BPMN XML via JSON proxy | Activity/Actor/Relation | Typed JSON (4 types) |
| **Schema enforcement** | None described | None described | None described | XML validation | Heuristic post-processing | Pydantic validators at every stage |
| **Entity resolution** | No | No | No | No | No | Yes (LLM-based, pre-conversion) |
| **Cross-section RAG** | No | Graph expert retrieval | No | No | No | Yes (FAISS + graded retrieval) |
| **Self-refinement** | No | No | No | No | No | Yes (multi-signal analyse-refine loop) |
| **Confidence labels** | No | No | No | No | No | Yes (high/medium/low per edge) |
| **Triplet verification** | No | No | No | No | No | Yes (decision-edge validation) |
| **Fault tolerance** | Not described | Not described | Not described | Not described | Not described | Checkpoint-resume with retry logic |
| **Target doc length** | Short-medium (recipes, game tasks) | Industrial SOPs (retrieval) | Customer care SOPs | Business processes | Process descriptions | Long-horizon (10,000+ words) |
| **Evaluation method** | End-to-end task execution | Retrieval accuracy | Task completion | Success rate + latency | F1 on entity/relation extraction | Hybrid graph comparison with SOP grounding |

*Table 9: Feature-by-feature comparison of AJent with contemporary SOP/process automation systems.*

### 5.3 Node Format Comparison: Graph-Based vs Code-Based

The choice of node representation has downstream implications for validation, execution, and human review. We identify two paradigms:

**Graph-Based Node Formats** (PADME, SOPRAG) represent nodes with semantic categories (HumanInput, InfoProcessing, InfoExtraction, Knowledge, Decision in PADME's case). These categories describe *what kind of step* a node represents, enabling role-based execution routing. However, the categories are defined informally --- there is no compile-time enforcement that a Decision node actually has branching options, or that an InfoExtraction node specifies a data source.

**Code-Based Node Formats** (AJent, BPMN Assistant) impose strict structural constraints. AJent's Pydantic-validated schema guarantees:
- Question nodes *must* have `options` with exactly two keys (Yes/No) --- violations are caught at parse time, not during execution.
- Instruction nodes *must* have a `next` pointer --- dangling instructions are impossible.
- Terminal nodes *cannot* have successors --- accidental continuation is prevented.

The BPMN Assistant [4] demonstrates the efficiency advantage of JSON over XML: approximately **43% reduction in generation latency** and **75% reduction in output tokens** compared to direct XML generation, while achieving higher success rates. AJent's JSON schema achieves similar efficiency gains by avoiding verbose XML/BPMN markup while adding Pydantic validation that BPMN Assistant's JSON proxy lacks.

Planetarium [6] reveals a critical insight about format strictness: GPT-4o achieves 96.1% syntactic correctness when generating PDDL code but only **24.8% semantic correctness**. This gap between syntactic validity and semantic accuracy motivates AJent's multi-layer validation: Pydantic ensures syntactic correctness, while the refinement loop's completeness and context checks target semantic correctness.

---

## 6. Self-Reflection Architecture: Theoretical Grounding and AJent's Implementation

AJent's self-refinement loop draws from and extends two foundational paradigms in LLM self-improvement: **Self-Refine** [10] and **Reflexion** [11]. This section analyzes how AJent adapts these general-purpose self-reflection mechanisms for the specific domain of graph construction, and argues that the structured nature of graph outputs enables stronger forms of self-correction than are possible in free-text generation.

### 6.1 From Self-Refine to Multi-Signal Analysis

Madaan et al. [10] introduce Self-Refine as a three-stage loop: **generate** $\rightarrow$ **feedback** $\rightarrow$ **refine**, where a single LLM produces an initial output, critiques it, and revises based on its own critique. This achieves approximately 20% absolute improvement across seven tasks. However, Self-Refine uses a *single, undifferentiated* feedback signal --- the LLM's general critique --- which can miss specific failure modes or produce vague, unhelpful feedback.

AJent's analysis phase decomposes Self-Refine's monolithic feedback into **four orthogonal quality signals**, each targeting a distinct failure mode:

| Signal | What it catches | Implementation | Cost |
|--------|----------------|----------------|------|
| Topological check | Orphans, dead ends, broken links | Pure Python (deterministic) | Zero (no LLM call) |
| Completeness check | Missing SOP steps/branches | LLM-based section-by-section audit | 1 LLM call |
| Context adjacency check | Illogical node connections | LLM-based edge evaluation | 1 LLM call |
| Triplet verification | Invalid decision-branch targets | LLM-based triplet batch validation | 1 LLM call per batch |

*Table 10: AJent's four analysis signals vs Self-Refine's single feedback.*

This decomposition is more effective than a single critique because:
1. **Deterministic signals cannot hallucinate.** The topological check is pure Python --- it *guarantees* that reported orphans are actually orphan nodes, unlike an LLM critique that might hallucinate issues.
2. **Specialized prompts produce targeted feedback.** The completeness check prompt is specifically engineered to audit SOP coverage section-by-section, producing actionable gap descriptions rather than vague "the graph could be more complete" feedback.
3. **Independent signals prevent masking.** A single critique might note "the graph looks mostly good" while missing a broken link. Independent signals ensure that topological validity is checked *regardless* of what the LLM thinks about completeness.

### 6.2 From Reflexion to Persistent Refinement State

Shinn et al. [11] introduce Reflexion, where agents maintain an **episodic memory** of linguistic reflections from prior attempts, enabling learning across trials without weight updates. Reflexion achieves 91% pass@1 on HumanEval by accumulating task-specific insights.

AJent adapts this principle through two mechanisms:

1. **Resolved-issue tracking.** AJent maintains a set of issues already resolved in prior iterations (`resolved_issues`). When the analyser flags an issue that was already addressed, it is skipped --- preventing the system from oscillating between fix-and-revert cycles. This is analogous to Reflexion's memory preventing repeated mistakes, but applied to graph patches rather than code attempts.

2. **Verified-triplet caching.** Decision-edge triplets verified as valid in prior iterations are cached (`verified_triplets`). Subsequent iterations skip these triplets, reducing LLM calls. New triplets introduced by patches are verified in the next iteration. This selective re-verification is more efficient than Reflexion's full re-evaluation.

Unlike Reflexion, which stores free-text reflections that may be imprecise, AJent's persistence is *structured*: issue signatures and triplet signatures are exact strings that enable precise deduplication. This eliminates the risk of the "memory" itself being unreliable.

### 6.3 Why Graph Outputs Enable Stronger Self-Correction

A fundamental advantage of AJent's self-refinement over general-purpose Self-Refine/Reflexion is that **workflow graphs are verifiable structures**. Free-text outputs can only be critiqued heuristically, but graphs admit deterministic invariant checks:

- **Connectivity invariants**: Every non-terminal node must have at least one outgoing edge. Every non-start node must have at least one incoming edge. These can be checked in $O(|V| + |E|)$ with no LLM involvement.
- **Type invariants**: Question nodes must have exactly two options. Terminal nodes must have zero successors. These are enforced by Pydantic validators.
- **Referential integrity**: Every `next` and `options` value must reference an existing node ID. Broken references are detected deterministically.

These invariants serve as **hard constraints** that the refinement loop *guarantees* in every iteration, regardless of LLM quality. The LLM-based checks (completeness, context, triplets) provide *soft signals* about semantic quality. This hybrid of hard deterministic constraints and soft LLM-based quality signals is what makes AJent's refinement more robust than pure self-reflection approaches.

### 6.4 Convergence Behavior

AJent's refinement loop converges through a combination of:
1. **Monotonic issue resolution**: Each iteration resolves flagged issues and records them as resolved. The resolved set grows monotonically, while the set of new issues shrinks (assuming the LLM's patches are generally correct).
2. **Rollback guards**: Patches that shrink the graph by $>30\%$ or remove the start node are rolled back, preventing catastrophic regression.
3. **Stabilization detection**: If the second refinement pass produces zero changes, the graph is declared stable.
4. **Hard iteration cap**: A maximum of 15 iterations prevents infinite loops from LLM-induced oscillation.

This is a stronger convergence guarantee than Self-Refine (which has no formal stopping criterion) or Reflexion (which relies on task success as a termination signal).

---

## 7. Hallucination Mitigation: A Multi-Layer Defense

Hallucination --- the generation of plausible but factually unsupported content [13] --- is a critical failure mode in SOP-to-graph conversion. A hallucinated node represents a step that doesn't exist in the SOP; a hallucinated edge represents a dependency that was never specified. In regulated domains (finance, healthcare, compliance), hallucinated steps in an automated workflow can lead to incorrect actions with material consequences.

AJent implements a **defense-in-depth** strategy against hallucination, with multiple independent layers each targeting a different hallucination vector.

### 7.1 Taxonomy of Hallucination in Graph Generation

We identify four categories of hallucination specific to SOP-to-graph conversion:

| Category | Description | Example | Risk Level |
|----------|-------------|---------|:----------:|
| **Node hallucination** | Generating a step not in the SOP | Adding "Notify compliance officer" when the SOP doesn't mention compliance | High |
| **Edge hallucination** | Creating a dependency not implied by the SOP | Connecting "Print report" to "Submit to regulator" when the SOP treats them as independent | Medium |
| **Detail hallucination** | Inventing specifics not in the source text | Adding "within 24 hours" deadline when the SOP specifies no timeframe | Medium |
| **Structural hallucination** | Imposing graph structure not supported by the SOP | Creating a decision branch where the SOP describes a linear sequence | High |

*Table 11: Taxonomy of hallucination types in SOP-to-graph conversion.*

### 7.2 Layer 1: RAG-Grounded Preprocessing

The first defense operates *before* conversion. By enriching each chunk with cross-reference context retrieved from the FAISS index and validated by the grading agent, AJent ensures the LLM has access to all relevant information from the SOP. This reduces **omission-driven hallucination** --- where the LLM invents content to fill gaps caused by missing cross-section context.

The entity resolution stage further reduces hallucination by preventing the LLM from treating terminological variants as distinct entities, which would manifest as duplicate nodes or spurious decision branches.

### 7.3 Layer 2: Structured Output Constraints

By requiring the LLM to produce typed `WorkflowNode` objects via `with_structured_output()`, AJent constrains the output space. The LLM cannot:
- Produce a node with an undefined type (only instruction, question, terminal, reference are allowed)
- Produce a question node without options
- Produce an instruction node without a next pointer
- Use free-form text for edge labels (only structured `next` and `options` fields)

These constraints eliminate an entire class of structural hallucinations that would be possible with free-text graph descriptions.

### 7.4 Layer 3: Confidence-Annotated Edges

AJent requires the LLM to self-assess each node's outgoing edges:
- **High confidence**: Edge is directly stated in the SOP ("After step 3, proceed to step 4")
- **Medium confidence**: Edge is inferred from context (implied ordering, logical necessity)
- **Low confidence**: Edge is a guess to maintain graph connectivity

This forced self-assessment acts as a **calibration mechanism**. Research on LLM calibration [14] shows that requiring explicit confidence ratings improves the correlation between stated confidence and actual accuracy. In AJent, low-confidence edges are candidates for human review or deletion --- they mark the boundary between extraction and hallucination.

### 7.5 Layer 4: Multi-Signal Refinement Verification

The self-refinement loop (Section 2.4) acts as a post-hoc hallucination detector:

- **Completeness check**: While primarily detecting missing content, the completeness check implicitly verifies that existing nodes correspond to SOP sections. Nodes that don't match any SOP section are hallucination candidates.
- **Context adjacency check**: Verifies that connected nodes are logically adjacent per the SOP. Hallucinated edges between unrelated steps are flagged as context violations.
- **Triplet verification**: Each decision-edge triplet is verified against the source SOP text. A hallucinated branch target (e.g., a "Yes" option pointing to a step that the SOP doesn't associate with that decision) is caught and corrected.

### 7.6 Layer 5: SOP Grounding in Evaluation

Even after the pipeline completes, the graph comparison framework (Section 2.5) provides a final hallucination audit. Unmatched auto-only nodes are checked against the SOP via embedding retrieval and LLM verification:
- Nodes grounded in the SOP are **auto advantages** (the system caught something the human missed)
- Nodes not grounded are **hallucinations** (the system invented something)

This post-hoc grounding check produces a **hallucination rate** metric: the fraction of auto-only nodes that are not SOP-grounded. Tracking this metric across experiments reveals systematic hallucination patterns and enables targeted prompt engineering.

### 7.7 Layer 6: Deterministic Schema Invariants

The final layer is the `SchemaValidator`, which applies deterministic fixes that are *incapable* of introducing hallucination:
- Converting question nodes without options to instructions (removes a structural hallucination)
- Nullifying next/options on terminal nodes (prevents continuation hallucination)
- Detecting instruction nodes without next pointers (flags potential missing edges without inventing them)

These six layers form a defense-in-depth strategy where each layer catches hallucinations that prior layers miss, and deterministic layers provide guarantees that LLM-based layers cannot.

---

## 8. Token Efficiency and the Chunking-First Architecture

A practical consideration for production deployment is computational cost. Long SOP processing requires many LLM calls, and each call's token count directly affects latency and API cost. AJent's architecture is designed to minimize total token consumption while maximizing the information available to each LLM call.

### 8.1 Why Chunking Before Conversion Saves Tokens

A naive approach to SOP conversion would send the full 10,000-word document to the LLM for each processing step. AJent's preprocessing chunking reduces the per-call context in the refinement stage:

| Approach | Step 1 Tokens | Step 2 Tokens (per chunk) | Total for 10 Chunks |
|----------|:------------:|:------------------------:|:-------------------:|
| Full-document-every-call | 10K | 10K | 10K + 10 $\times$ 10K = 110K |
| AJent (chunk-level refinement) | 10K (initial graph) | ~1K (chunk) + ~3K (current graph) | 10K + 10 $\times$ 4K = 50K |

*Table 12: Estimated input token comparison (10,000-word SOP, 10 chunks).*

The key efficiency comes from the refinement stage: each chunk-level patch audit receives only the chunk text (~1K tokens), the current graph as compact JSON (~3K tokens), and the adjacency map (~0.5K tokens) --- not the full SOP. The initial graph-first call sees the full document, but this is a one-time cost.

### 8.2 Entity Resolution as Token Saver

Entity resolution has a secondary benefit beyond semantic correctness: **token reduction through alias elimination**. When the SOP uses five different names for the same team, each mention of each alias consumes tokens. After entity resolution replaces all aliases with canonical forms, the enriched text is shorter and more information-dense.

More importantly, entity resolution prevents the LLM from creating duplicate nodes for the same entity, which would cause wasted tokens in every subsequent pipeline stage that processes those redundant nodes.

### 8.3 Graded Retrieval as Context-Quality Filter

The RAG enrichment stage's grading step is explicitly designed to prevent **context bloat**. Without grading, every retrieved chunk would be appended to the enriched context, inflating the input for downstream calls. The relevance grader rejects irrelevant retrievals, and the condensation agent compresses accepted retrievals into 2--4 sentence notes. This ensures that cross-reference context adds information density, not noise.

### 8.4 Compact Graph Representation

AJent uses two graph representations optimized for different consumers:
- **Full JSON** (for the resolver/patcher): Complete node details including all metadata. Used when the LLM needs to produce precise patches.
- **Compact text representation** (for the analyser): A compressed format showing only ID, type, text, and connections --- approximately 60--70% smaller than full JSON. Used for completeness and context checks where the LLM needs to understand structure but not produce patches.

This dual-representation approach reduces tokens in analysis calls (which run every iteration) while preserving full detail for refinement calls (which run only when issues are found).

---

## 9. Design Decisions and Architectural Insights

### 9.1 Why Graph-First Over Bottom-Up

The fundamental architectural choice in AJent is producing the workflow graph *directly* from the SOP rather than constructing it incrementally from chunks. This is motivated by the observation that SOP procedures are inherently graph-structured: steps have dependencies, decisions create branches, and branches converge. Bottom-up construction requires a lossy intermediate (text outline, entity list, or procedure card) that must be reassembled --- and every reassembly step risks losing structural information.

The graph-first approach is feasible because modern LLMs with structured output capabilities can produce well-formed JSON graphs of 50--200 nodes in a single call. The pattern guide (Table 1) provides the LLM with concrete modeling templates that resolve common ambiguities (multi-way decisions, nested conditionals, retry loops) before they propagate.

### 9.2 Why Multi-Pass Patch Refinement

A single LLM call, even with the full SOP, cannot capture every detail of a 10,000-word document. The initial graph captures the correct *structure* (decision topology, branch convergence, loop back-edges) but may miss *specifics* (exact field names, threshold values, system names, role assignments). The chunk-by-chunk patch refinement addresses this by auditing the graph against each section individually, with cross-reference context from RAG enrichment providing necessary inter-section context.

Two passes are performed because the first pass's additions may create new gaps (e.g., adding a node that should connect to another section's node), and the second pass catches these. If the second pass produces zero changes, the graph is stable and further passes are unnecessary.

### 9.3 Why Confidence Labels

Not all edges in an auto-generated graph are equally reliable. Edges that directly correspond to SOP text ("If fraud is detected, escalate to supervisor") are high-confidence. Edges that the LLM infers from context ("After completing the form, proceed to verification") are medium-confidence. Edges that the LLM guesses to maintain graph connectivity are low-confidence.

By annotating each node with a confidence label, AJent enables two downstream capabilities: (a) prioritized human-in-the-loop review, where domain experts focus attention on low-confidence edges, and (b) risk-aware execution, where agents treat high-confidence paths differently from uncertain ones.

### 9.4 Fault Tolerance for Long Documents

Production SOPs may require 50--100 LLM calls for full processing (preprocessing + conversion + refinement). At this scale, API rate limits, transient server errors, and network interruptions are not exceptional --- they are expected. AJent's checkpoint system writes the graph state after every chunk and every refinement iteration, enabling exact resumption. The `safe_invoke()` wrapper distinguishes between retryable errors (429, 5xx) and fatal errors (401, 403), applying appropriate retry-or-halt logic.

---

## 10. Conclusion

We have presented AJent, a modular framework for converting long-horizon Standard Operating Procedures into executable workflow graphs. AJent's three core contributions --- graph-first conversion, RAG-augmented preprocessing with entity resolution, and multi-signal self-refinement --- address the scalability and quality challenges that prior approaches face on production-length SOPs.

The graph-first architecture eliminates lossy intermediate representations by producing typed JSON graphs directly from enriched SOP text, while chunk-level patch refinement preserves fine-grained detail. The preprocessing pipeline ensures consistent terminology and resolved cross-references before conversion begins. The self-refinement loop provides convergent quality assurance through four complementary analysis signals and issue-by-issue patch resolution with full graph context.

The hybrid graph comparison framework enables rigorous evaluation with SOP-grounded attribution, distinguishing genuine coverage gaps from hallucinations and auto advantages from human extrapolations.

We view AJent as a step toward production-grade SOP automation --- capable of handling the length, complexity, and terminological inconsistency of real-world procedural documents while providing the transparency (confidence labels, checkpoint trails, structured comparison reports) needed for deployment in regulated domains.

---

## 11. Limitations

We acknowledge several limitations of the current work:

1. **LLM Dependency**: AJent relies on a capable LLM for all non-deterministic pipeline stages. Graph quality is bounded by the LLM's ability to understand procedural logic and produce valid structured output. While the architecture is model-agnostic, performance may vary across LLMs.

2. **Binary Decision Constraint**: All decisions are decomposed into binary Yes/No questions. While this simplifies validation and execution, some multi-way decisions may be more naturally represented as multi-branch nodes. The chain of binary questions increases graph size and may feel unnatural for domain experts reviewing the output.

3. **Evaluation Coverage**: The proposed evaluation is limited to English-language SOPs. Multilingual procedures and procedures with embedded diagrams, tables, or forms are not currently addressed.

4. **Computational Cost**: The multi-pass refinement loop and multiple analysis checks incur significant LLM usage. For very long SOPs, the total token consumption may be substantial. The caching and checkpoint systems mitigate re-work but do not reduce first-run cost.

5. **Source Quality Dependency**: AJent cannot compensate for fundamentally incomplete or incorrect source SOPs. If the SOP omits critical steps or contains contradictory instructions, the generated graph will reflect these deficiencies.

---

## References

[1] D. Garg, S. Zeng, A. L. Narayanan, S. Ganesh, and L. Ardon, "PADME: Procedure Aware DynaMic Execution," arXiv preprint arXiv:2510.11281, 2025.

[2] L. Lin, Z. Zhu, T. Zhang, and Y. Wen, "SOPRAG: Multi-view Graph Experts Retrieval for Industrial Standard Operating Procedures," arXiv preprint arXiv:2602.01858, 2026.

[3] M. Kulkarni, "Agent-S: LLM Agentic Workflow to Automate Standard Operating Procedures," arXiv preprint arXiv:2503.15520, 2025.

[4] J. T. Licardo, N. Tankovic, and D. Etinger, "BPMN Assistant: An LLM-Based Approach to Business Process Modeling," arXiv preprint arXiv:2509.24592, 2025.

[5] S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," arXiv preprint arXiv:2210.03629, 2023.

[6] M. Zuo, F. P. Velez, X. Li, M. L. Littman, and S. H. Bach, "Planetarium: A Rigorous Benchmark for Translating Text to Structured Planning Languages," arXiv preprint arXiv:2407.03321, 2024.

[7] Y. Wu et al., "AgentKit: Structured LLM Reasoning with Dynamic Graphs," in First Conference on Language Modeling, 2024.

[8] Y. Wu et al., "SPRING: Studying Papers and Reasoning to Play Games," Advances in Neural Information Processing Systems, vol. 36, 2024.

[9] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," Advances in Neural Information Processing Systems, vol. 33, 2020.

[10] A. Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback," Advances in Neural Information Processing Systems, vol. 36, 2023.

[11] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language Agents with Verbal Reinforcement Learning," Advances in Neural Information Processing Systems, vol. 36, 2023.

[12] J. Neuberger, L. Ackermann, H. van der Aa, and S. Jablonski, "A Universal Prompting Strategy for Extracting Process Model Information from Natural Language Text using Large Language Models," arXiv preprint arXiv:2407.18540, 2024.

[13] L. Huang et al., "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions," ACM Transactions on Information Systems, 2024.

[14] S. Kadavath et al., "Language Models (Mostly) Know What They Know," arXiv preprint arXiv:2207.05221, 2022.

[15] I. Fujisawa et al., "ProcBench: Benchmark for Multi-Step Reasoning and Following Procedure," arXiv preprint arXiv:2410.03117, 2024.

[16] F. Monti, F. Leotta, J. Mangler, M. Mecella, and S. Rinderle-Ma, "NL2ProcessOps: Towards LLM-Guided Code Generation for Process Execution," in International Conference on Business Process Management, Springer, 2024, pp. 127--143.

---

## Appendix A: Node Schema Specification

```json
{
  "id": "check_fraud_score",
  "type": "question",
  "text": "Is the fraud score above the threshold?",
  "next": null,
  "options": {"Yes": "escalate_to_supervisor", "No": "proceed_normal"},
  "external_ref": null,
  "role": "Fraud Analyst",
  "system": "Fraud Detection System",
  "confidence": "high"
}
```

## Appendix B: Pattern Guide Examples

### B.1 Multi-Option to Binary Chain (P1)

**SOP**: "If dispute code is 103, do X. If 1047, do Y. Otherwise do Z."

```json
[
  {"id": "is_code_103_question", "type": "question",
   "text": "Is the dispute code 103?",
   "options": {"Yes": "handle_103", "No": "is_code_1047_question"}},
  {"id": "is_code_1047_question", "type": "question",
   "text": "Is the dispute code 1047?",
   "options": {"Yes": "handle_1047", "No": "handle_other_codes"}}
]
```

### B.2 Retry Loop with Back-Edge (P7)

**SOP**: "Submit request. If it fails, wait 1 day and retry."

```json
[
  {"id": "submit_request", "type": "instruction",
   "text": "Submit the request", "next": "did_request_succeed_question"},
  {"id": "did_request_succeed_question", "type": "question",
   "text": "Did the request succeed?",
   "options": {"Yes": "next_step", "No": "wait_one_day"}},
  {"id": "wait_one_day", "type": "instruction",
   "text": "Wait 1 business day", "next": "submit_request"}
]
```
