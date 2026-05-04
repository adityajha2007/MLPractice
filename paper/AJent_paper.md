# AJent: Scalable Conversion of Long-Horizon Standard Operating Procedures into Executable Workflow Graphs

**Aditya Jha**

---

## Abstract

Standard Operating Procedures (SOPs) encode critical institutional knowledge as free-form natural language, yet their length, inconsistent terminology, and complex branching logic make them resistant to automated parsing by large language model (LLM) agents. We introduce **AJent**, a modular framework that converts arbitrarily long SOP documents into structured, executable workflow graphs represented as typed JSON DAGs. Unlike prior approaches that rely on bottom-up chunk-and-merge strategies or manual graph construction, AJent employs a **graph-first** conversion pipeline: the LLM produces a complete workflow graph directly from the enriched SOP text, followed by chunk-level patch-based refinement that preserves global coherence. Central to AJent is a four-stage **RAG-augmented preprocessing** pipeline --- semantic chunking, FAISS-based cross-chunk retrieval, graded context condensation, and entity resolution --- that standardizes terminology and resolves cross-section dependencies before conversion. Post-conversion, a **self-refinement loop** iterates through topological validation, LLM-based completeness and context audits, decision-edge triplet verification, and deterministic schema enforcement until the graph stabilizes. We further contribute a **hybrid graph comparison** framework that combines embedding-based candidate selection with LLM-verified semantic matching to evaluate auto-generated graphs against human-curated baselines with SOP-grounded attribution. AJent's architecture is designed for production-length SOPs (10,000+ words) where prior methods degrade, and includes checkpoint-based fault tolerance for reliable processing under API rate limits. We propose evaluation on business process, onboarding, and compliance SOPs, comparing against bottom-up (PADME-style), edge-vertex (Agent-S-style), and direct prompting baselines, adopting established metrics from ProcBench [15], PADME [1], and the Universal Prompting framework [12] alongside novel SOP-grounding metrics.

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

The remainder of this paper is organized as follows: Section 2 surveys related work across five research threads. Section 3 details the AJent methodology. Section 4 provides a dimension-by-dimension comparative analysis with contemporary systems, grounding each comparison in published results. Section 5 analyzes AJent's self-reflection architecture and its relationship to Self-Refine [10] and Reflexion [11]. Section 6 presents the multi-layer hallucination mitigation strategy. Section 7 discusses token efficiency. Section 8 describes the experimental setup with evaluation metrics adopted from established benchmarks. Section 9 covers key design decisions, and Sections 10--11 conclude with limitations and future work.

---

## 2. Related Work

### 2.1 LLM-Based Procedure Structuring

Learning to autonomously structure long-horizon procedures from natural language has attracted significant recent attention. **PADME** [1] introduces a two-phase Teach-Execute framework where a structuring agent converts free-form SOPs into decision graphs through iterative document chunking and sequential node extraction with context carryover. PADME formalizes the decision graph as $G = (V, E)$ where each node $v \in V$ is a typed operator with signature $f_v: \mathcal{X}_v \rightarrow \mathcal{Y}_v$, and edges encode input-output dependencies. Nodes are categorized into five classes: HumanInput, InfoProcessing, InfoExtraction, Knowledge, and Decision. PADME achieves state-of-the-art on four benchmarks (ALFWorld, ScienceWorld, Recipe, Business Process), with Final Match (FM) scores of 0.87 on Recipes and 0.74 on Business Processes, outperforming ReAct [5], Chain-of-Thought [17], and SPRING [8] baselines.

**SOPRAG** [2] takes a top-down approach, introducing a Procedure Card layer that functions as a search-space pruner. Rather than flat chunking, SOPRAG replaces traditional retrieval with specialized Entity, Causal, and Flow graph experts in a Mixture-of-Experts architecture, with an LLM-guided gating mechanism that aligns retrieval outcomes with operator intent.

**Agent-S** [3] approaches SOP automation from the execution side, categorizing SOP steps as either user interactions or API calls and using a Global Action Repository (GAR) with execution memory. The agent dynamically chooses actions, interacts with environments, and uses feedback to inform next decisions.

While these approaches advance the field, none addresses the compound challenges of long-document processing: cross-section dependency resolution, terminology standardization, multi-pass quality assurance, and fault-tolerant checkpoint recovery.

### 2.2 Process Information Extraction

Neuberger et al. [12] systematically investigate LLM-based extraction of process model information, proposing a **Universal Prompting Strategy** built on three pillars: quantity of examples, definitional specificity, and format rigor. Their evaluation across eight LLMs and three datasets demonstrates an 8% F1 improvement over prior ML methods for detecting activities, actors, and relations. Crucially, they convert extracted information into actual process models using a heuristic algorithm, validating that LLM outputs can feed downstream automation. Their work establishes that careful prompt engineering is "universally applicable" across models --- a finding that motivates AJent's detailed pattern guide (Table 1).

**ProcBench** [15] introduces comprehensive metrics for evaluating multi-step procedure adherence: Prefix Match Length (PML), Prefix Accuracy (PA), Sequential Match (SM), and Final Match (FM). These metrics capture both partial and complete correctness, measuring how long a method stays aligned before drifting (PML/PA) and whether the final outcome is correct (FM). We adopt these metrics in our evaluation framework (Section 8).

### 2.3 Business Process Modeling with LLMs

The **BPMN Assistant** [4] addresses a key representation question: should LLMs generate business process models as XML or through an intermediate format? By introducing a JSON-based proxy representation with atomic edit operations, the BPMN Assistant achieves approximately **43% reduction in generation latency** and **75% reduction in output tokens** compared to direct XML generation, while maintaining higher or equivalent success rates across GPT-5.1, Claude 4.5 Sonnet, and DeepSeek V3. This empirically validates the advantage of structured JSON over verbose markup formats --- a design choice AJent shares.

**Planetarium** [6] provides a rigorous benchmark for text-to-PDDL (Planning Domain Definition Language) translation, revealing a critical gap: GPT-4o achieves **96.1% syntactic correctness** but only **24.8% semantic correctness** when generating structured planning code. This 72-point gap between syntactic parsing and semantic validity directly motivates AJent's multi-layer validation architecture, where Pydantic schema enforcement handles syntactic correctness and the self-refinement loop targets semantic correctness.

**NL2ProcessOps** [16] explores LLM-guided code generation for business process execution, generating Python APIs from SOP descriptions. This work demonstrates the viability of code-based (rather than diagram-based) process representations, aligning with AJent's use of typed JSON with Pydantic validation.

### 2.4 Self-Refinement in LLM Systems

**Self-Refine** [10] introduces a three-stage iterative loop (generate $\rightarrow$ feedback $\rightarrow$ refine) achieving approximately **20% absolute improvement** across seven tasks. The approach requires no additional training --- it operates purely at inference time using a single LLM. However, Self-Refine uses a single undifferentiated feedback signal, which can produce vague or unhelpful critiques.

**Reflexion** [11] extends self-improvement with an episodic memory of linguistic reflections, achieving **91% pass@1 on HumanEval** (surpassing GPT-4's 80%). The key insight is that agents can learn from prior attempts via stored natural-language reflections without weight updates. AJent adapts both paradigms --- decomposing Self-Refine's monolithic feedback into four orthogonal signals, and implementing Reflexion's memory as structured issue/triplet caches (Section 5).

### 2.5 Graph-Based LLM Reasoning

**AgentKit** [7] requires users to manually decompose tasks into node-level graphs with per-node prompts. **SPRING** [8] parses game documentation into a fixed DAG of interdependent questions traversed in topological order. Both require manual graph construction, limiting scalability. AJent automatically constructs graphs from free-form text, eliminating manual engineering while achieving structured reasoning benefits.

### 2.6 Hallucination in LLMs

Huang et al. [13] provide a comprehensive taxonomy of LLM hallucination, identifying it as the generation of "plausible yet nonfactual content." They survey detection methods, mitigation strategies, and note that retrieval-augmented approaches have "current limitations" in combating hallucination. AJent's defense-in-depth strategy (Section 6) addresses this through six independent layers, combining RAG grounding with structured output constraints, confidence annotation, and multi-signal verification.

---

## 3. AJent Methodology

AJent transforms unstructured SOP documents into executable workflow graphs through a three-phase pipeline: **Preprocessing** (Section 3.1), **Conversion** (Section 3.2), and **Refinement** (Section 3.3). Figure 1 illustrates the end-to-end architecture.

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

### 3.1 Graph Representation

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

Unlike XML-based representations (BPMN [4]) or free-form text outlines, this strict JSON schema enables both deterministic validation and LLM-based structured output via `with_structured_output()`, ensuring that every LLM call produces schema-compliant nodes. The BPMN Assistant [4] empirically validates this format choice: JSON intermediate representations achieve 75% fewer output tokens and 43% lower latency than direct XML generation.

### 3.2 Phase 1: RAG-Augmented Preprocessing

Long SOPs suffer from three problems that must be resolved *before* conversion: (a) cross-section dependencies are implicit, (b) terminology varies across sections, and (c) the full document may exceed the LLM's effective context window. AJent addresses these through a four-node LangGraph pipeline.

#### 3.2.1 Semantic Chunking

An LLM segments the raw SOP into semantically coherent chunks at natural process boundaries --- phase transitions, department handoffs, or decision-branch entry points. Unlike fixed-length chunking, this preserves logical units intact: a multi-step sub-procedure stays in one chunk, and closely related decision branches are not split mid-logic.

#### 3.2.2 FAISS Indexing

All chunks are embedded using a local embedding model (BAAI/bge-base-en-v1.5) and indexed in a FAISS vector store. This enables efficient cross-chunk retrieval without API cost, as the embedding model runs entirely on CPU.

#### 3.2.3 RAG Enrichment

For each chunk, a Dependency Analyst agent identifies dangling references --- mentions of other sections, teams, systems, or processes not fully defined within the chunk. For each reference, it generates a search query, retrieves the top-$k$ most similar chunks from the FAISS index (excluding the chunk itself), and a Relevance Grader agent filters retrievals for strict relevance. Accepted retrievals are condensed by a Context Condensation agent into a concise factual note (2--4 sentences) capturing only cross-chunk dependencies, upstream triggers, downstream handoffs, and shared entities. This condensed context is appended to the chunk for the converter's consumption.

This three-agent retrieval pipeline (generate queries $\rightarrow$ retrieve and grade $\rightarrow$ condense) avoids the common failure mode of RAG systems where noisy retrievals degrade downstream quality [13]. The grading step rejects irrelevant retrievals, and the condensation step prevents context bloat.

#### 3.2.4 Entity Resolution

A Terminology Standardization agent analyzes all chunks simultaneously, identifies groups of synonymous terms (e.g., "CBRD team", "Credit Bureau Reporting Disputes team", "the CBRD"), and maps them to canonical forms. Aliases are then replaced with canonical names in both the chunk text and any retrieved context. This ensures the converter sees consistent terminology, preventing duplicate nodes for the same entity described with different names --- a prevalent issue in production SOPs authored by multiple writers.

**Content-Based Caching.** Preprocessing results are keyed by the SHA-256 hash of the input document and persisted to disk. On subsequent runs with the same document, the cached chunks, enriched chunks, and entity maps are loaded directly, with only the FAISS vector store rebuilt from cached chunks (as FAISS objects are not serializable). This eliminates redundant LLM calls during iterative development.

### 3.3 Phase 2: Graph-First Conversion

The core insight of AJent's conversion strategy is that **producing the graph directly avoids the lossy intermediates** of prior approaches. Text outlines lose temporal dependencies. Entity lists lose contextual flow. Procedure cards lose decision scope. By prompting the LLM to output structured JSON graph nodes from the start, AJent preserves all structural information.

#### 3.3.1 Step 1: Full SOP to Initial Graph

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

*Table 1: AJent pattern guide for graph-first conversion. These patterns operationalize the "definitional specificity" and "format rigor" principles identified by Neuberger et al. [12] as key to effective LLM-based process extraction.*

The LLM produces a structured `InitialGraph` output containing a reasoning chain and a list of `WorkflowNode` objects. The graph undergoes immediate schema validation and topological checking (orphan detection, dead-end detection, broken link detection) before proceeding.

#### 3.3.2 Step 2: Chunk-by-Chunk Patch Refinement

While the initial graph captures the global structure, it may miss fine-grained details within individual sections. AJent addresses this through a multi-pass patch refinement:

For each enriched chunk (now including its cross-reference context), a Graph Refinement Engineer agent performs a line-by-line audit:

1. **Extract**: Read the SOP chunk sentence by sentence, listing every discrete detail (actions, decisions, systems, roles, field names, thresholds, temporal dependencies, exception paths).
2. **Match**: For each extracted detail, check whether the current graph covers it (COVERED), partially covers it (PARTIAL), or misses it (MISSING).
3. **Patch**: For every MISSING or PARTIAL detail, produce add/modify/remove operations as a structured `GraphPatch`.

Each patch is validated for schema compliance, topological integrity, and sanity (rejecting patches that shrink the graph by $>30\%$ or remove the start node). Two passes over all chunks are performed; if the second pass produces zero changes, the graph is declared stable.

**Checkpoint-Based Fault Tolerance.** After every chunk, the current graph state is checkpointed to disk. If the pipeline is interrupted (API rate limit, server error, network failure), it can resume from the last checkpoint via the `--resume` flag. The `safe_invoke()` wrapper implements automatic retry with backoff (up to 2 retries with 500-second wait) for rate-limit and server errors, and halts immediately on authentication errors. This makes AJent suitable for processing long SOPs that require dozens of LLM calls over extended periods.

### 3.4 Phase 3: Self-Refinement Loop

After conversion, AJent enters a LangGraph-orchestrated analyse-refine loop that iterates until the graph passes all quality checks or reaches a maximum iteration count.

#### 3.4.1 Analysis (Multi-Signal Quality Assessment)

Each analysis iteration evaluates the graph through four complementary signals:

1. **Topological Check** (deterministic, zero-cost): Pure Python scan for orphan nodes (unreferenced non-start nodes), dead ends (non-terminal nodes with no outgoing edges), and broken links (references to undefined node IDs).

2. **Completeness Check** (LLM-based): A Process Quality Auditor breaks the SOP into its natural sections and, for each section, finds the graph nodes that cover it. Every section is marked as `covered` or `missing`. The check explicitly avoids false positives for correct modeling patterns (multi-way decisions as binary chains, loops as back-edges, field lists as single nodes).

3. **Context Adjacency Check** (LLM-based): A Process Flow Analyst verifies that connected nodes are logically adjacent --- that transitions make sense given the SOP, no intermediate steps are skipped, and edge directions are correct. Back-edges for loops and cross-system multi-node sequences are recognized as valid.

4. **Triplet Verification** (LLM-based): All decision-node triplets $(source, edge\_label, target)$ are extracted and verified against the SOP. Invalid triplets (wrong branch target, incorrect condition, mismatched answer) are flagged. Previously verified triplets are cached to avoid redundant LLM calls across iterations.

The graph is declared complete only if all four signals pass simultaneously.

#### 3.4.2 Refinement (Issue-by-Issue Patch Resolution)

Each flagged issue is resolved individually through a Graph Patch Resolver that sees the **full graph** plus the **full SOP** and produces a coordinated `GraphPatch`. This full-context approach, as opposed to local 2-hop-window repairs, enables the resolver to:

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

### 3.5 Graph Comparison Framework

To evaluate auto-generated graphs against human-curated baselines, AJent includes a **hybrid comparison framework** that combines embedding retrieval with LLM-verified semantic matching.

#### 3.5.1 Node Alignment (Hybrid: Embeddings + LLM)

1. **Phase 1 --- Embedding Retrieval**: All node texts from both graphs are embedded using bge-base-en-v1.5. A cosine similarity matrix is computed, and for each human node, the top-$K$ most similar auto nodes are selected as candidates.

2. **Phase 2 --- LLM Matching**: Candidates are sent to an LLM in batches. The LLM performs semantic matching, allowing one-to-many alignment (a single human node may map to multiple granular auto nodes). Matching is based on meaning, not wording, with type-awareness (decisions rarely match instructions).

#### 3.5.2 Edge Comparison (Deterministic + LLM)

1. **Phase 1 --- Deterministic Path Check**: For each human edge $(h_{src}, h_{tgt})$, the system checks whether the aligned auto nodes have a direct edge or a short path (up to 3 hops) between them.

2. **Phase 2 --- LLM Validation**: Unmatched *decision* edges are sent to an LLM with subgraph context from both graphs. The LLM determines whether the auto graph semantically preserves the flow, even if the structure differs. Non-decision edges are classified structurally only, as they are typically trivial.

#### 3.5.3 SOP Grounding

For nodes that exist in only one graph, a grounding check determines whether the content is actually present in the SOP document:

- **Auto-only nodes** grounded in the SOP = **Auto Advantages** (content the human missed)
- **Auto-only nodes** not grounded = **Hallucinations**
- **Human-only nodes** grounded in the SOP = **True Gaps** (content AJent missed)
- **Human-only nodes** not grounded = **Human Extrapolations** (added beyond the SOP)

This four-way attribution provides nuanced evaluation beyond simple precision/recall.

---

## 4. Comparative Analysis with Contemporary Approaches

This section provides a detailed comparison between AJent and the systems whose ideas informed its design. We organize the comparison across parsing strategy, node format, and evaluation approach --- grounding each comparison in published empirical results where available.

### 4.1 SOP Parsing Strategies: A Taxonomy

The fundamental architectural choice in any SOP-to-graph system is *how* the procedural text is decomposed and restructured. We identify three distinct paradigms in the current literature, each with characteristic strengths and failure modes, and position AJent's graph-first approach as a fourth.

#### 4.1.1 Bottom-Up Parsing (PADME [1])

PADME introduces the concept of **iteratively chunking documents and sequentially extracting information**, retaining context from previously processed chunks to build a continuous understanding. The structuring agent segments the procedure into units $S_k$ ($k = 1, \ldots, m$), converts each segment into a local decision subgraph, and merges subgraphs into a global DAG $G = (V, E)$.

PADME demonstrates that this approach works well for short-to-medium procedures: on ALFWorld (5--15 steps), it achieves SM=0.62 and FM=0.69; on ScienceWorld (20--50 steps), PA=0.44 and FM=0.74. However, the sequential chunk-then-merge pipeline faces compounding problems as document length grows:

1. **Cross-section dependency loss.** Each chunk is processed with only a summary of prior chunks' node IDs as context. In a 30-chunk SOP where chunk 25 references a decision made in chunk 3, the carryover context has been compressed through 22 intermediate summaries, losing specifics.
2. **Merge-stage fragility.** The final merge agent must deduplicate, fix cross-chunk edges, and ensure global connectivity across all independently-produced subgraphs --- itself a complex reasoning task that scales with chunk count.
3. **Duplicate node creation.** Without entity resolution, sections written by different authors using variant terminology (e.g., "CBRD team" vs "Credit Bureau Reporting Disputes") produce duplicate sub-procedures.

AJent addresses these by (a) producing the global graph structure in a single call before chunk-level refinement, (b) performing entity resolution *before* conversion to prevent duplicates, and (c) using intra-document RAG to make cross-section dependencies explicit.

#### 4.1.2 Top-Down Parsing (SOPRAG [2])

SOPRAG introduces a **Procedure Card** layer that functions as a search-space pruner, extracting overarching goals and major decision points first before processing micro-steps. The framework uses specialized Entity, Causal, and Flow graph experts in a Mixture-of-Experts architecture, replacing flat chunking with structured multi-view retrieval.

*Strengths:* The top-down approach captures the global procedure skeleton before filling in detail. The Procedure Card eliminates computational noise by constraining subsequent processing scope.

*Limitations:* The Procedure Card is optimized for *retrieval* (selecting the right SOP from a corpus) rather than *conversion* (transforming an SOP into an executable graph). The macro-skeleton may impose rigid structure that doesn't accommodate fine-grained conditional logic within sections.

AJent incorporates the top-down insight differently: the graph-first conversion step implicitly produces a macro-skeleton (the initial graph's decision topology), which is then refined with chunk-level detail in the patch stage. This achieves the top-down benefit without a separate skeleton extraction step.

#### 4.1.3 Edge-Vertex Parsing (Agent-S [3], Universal Prompting [12])

Agent-S and the Universal Prompting Strategy [12] employ a **two-stage decomposition**: first extract all entities/activities, then map relationships/edges. Neuberger et al. [12] demonstrate that carefully engineered prompts can achieve 8% F1 improvement over ML baselines for extracting activities, actors, and relations from process descriptions.

*Limitations:*
1. **Contextual flow loss.** When extracting entities in isolation, the LLM loses the procedural context that determines edge semantics. "Verify account status" and "Update account status" may be identified as entities, but their ordering and dependency are ambiguous without the surrounding procedural flow.
2. **Combinatorial edge mapping.** In the second stage, the LLM must reason over all possible edges among $N$ entities --- an $O(N^2)$ task. For long SOPs producing 100+ entities, this exceeds practical LLM reasoning capacity.
3. **No iterative refinement.** Both Agent-S and Universal Prompting produce the graph in a single pass with no quality feedback loop.

#### 4.1.4 Graph-First Parsing (AJent)

AJent's approach is a **synthesis** of the above paradigms: it captures global structure like top-down methods, processes detail at the chunk level like bottom-up methods, and uses structured output like edge-vertex methods --- but without the lossy intermediates of any individual approach. The key insight is that modern LLMs with structured output capabilities can produce well-formed JSON graphs of 50--200 nodes in a single call, making the graph itself the natural "first output" rather than an intermediate text representation.

### 4.2 Feature Comparison Matrix

| Dimension | PADME [1] | SOPRAG [2] | Agent-S [3] | BPMN Asst. [4] | Universal Prompting [12] | **AJent** |
|-----------|:---------:|:----------:|:-----------:|:---------:|:----------------:|:---------:|
| **Parsing strategy** | Bottom-up chunk-merge | Top-down Procedure Card | Execution agent + GAR | LLM function calling | Two-stage entity-relation | Graph-first + patch |
| **Intermediate repr.** | Subgraph per chunk | Macro skeleton | Action labels | JSON edit operations | Entity-relation pairs | Direct JSON graph |
| **Node types** | 5 (HumanInput, InfoProc, InfoExtract, Knowledge, Decision) | Multi-view graph | Action / User Prompt | BPMN elements | Activity / Actor / Relation | 4 (instruction, question, terminal, reference) |
| **Schema enforcement** | Informal categories | Not described | GAR matching | XML validation | Heuristic post-processing | Pydantic validators at every stage |
| **Entity resolution** | No | No | No | No | No | Yes (LLM-based, pre-conversion) |
| **Cross-section RAG** | No | Graph expert retrieval | External knowledge source | No | No | Yes (FAISS + graded retrieval) |
| **Self-refinement** | No | No | Retry on failure | No | No | Yes (multi-signal analyse-refine loop) |
| **Confidence labels** | No | No | No | No | No | Yes (high/medium/low per edge) |
| **Triplet verification** | No | No | No | No | No | Yes (decision-edge validation) |
| **Fault tolerance** | Not described | Not described | Dynamic retry | Not described | Not described | Checkpoint-resume with retry logic |
| **Target doc length** | Short-medium | Industrial SOPs | Customer care SOPs | Business processes | Process descriptions | Long-horizon (10,000+ words) |
| **Primary evaluation** | PML, PA, SM, FM [15] | Retrieval accuracy | Task completion | Success rate, latency, tokens | Entity/Relation F1 | Hybrid graph comparison + SOP grounding |

*Table 2: Feature-by-feature comparison of AJent with contemporary SOP/process automation systems.*

### 4.3 Node Format: Graph-Based vs Code-Based Representations

The choice of node representation has downstream implications for validation, execution, and the syntactic-semantic correctness gap revealed by Planetarium [6].

**Graph-Based Node Formats** (PADME, SOPRAG) represent nodes with semantic categories. PADME's five categories describe *what kind of step* a node represents, enabling role-based execution routing. However, the categories are defined informally --- there is no compile-time enforcement that a Decision node actually has branching options, or that an InfoExtraction node specifies a data source.

**Code-Based Node Formats** (AJent, BPMN Assistant) impose strict structural constraints. AJent's Pydantic-validated schema guarantees:
- Question nodes *must* have `options` with exactly two keys (Yes/No) --- violations are caught at parse time, not during execution.
- Instruction nodes *must* have a `next` pointer --- dangling instructions are impossible.
- Terminal nodes *cannot* have successors --- accidental continuation is prevented.

Planetarium [6] quantifies why this matters: GPT-4o achieves 96.1% syntactic correctness when generating PDDL but only **24.8% semantic correctness**. The 72-point gap shows that syntactic validity is necessary but radically insufficient. AJent addresses both sides: Pydantic ensures syntactic correctness (analogous to Planetarium's 96%), while the refinement loop targets semantic correctness (analogous to their 24% --- the gap AJent aims to close).

### 4.4 Evaluation Methodology Comparison

Different systems adopt fundamentally different evaluation approaches, reflecting their different goals:

| System | Evaluation Approach | What It Measures | Limitation |
|--------|-------------------|-----------------|------------|
| PADME [1] | End-to-end task execution (PML, PA, SM, FM) | Does the graph lead to correct actions? | Requires executable environment; doesn't evaluate graph structure directly |
| Universal Prompting [12] | Entity/Relation F1 against gold annotations | Are the right activities and relations extracted? | Doesn't evaluate graph connectivity or decision logic |
| BPMN Assistant [4] | Success rate + latency + token count | Does the edit operation succeed? How efficient is it? | Measures individual edits, not full graph quality |
| Planetarium [6] | Syntactic parsing + solvability + semantic equivalence | Is the generated PDDL correct? | Limited to planning domains |
| **AJent** | Hybrid node/edge alignment + SOP grounding | Is the graph structurally and semantically faithful to the SOP? | Requires human gold-standard graphs |

*Table 3: Evaluation methodology comparison across contemporary systems.*

AJent's evaluation framework bridges these approaches by combining structural metrics (node/edge F1, analogous to Universal Prompting's entity/relation F1) with execution-oriented metrics (adapted PML/PA/SM/FM from PADME [1]/ProcBench [15]) and a novel SOP-grounding analysis that no prior system provides.

---

## 5. Self-Reflection Architecture: Theoretical Grounding

AJent's self-refinement loop draws from and extends two foundational paradigms in LLM self-improvement: **Self-Refine** [10] and **Reflexion** [11]. This section analyzes how AJent adapts these general-purpose self-reflection mechanisms for graph construction, and argues that the structured nature of graph outputs enables stronger forms of self-correction than are possible in free-text generation.

### 5.1 From Self-Refine to Multi-Signal Analysis

Madaan et al. [10] introduce Self-Refine as a three-stage loop: **generate** $\rightarrow$ **feedback** $\rightarrow$ **refine**, achieving approximately 20% absolute improvement across seven tasks. However, Self-Refine uses a *single, undifferentiated* feedback signal which can miss specific failure modes or produce vague feedback.

AJent decomposes Self-Refine's monolithic feedback into **four orthogonal quality signals**, each targeting a distinct failure mode:

| Signal | What it catches | Implementation | Cost |
|--------|----------------|----------------|------|
| Topological check | Orphans, dead ends, broken links | Pure Python (deterministic) | Zero (no LLM call) |
| Completeness check | Missing SOP steps/branches | LLM-based section-by-section audit | 1 LLM call |
| Context adjacency check | Illogical node connections | LLM-based edge evaluation | 1 LLM call |
| Triplet verification | Invalid decision-branch targets | LLM-based triplet batch validation | 1 LLM call per batch |

*Table 4: AJent's four analysis signals vs Self-Refine's single feedback.*

This decomposition is more effective than a single critique because:
1. **Deterministic signals cannot hallucinate.** The topological check is pure Python --- it *guarantees* that reported orphans are actually orphan nodes, unlike an LLM critique that might hallucinate issues.
2. **Specialized prompts produce targeted feedback.** The completeness check is specifically engineered to audit SOP coverage section-by-section, producing actionable gap descriptions rather than vague "the graph could be more complete" feedback.
3. **Independent signals prevent masking.** A single critique might note "the graph looks mostly good" while missing a broken link. Independent signals ensure that topological validity is checked *regardless* of what the LLM thinks about completeness.

### 5.2 From Reflexion to Persistent Refinement State

Shinn et al. [11] introduce Reflexion, where agents maintain an **episodic memory** of linguistic reflections from prior attempts, achieving 91% pass@1 on HumanEval. AJent adapts this principle through two structured mechanisms:

1. **Resolved-issue tracking.** AJent maintains a set of issues resolved in prior iterations (`resolved_issues`). When the analyser flags an already-addressed issue, it is skipped --- preventing oscillation between fix-and-revert cycles. This is analogous to Reflexion's memory preventing repeated mistakes, but applied to graph patches with exact string signatures rather than imprecise natural-language reflections.

2. **Verified-triplet caching.** Decision-edge triplets verified as valid in prior iterations are cached (`verified_triplets`). Subsequent iterations skip these triplets, reducing LLM calls. New triplets introduced by patches are verified in the next iteration.

### 5.3 Why Graph Outputs Enable Stronger Self-Correction

A fundamental advantage over general-purpose Self-Refine/Reflexion is that **workflow graphs are verifiable structures**. Free-text outputs can only be critiqued heuristically, but graphs admit deterministic invariant checks:

- **Connectivity invariants**: Every non-terminal node must have at least one outgoing edge. Every non-start node must have at least one incoming edge. Checkable in $O(|V| + |E|)$ with no LLM.
- **Type invariants**: Question nodes must have exactly two options. Terminal nodes must have zero successors. Enforced by Pydantic validators.
- **Referential integrity**: Every `next` and `options` value must reference an existing node ID. Broken references are detected deterministically.

These serve as **hard constraints** that the refinement loop *guarantees*, regardless of LLM quality. The LLM-based checks provide *soft signals* about semantic quality. This hybrid of hard deterministic constraints and soft LLM-based signals is what makes AJent's refinement more robust than pure self-reflection.

### 5.4 Convergence Guarantees

AJent's refinement loop converges through:
1. **Monotonic issue resolution**: The resolved set grows monotonically; the set of new issues shrinks.
2. **Rollback guards**: Patches that shrink the graph by $>30\%$ or remove the start node are rolled back.
3. **Stabilization detection**: Zero-change passes terminate the loop early.
4. **Hard iteration cap**: Maximum 15 iterations prevents infinite loops.

This is a stronger convergence guarantee than Self-Refine (no formal stopping criterion) or Reflexion (relies on task success as termination signal).

---

## 6. Hallucination Mitigation: A Multi-Layer Defense

Hallucination --- the generation of plausible but factually unsupported content [13] --- is a critical failure mode in SOP-to-graph conversion. In regulated domains (finance, healthcare, compliance), hallucinated steps in an automated workflow can lead to incorrect actions with material consequences. AJent implements a **defense-in-depth** strategy with six layers, each targeting a different hallucination vector.

### 6.1 Taxonomy of Hallucination in Graph Generation

We identify four categories of hallucination specific to SOP-to-graph conversion:

| Category | Description | Example | Risk |
|----------|-------------|---------|:----:|
| **Node hallucination** | Generating a step not in the SOP | Adding "Notify compliance officer" when the SOP doesn't mention compliance | High |
| **Edge hallucination** | Creating a dependency not implied by the SOP | Connecting "Print report" to "Submit to regulator" when independent | Medium |
| **Detail hallucination** | Inventing specifics not in the source | Adding "within 24 hours" when the SOP specifies no timeframe | Medium |
| **Structural hallucination** | Imposing unsupported graph structure | Creating a decision branch where the SOP describes a linear sequence | High |

*Table 5: Taxonomy of hallucination types in SOP-to-graph conversion.*

### 6.2 The Six Layers

**Layer 1: RAG-Grounded Preprocessing.** By enriching each chunk with cross-reference context retrieved from FAISS and validated by the grading agent, AJent ensures the LLM has access to all relevant SOP information. This reduces **omission-driven hallucination** --- where the LLM invents content to fill gaps caused by missing cross-section context. Entity resolution further prevents the LLM from treating terminological variants as distinct entities, which would manifest as duplicate nodes or spurious branches.

**Layer 2: Structured Output Constraints.** Requiring typed `WorkflowNode` objects via `with_structured_output()` constrains the output space. The LLM cannot produce undefined types, question nodes without options, instructions without successors, or free-form edge labels. These constraints eliminate an entire class of structural hallucinations possible with free-text graph descriptions.

**Layer 3: Confidence-Annotated Edges.** AJent requires the LLM to self-assess each node's outgoing edges as high/medium/low confidence. Research on LLM calibration [14] shows that requiring explicit confidence ratings improves the correlation between stated confidence and actual accuracy. Low-confidence edges mark the boundary between extraction and hallucination, enabling targeted human review.

**Layer 4: Multi-Signal Refinement Verification.** The self-refinement loop (Section 3.4) acts as a post-hoc hallucination detector. The completeness check implicitly verifies that existing nodes correspond to SOP sections. The context adjacency check flags hallucinated edges between unrelated steps. Triplet verification catches hallucinated branch targets.

**Layer 5: SOP Grounding in Evaluation.** The graph comparison framework (Section 3.5) provides a final hallucination audit. Unmatched auto-only nodes are checked against the SOP via embedding retrieval and LLM verification, producing a **hallucination rate** metric that reveals systematic patterns.

**Layer 6: Deterministic Schema Invariants.** The `SchemaValidator` applies deterministic fixes incapable of introducing hallucination: converting optionless questions to instructions, nullifying terminal successors, flagging missing instruction edges. Deterministic layers provide guarantees that LLM-based layers cannot.

### 6.3 Comparison with Existing Hallucination Mitigation

Huang et al. [13] note that retrieval-augmented approaches have "current limitations" in combating hallucination, primarily because retrieved context may itself be noisy or irrelevant. AJent's graded retrieval pipeline directly addresses this: the relevance grader rejects noisy retrievals, and the condensation agent compresses accepted context into information-dense notes. The combination of RAG (Layer 1) with structured output (Layer 2), self-assessment (Layer 3), and multi-signal verification (Layer 4) provides a more comprehensive defense than RAG alone.

Planetarium's [6] finding that syntactic correctness (96%) vastly exceeds semantic correctness (24%) maps directly to AJent's architecture: Layers 2 and 6 handle syntactic correctness (schema validity), while Layers 1, 3, 4, and 5 target the harder problem of semantic correctness (SOP faithfulness).

---

## 7. Token Efficiency and the Chunking-First Architecture

A practical consideration for production deployment is computational cost. AJent's architecture minimizes total token consumption while maximizing information density per LLM call.

### 7.1 Chunking Before Conversion Saves Tokens

A naive approach sends the full document to the LLM for every processing step. AJent's chunk-level refinement reduces per-call context:

| Approach | Step 1 Tokens | Step 2 Tokens (per chunk) | Total for 10 Chunks |
|----------|:------------:|:------------------------:|:-------------------:|
| Full-document-every-call | 10K | 10K | 10K + 10 $\times$ 10K = 110K |
| AJent (chunk-level refinement) | 10K (initial graph) | ~1K (chunk) + ~3K (graph) | 10K + 10 $\times$ 4K = 50K |

*Table 6: Estimated input token comparison (10,000-word SOP, 10 chunks).*

The BPMN Assistant [4] provides empirical validation of this efficiency principle: their JSON intermediate representation achieves 75% fewer output tokens than direct XML generation. AJent's approach similarly benefits from compact JSON over verbose alternatives.

### 7.2 Entity Resolution as Token Saver

Entity resolution has a secondary benefit beyond semantic correctness: **token reduction through alias elimination**. When the SOP uses five names for the same team, each mention consumes tokens. After resolution, the enriched text is shorter and more information-dense. More importantly, entity resolution prevents duplicate nodes that would waste tokens in every subsequent pipeline stage.

### 7.3 Dual Graph Representations

AJent uses two representations optimized for different consumers:
- **Full JSON** (for the patcher): Complete node details. Used when the LLM needs to produce precise patches.
- **Compact text** (for the analyser): ID, type, text, and connections only --- approximately **60--70% smaller** than full JSON. Used for completeness and context checks.

This reduces tokens in analysis calls (which run every iteration) while preserving full detail for refinement calls (which run only when issues are found).

---

## 8. Experimental Setup and Proposed Evaluation

### 8.1 Baselines

We compare AJent against four conversion approaches, all using the same underlying LLM:

1. **Bottom-Up (PADME-style)** [1]: Chunk $\rightarrow$ per-chunk extraction with context carryover $\rightarrow$ merge. Mirrors PADME's Teach phase.
2. **Edge-Vertex (Agent-S-style)** [3]: Extract all entities without connections $\rightarrow$ map edges in second pass.
3. **Direct Prompting (Zero-Shot)**: Full SOP to LLM with graph-generation prompt, no preprocessing or refinement.
4. **ReAct-Style** [5]: Interleaving reasoning-and-acting, generating nodes incrementally with reflection on partial graph.

### 8.2 Datasets

| Dataset | Description | Avg. Length (words) | Complexity |
|---------|-------------|--------------------:|-----------|
| Business Process SOPs | Financial operations, claims, disputes | 5,000--15,000 | High branching, cross-system flows |
| Onboarding SOPs | Employee/vendor onboarding | 1,000--5,000 | Moderate branching, role-based |
| Compliance SOPs | Regulatory, audit procedures | 3,000--10,000 | Heavy conditionals, external references |

*Table 7: Proposed evaluation datasets.*

For each SOP, a human domain expert creates a gold-standard workflow graph. Inter-annotator agreement is measured on a subset.

### 8.3 Adopted Evaluation Metrics

We adopt metrics from three established frameworks, covering complementary dimensions of graph quality:

#### 8.3.1 Structural Metrics (adapted from Universal Prompting [12])

Neuberger et al. [12] evaluate process extraction via entity F1 and relation F1. We adapt these to graph comparison:

| Metric | Origin | Our Adaptation |
|--------|--------|---------------|
| Entity F1 | Universal Prompting [12] | **Node F1**: Harmonic mean of node precision (fraction of auto nodes aligned to human) and recall (fraction of human nodes covered by auto) |
| Relation F1 | Universal Prompting [12] | **Edge F1**: Harmonic mean of edge precision and recall, using hybrid deterministic + LLM matching |
| Type Accuracy | Novel | Fraction of aligned node pairs where types agree (instruction vs question vs terminal) |

#### 8.3.2 Procedural Fidelity Metrics (from ProcBench [15] / PADME [1])

PADME adopts ProcBench's metrics for measuring how faithfully a system follows procedural logic. We adapt these for graph-to-graph comparison by treating the topological ordering of the auto graph as the predicted sequence and the human graph's ordering as the target:

| Metric | Definition | What It Reveals |
|--------|-----------|-----------------|
| **Prefix Match Length (PML)** | Longest prefix of the auto graph's topological order matching the human graph | How far the auto graph stays aligned before diverging |
| **Prefix Accuracy (PA)** | PML normalized by max(auto length, human length) | Length-normalized alignment measure |
| **Sequential Match (SM)** | Binary: 1 if PA = 1.0 (exact topological match) | Whether the graph structures are identical in ordering |
| **Final Match (FM)** | Binary: 1 if the terminal node semantics match | Whether both graphs reach the same conclusion |

*Table 8: Procedural fidelity metrics adapted from ProcBench [15] and PADME [1].*

Note: PADME reports PML=2.64, PA=0.69, SM=0.64, FM=0.87 on Recipes and PML=5.15, PA=0.71, SM=0.30, FM=0.74 on Business Processes. These establish the current state-of-the-art benchmarks.

#### 8.3.3 Efficiency Metrics (from BPMN Assistant [4])

Following the BPMN Assistant's efficiency evaluation, we report:

| Metric | Definition |
|--------|-----------|
| **Total LLM Calls** | Number of API calls across the full pipeline |
| **Total Output Tokens** | Aggregate tokens generated |
| **Wall-Clock Time** | End-to-end processing time |
| **Success Rate** | Fraction of SOPs producing a valid, connected graph |

#### 8.3.4 Semantic Correctness Metrics (inspired by Planetarium [6])

Planetarium reveals the critical gap between syntactic and semantic correctness. We adapt this decomposition:

| Metric | Definition | Analogy to Planetarium |
|--------|-----------|----------------------|
| **Schema Validity Rate** | Fraction of nodes passing Pydantic validation | Syntactic parsing rate (Planetarium: 96%) |
| **Topological Validity Rate** | Fraction of runs producing a connected, acyclic graph with no orphans/dead-ends | Solvability rate (Planetarium: 94%) |
| **Semantic Correctness** | Structural Score (avg of Node F1 + Edge F1) against human gold standard | Semantic equivalence (Planetarium: 24%) |

*Table 9: Syntactic-semantic decomposition inspired by Planetarium [6].*

The hypothesis: all methods will achieve high schema validity (>90%) but diverge sharply on semantic correctness, paralleling Planetarium's findings. AJent's multi-layer refinement is designed to close this gap.

#### 8.3.5 Novel Metrics: SOP Grounding

No prior system evaluates *why* unmatched nodes exist. AJent's SOP grounding analysis provides:

| Metric | Definition |
|--------|-----------|
| **Auto Advantage Rate** | Fraction of auto-only nodes grounded in the SOP (content the human missed) |
| **Hallucination Rate** | Fraction of auto-only nodes NOT grounded in the SOP |
| **True Gap Rate** | Fraction of human-only nodes grounded in the SOP (content the auto missed) |
| **Granularity Ratio** | Auto nodes per covered human node (measures decomposition granularity) |

### 8.4 Proposed Evaluation Tables

#### 8.4.1 Graph Quality (Structural + Procedural Fidelity)

| Metric | AJent | Bottom-Up | Edge-Vertex | Direct | ReAct |
|--------|:-----:|:---------:|:-----------:|:------:|:-----:|
| Node F1 | | | | | |
| Edge F1 | | | | | |
| Structural Score | | | | | |
| Type Accuracy | | | | | |
| PML | | | | | |
| PA | | | | | |
| SM | | | | | |
| FM | | | | | |

*Table 10: Graph quality comparison (to be filled).*

#### 8.4.2 Syntactic vs Semantic Correctness (Planetarium-style)

| Method | Schema Validity | Topological Validity | Semantic Correctness | Gap (Schema - Semantic) |
|--------|:--------------:|:-------------------:|:--------------------:|:----------------------:|
| AJent | | | | |
| Bottom-Up | | | | |
| Edge-Vertex | | | | |
| Direct | | | | |

*Table 11: Syntactic-semantic decomposition (to be filled). We expect all methods to achieve high schema validity but diverge on semantic correctness, paralleling Planetarium's [6] finding of a 72-point gap.*

#### 8.4.3 Long-Document Scalability

| Document Length (words) | AJent Node F1 | Bottom-Up Node F1 | Edge-Vertex Node F1 | Direct Node F1 |
|------------------------:|:-------------:|:------------------:|:--------------------:|:--------------:|
| 1,000--2,000 | | | | |
| 2,000--5,000 | | | | |
| 5,000--10,000 | | | | |
| 10,000--20,000 | | | | |
| 20,000+ | | | | |

*Table 12: Scalability analysis (to be filled).*

#### 8.4.4 SOP Grounding Analysis

| Method | Auto Advantages | Hallucinations | True Gaps | Hallucination Rate | Auto Advantage Rate |
|--------|:--------------:|:--------------:|:---------:|:-----------------:|:------------------:|
| AJent | | | | | |
| Bottom-Up | | | | | |
| Edge-Vertex | | | | | |
| Direct | | | | | |

*Table 13: SOP grounding analysis (to be filled).*

#### 8.4.5 Efficiency (BPMN-Assistant-style)

| Method | LLM Calls | Output Tokens | Wall-Clock Time (min) | Success Rate |
|--------|:---------:|:------------:|:---------------------:|:------------:|
| AJent | | | | |
| Bottom-Up | | | | |
| Edge-Vertex | | | | |
| Direct | | | | |

*Table 14: Computational efficiency (to be filled).*

#### 8.4.6 Component Ablation

| Configuration | Node F1 | Edge F1 | Structural Score | Hallucination Rate |
|---------------|:-------:|:-------:|:----------------:|:-----------------:|
| Full AJent | | | | |
| -- w/o Entity Resolution | | | | |
| -- w/o RAG Enrichment | | | | |
| -- w/o Patch Refinement (Step 2) | | | | |
| -- w/o Self-Refinement Loop | | | | |
| -- w/o Pattern Guide | | | | |
| -- w/o Confidence Labels | | | | |

*Table 15: Ablation study (to be filled).*

---

## 9. Design Decisions and Architectural Insights

### 9.1 Why Graph-First Over Bottom-Up

The fundamental architectural choice in AJent is producing the workflow graph *directly* from the SOP rather than constructing it incrementally from chunks. This is motivated by the observation that SOP procedures are inherently graph-structured: steps have dependencies, decisions create branches, and branches converge. Bottom-up construction requires a lossy intermediate that must be reassembled --- and every reassembly step risks losing structural information.

The graph-first approach is feasible because modern LLMs with structured output capabilities can produce well-formed JSON graphs of 50--200 nodes in a single call. The pattern guide (Table 1) provides concrete modeling templates that resolve common ambiguities (multi-way decisions, nested conditionals, retry loops) before they propagate --- operationalizing the "definitional specificity" principle from Neuberger et al. [12].

### 9.2 Why Multi-Pass Patch Refinement

A single LLM call, even with the full SOP, cannot capture every detail of a 10,000-word document. The initial graph captures the correct *structure* (decision topology, branch convergence, loop back-edges) but may miss *specifics* (exact field names, threshold values, system names, role assignments). The chunk-by-chunk patch refinement audits the graph against each section individually, with cross-reference context from RAG enrichment providing necessary inter-section context.

Two passes are performed because the first pass's additions may create new gaps (e.g., a node that should connect to another section), and the second pass catches these. If the second pass produces zero changes, the graph is stable.

### 9.3 Why Confidence Labels

Not all edges in an auto-generated graph are equally reliable. By annotating each node with a confidence label, AJent enables two downstream capabilities: (a) prioritized human-in-the-loop review, where domain experts focus on low-confidence edges, and (b) risk-aware execution, where agents treat high-confidence paths differently from uncertain ones. This is grounded in Kadavath et al.'s [14] finding that requiring explicit confidence ratings improves LLM calibration.

### 9.4 Fault Tolerance for Long Documents

Production SOPs may require 50--100 LLM calls for full processing. At this scale, API rate limits and network interruptions are expected, not exceptional. AJent's checkpoint system writes graph state after every chunk and every refinement iteration, enabling exact resumption. The `safe_invoke()` wrapper distinguishes retryable errors (429, 5xx) from fatal errors (401, 403), applying appropriate retry-or-halt logic.

---

## 10. Conclusion

We have presented AJent, a modular framework for converting long-horizon Standard Operating Procedures into executable workflow graphs. AJent's three core contributions --- graph-first conversion, RAG-augmented preprocessing with entity resolution, and multi-signal self-refinement --- address the scalability and quality challenges that prior approaches face on production-length SOPs.

The graph-first architecture eliminates lossy intermediate representations by producing typed JSON graphs directly from enriched SOP text, while chunk-level patch refinement preserves fine-grained detail. The preprocessing pipeline ensures consistent terminology and resolved cross-references before conversion begins. The self-refinement loop provides convergent quality assurance through four complementary analysis signals, drawing on and extending the Self-Refine [10] and Reflexion [11] paradigms with domain-specific adaptations that exploit the verifiable structure of graph outputs.

The hybrid graph comparison framework enables rigorous evaluation with SOP-grounded attribution, adopting established metrics from ProcBench [15], Universal Prompting [12], BPMN Assistant [4], and Planetarium [6] alongside novel grounding metrics --- distinguishing genuine coverage gaps from hallucinations and auto advantages from human extrapolations.

We view AJent as a step toward production-grade SOP automation --- capable of handling the length, complexity, and terminological inconsistency of real-world procedural documents while providing the transparency (confidence labels, checkpoint trails, structured comparison reports) needed for deployment in regulated domains.

---

## 11. Limitations

We acknowledge several limitations of the current work:

1. **LLM Dependency**: AJent relies on a capable LLM for all non-deterministic pipeline stages. Graph quality is bounded by the LLM's ability to understand procedural logic and produce valid structured output. While the architecture is model-agnostic, performance may vary across LLMs.

2. **Binary Decision Constraint**: All decisions are decomposed into binary Yes/No questions. While this simplifies validation and execution, some multi-way decisions may be more naturally represented as multi-branch nodes. The chain of binary questions increases graph size and may feel unnatural for domain experts.

3. **Evaluation Coverage**: The proposed evaluation is limited to English-language SOPs. Multilingual procedures and procedures with embedded diagrams, tables, or forms are not currently addressed.

4. **Computational Cost**: The multi-pass refinement loop incurs significant LLM usage. For very long SOPs, total token consumption may be substantial. Caching and checkpoints mitigate re-work but do not reduce first-run cost.

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

[17] J. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," Advances in Neural Information Processing Systems, vol. 35, 2022.

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
