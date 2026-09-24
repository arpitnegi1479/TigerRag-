# Agentic GraphRAG Intelligence System — Backend Build Brief (Refined)

You are a senior backend/AI engineer building the backend for a hackathon system that proves — with real data, not staged demos — how **Traditional RAG**, **GraphRAG**, and **Agentic GraphRAG** actually differ in behavior and quality on the same corpus and the same benchmark questions.

## The one rule that overrides everything else

**Nothing may be faked.** No hardcoded graph paths, no invented citations, no manually written agent traces, no fabricated metrics, no simulated retrieval, no per-mode "different" datasets, no business logic hidden in the frontend. If a piece can't be built for real yet, wall it off behind a clearly named interface and say so in the docs — don't fake the output to make a demo look finished. This constraint matters more than feature completeness.

Everything downstream in this brief exists in service of that rule: the point is to let three retrieval architectures compete honestly on identical ground, and to be able to show *why* one wins on a given question — with receipts.

## What "done" looks like

A user asks something like *"What's the relationship between Entity A and Entity D, and what evidence supports it?"* and:
- **RAG** returns whatever isolated chunks it found (often insufficient for multi-hop questions — that's the point).
- **GraphRAG** walks entity → relationship → entity across the graph and returns an actual traversed path with provenance.
- **Agentic GraphRAG** classifies the query, decides for itself which tools to call and in what order, notices if evidence is missing or conflicting, goes and gets more, verifies it, and only then answers — with a trace showing the real decisions it made.

The difference between the three should be visible and explainable from real execution data, not asserted.

## Product surface

Ingest documents → chunk → embed → store vectors → extract entities/relationships → build a provenance-preserving knowledge graph → answer queries via all three retrieval modes → verify and score evidence → run a fixed benchmark across all three modes → expose metrics, failure analysis, and execution traces through a clean REST API a frontend can consume without reimplementing any of this logic.

## Stack (use these unless you hit a real blocker)

Python 3.11+, FastAPI, Pydantic Settings, async where it earns its keep. Qdrant for vectors, Neo4j for the graph (NetworkX only for local test utilities, never as the main graph engine), Postgres for app metadata/queries/traces/evaluation runs, local filesystem for document storage (structured so it can become S3 later). Keep the LLM and embedding providers behind thin interfaces — don't hardwire the whole app to one vendor or one framework (LangChain/LlamaIndex pieces are fine as components, not as the architecture). pytest + httpx for tests. All secrets via environment variables, never hardcoded, never committed.

## Architecture shape

A conventional layered backend works well here: `api/routes` (documents, queries, evaluation, metrics, graph, health) → `services` (ingestion, rag, graphrag, agent, evidence, evaluation, metrics) → `repositories` (document/chunk/query/trace/vector/graph) → `models`/`schemas` → `llm` (provider abstraction + a `prompts/` folder of actual prompt files, not strings buried in Python) → `core` (config/logging/exceptions). Adapt this if you have a good reason — the goal is separation of concerns and independently testable stages, not a specific folder name.

## Domain model (the essentials)

- **Document**: id, title, source, file_type, content, metadata, status (`PENDING → PROCESSING → COMPLETED/FAILED`), timestamps.
- **Chunk**: id, document_id, content, page/section, chunk_index, metadata (must carry enough to trace back to page/source), embedding reference.
- **Entity**: id, canonical_name, type, description, confidence, metadata. Normalize mentions ("Microsoft Corp." / "Microsoft") into a canonical entity deliberately — don't silently merge without a strategy, and keep the original mention alongside the canonical form.
- **Relationship**: id, source/target entity, type, confidence, **source_document_id + source_chunk_id** (this is the provenance link — an edge without it is lower-quality evidence and should be treated that way downstream).

Provenance is the spine of the whole system: every graph edge and every citation should be traceable back to a document → chunk → page. Design for that from the start rather than bolting it on later.

## Ingestion

`POST /api/documents` accepting PDF/TXT/Markdown/DOCX/HTML, with real validation (extension, MIME where available, size limits, malformed-file handling — don't trust the filename). Pipeline: validate → store → extract text → clean → chunk (configurable size/overlap, defaults like 800/120, don't assume they're optimal) → embed → store vectors → extract entities/relationships → persist to graph with provenance → mark COMPLETED. Each stage should be independently testable and swappable.

## The three retrieval modes

Implement them as **independently callable services** (`rag_service.answer(query)`, `graphrag_service.answer(query)`, `agentic_service.answer(query)`) — a comparison layer orchestrates them, it doesn't merge their internals. That separation is what makes the benchmark meaningful.

**Traditional RAG** (`POST /api/query/rag`): embed query → vector search → optional rerank → top-K chunks → grounded generation → citations that only reference chunks actually retrieved. Return answer, citations, confidence, evidence status, retrieval stats, latency/token metrics.

**GraphRAG** (`POST /api/query/graphrag`): entity detection/linking → graph search and relationship traversal (bounded — configurable max depth/nodes, cycle protection, never unbounded) → pull source documents behind the path → grounded answer with the actual traversed path and its provenance attached.

**Agentic GraphRAG** (`POST /api/query/agentic`) — the core differentiator, and the part most tempting to fake. Build a real controlled agent loop, not a fixed `search_entity → traverse_graph → search_documents → verify` sequence dressed up as "agentic." Give it a small tool registry (search_documents, search_entity, traverse_graph, find_relationship, get_source, verify_claim, rewrite_query) and let it genuinely choose which to call, in what order, based on explicit state (query, entities, hypotheses, retrieved evidence, graph paths, missing information, conflicts, verification results). It should be able to decide it has enough evidence and stop, decide it needs more and go get it, detect conflicting evidence and report both sides rather than silently picking one, and return `INSUFFICIENT_EVIDENCE` honestly rather than filling the gap with a hallucination. Hard safety limits (max steps, max tool calls, max retrieval iterations, max graph depth) prevent runaway loops — these are floors, not the intended behavior.

Every agent query produces a **sanitized execution trace** (action, tool, input, result summary/count, latency, status per step) — never raw hidden chain-of-thought. The answer itself should include a short human-readable reasoning summary ("identified two entities, found a 3-hop path, retrieved supporting docs, verified all three relationships") rather than exposing internal reasoning.

Confidence should be assembled from observable signals — retrieval relevance, evidence coverage, verification results, provenance quality, cross-source agreement, graph confidence — not obtained by asking the LLM to self-report a number.

## Comparison endpoint

`POST /api/query/compare` runs all three modes against the identical query and corpus, and returns per-mode answer/status/confidence/citations/retrieval stats/graph paths/agent steps/latency/cost side by side, plus a short structured explanation of *why* they differ (e.g. RAG's scope is semantic chunks, GraphRAG's is the entity subgraph, Agentic's is adaptive). Use `null`/empty rather than inventing a value for a field that doesn't apply to a given mode.

## Evaluation

A fixed benchmark (~50–100 hand-designed questions, not randomly generated) spanning factual, entity, relationship, multi-hop, temporal, comparison, cause/effect, and conflict categories, each with expected answer, required entities/relationships, and gold sources. Run RAG/GraphRAG/Agentic against the same set and compute real precision/recall, answer correctness, faithfulness, citation accuracy, graph metrics (nodes explored, depth, path accuracy), agent metrics (steps, tool calls, rewrites, verification attempts), and latency/token/cost — all computed from actual execution, never hand-entered. Persist enough per run (`EvaluationRun` + `EvaluationResult`, with dataset version and model/config captured) to reproduce it later, and surface per-question failure analysis explaining *why* each mode succeeded or failed — generated from real execution data, not authored copy.

`GET /api/metrics` aggregates this. Treat any example numbers you see anywhere as shape illustrations only, never literal targets to hardcode.

## Security and prompt-injection defense

Standard hygiene (env-based secrets, file validation/size limits, request validation, structured logging that never leaks stack traces to clients, rate limiting where practical). The one worth calling out specifically: **treat every ingested document as untrusted data**. Text inside a document that says "ignore previous instructions" is evidence to reason about, never an instruction to follow. Keep retrieved evidence clearly delimited in prompts (e.g. inside an explicit `<retrieved_evidence>` block) and tell the model it's untrusted source material, not commands.

## Suggested build order

Foundation (app boots, config, logging, health, Docker) → Ingestion (real documents, real chunking/embeddings/vector storage) → Traditional RAG end-to-end → Knowledge graph construction with provenance → GraphRAG (verify on real multi-hop questions) → Agentic GraphRAG (verify the agent's behavior actually *changes* across different query types — this is the thing to scrutinize hardest) → Comparison endpoint → Evaluation framework → Hardening (security, tests, docs). After each phase: run it, hit the API, read the logs, fix what's broken, then move on — don't stack untested modules.

## Before writing any code

1. Inspect whatever exists in the repo already (backend, frontend, databases, env files, routes, AI integrations).
2. Write a short assessment: current state, architecture gaps, what's reusable, what needs to change, risks, and your implementation plan.
3. Proceed with implementation directly unless there's a genuine architectural fork that can't be safely inferred — don't stall waiting for approval on things you can reasonably decide yourself.

## Definition of done, condensed

Runs locally with Docker (Postgres + Qdrant + Neo4j). All five document types ingest, chunk, embed, and land in the vector store. RAG, GraphRAG, and Agentic GraphRAG each independently answer real queries against the same corpus with real citations/graph paths/traces. The agent visibly makes different tool-call sequences for different query types and can say "insufficient evidence" instead of hallucinating. The comparison endpoint runs all three against one query and explains the differences. The evaluation framework runs the fixed benchmark through all three modes and computes every metric from real execution. No hardcoded secrets, real file/input validation, prompt-injection defense in place. Unit, integration, API, and security tests exist and pass.
