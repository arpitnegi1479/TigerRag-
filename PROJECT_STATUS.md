# Agentic GraphRAG Backend — Project Status

## What this project is

A backend proving, with real execution rather than assertion, how Traditional RAG, GraphRAG, and Agentic GraphRAG behave differently on the same corpus and benchmark questions. **The core non-negotiable rule governing every round of this project: nothing may be faked** — no hardcoded graph paths, invented citations, fabricated metrics, simulated retrieval, or fixed pipelines dressed up as "agentic." If something can't be built for real yet, it gets isolated and documented honestly rather than faked.

## Stack

Python 3.11+/FastAPI, PostgreSQL, Qdrant (vectors), Neo4j (graph), Gemini (LLM + embeddings, model `gemini-2.5-flash` / `gemini-embedding-001`). Docker Compose brings up Postgres/Qdrant/Neo4j locally.

To run:
```
docker compose up -d postgres qdrant neo4j
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m pytest -q
```

## What's done and verified (Sections 1-4)

- **Section 1 — Infrastructure**: Postgres, Qdrant, and Neo4j all confirmed live via direct checks (real Cypher queries, real Qdrant client calls, real psycopg connections) AND via real HTTP requests against a running FastAPI server (upload, RAG, GraphRAG, Agentic all returning HTTP 200). Health endpoint accurately reports `ok`/`degraded`. Fixed real bugs along the way: Qdrant requires UUID point IDs (was using raw strings), the installed Qdrant client uses `query_points` not the removed `search` method (this bug was previously being silently swallowed by a broad exception handler), Postgres DSN/timeout string bug, JSONB metadata serialization bug.

- **Section 2 — Real embeddings**: Gemini embeddings (`gemini-embedding-001`, 768-dim) wired in via a config-driven provider, kept separate from the deterministic 384-dim fallback in its own Qdrant collection (`document_chunks_gemini_embedding_001` vs `document_chunks_deterministic_384`) to avoid dimension mixing. Retrieval-quality check confirmed real embeddings correctly ranked a semantically-relevant document top for a paraphrased query, while the deterministic fallback got it backwards.

- **Section 3 — Comparison endpoint**: `POST /api/query/compare` orchestrates all three services, measures per-mode latency, preserves each mode's status/citations/paths/steps, and correctly surfaces `INSUFFICIENT_EVIDENCE`/`CONFLICTING_EVIDENCE` without masking them.

- **Agent-loop fix (between sections 3 and 4)**: Found and fixed a real regression where the agent returned `INSUFFICIENT_EVIDENCE` on a query plain RAG could answer, because the fallback decision policy kept retrying `TRAVERSE_GRAPH` after graph failure instead of falling back to already-retrieved document evidence. Fixed so graph failure now triggers `GET_SOURCE` → `VERIFY_CLAIM` instead of repeating traversal. Verified live: agent now matches RAG's evidence outcome while its trace honestly shows the graph attempt.

- **Section 4 — Remaining endpoints**: `GET /api/documents`, `GET /api/graph/entity/{id}`, `GET /api/graph/path`, `GET /api/query/{query_id}/trace` (backed by real Postgres persistence in an `agent_traces` table), `GET /api/health` — all live-verified over real HTTP. Fixed a real duplicate-relationship bug in the entity endpoint's Cypher query. Proved `rewrite_query` is not dead code: forced a scenario with empty initial retrieval, confirmed the agent rewrites the query and retries successfully.

Test suite was at **28 passing** as of the end of section 4.

## Section 5a — Benchmark dataset (IN PROGRESS, NOT CONFIRMED COMPLETE)

Work done:
- Built a 6-document interconnected benchmark corpus (`data/benchmark_corpus/`) with recurring entities (Helios Research Institute, Meridian Labs, Atlas Sensor, Zephyr Station, Nereid Basin, Lumen Shelf, Celeste Analytics, Nova Dynamics), a 3-hop evidence chain (Atlas Sensor → Zephyr Station → Nereid Basin → Lumen Shelf), and a deliberate licensing contradiction between two documents (`05_license_record.txt` says Meridian licensed Atlas Sensor to Helios; `06_counter_record.txt` contradicts this and says it was licensed to Nova Dynamics instead).
- Built `data/benchmark_questions.json` — confirmed at exactly 50 questions across all 9 required categories (factual, document lookup, entity, relationship, multi-hop, temporal, comparison, cause/effect, conflict).
- Ingested the corpus for real through the live pipeline — confirmed in Neo4j, Qdrant (6 points, 768-dim), and Postgres.
- **Important known limitation, documented honestly**: Gemini's generation quota was hit repeatedly during this section. Real extraction fell back to the heuristic path, which produced noisy/incorrect relationships. To keep the corpus semantically testable, the gold graph edges (the ones the multi-hop chain and the conflict depend on) were **manually annotated directly into Neo4j** with `benchmark_annotation: true` metadata and real provenance fields — NOT extracted by Gemini. This is documented in `data/benchmark_dataset.md` and must stay documented as such; these are constructed gold facts for benchmark purposes, not model output, and should never be presented as if Gemini produced them.
- Debugged and fixed several real issues found via live spot-checks: heuristic-extracted noise edges were deleted (kept only the annotated gold edges), a query-parsing bug where "Did Meridian Labs" was mis-parsed as one entity, conflict detection was too broad (flagging unrelated multi-hop evidence as conflicting) and was narrowed to same-endpoint-pair claims only, and relationship classification was missing "license"/"licensed" as a relationship cue.
- Final spot-checks (as last confirmed) showed correct results: the multi-hop question returned `VERIFIED` with a real traversed path, and the conflict question returned `CONFLICTING_EVIDENCE` with both source documents cited.

**What is NOT yet confirmed and must be checked before section 5a is considered done:**
1. **The full `pytest -q` suite was run after the last two code changes (conflict-detection scoping + relationship-classification keyword fix) but the output was never seen/confirmed** — credits ran out before this could be verified. This is a real risk: those changes could have regressed the original conflict-detection test or something else. **Run the full suite and confirm a clean pass before doing anything else.**
2. **The final benchmark question sample/breakdown printout was queued but never confirmed to have printed successfully.** Re-run it and confirm the sample and category breakdown look correct.
3. No formal "section 5a complete" honest-status report was ever written. Once 1 and 2 above are confirmed clean, write that report before moving to section 5b.

## Not yet started

- **Section 5b** — evaluation runner: execute the benchmark against all three modes, compute precision@K, recall@K, answer correctness, faithfulness/groundedness, citation accuracy, graph metrics, agent metrics, latency/token usage — all from real execution.
- **Section 5c** — persistence (`EvaluationRun`/`EvaluationResult`), `POST /api/evaluate`, `GET /api/metrics`, per-question failure analysis.
- **Section 6** — hardening: security test suite (malicious file extension, oversized file, prompt-injection document, malformed request, missing credentials), rate limiting (not implemented anywhere yet), full `docs/` folder, and a final line-by-line check against the PRD's Definition of Done.

## Working rules for this project (apply to every future round)

1. No fabricated results, ever — no hardcoded metrics, no invented citations, no pretending a fallback result came from a real model call.
2. Report back honestly after each section/subsection, in the same format used throughout: what's real, what's still stubbed, what pushed back on the plan.
3. Don't silently do multiple sections in one pass — stop and report at natural checkpoints.
4. Verify against live infrastructure with real HTTP requests wherever possible, not just direct service-layer Python calls.
5. When something looks suspicious (a metric that's too clean, a result that seems backwards), stop and investigate before reporting it as evidence — this project has caught several real bugs this way and that discipline should continue.
