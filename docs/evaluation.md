# Evaluation Runner

## Running

`python scripts/run_benchmark.py` runs the complete benchmark. Use `--question-id ID`, repeated `--question-id`, `--ids ID1,ID2`, `--category NAME`, or `--limit N` to select a subset. Results are written to an atomic JSON checkpoint under `data/evaluation_runs/` after every question and mode. Resume an interrupted run with the same `--output` path and `--resume`; the runner rejects a different dataset version, configuration, or selection. The runner refuses to execute unless live Neo4j benchmark annotations, Qdrant benchmark vectors, and all six Postgres benchmark documents are present.

The smoke run selects one question per category and runs all three modes. The full 50-question run is intentionally not run until its quota estimate is reviewed.

## Metrics and Granularity

Retrieval gold data is document-level (`gold_sources`), not chunk-level. Precision@K uses the unique cited document IDs intersecting `gold_sources`, divided by configured K. Recall@K divides relevant retrieved documents by the number of gold source documents. A score is `null` when there are no gold source documents.

Answer correctness is deterministic. A normalized exact containment match against the expected answer or an acceptable variant scores 1. Otherwise token-level F1 against the expected answer is used and the method is recorded. Conflict questions, and any question declaring `expected_status`, are scored by exact evidence-status match. No LLM judge or model confidence is used.

Faithfulness splits the returned answer into sentence/semicolon/newline claims and sends each claim with the concatenated text of the chunks actually cited by that mode to the existing `verify_claim` service. `SUPPORTS` scores 1, `PARTIALLY_SUPPORTS` 0.5, and `CONTRADICTS`/`DOES_NOT_SUPPORT` 0. The verification verdict and method are stored per claim. A null score means there are no claims or no cited evidence.

The API does not attach individual citations to individual claims. Citation accuracy therefore records every claim/citation pair and computes non-stopword token coverage of the claim in that cited chunk; coverage of at least 0.5 counts as supported. The score is the supported pair fraction. This is a transparent lexical attribution proxy, not a semantic entailment judge; its pair-level details are retained in the result.

GraphRAG/Agentic graph metrics use returned graph paths/tool results: distinct exposed nodes, returned edges, maximum observed path depth, and whether a returned path contains all required entities. Backend-internal search expansions are not exposed and are not inferred. Graph metrics are `null` for RAG. Agent metrics are `null` outside Agentic GraphRAG and count returned steps, tool calls, rewrites, verification attempts, and successful/failed calls.

Performance records monotonic end-to-end and service latency. Gemini latency, attempts, token counts, and fallback events are recorded from actual provider calls. Prompt/response token totals are `null` if the provider response supplies no usage metadata; estimated cost is `null` because this project does not pin a verifiable price schedule. Fields that do not apply to a mode remain `null`.

## Degradation and Reproducibility

Each execution records Gemini HTTP attempts and logical requests by step (retrieval embedding, answer generation, agent decision, service verification, and evaluation faithfulness verification). A retry is another HTTP attempt but remains part of its logical request. A result is marked `degraded` if a Gemini logical request fails or any LLM-dependent step uses a fallback; otherwise it is `clean`. Summaries keep these populations separate. Deterministic embedding use, lexical verification, agent rule-based decisions, and generation fallback are explicitly recorded as fallback events.

Gemini requests made inside an evaluation trace are paced by `--pace-seconds` (default 0.5) and use at most two HTTP attempts, honoring numeric `Retry-After` within a 10-second bound. Runs can be interrupted after a mode checkpoint and resumed. No cache is used; each result records `cache_status: fresh`.

Each per-mode `EvaluationResult` includes run/dataset/question identifiers, raw `QueryResult`, citations, scores, metrics, LLM telemetry, execution configuration, quality classification, and cache status. Model, embedding model/dimension, top-k, graph depth, agent limits, temperature, dataset version, retry limit, pacing, and cache state are captured with every result. Temperature is 0 for Gemini generation.

The deterministic verifier implementation is versioned in the captured configuration so result sets remain reproducible when fallback semantics change. New runs use `sentence-scoped-negation-v3`; resuming a checkpoint created with `claim-local-negation-v2` restores that legacy fallback for unfinished executions rather than mixing score semantics.

The run summary reports actual Gemini HTTP attempts and logical requests per question and mode. Its full-dataset call projection linearly scales the observed mean execution call count to 50 questions × 3 modes; it is an estimate, not a quota guarantee, because category, fallback, and retry behavior vary. The runner's pacing clock is shared across question/mode traces so the configured minimum interval applies across the entire run.

## Persistence and API

The runner atomically checkpoints JSON after each completed mode and persists each individual result plus run metadata to the Postgres `evaluation_runs` and `evaluation_results` tables. The `EvaluationResult` fields remain the per-execution API payload. `POST /api/evaluate` accepts the full dataset by default or a question ID subset, category, or limit; its response includes the run identifier and current summary. Send `{"resume_run_id": "..."}` to resume the same checkpoint and original configuration. `GET /api/evaluation/runs`, `GET /api/evaluation/runs/{run_id}`, and `GET /api/evaluation/runs/{run_id}/failures` expose persisted data. `GET /api/metrics` aggregates saved results and keeps clean and degraded scores separate. The evaluation API is synchronous; use a long client timeout for larger runs.
