# PRD Definition of Done Review

Status reviewed on 2026-09-30 against `agentic-graphrag-master-prompt-refined.md`. This checklist separates verified implementation from the still-running full evaluation.

| PRD requirement | Status | Evidence / remaining condition |
| --- | --- | --- |
| Run Postgres, Qdrant, and Neo4j locally with Docker | Verified running | All three containers were live during the benchmark and HTTP checks. Fresh Compose starts now require `POSTGRES_PASSWORD` and `NEO4J_PASSWORD` in ignored `.env`; Compose configuration validates when these required values are supplied. |
| Ingest PDF, TXT, Markdown, DOCX, and HTML | Verified | `tests/test_document_ingestion.py` uploads all five through the FastAPI route and checks completed ingestion. |
| Validate upload type, size, parser input, and malformed requests | Verified | `tests/test_security.py` covers disallowed extensions, oversized upload rejection, MIME mismatch, malformed PDF, and invalid request data; parser errors do not expose stack traces. |
| RAG returns retrieved evidence and citations | Verified | Existing real-ingestion/RAG tests and the 9-question smoke artifact exercise retrieval and citation behavior. |
| GraphRAG returns traversed graph evidence with provenance | Verified | Graph/GraphRAG tests and the live benchmark audit cover graph traversal and annotation provenance. |
| Agentic GraphRAG uses a bounded tool loop with observable trace | Verified | Agent-loop tests verify tool decisions, steps, and limits. |
| Comparison executes three independent modes | Verified | Comparison API/service tests cover the three-mode result contract. |
| Evaluation runs the fixed dataset through all three modes and computes metrics from execution | In progress | Runner, deterministic scoring, telemetry, checkpoint/resume, smoke results, and call projection are implemented. The v2 pre-fix baseline resumed and is at 116/150 executions with 30 clean and 86 degraded. Finish it, then inspect targeted v3 live results before closing this line. |
| Persist evaluation runs/results and expose metrics/failure analysis | Verified | Live HTTP checks returned persisted run metadata, results, failure indicators, and aggregate metrics from Postgres. Checkpoint filesystem paths are withheld from responses. |
| No committed credentials; safe secret handling | Verified in source; local setup required | Credential literals were removed from application and Compose configuration; defaults are empty and debug defaults off. Provide local Postgres/Neo4j passwords before fresh Compose startup. |
| Prompt-injection defense treats documents as untrusted | Verified | Evidence and serialized agent state escape angle brackets before prompt interpolation; a test uploads a closing-tag injection through retrieval and checks it remains inside the evidence boundary. |
| Security, unit, integration, and API tests pass | Verified | Final complete suite after verifier-version-aware resume: 70 passed, 2 upstream Starlette/httpx deprecation warnings. |
| Final backend completion | Pending | Requires successful completion of all 150 baseline executions, a targeted post-fix live rerun with inspected answers/citations, review of clean/degraded counts, and a post-run live-store integrity audit. Do not represent an interrupted or quota-degraded run as a clean benchmark. |

## Operational Notes

- Rate limiting is in-process and keyed by the socket peer address; multiple workers or replicas need a shared upstream limiter for a global limit.
- Gemini quota degradation is an observed system limitation. The runner reports fallback usage and separates degraded from clean results rather than hiding or tuning them away.
- Run `python -m pytest -q` for the isolated suite. Start services with `docker compose up -d postgres qdrant neo4j` only after required passwords are configured in `.env`.
