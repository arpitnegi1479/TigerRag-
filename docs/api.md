# API Guide

Base URL for local development: `http://127.0.0.1:8000`.

## Evaluation

`POST /api/evaluate` starts a benchmark run. With no selection fields it selects the complete dataset; use `question_ids`, `category`, or `limit` to narrow the run. `top_k`, `max_graph_depth`, and `pacing_seconds` are captured with the run. A full run executes each of the 50 questions through RAG, GraphRAG, and Agentic GraphRAG and can take a long time; prefer a subset when checking configuration or quota.

Example subset request:

```json
{
  "question_ids": ["f01", "m01"],
  "top_k": 5,
  "max_graph_depth": 3,
  "pacing_seconds": 0.5
}
```

The response returns `run_id`, `dataset_version`, status, and summary. Each mode execution is persisted independently in Postgres and atomically checkpointed to `data/evaluation_runs/`. Resume a paused checkpoint with:

```json
{"resume_run_id": "<run-id>"}
```

A resume uses the original dataset selection and configuration. It skips finished question/mode pairs and retries only missing pairs.

- `GET /api/evaluation/runs?limit=50` lists run metadata.
- `GET /api/evaluation/runs/{run_id}` returns run metadata and its individual results.
- `GET /api/evaluation/runs/{run_id}/failures` reports score- and telemetry-derived failure indicators per question/mode.
- `GET /api/metrics` aggregates persisted execution scores and latency, keeping clean and degraded populations separate.

Retrieval scores are document-level against `gold_sources`. Answer correctness is deterministic and records its method; no model confidence is used as a score. Faithfulness uses the existing verifier on cited evidence. Citation support is a lexical attribution proxy because the query result schema does not associate individual citations with individual claims. Null values mean a metric does not apply or could not be computed.

Evaluation requires the live Neo4j, Qdrant, and Postgres benchmark stores. It fails closed if those stores or benchmark provenance are unavailable. API execution is synchronous, so clients should use a long request timeout. Gemini call counts, fallback events, and degraded classifications are part of every result; an HTTP 200 run status does not mean every execution was clean.
