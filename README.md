# Agentic GraphRAG Backend

This repository contains the backend scaffold for the system described in the project brief. It follows the layered architecture required for real multi-mode comparison between traditional RAG, GraphRAG, and Agentic GraphRAG.

## Current status

Sections 1-5a are implemented and verified. Section 5b's resumable runner, real execution scoring, Gemini fallback telemetry, and call projections are implemented; the 9-question smoke run is complete, and the full 50-question run is resumed at 116/150 executions in `data/evaluation_runs/section5b-full-20260930.json`. Section 5c provides persisted evaluation runs/results and evaluation/metrics APIs. Section 6 adds input hardening, request rate limiting, security tests, and documentation. The backend is not fully closed until the baseline completes and the post-fix verifier cases are checked. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for details and [docs/](docs/) for API, security, evaluation, benchmark, and PRD Definition of Done material.

## Backend architecture

- api/routes: HTTP handlers
- services: ingestion, rag, graphrag, agentic, evaluation, metrics
- repositories: storage abstractions for documents, graph, and vectors
- models/schemas: domain and API models
- llm: provider abstraction and prompts
- core: configuration, logging, exceptions

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Set POSTGRES_PASSWORD and NEO4J_PASSWORD in .env before starting Docker.
uvicorn app.main:app --reload
```

Start the required stores with `docker compose up -d postgres qdrant neo4j`. The default test suite is isolated from live stores; run it with `python -m pytest -q`.

Evaluation API: `POST /api/evaluate`, `GET /api/evaluation/runs`, `GET /api/evaluation/runs/{run_id}`, `GET /api/evaluation/runs/{run_id}/failures`, and `GET /api/metrics`. See [docs/api.md](docs/api.md). A full benchmark consumes substantial Gemini quota; inspect the runner's observed call projection and keep degraded results separate from clean results.

## Docker

```bash
docker compose up --build
```

## Notes

The project deliberately separates modes so they can be benchmarked on the same corpus and query set without sharing implementation shortcuts.

## Structured graph extraction

When `GEMINI_API_KEY` is configured, ingestion requests strict JSON graph extraction from Gemini using the typed entity and relationship schema in `app/schemas/extraction.py`. Provider output is validated with Pydantic and retried once after a validation failure. If no provider is configured, or both attempts fail, ingestion logs the failure and uses the low-confidence local heuristic fallback so ingestion remains usable without silently claiming LLM-quality extraction.
