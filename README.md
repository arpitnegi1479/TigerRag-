# Agentic GraphRAG Backend

This repository contains the backend scaffold for the system described in the project brief. It follows the layered architecture required for real multi-mode comparison between traditional RAG, GraphRAG, and Agentic GraphRAG.

## Current status

This is an initial working scaffold for the backend foundation, not a complete production deployment. It includes:

- FastAPI application bootstrap
- Environment-based configuration
- Domain and API schema models
- HTTP routes for health, documents, queries, graph, evaluation, and metrics
- Service layer boundaries for ingestion, RAG, GraphRAG, Agentic GraphRAG, and benchmarking
- Prompt files for grounded generation
- Docker and environment setup
- Initial smoke tests

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
uvicorn app.main:app --reload
```

## Docker

```bash
docker compose up --build
```

## Notes

The project deliberately separates modes so they can be benchmarked on the same corpus and query set without sharing implementation shortcuts.

## Structured graph extraction

When `GEMINI_API_KEY` is configured, ingestion requests strict JSON graph extraction from Gemini using the typed entity and relationship schema in `app/schemas/extraction.py`. Provider output is validated with Pydantic and retried once after a validation failure. If no provider is configured, or both attempts fail, ingestion logs the failure and uses the low-confidence local heuristic fallback so ingestion remains usable without silently claiming LLM-quality extraction.
