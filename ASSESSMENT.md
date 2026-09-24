# Backend Assessment

## Current state

The repository started empty. There is no pre-existing backend or frontend implementation to preserve. Because of that, the architecture is being created from scratch to match the specification in the project brief.

## Architecture gaps

The main gaps are the absence of:

- API bootstrap and route structure
- ingestion pipeline with file validation and chunking
- vector and graph storage integration
- benchmark and evaluation flow
- agentic tool loop with execution trace sanitization

## Reusable elements

There are no existing system components to reuse, so the scaffold will be greenfield and intentionally modular.

## What needs to change

The app must be built around clear responsibilities: documents, vector retrieval, graph provenance, agent control loop, and evaluation execution. The system must be able to compare modes honestly and generate outputs only from executed retrieval and verification steps.

## Risks

The biggest risk is faking the difference between RAG, GraphRAG, and Agentic GraphRAG. The implementation must avoid hand-written traces, fabricated graph paths, and demo-only metrics. Provenance, source checks, and execution data must be real.

## Implementation plan

1. Create the app foundation with configuration, logging, and health endpoints.
2. Define domain models and API schemas.
3. Add route shells and service interfaces.
4. Add ingestion, vector, and graph repository abstractions.
5. Implement retrieval and comparison services.
6. Build the agent tool loop and trace structure.
7. Add benchmark and evaluation logic.
8. Add tests and Docker scaffolding.
