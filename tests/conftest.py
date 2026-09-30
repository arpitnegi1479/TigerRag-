import os

os.environ.update(
    {
        "GRAPH_BACKEND": "memory",
        "QDRANT_URL": ":memory:",
        "POSTGRES_DSN": "",
        "GEMINI_API_KEY": "",
        "EMBEDDING_PROVIDER": "deterministic",
    }
)

import pytest


@pytest.fixture(autouse=True)
def isolate_repository_singletons():
    from app.repositories.graph_repository import GraphRepository
    from app.repositories.postgres_repository import PostgresDocumentRepository
    from app.repositories.vector_repository import VectorRepository

    repositories = (GraphRepository, PostgresDocumentRepository, VectorRepository)
    for repository in repositories:
        repository._default_instance = None

    yield

    for repository in repositories:
        repository._default_instance = None