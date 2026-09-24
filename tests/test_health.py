from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Agentic GraphRAG backend is running." in response.json()["message"]


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert set(data["services"]) == {"postgres", "qdrant", "neo4j", "gemini"}


def test_compare_endpoint():
    payload = {"query": "What is the relationship between Entity A and Entity D?", "top_k": 3, "max_depth": 2}
    response = client.post("/api/query/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) >= {"query", "rag", "graphrag", "agentic", "explanation"}
