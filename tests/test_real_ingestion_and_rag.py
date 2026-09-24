from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_upload_and_query_real_text_document_changes_with_indexed_content():
    first_doc = "Alpha Corp is a software company with offices in Berlin and a strong focus on machine learning."
    second_doc = "Beta Systems builds cloud storage hardware and has no machine learning division in Berlin."

    first_response = client.post(
        "/api/documents/",
        files={"file": ("alpha.txt", first_doc, "text/plain")},
    )
    second_response = client.post(
        "/api/documents/",
        files={"file": ("beta.txt", second_doc, "text/plain")},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["status"] == "COMPLETED"
    assert second_response.json()["status"] == "COMPLETED"

    rag_response = client.post("/api/query/rag", json={"query": "machine learning in Berlin"})
    assert rag_response.status_code == 200
    payload = rag_response.json()
    assert payload["mode"] == "rag"
    assert payload["citations"]
