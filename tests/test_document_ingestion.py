from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_document_upload_and_processing():
    content = "This is a sample document. It contains multiple sentences for testing ingestion and chunking.\n\nIt should also be safe to store and process." \
        "\n\nThis is the final paragraph."

    response = client.post(
        "/api/documents/",
        files={"file": ("sample.txt", content, "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.txt"
    assert data["status"] == "COMPLETED"
    assert data["chunks"] >= 1
