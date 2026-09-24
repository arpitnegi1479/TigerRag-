from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_verify_claim_endpoint_returns_structured_verdict():
    response = client.post(
        "/api/query/verify-claim",
        json={
            "claim": "Microsoft uses Azure",
            "evidence": "Microsoft uses Azure for cloud products.",
        },
    )

    assert response.status_code == 200
    assert response.json()["verdict"] == "SUPPORTS"
    assert response.json()["explanation"]