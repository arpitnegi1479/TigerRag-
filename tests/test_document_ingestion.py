from io import BytesIO

from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from app.main import app
from app.repositories.postgres_repository import PostgresDocumentRepository
from app.repositories.vector_repository import VectorRepository


client = TestClient(app)


def _valid_pdf() -> bytes:
    stream = b"BT /F1 12 Tf 50 200 Td (Generated PDF text.) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode())
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    )
    return bytes(pdf)


def _valid_docx() -> bytes:
    document = DocxDocument()
    document.add_paragraph("Generated DOCX text.")
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def test_all_supported_document_types_ingest_through_api():
    documents = [
        ("format.txt", "text/plain", b"Plain text content."),
        ("format.md", "text/markdown", b"# Markdown\n\nMarkdown content."),
        ("format.html", "text/html", b"<h1>HTML</h1><p>HTML content.</p>"),
        ("format.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", _valid_docx()),
        ("format.pdf", "application/pdf", _valid_pdf()),
    ]

    for filename, content_type, payload in documents:
        response = client.post(
            "/api/documents/",
            files={"file": (filename, payload, content_type)},
        )

        assert response.status_code == 200, f"{filename}: {response.text}"
        assert response.json()["status"] == "COMPLETED"
        assert response.json()["chunks"] >= 1
        assert VectorRepository.get_default().count_documents([filename]) == 1


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


def test_postgres_repository_counts_only_requested_documents():
    repository = PostgresDocumentRepository(dsn="")
    repository.create_document("benchmark.txt", "Benchmark", "text", ".txt", "benchmark.txt")
    repository.create_document("other.txt", "Other", "text", ".txt", "other.txt")

    assert repository.count_documents() == 2
    assert repository.count_documents(["benchmark.txt"]) == 1
    assert repository.count_documents([]) == 0
