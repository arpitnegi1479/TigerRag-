from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.exceptions import IngestionError
from app.services.ingestion_service import IngestionService
from app.repositories.postgres_repository import PostgresDocumentRepository

router = APIRouter(prefix="/api/documents", tags=["documents"])
ingestion_service = IngestionService()


@router.get("/")
def list_documents() -> dict:
    return {"documents": PostgresDocumentRepository.get_default().list_documents()}


@router.post("/")
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    try:
        file_bytes = await file.read()
        result = ingestion_service.process_upload(file.filename, file_bytes, content_type=file.content_type)
        return {
            "message": "Document ingested successfully.",
            "filename": result["filename"],
            "content_type": result["content_type"],
            "status": result["status"],
            "chunks": result["chunks"],
            "text_length": result["text_length"],
        }
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
