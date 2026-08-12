from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.agents.graph import agent_graph
from src.ingestion.ingestion import run_ingestion

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


@router.post("/query")
def query(request: ChatRequest):
    """
    Send a natural-language banking question to the LangGraph agent.
    """

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:
        result = agent_graph.invoke(
            {
                "query": request.query,
            }
        )

        return {
            "success": True,
            "query": request.query,
            "route": result.get("route"),
            "query_path": result.get("route"),
            "answer": result.get("final_response"),
            "sources": result.get("sources", []),
            "citations": result.get("sources", []),
            "sql_query": result.get("sql_query"),
            "sql_result": result.get("sql_result"),
            "confidence_score": result.get("confidence_score"),
            "retry_count": result.get("retry_count", 0),
        }

    except Exception as exc:
        return {
            "success": False,
            "query": request.query,
            "error": str(exc),
        }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a PDF and run the existing Docling ingestion pipeline.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    original_name = Path(file.filename).name

    if Path(original_name).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are currently supported.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    destination = upload_dir / original_name

    try:
        destination.write_bytes(contents)

        result = run_ingestion(str(destination))

        return {
            "success": True,
            "filename": original_name,
            "status": result.get("status", "success"),
            "document_id": result.get("document_id"),
            "chunks_created": result.get(
                "chunks_ingested",
                result.get("chunks_created", 0),
            ),
        }

    except Exception as exc:
        return {
            "success": False,
            "filename": original_name,
            "error": str(exc),
        }
