from pathlib import Path
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.agents.graph import agent_graph
from src.ingestion.ingestion import run_ingestion
from src.common.logger import logger
from src.services.rag_service import stream_rag_answer

import json


router = APIRouter()


# ============================================================
# Request Schema
# ============================================================


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


# ============================================================
# Helper: Build Agent Input
# ============================================================


def build_agent_input(
    request: ChatRequest,
):
    """
    Build the initial LangGraph state.

    Conversation messages are preserved through the
    LangGraph checkpoint/thread.

    Transient routing/RAG/SQL fields are reset for each
    request so state from a previous turn does not leak
    into the current response.
    """

    logger.debug(
        "Building agent input | session_id=%s",
        request.session_id,
    )

    return {
        # ----------------------------------------------------
        # Current user request
        # ----------------------------------------------------
        "query": request.query,
        "original_query": request.query,
        # ----------------------------------------------------
        # Retry / routing state
        #
        # These values are intentionally reset for every
        # request. They must not be inherited from a previous
        # conversation turn through the checkpoint.
        # ----------------------------------------------------
        "retry_count": 0,
        "max_retries": 1,
        "rewritten_query": None,
        "retry_required": False,
        "fast_small_talk_checked": False,
        "guardrail": None,
        # ----------------------------------------------------
        # RAG transient state
        # ----------------------------------------------------
        "retrieval_quality": None,
        "rag_response": {},
        "sources": [],
        # ----------------------------------------------------
        # SQL transient state
        # ----------------------------------------------------
        "sql_response": {},
        # ----------------------------------------------------
        # Conversation memory
        #
        # This remains persisted through LangGraph
        # checkpointing / thread_id.
        # ----------------------------------------------------
        "messages": [HumanMessage(content=request.query)],
    }


# ============================================================
# Helper: Build LangGraph Config
# ============================================================


def build_graph_config(
    session_id: str | None,
):
    """
    Build LangGraph configuration.

    session_id becomes LangGraph thread_id.
    """

    thread_id = session_id if session_id else "api-default-session"

    logger.debug(
        "Building graph config | thread_id=%s",
        thread_id,
    )

    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


# ============================================================
# Normal Query Endpoint
# ============================================================


@router.post("/query")
def query(
    request: ChatRequest,
):
    """
    Send a natural-language banking question to LangGraph.
    """

    logger.info("=" * 60)
    logger.info("       New Query : %s", request.query)
    logger.info("=" * 60)

    if not request.query.strip():

        logger.info(
            "Rejected empty query | session_id=%s",
            request.session_id,
        )

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:

        # ----------------------------------------------------
        # Execute graph
        # ----------------------------------------------------

        result = agent_graph.invoke(
            build_agent_input(request),
            config=build_graph_config(request.session_id),
        )

        # ----------------------------------------------------
        # Route
        # ----------------------------------------------------

        route = result.get("route")

        logger.info(
            "Graph execution completed | route=%s",
            route,
        )

        # ----------------------------------------------------
        # RAG / BOTH
        # ----------------------------------------------------

        is_rag_route = route in {
            "rag",
            "both",
        }

        sources = (
            result.get(
                "sources",
                [],
            )
            if is_rag_route
            else []
        )

        confidence_score = result.get("retrieval_quality") if is_rag_route else None

        # ----------------------------------------------------
        # SQL / BOTH
        # ----------------------------------------------------

        is_sql_route = route in {
            "sql",
            "both",
        }

        sql_response = result.get(
            "sql_response",
            {},
        )

        sql_query = sql_response.get("sql") if is_sql_route else None

        sql_result = (
            sql_response.get(
                "rows",
                [],
            )
            if is_sql_route
            else []
        )

        # ----------------------------------------------------
        # API Response
        #
        # route and source intentionally have the same value:
        #
        # rag          -> rag
        # sql          -> sql
        # both         -> both
        # small_talks  -> small_talks
        # ----------------------------------------------------

        return {
            "success": True,
            "query": request.query,
            "route": route,
            "query_path": route,
            "source": route,
            "answer": result.get("final_response"),
            # RAG / BOTH only
            "sources": sources,
            "citations": sources,
            # SQL / BOTH only
            "sql_query": sql_query,
            "sql_result": sql_result,
            # RAG / BOTH only
            "confidence_score": confidence_score,
            # Kept for evaluation/debugging
            "retry_count": result.get(
                "retry_count",
                0,
            ),
        }

    except Exception as exc:

        logger.exception(
            "Query processing failed | session_id=%s",
            request.session_id,
        )

        return {
            "success": False,
            "query": request.query,
            "error": str(exc),
        }


# ============================================================
# Document Upload / Ingestion
# ============================================================


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a PDF and run the Docling ingestion pipeline.
    """
    
    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    original_name = Path(file.filename).name

    if Path(original_name).suffix.lower() != ".pdf":

        logger.info(
            "Document upload rejected: unsupported file type | " "filename=%s",
            original_name,
        )

        raise HTTPException(
            status_code=400,
            detail=("Only PDF files are currently supported."),
        )

    contents = await file.read()

    if not contents:

        logger.info(
            "Document upload rejected: empty file | " "filename=%s",
            original_name,
        )

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    upload_dir = Path("data/uploads")

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = upload_dir / original_name

    try:

        destination.write_bytes(contents)

        result = run_ingestion(str(destination))

        # ----------------------------------------------------
        # Duplicate document
        # ----------------------------------------------------

        if result.get("status") == "already_exists":

            return {
                "success": True,
                "filename": original_name,
                "status": "already_exists",
                "document_id": result.get("document_id"),
                "chunks_created": 0,
                "message": result.get("message"),
            }

        # ----------------------------------------------------
        # New document
        # ----------------------------------------------------

        chunks_created = result.get(
            "chunks_ingested",
            result.get(
                "chunks_created",
                0,
            ),
        )

        logger.info(
            "Document ingestion completed | filename=%s | "
            "chunks_created=%s",
            original_name,
            chunks_created,
        )

        return {
            "success": True,
            "filename": original_name,
            "status": result.get(
                "status",
                "success",
            ),
            "document_id": result.get("document_id"),
            "chunks_created": chunks_created,
        }

    except Exception:

        logger.exception(
            "Document ingestion failed | filename=%s",
            original_name,
        )

        return {
            "success": False,
            "filename": original_name,
            "error": "Document ingestion failed.",
        }


# ============================================================
# Streaming Query Endpoint
# ============================================================


@router.post("/query/stream")
def stream_query(
    request: ChatRequest,
):
    """
    Stream LangGraph execution events.
    """

    logger.info("=" * 60)
    logger.info(
        "Streaming query received |  query :=%s",
        request.query,
    )

    if not request.query.strip():

        logger.info(
            "Rejected empty streaming query | session_id=%s",
            request.session_id,
        )

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    def event_generator():

        try:

            # ------------------------------------------------
            # Initial status
            # ------------------------------------------------

            logger.debug(
                "Streaming processing started | session_id=%s",
                request.session_id,
            )

            yield (
                "event: status\n"
                "data: " + json.dumps({"message": ("Processing request...")}) + "\n\n"
            )

            # ------------------------------------------------
            # Stream graph events
            # ------------------------------------------------

            for event in agent_graph.stream(
                build_agent_input(request),
                config=build_graph_config(request.session_id),
                stream_mode="updates",
            ):

                for (
                    node_name,
                    node_data,
                ) in event.items():

                    logger.debug(
                        "Streaming graph node completed | " "session_id=%s | node=%s",
                        request.session_id,
                        node_name,
                    )

                    # ----------------------------------------
                    # Node event
                    # ----------------------------------------

                    yield (
                        "event: node\n"
                        "data: "
                        + json.dumps(
                            {
                                "node": node_name,
                            }
                        )
                        + "\n\n"
                    )

                    # ----------------------------------------
                    # Route
                    # ----------------------------------------

                    if node_name == "classifier":

                        route = node_data.get("route")

                        if route:

                            logger.info(
                                "Streaming route selected | " "route=%s",
                                route,
                            )

                            yield (
                                "event: status\n"
                                "data: "
                                + json.dumps(
                                    {
                                        "message": (f"Route selected: " f"{route}"),
                                        "route": route,
                                    }
                                )
                                + "\n\n"
                            )

                    # ----------------------------------------
                    # Query rephrase
                    # ----------------------------------------

                    if node_name == "rephrase":

                        rewritten_query = node_data.get("rewritten_query")

                        if rewritten_query:

                            logger.info("Streaming RAG query rewritten ")

                            yield (
                                "event: status\n"
                                "data: "
                                + json.dumps(
                                    {
                                        "message": (
                                            "Retrying with " "rewritten query..."
                                        ),
                                        "rewritten_query": rewritten_query,
                                    }
                                )
                                + "\n\n"
                            )

                    # ----------------------------------------
                    # RAG retrieval
                    # ----------------------------------------

                    if node_name == "rag":

                        retrieval_quality = node_data.get("retrieval_quality")

                        retry_count = node_data.get(
                            "retry_count",
                            0,
                        )

                        logger.info("Streaming RAG completed ")

                        yield (
                            "event: retrieval\n"
                            "data: "
                            + json.dumps(
                                {
                                    "retrieval_quality": retrieval_quality,
                                    "retry_count": retry_count,
                                }
                            )
                            + "\n\n"
                        )

                    # ----------------------------------------
                    # Final response
                    # ----------------------------------------

                    if node_name in {
                        "merge",
                        "small_talks",
                    }:

                        final_response = node_data.get("final_response")

                        if final_response:

                            logger.debug(
                                "Streaming final response generated | "
                                "session_id=%s | node=%s",
                                request.session_id,
                                node_name,
                            )

                            yield (
                                "event: answer\n"
                                "data: "
                                + json.dumps({"answer": final_response})
                                + "\n\n"
                            )

            # ------------------------------------------------
            # Done
            # ------------------------------------------------

            logger.info("Streaming query completed")

            yield (
                "event: done\n" "data: " + json.dumps({"status": "completed"}) + "\n\n"
            )

        except Exception:

            logger.exception(
                "Streaming query failed | session_id=%s",
                request.session_id,
            )

            yield (
                "event: error\n"
                "data: " + json.dumps({"error": "Streaming query failed."}) + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
