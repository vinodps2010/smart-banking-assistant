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


from src.common.guardrails import guard_input, guard_output,guard_sql_result

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
            "answer": guard_output(result.get("final_response")),
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
            "Document ingestion completed | filename=%s | " "chunks_created=%s",
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
# RAG Token Streaming + Metadata
# ============================================================


@router.post("/query/stream")
def stream_query(
    request: ChatRequest,
):
    """
    Streaming query endpoint.

    Supports:
    - RAG token streaming
    - SQL/BOTH/Small talk normal response
    - Metadata event for sources and confidence
    """

    logger.info("=" * 60)
    logger.info(
        "Streaming query received | query=%s",
        request.query,
    )
    logger.info("=" * 60)

    if not request.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    def event_generator():

        try:

            guard_input(request.query)

            # ------------------------------------------------
            # Execute LangGraph
            # ------------------------------------------------

            result = agent_graph.invoke(
                build_agent_input(request),
                config=build_graph_config(request.session_id),
            )

            route = result.get("route")

            logger.info(
                "Streaming route selected | route=%s",
                route,
            )

            # =================================================
            # RAG STREAMING
            # =================================================

            if route == "rag":

                logger.info("Starting RAG token streaming")

                # ---------------------------------------------
                # Extract sources
                # ---------------------------------------------

                sources = result.get(
                    "sources",
                    [],
                )

                if not sources:

                    sources = result.get(
                        "rag_response",
                        {},
                    ).get(
                        "sources",
                        [],
                    )

                # ---------------------------------------------
                # Send metadata event
                # ---------------------------------------------

                yield (
                    "event: metadata\n"
                    "data: "
                    + json.dumps(
                        {
                            "sources": sources,
                            "confidence_score": result.get("retrieval_quality"),
                            "retry_count": result.get(
                                "retry_count",
                                0,
                            ),
                        }
                    )
                    + "\n\n"
                )

                # ---------------------------------------------
                # Stream tokens
                # ---------------------------------------------

                for token in stream_rag_answer(request.query, sources):

                    yield (
                        "event: token\n"
                        "data: " + json.dumps({"token": token}) + "\n\n"
                    )

            # =================================================
            # NON RAG RESPONSE
            # =================================================

            else:

                final_response = result.get(
                    "final_response",
                    "",
                )

                sql_response = result.get(
                    "sql_response",
                    {},
                ).get(
                    "rows",
                    [],
                )

                yield (
                    "event: answer\n"
                    "data: "
                    + json.dumps(
                        {
                            "answer": guard_output(final_response),
                            "route": route,
                            "sql_result": guard_sql_result(sql_response),
                            "sources": result.get(
                                "sources",
                                [],
                            ),
                            "confidence_score": result.get("retrieval_quality"),
                            "retry_count": result.get(
                                "retry_count",
                                0,
                            ),
                        },
                        default=str,
                    )
                    + "\n\n"
                )

            # =================================================
            # Completed
            # =================================================

            yield (
                "event: done\n" "data: " + json.dumps({"status": "completed"}) + "\n\n"
            )

        except Exception as exc:

            logger.exception("Streaming query failed")

            # yield ("event: error\n" "data: " + json.dumps({"error": str(exc)}) + "\n\n")
            yield (
                "event: error\n"
                "data: "
                + json.dumps({"error": str(exc), "guardrail": "toxicity"})
                + "\n\n"
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
