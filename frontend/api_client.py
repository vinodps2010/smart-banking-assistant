import json
import os
import uuid
from datetime import datetime

import httpx

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

# FastAPI backend is now ready.
MOCK_MODE = False


class BankingAPIClient:
    """
    Client responsible for communication between
    Streamlit frontend and FastAPI backend.
    """

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

        # --------------------------------------------------------------
        # Default HTTP timeout
        #
        # Upload/ingestion can take several minutes because:
        #   PDF upload
        #   -> Docling
        #   -> chunking
        #   -> embeddings
        #   -> PostgreSQL/pgvector
        # --------------------------------------------------------------

        self.timeout = httpx.Timeout(
            connect=10.0,
            read=600.0,
            write=120.0,
            pool=10.0,
        )

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health_check(self):
        """
        Check FastAPI backend health.
        """

        if MOCK_MODE:
            return {
                "status": "healthy",
                "service": "Smart Banking Assistant API",
                "version": "1.0.0",
                "mode": "mock",
            }

        try:
            response = httpx.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )

            response.raise_for_status()

            return response.json()

        except Exception as exc:
            return {
                "status": "offline",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Normal Chat Query
    # ------------------------------------------------------------------

    def query_assistant(
        self,
        query: str,
        session_id: str | None = None,
    ):
        """
        Send one user query to LangGraph through FastAPI.

        LangGraph decides:
            RAG / SQL / BOTH
        """

        if not session_id:
            session_id = str(uuid.uuid4())

        if MOCK_MODE:
            return self._mock_query_response(
                query=query,
                session_id=session_id,
            )

        payload = {
            "query": query,
            "session_id": session_id,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/v1/query",
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

            return response.json()

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": (
                    "The request timed out while " "processing the banking query."
                ),
                "session_id": session_id,
            }

        except httpx.HTTPStatusError as exc:
            return {
                "success": False,
                "error": (
                    f"FastAPI returned HTTP "
                    f"{exc.response.status_code}: "
                    f"{exc.response.text}"
                ),
                "session_id": session_id,
            }

        except httpx.RequestError as exc:
            return {
                "success": False,
                "error": (f"Unable to connect to FastAPI: {exc}"),
                "session_id": session_id,
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "session_id": session_id,
            }

    # ------------------------------------------------------------------
    # Streaming Query
    # ------------------------------------------------------------------

    def stream_query_assistant(
        self,
        query: str,
        session_id: str | None = None,
    ):
        """
        Stream LangGraph execution events from FastAPI.

        FastAPI endpoint:
            POST /api/v1/query/stream

        The endpoint uses Server-Sent Events (SSE).

        Yields dictionaries such as:

            {
                "event": "status",
                "data": {
                    "message": "Processing request..."
                }
            }

            {
                "event": "node",
                "data": {
                    "node": "classifier"
                }
            }

            {
                "event": "answer",
                "data": {
                    "answer": "..."
                }
            }

            {
                "event": "done",
                "data": {
                    "status": "completed"
                }
            }
        """

        if not session_id:
            session_id = str(uuid.uuid4())

        if MOCK_MODE:
            yield {
                "event": "status",
                "data": {"message": "Mock streaming mode"},
            }

            yield {
                "event": "answer",
                "data": {
                    "answer": ("This is a mock streaming response."),
                },
            }

            yield {
                "event": "done",
                "data": {"status": "completed"},
            }

            return

        payload = {
            "query": query,
            "session_id": session_id,
        }

        try:

            # ----------------------------------------------------------
            # Streaming timeout
            #
            # Read timeout is deliberately long because RAG may involve:
            #   vector search
            #   FTS
            #   RRF
            #   Cohere reranking
            #   query retry
            #   LLM generation
            # ----------------------------------------------------------

            stream_timeout = httpx.Timeout(
                connect=10.0,
                read=600.0,
                write=120.0,
                pool=10.0,
            )

            with httpx.stream(
                "POST",
                f"{self.base_url}/api/v1/query/stream",
                json=payload,
                timeout=stream_timeout,
            ) as response:

                response.raise_for_status()

                current_event = None

                for line in response.iter_lines():

                    if not line:
                        continue

                    # --------------------------------------------------
                    # SSE event line
                    # --------------------------------------------------

                    if line.startswith("event:"):

                        current_event = line[len("event:") :].strip()

                        continue

                    # --------------------------------------------------
                    # SSE data line
                    # --------------------------------------------------

                    if line.startswith("data:"):

                        raw_data = line[len("data:") :].strip()

                        try:
                            data = json.loads(raw_data)
                        except json.JSONDecodeError:
                            data = {"message": raw_data}

                        yield {
                            "event": (current_event or "message"),
                            "data": data,
                        }

                        current_event = None

        except httpx.TimeoutException as exc:

            yield {
                "event": "error",
                "data": {
                    "error": ("Streaming request timed out: " f"{exc}"),
                },
            }

        except httpx.HTTPStatusError as exc:

            yield {
                "event": "error",
                "data": {
                    "error": (
                        f"FastAPI returned HTTP "
                        f"{exc.response.status_code}: "
                        f"{exc.response.text}"
                    ),
                },
            }

        except httpx.RequestError as exc:

            yield {
                "event": "error",
                "data": {
                    "error": (f"Unable to connect to FastAPI: " f"{exc}"),
                },
            }

        except Exception as exc:

            yield {
                "event": "error",
                "data": {
                    "error": str(exc),
                },
            }

    # ============================================================
    # Unified Streaming Query
    # ============================================================

    def stream_query(
        self,
        query: str,
        session_id: str | None = None,
    ):
        """
        Unified streaming API consumer.

        Backend decides:
            RAG:
                token streaming

            SQL/BOTH/SMALL_TALK:
                answer event

        Returns normalized events for Streamlit UI.
        """

        try:

            with httpx.stream(
                "POST",
                f"{self.base_url}/api/v1/query/stream",
                json={
                    "query": query,
                    "session_id": session_id,
                },
                timeout=None,
            ) as response:

                response.raise_for_status()

                current_event = None

                for line in response.iter_lines():

                    if not line:
                        continue

                    # --------------------------------------------
                    # Event name
                    # --------------------------------------------

                    if line.startswith("event:"):

                        current_event = line.replace(
                            "event:",
                            "",
                        ).strip()

                    # --------------------------------------------
                    # Event payload
                    # --------------------------------------------

                    elif line.startswith("data:"):

                        data_text = line.replace(
                            "data:",
                            "",
                        ).strip()

                        data = json.loads(data_text)

                        # ----------------------------------------
                        # RAG token streaming
                        # ----------------------------------------

                        if current_event == "token":

                            token = data.get(
                                "token",
                                "",
                            )

                            if token:

                                yield {
                                    "type": "token",
                                    "value": token,
                                }

                        # ----------------------------------------
                        # SQL / BOTH / SMALL TALK response
                        # ----------------------------------------

                        elif current_event == "answer":

                            # print("API CLIENT ANSWER EVENT:", data)
                            yield {
                                "type": "answer",
                                "value": data.get(
                                    "answer",
                                    "",
                                ),
                                "route": data.get(
                                    "route",
                                    "",
                                ),
                                "sql_result": data.get(
                                    "sql_result",
                                    [],
                                ),
                                "sources": data.get(
                                    "sources",
                                    [],
                                ),
                                "confidence_score": data.get(
                                    "confidence_score",
                                ),
                                "retry_count": data.get(
                                    "retry_count",
                                    0,
                                ),
                            }

                        # ----------------------------------------
                        # Completed
                        # ----------------------------------------

                        elif current_event == "done":

                            yield {"type": "done"}

                        # ----------------------------------------
                        # metadata
                        # ----------------------------------------

                        elif current_event == "metadata":

                            yield {
                                "type": "metadata",
                                "route": data.get("route", "rag"),
                                "sources": data.get("sources", []),
                                "confidence_score": data.get("confidence_score"),
                                "retry_count": data.get(
                                    "retry_count",
                                    0,
                                ),
                            }

                        # ----------------------------------------
                        # Error
                        # ----------------------------------------

                        elif current_event == "error":

                            yield {
                                "type": "error",
                                "value": data.get(
                                    "error",
                                    "Unknown streaming error",
                                ),
                            }

        except Exception as exc:

            yield {
                "type": "error",
                "value": str(exc),
            }

    # ------------------------------------------------------------------
    # RAG Token Streaming
    # ------------------------------------------------------------------

    def stream_rag_tokens(
        self,
        query: str,
        session_id: str | None = None,
    ):

        for event in self.stream_query_assistant(
            query=query,
            session_id=session_id,
        ):

            if event.get("event") == "token":

                token = event.get(
                    "data",
                    {},
                ).get("token")

                if token:
                    yield token

            elif event.get("event") == "error":

                yield (
                    "\n\n⚠️ Streaming failed: "
                    + event.get(
                        "data",
                        {},
                    ).get(
                        "error",
                        "",
                    )
                )

    # ------------------------------------------------------------------
    # Document Upload / Ingestion
    # ------------------------------------------------------------------

    def upload_document(
        self,
        uploaded_file,
    ):
        """
        Upload a PDF to FastAPI.

        FastAPI endpoint:
            POST /api/v1/upload

        Backend pipeline:
            PDF
            -> Docling
            -> chunking
            -> embeddings
            -> PostgreSQL/pgvector
        """

        if MOCK_MODE:
            return {
                "success": True,
                "document_id": (f"DOC-" f"{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                "file_name": uploaded_file.name,
                "chunks_created": 12,
                "status": "completed",
                "mode": "mock",
            }

        try:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            upload_timeout = httpx.Timeout(
                connect=10.0,
                read=600.0,
                write=120.0,
                pool=10.0,
            )

            response = httpx.post(
                f"{self.base_url}/api/v1/upload",
                files=files,
                timeout=upload_timeout,
            )

            response.raise_for_status()

            return response.json()

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": (
                    "Document processing timed out. "
                    "The PDF may still be processing "
                    "on the server. "
                    "Please check the FastAPI console."
                ),
            }

        except httpx.HTTPStatusError as exc:
            return {
                "success": False,
                "error": (
                    f"FastAPI returned HTTP "
                    f"{exc.response.status_code}: "
                    f"{exc.response.text}"
                ),
            }

        except httpx.RequestError as exc:
            return {
                "success": False,
                "error": (f"Unable to connect to FastAPI: " f"{exc}"),
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Mock Response
    # ------------------------------------------------------------------

    def _mock_query_response(
        self,
        query: str,
        session_id: str,
    ):
        """
        Temporary responses for frontend testing.
        """

        query_lower = query.lower()

        # --------------------------------------------------------------
        # RDBMS / SQL path
        # --------------------------------------------------------------

        if any(
            word in query_lower
            for word in [
                "balance",
                "transaction",
                "account",
                "customer",
                "loan outstanding",
                "purchase history",
            ]
        ):
            return {
                "success": True,
                "answer": (
                    "The requested banking data was "
                    "retrieved from the "
                    "transactional database."
                ),
                "query_path": "RDBMS",
                "route": "sql",
                "citations": [],
                "sources": [],
                "sql_query": (
                    "SELECT account_id, account_type " "FROM accounts " "LIMIT 10;"
                ),
                "sql_result": [
                    {
                        "account_id": 101,
                        "account_type": "Savings",
                    }
                ],
                "retry_count": 0,
                "confidence_score": 0.95,
                "session_id": session_id,
            }

        # --------------------------------------------------------------
        # Hybrid path
        # --------------------------------------------------------------

        if any(
            word in query_lower
            for word in [
                "qualify",
                "eligible",
                "customer loan",
                "my loan policy",
            ]
        ):
            return {
                "success": True,
                "answer": (
                    "The response was generated using "
                    "both customer banking data and "
                    "relevant banking policy documents."
                ),
                "query_path": "BOTH",
                "route": "both",
                "citations": [
                    {
                        "source": "loan_policy.pdf",
                        "page": 3,
                    }
                ],
                "sources": [],
                "sql_query": (
                    "SELECT account_id, outstanding " "FROM loan_accounts " "LIMIT 10;"
                ),
                "sql_result": [
                    {
                        "account_id": 101,
                        "outstanding": 250000.00,
                    }
                ],
                "retry_count": 0,
                "confidence_score": 0.92,
                "session_id": session_id,
            }

        # --------------------------------------------------------------
        # RAG path
        # --------------------------------------------------------------

        return {
            "success": True,
            "answer": (
                "Based on the available banking "
                "documents, the relevant policy "
                "information has been retrieved "
                "and summarized."
            ),
            "query_path": "RAG",
            "route": "rag",
            "citations": [
                {
                    "source": "banking_policy.pdf",
                    "page": 2,
                }
            ],
            "sources": [],
            "sql_query": None,
            "sql_result": None,
            "retry_count": 0,
            "confidence_score": 0.89,
            "session_id": session_id,
        }


# ----------------------------------------------------------------------
# Global API client used by Streamlit pages
# ----------------------------------------------------------------------

api_client = BankingAPIClient()
