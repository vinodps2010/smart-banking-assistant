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

        # Timeout configuration
        #
        # Upload/ingestion can take several minutes because:
        # 1. PDF is uploaded
        # 2. Docling parses the PDF
        # 3. Tables/images are extracted
        # 4. Embeddings are generated
        # 5. Chunks are stored in PostgreSQL/pgvector
        #
        # Therefore read timeout is deliberately long.
        self.timeout = httpx.Timeout(
            connect=10.0,
            read=600.0,
            write=120.0,
            pool=10.0,
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self):
        """Check FastAPI backend health."""

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

        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def query_assistant(
        self,
        query: str,
        session_id: str | None = None,
    ):
        """
        Send one user query to LangGraph through FastAPI.

        LangGraph internally decides:
        RAG / RDBMS / BOTH.
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

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
            }

    # ------------------------------------------------------------------
    # Document upload / ingestion
    # ------------------------------------------------------------------

    def upload_document(self, uploaded_file):
        """
        Upload a PDF to FastAPI.

        FastAPI endpoint:
            POST /api/v1/ingest

        The backend then runs the existing:
            Docling -> chunking -> embedding -> PostgreSQL/pgvector
        ingestion pipeline.
        """

        if MOCK_MODE:
            return {
                "success": True,
                "document_id": (f"DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
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

            # IMPORTANT:
            # Ingestion can take several minutes.
            #
            # connect = time allowed to establish connection
            # write   = time allowed to upload the PDF
            # read    = time allowed for FastAPI to finish
            #           Docling + embedding + DB storage
            #
            timeout = httpx.Timeout(
                connect=10.0,
                read=600.0,
                write=120.0,
                pool=10.0,
            )

            response = httpx.post(
                f"{self.base_url}/api/v1/upload",
                files=files,
                timeout=timeout,
            )

            response.raise_for_status()

            return response.json()

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": (
                    "Document processing timed out. "
                    "The PDF may still be processing on the server. "
                    "Please check the FastAPI console."
                ),
            }

        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": (
                    f"FastAPI returned HTTP "
                    f"{e.response.status_code}: "
                    f"{e.response.text}"
                ),
            }

        except httpx.RequestError as e:
            return {
                "success": False,
                "error": (f"Unable to connect to FastAPI: {str(e)}"),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Mock response
    # ------------------------------------------------------------------

    def _mock_query_response(
        self,
        query: str,
        session_id: str,
    ):
        """
        Temporary responses for frontend testing.

        These simulate the actual response expected from
        the LangGraph + FastAPI backend.
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
                    "The requested banking data was retrieved "
                    "from the transactional database."
                ),
                "query_path": "RDBMS",
                "citations": [],
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
                    "The response was generated using both "
                    "customer banking data and relevant "
                    "banking policy documents."
                ),
                "query_path": "BOTH",
                "citations": [
                    {
                        "source": "loan_policy.pdf",
                        "page": 3,
                    }
                ],
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
                "Based on the available banking documents, "
                "the relevant policy information has been "
                "retrieved and summarized."
            ),
            "query_path": "RAG",
            "citations": [
                {
                    "source": "banking_policy.pdf",
                    "page": 2,
                }
            ],
            "sql_query": None,
            "sql_result": None,
            "retry_count": 0,
            "confidence_score": 0.89,
            "session_id": session_id,
        }


# Global API client used by Streamlit pages
api_client = BankingAPIClient()
