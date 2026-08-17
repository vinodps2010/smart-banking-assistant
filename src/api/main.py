from fastapi import FastAPI

from src.api.routes import router
from src.common.logger import logger

app = FastAPI(
    title="Smart Banking Assistant API",
    version="0.1.0",
    description="API for the Smart Banking RAG + SQL assistant.",
)


app.include_router(
    router,
    prefix="/api/v1",
)


# ============================================================
# Application Lifecycle
# ============================================================


@app.on_event("startup")
def startup_event():
    """
    Application startup logging.
    """

    

@app.on_event("shutdown")
def shutdown_event():
    """
    Application shutdown logging.
    """

    

# ============================================================
# Health Check
# ============================================================


@app.get("/health")
def health():
    """
    API health check.
    """

    logger.debug(
        "Health check requested",
    )

    return {
        "status": "ok",
        "service": "smart-banking-assistant",
    }
