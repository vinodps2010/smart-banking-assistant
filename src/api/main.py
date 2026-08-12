from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="Smart Banking Assistant API",
    version="0.1.0",
    description="API for the Smart Banking RAG + SQL assistant.",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "smart-banking-assistant",
    }
