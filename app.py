from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(title="Smart Banking Assistant")


app.include_router(router, prefix="/api")

# uv run uvicorn app:app --reload
