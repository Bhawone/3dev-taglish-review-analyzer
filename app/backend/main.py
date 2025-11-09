from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.inference.predictor import analyze_text


class AnalyzeRequest(BaseModel):
    review: str


app = FastAPI(
    title="Taglish Review Analyzer",
    description="Prototype API for sentiment, aspect, and deception analysis.",
    version="0.1.0",
)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        return analyze_text(request.review)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


