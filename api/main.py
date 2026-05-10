# api/main.py

from contextlib import asynccontextmanager

import io
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
import csv
from fastapi.responses import StreamingResponse

from . import pipeline
from .schemas import (
    LeadFeatures,
    PredictionResponse,
    BatchPredictionResponse,
    CSVPredictionResponse,
)


# ── Lifespan: replaces the deprecated @app.on_event("startup") ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs at startup.
    Code after `yield` runs at shutdown (use for cleanup if needed).
    """
    pipeline.load_models()
    yield
    # nothing to clean up for in-memory models


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lead Scoring API",
    version="1.0.0",
    description="Predicts lead conversion probability using a trained LightGBM pipeline.",
    lifespan=lifespan,
)

@app.get("/", tags=["Root"])
def root():
    return {"message": "Lead Scoring API is running. Visit /docs for the interactive docs."}

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Returns model load status.
    Use this to verify the server is ready before sending predictions.
    """
    loaded = pipeline.is_loaded()
    return {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
    }


# ── Single prediction (JSON) ──────────────────────────────────────────────────
@app.post(
    "/predict/single",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Score a single lead via JSON",
)
def predict_single(lead: LeadFeatures):
    """
    Send one lead as a JSON body.
    Field names must match the original CSV column names exactly.
    Missing fields are allowed — the pipeline handles nulls internally.
    """
    try:
        df     = lead.to_dataframe()
        result = pipeline.predict_single(df)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        )


# ── Batch prediction (JSON list) ──────────────────────────────────────────────
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Score multiple leads via JSON array",
)
def predict_batch_json(leads: list[LeadFeatures]):
    """
    Send a list of leads as a JSON array.
    Returns scores for every lead in the same order.
    """
    if not leads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body contains no leads.",
        )
    try:
        df      = pd.concat([lead.to_dataframe() for lead in leads], ignore_index=True)
        results = pipeline.predict_batch(df)
        responses = [
            PredictionResponse(**row)
            for row in results[["conversion_probability", "label", "threshold_used"]]
            .to_dict(orient="records")
        ]
        return BatchPredictionResponse(total_leads=len(responses), results=responses)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {e}",
        )


# ── CSV upload prediction ─────────────────────────────────────────────────────
@app.post(
    "/predict/csv",
    response_model=CSVPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Score leads from an uploaded CSV file",
)
def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are accepted.",
        )

    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Could not parse CSV: {e}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV is empty.",
        )

    try:
        result_df = pipeline.predict_batch(df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        )

    # Stream CSV back instead of JSON
    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )