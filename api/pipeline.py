# api/pipeline.py

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── CRITICAL: Import transformers so pickle can find the classes ───────────────
# joblib.load() needs these class definitions in scope to reconstruct the pipeline.
# Without this import, you get AttributeError on startup.
from .transformers import DataCleaner, RareCategoryCollapser, Winsorizer  # noqa: F401

# ── Configuration ─────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_PATH     = MODELS_DIR / "lead_scoring_model.pkl"
THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.pkl"

# ── Expected columns (original CSV names) ─────────────────────────────────────
_EXPECTED_COLUMNS = [
    "Lead Origin", "Lead Source", "Do Not Email", "Do Not Call",
    "TotalVisits", "Total Time Spent on Website", "Page Views Per Visit",
    "Last Activity", "Country", "Specialization",
    "How did you hear about X Education", "What is your current occupation",
    "What matters most to you in choosing a course", "Search", "Magazine",
    "Newspaper Article", "X Education Forums", "Newspaper",
    "Digital Advertisement", "Through Recommendations",
    "Receive More Updates About Our Courses", "Tags", "Lead Quality",
    "Update me on Supply Chain Content", "Get updates on DM Content",
    "Lead Profile", "City", "Asymmetrique Activity Index",
    "Asymmetrique Profile Index", "Asymmetrique Activity Score",
    "Asymmetrique Profile Score", "I agree to pay the amount through cheque",
    "A free copy of Mastering The Interview", "Last Notable Activity",
]

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing expected columns as NaN before pipeline entry."""
    for col in _EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ── Module-level state (loaded once at startup via lifespan) ──────────────────
_model_pipeline    = None
_optimal_threshold = None


def load_models() -> None:
    """
    Load model artifacts from disk into module-level globals.
    Called exactly once during FastAPI lifespan startup.
    Raises FileNotFoundError with a clear message if artifacts are missing.
    """
    global _model_pipeline, _optimal_threshold

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run the notebook to generate and save the .pkl file."
        )
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold artifact not found at {THRESHOLD_PATH}. "
            "Run the notebook to generate and save the .pkl file."
        )

    _model_pipeline    = joblib.load(MODEL_PATH)
    _optimal_threshold = joblib.load(THRESHOLD_PATH)


def is_loaded() -> bool:
    """Check whether models are currently loaded."""
    return _model_pipeline is not None and _optimal_threshold is not None


def _label(proba: float) -> str:
    """3-tier label based on threshold."""
    if proba >= _optimal_threshold * 1.4:
        return "Hot"
    elif proba >= _optimal_threshold:
        return "Warm"
    return "Cold"


def _score_to_response(proba: float) -> dict:
    """Convert a single probability float to API response dict."""
    return {
        "conversion_probability": round(float(proba), 4),
        "label": _label(proba),
        "threshold_used": round(float(_optimal_threshold), 4),
    }


# ── Public prediction functions ───────────────────────────────────────────────

def predict_single(df: pd.DataFrame) -> dict:
    """
    Predict conversion for a single lead.
    Input:  1-row DataFrame with original CSV column names.
    Output: dict matching PredictionResponse schema.
    """
    df = _ensure_columns(df)   
    proba = _model_pipeline.predict_proba(df)[:, 1][0]
    return _score_to_response(proba)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict conversion for multiple leads.
    Input:  n-row DataFrame with original CSV column names.
    Output: same DataFrame with 3 new columns appended.
    """
    df = _ensure_columns(df) 
    probas = _model_pipeline.predict_proba(df)[:, 1]

    result = df.copy()
    result["conversion_probability"] = np.round(probas, 4)
    result["label"]                  = [_label(p) for p in probas]
    result["threshold_used"]         = round(float(_optimal_threshold), 4)

    return result