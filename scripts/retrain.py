"""
scripts/retrain.py
------------------
Retrains the lead-scoring pipeline on fresh data and overwrites the saved
artifacts only if the new model's ROC-AUC doesn't regress more than MIN_DELTA
below the current benchmark.

Artifacts managed:
  models/final_pipeline.pkl      ← sklearn Pipeline
  models/optimal_threshold.pkl   ← float, recomputed on holdout after retrain
  models/metrics.json            ← {roc_auc, updated_at}

Usage:
  python scripts/retrain.py
  python scripts/retrain.py --data path/to/new_data.csv   # override data path
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# ── resolve project root & make api/ importable ───────────────────────────────
ROOT = Path(__file__).resolve().parents[1]   # lead-scoring/
sys.path.insert(0, str(ROOT))                # lets pickle find api.transformers

# ── paths (all relative to project root) ──────────────────────────────────────
DEFAULT_DATA_PATH  = ROOT / "data"   / "lead_scoring.csv"
MODEL_PATH         = ROOT / "models" / "final_pipeline.pkl"
THRESHOLD_PATH     = ROOT / "models" / "optimal_threshold.pkl"
METRICS_PATH       = ROOT / "models" / "metrics.json"

# ── constants ─────────────────────────────────────────────────────────────────
TARGET       = "Converted"
HOLDOUT_SIZE = 0.20
RANDOM_STATE = 42
MIN_DELTA    = 0.01   # new ROC-AUC must be >= current - MIN_DELTA to overwrite

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pipeline() -> object:
    """Load the current production pipeline from disk."""
    log.info("Loading pipeline: %s", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_current_auc() -> float:
    """Read the benchmark ROC-AUC stored in metrics.json."""
    with open(METRICS_PATH) as f:
        return float(json.load(f)["roc_auc"])


def compute_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find the classification threshold that maximises Youden's J statistic
    (TPR - FPR), identical to the logic used during the original training run.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def save_artifacts(pipeline: object, threshold: float, roc_auc: float) -> None:
    """Overwrite pipeline pkl, threshold pkl, and metrics.json atomically."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    log.info("Saved pipeline       → %s", MODEL_PATH)

    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump(threshold, f)
    log.info("Saved threshold      → %s (%.4f)", THRESHOLD_PATH, threshold)

    metrics = {
        "roc_auc":    round(roc_auc, 4),
        "threshold":  round(threshold, 4),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics.json   → %s", METRICS_PATH)


# ── main ──────────────────────────────────────────────────────────────────────

def main(data_path: Path) -> None:
    # 1. Clone current pipeline — preserves architecture + hyperparams,
    #    gives us an unfitted copy so we don't mutate the production object.
    current_pipeline = load_pipeline()
    new_pipeline     = clone(current_pipeline)

    # 2. Load raw data and split
    log.info("Loading data: %s", data_path)
    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,           # preserve class ratio in both splits
    )
    log.info(
        "Split complete — train: %d rows | holdout: %d rows | positive rate: %.2f%%",
        len(X_train), len(X_holdout), y_holdout.mean() * 100,
    )

    # 3. Fit the cloned pipeline end-to-end on training data
    #    DataCleaner → RareCategoryCollapser → ColumnTransformer → LGBM
    log.info("Fitting new pipeline...")
    new_pipeline.fit(X_train, y_train)

    # 4. Evaluate on holdout
    y_proba  = new_pipeline.predict_proba(X_holdout)[:, 1]
    new_auc  = roc_auc_score(y_holdout, y_proba)
    log.info("New  ROC-AUC: %.4f", new_auc)

    # 5. Load benchmark
    current_auc = load_current_auc()
    log.info("Current ROC-AUC: %.4f", current_auc)

    # 6. Gate: overwrite only if regression is within tolerance
    gate = current_auc - MIN_DELTA
    delta = new_auc - current_auc

    if new_auc >= gate:
        log.info(
            "✓  New model passes gate  (%.4f >= %.4f, Δ = %+.4f) — updating artifacts.",
            new_auc, gate, delta,
        )
        new_threshold = compute_optimal_threshold(y_holdout, y_proba)
        log.info("Recomputed optimal threshold: %.4f", new_threshold)
        save_artifacts(new_pipeline, new_threshold, new_auc)
    else:
        log.warning(
            "✗  New model below gate  (%.4f < %.4f, Δ = %+.4f) — artifacts NOT updated.",
            new_auc, gate, delta,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain lead-scoring pipeline.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to raw CSV (default: data/lead_scoring.csv)",
    )
    args = parser.parse_args()
    main(args.data)