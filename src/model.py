"""Model building, metrics, and save/load helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES

MODEL_VERSION = "1.0.0"

# Where the trained artifacts live.
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"rmse": round(self.rmse, 4), "mae": round(self.mae, 4), "r2": round(self.r2, 4)}


def candidate_models() -> dict[str, Pipeline]:
    """The models I compare. Linear is the baseline, the tree ensembles are
    what usually win on this dataset because strength is non-linear in the mix.
    """
    return {
        "linear_regression": Pipeline(
            [("scale", StandardScaler()), ("reg", LinearRegression())]
        ),
        "random_forest": Pipeline(
            [("reg", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))]
        ),
        "gradient_boosting": Pipeline(
            [(
                "reg",
                GradientBoostingRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            )]
        ),
    }


def evaluate(model: Pipeline, X, y_true) -> Metrics:
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))
    return Metrics(rmse=rmse, mae=mae, r2=r2)


def save_model(model: Pipeline, metadata: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_model() -> tuple[Pipeline, dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    metadata: dict[str, Any] = {"model_version": MODEL_VERSION, "features": FEATURE_NAMES}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
    return model, metadata
