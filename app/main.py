"""FastAPI service for concrete compressive strength prediction.

The model is loaded once when the app starts and reused for every request.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from src.features import FEATURE_NAMES
from src.model import MODEL_VERSION, load_model

from .schemas import (
    BatchPrediction,
    BatchRequest,
    ConcreteMix,
    HealthResponse,
    Prediction,
)

# Holds the loaded model and its metadata for the life of the process.
state: dict = {"model": None, "metadata": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, metadata = load_model()
    state["model"] = model
    state["metadata"] = metadata
    yield
    state.clear()


app = FastAPI(
    title="Concrete Compressive Strength API",
    description=(
        "Predict the compressive strength (MPa) of a concrete mix from its "
        "eight ingredients. Model trained on the UCI Concrete Compressive "
        "Strength dataset."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def _to_frame(mixes: list[ConcreteMix]) -> pd.DataFrame:
    """Turn validated inputs into a DataFrame with columns in model order."""
    rows = [[getattr(m, name) for name in FEATURE_NAMES] for m in mixes]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def _version() -> str:
    return state["metadata"].get("model_version", MODEL_VERSION)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check. Reports whether the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=state["model"] is not None,
        model_version=_version(),
    )


@app.post("/predict", response_model=Prediction, tags=["predict"])
def predict(mix: ConcreteMix) -> Prediction:
    """Predict strength for a single mix."""
    frame = _to_frame([mix])
    value = float(state["model"].predict(frame)[0])
    return Prediction(predicted_strength_mpa=round(value, 2), model_version=_version())


@app.post("/predict/batch", response_model=BatchPrediction, tags=["predict"])
def predict_batch(request: BatchRequest) -> BatchPrediction:
    """Predict strength for a list of mixes in one call."""
    frame = _to_frame(request.items)
    values = state["model"].predict(frame)
    preds = [
        Prediction(predicted_strength_mpa=round(float(v), 2), model_version=_version())
        for v in values
    ]
    return BatchPrediction(predictions=preds, model_version=_version())
