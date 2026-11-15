"""Pydantic request and response models for the API.

The field constraints double as documentation and as a first line of defence.
The bounds are a bit wider than the training data range so the API does not
reject unusual but physically plausible mixes, while still catching obvious
nonsense like negative water or a 100 year cure time.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# One worked example, reused in the OpenAPI docs. These numbers are a real row
# from the dataset that cures to roughly 80 MPa.
_EXAMPLE_MIX = {
    "cement": 540.0,
    "blast_furnace_slag": 0.0,
    "fly_ash": 0.0,
    "water": 162.0,
    "superplasticizer": 2.5,
    "coarse_aggregate": 1040.0,
    "fine_aggregate": 676.0,
    "age": 28,
}


class ConcreteMix(BaseModel):
    """A single concrete mix. All component masses are kg per cubic metre."""

    cement: float = Field(..., ge=0, le=1000, description="Cement, kg/m^3")
    blast_furnace_slag: float = Field(..., ge=0, le=1000, description="Blast furnace slag, kg/m^3")
    fly_ash: float = Field(..., ge=0, le=1000, description="Fly ash, kg/m^3")
    water: float = Field(..., gt=0, le=500, description="Water, kg/m^3")
    superplasticizer: float = Field(..., ge=0, le=100, description="Superplasticizer, kg/m^3")
    coarse_aggregate: float = Field(..., ge=0, le=2000, description="Coarse aggregate, kg/m^3")
    fine_aggregate: float = Field(..., ge=0, le=2000, description="Fine aggregate, kg/m^3")
    age: int = Field(..., ge=1, le=3650, description="Curing age in days")

    model_config = {"json_schema_extra": {"examples": [_EXAMPLE_MIX]}}


class BatchRequest(BaseModel):
    items: list[ConcreteMix] = Field(..., min_length=1, max_length=1000)

    model_config = {"json_schema_extra": {"examples": [{"items": [_EXAMPLE_MIX]}]}}


class Prediction(BaseModel):
    predicted_strength_mpa: float = Field(..., description="Predicted 28 day style compressive strength, MPa")
    model_version: str


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
