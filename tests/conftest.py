"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # Using the context manager runs the lifespan, which loads the model.
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_mix():
    return {
        "cement": 540.0,
        "blast_furnace_slag": 0.0,
        "fly_ash": 0.0,
        "water": 162.0,
        "superplasticizer": 2.5,
        "coarse_aggregate": 1040.0,
        "fine_aggregate": 676.0,
        "age": 28,
    }
