"""API tests driven through FastAPI's TestClient. No network needed."""


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_predict_returns_plausible_value(client, valid_mix):
    resp = client.post("/predict", json=valid_mix)
    assert resp.status_code == 200
    body = resp.json()
    value = body["predicted_strength_mpa"]
    # Real concrete strengths sit well inside this range. The dataset tops out
    # around 82 MPa, so anything outside 0 to 150 means the model is broken.
    assert isinstance(value, (int, float))
    assert 0 < value < 150
    assert body["model_version"]


def test_predict_batch(client, valid_mix):
    other = dict(valid_mix)
    other["age"] = 3  # younger concrete should be weaker
    resp = client.post("/predict/batch", json={"items": [valid_mix, other]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert all(0 < p["predicted_strength_mpa"] < 150 for p in preds)
    # The 28 day mix should not be weaker than the 3 day one.
    assert preds[0]["predicted_strength_mpa"] >= preds[1]["predicted_strength_mpa"]


def test_predict_rejects_missing_field(client, valid_mix):
    bad = dict(valid_mix)
    del bad["cement"]
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_negative_water(client, valid_mix):
    bad = dict(valid_mix)
    bad["water"] = -5.0
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_age(client, valid_mix):
    bad = dict(valid_mix)
    bad["age"] = 0  # must be at least 1 day
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
