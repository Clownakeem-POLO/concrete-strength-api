# Concrete Compressive Strength API

**Live API:** https://concrete-strength-api-2lr4.onrender.com &middot; **interactive docs:** https://concrete-strength-api-2lr4.onrender.com/docs
(Free Render instance, so the first request after it has been idle takes ~50 seconds to wake.)

Predict the compressive strength of a concrete mix (in MPa) from its eight ingredients, served as a REST API.

This is a small but complete MLOps slice: train a model, serve it behind a clean FastAPI service, containerize it, test it, and wire up CI. I built it as a portfolio piece that sits where mechanical/materials engineering meets ML. Concrete strength is a problem I actually understand from the materials side, which is why I picked it over yet another generic demo dataset.

## The problem

Concrete gains strength as it cures, and the final strength depends on the mix: how much cement, how much water, what supplementary materials (slag, fly ash), how much superplasticizer, the aggregate split, and how long it cures. Getting a strength estimate from the mix design, before you pour and wait 28 days, is genuinely useful. This model does that.

It is a regression problem: eight numeric inputs, one numeric output (strength in MPa).

## Dataset

The real UCI "Concrete Compressive Strength" dataset by I-Cheng Yeh. 1030 mixes, 8 input features, 1 target.

- Source: UCI Machine Learning Repository, Concrete Compressive Strength.
- Raw file: `data/Concrete_Data.xls` (committed so training is reproducible).

Features (all component masses are kg per cubic metre of concrete):

| Feature | Meaning | Unit |
|---|---|---|
| `cement` | Cement | kg/m^3 |
| `blast_furnace_slag` | Ground granulated blast furnace slag | kg/m^3 |
| `fly_ash` | Fly ash | kg/m^3 |
| `water` | Water | kg/m^3 |
| `superplasticizer` | Superplasticizer admixture | kg/m^3 |
| `coarse_aggregate` | Coarse aggregate | kg/m^3 |
| `fine_aggregate` | Fine aggregate (sand) | kg/m^3 |
| `age` | Curing age | days |

Target: `compressive_strength` in MPa.

## Model and metrics

`train.py` compares three models and keeps the best by RMSE on a held-out 20% test set (206 mixes, fixed random seed so the split is reproducible).

| Model | RMSE (MPa) | MAE (MPa) | R2 |
|---|---|---|---|
| Linear regression (baseline) | 9.80 | 7.75 | 0.628 |
| Random forest | 5.59 | 3.80 | 0.879 |
| Gradient boosting (**selected**) | **4.65** | **3.30** | **0.916** |

Gradient boosting wins. An average error around 3.3 MPa on a target that ranges from about 2 to 83 MPa is a solid result for this dataset, and it matches what the published literature gets with tree ensembles. Strength is non-linear in the mix (the water to cement ratio especially), so the linear baseline leaves a lot on the table.

The trained model (`model/model.joblib`) and its metadata (`model/metadata.json`) are committed, so the API runs the moment you clone the repo. No training step needed to serve.

## API

FastAPI, with automatic OpenAPI docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check, reports if the model is loaded |
| POST | `/predict` | Predict strength for one mix |
| POST | `/predict/batch` | Predict for a list of mixes |

Inputs are validated by a Pydantic schema with per-field bounds (no negative water, curing age between 1 and 3650 days, and so on). Bad input returns a `422` with a clear error. The model is loaded once at startup, not per request.

### Examples

Health:

```bash
curl http://localhost:8000/health
```

```json
{"status":"ok","model_loaded":true,"model_version":"1.0.0"}
```

Single prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cement": 540.0,
    "blast_furnace_slag": 0.0,
    "fly_ash": 0.0,
    "water": 162.0,
    "superplasticizer": 2.5,
    "coarse_aggregate": 1040.0,
    "fine_aggregate": 676.0,
    "age": 28
  }'
```

```json
{"predicted_strength_mpa":76.86,"model_version":"1.0.0"}
```

Batch:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [
    {"cement":540,"blast_furnace_slag":0,"fly_ash":0,"water":162,"superplasticizer":2.5,"coarse_aggregate":1040,"fine_aggregate":676,"age":28},
    {"cement":540,"blast_furnace_slag":0,"fly_ash":0,"water":162,"superplasticizer":2.5,"coarse_aggregate":1040,"fine_aggregate":676,"age":3}
  ]}'
```

```json
{"predictions":[{"predicted_strength_mpa":76.86,"model_version":"1.0.0"},{"predicted_strength_mpa":53.43,"model_version":"1.0.0"}],"model_version":"1.0.0"}
```

A validation error (negative water) returns `422`:

```json
{"detail":[{"type":"greater_than","loc":["body","water"],"msg":"Input should be greater than 0"}]}
```

(These are the model's real outputs for those mixes. The 3 day mix cures to less strength than the 28 day one, which is what you want to see.)

## Run it locally

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# Optional: retrain from the raw data (a model is already committed)
python train.py

# Serve
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs.

## Run the tests

```bash
pytest -q
```

The tests use FastAPI's `TestClient`, so they start the app in-process with no network or running server. They cover `/health`, a valid prediction (asserting a plausible numeric result), batch prediction, and three validation-error cases.

## Run with Docker

```bash
docker build -t concrete-strength-api .
docker run -p 8000:8000 concrete-strength-api
```

Or with compose:

```bash
docker compose up --build
```

The image uses a slim Python base, runs as a non-root user, and has a container health check against `/health`.

## Deploy it live

The repo ships a `render.yaml` for Render's free tier (Docker runtime). Push this repo to GitHub, create a new Blueprint on Render pointing at it, and Render builds the Dockerfile and serves it with a public URL. The health check path is already set to `/health`. The free instance sleeps when idle and cold starts in a few seconds.

Hugging Face Spaces (Docker SDK) works too: create a Docker Space, add this repo, and it uses the same Dockerfile. Set the Space's app port to 8000.

## Project layout

```
concrete-strength-api/
  app/
    main.py          FastAPI app and routes
    schemas.py       Pydantic request/response models
  src/
    features.py      Feature names, units, raw column map
    data.py          Load and clean the UCI dataset
    model.py         Model definitions, metrics, save/load
  tests/
    conftest.py      Shared fixtures
    test_api.py      API tests via TestClient
  data/
    Concrete_Data.xls
  model/
    model.joblib     Trained model (committed)
    metadata.json    Metrics and training info
  train.py           Train, compare, save the best model
  Dockerfile
  docker-compose.yml
  render.yaml
  requirements.txt
  .github/workflows/ci.yml
  LICENSE
```

## Honest limitations

- The dataset is 1030 rows from one research program. The model learns that distribution, not every concrete on earth. Exotic admixtures or aggregates outside the training range will give shaky predictions.
- The age feature ranges from 1 to 365 days in the data. Predictions far outside that, for very old concrete, are extrapolation.
- There is no uncertainty estimate on a prediction. For engineering use you would want a confidence interval, not just a point value. That is the obvious next step.
- No authentication or rate limiting on the API. It is a demo service, not hardened for public production traffic.
- The model is a point-in-time artifact. In a real pipeline I would track data and model versions, monitor drift, and retrain on a schedule.

## License

MIT. See `LICENSE`.
