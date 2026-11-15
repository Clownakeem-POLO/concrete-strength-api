"""Train the concrete compressive strength model.

Loads the UCI data, compares a linear baseline against two tree ensembles,
keeps the best on a held-out test set, prints the metrics, and saves the
winner plus its feature list to model/.

Run it with:
    python train.py
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone

import sklearn
from sklearn.model_selection import train_test_split

from src.data import load_dataset
from src.features import FEATURE_NAMES, TARGET_NAME
from src.model import (
    MODEL_VERSION,
    candidate_models,
    evaluate,
    save_model,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    print("Loading dataset...")
    df = load_dataset()
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  train={len(X_train)}  test={len(X_test)}")

    results = {}
    print("\nComparing models (metrics on the held-out test set):")
    print(f"{'model':<22}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = (model, metrics)
        print(f"{name:<22}{metrics.rmse:>10.3f}{metrics.mae:>10.3f}{metrics.r2:>10.3f}")

    # Pick the model with the lowest RMSE on the test set.
    best_name = min(results, key=lambda n: results[n][1].rmse)
    best_model, best_metrics = results[best_name]
    print(f"\nBest model: {best_name}")
    print(f"  RMSE={best_metrics.rmse:.3f} MPa  MAE={best_metrics.mae:.3f} MPa  R2={best_metrics.r2:.3f}")

    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": best_name,
        "features": FEATURE_NAMES,
        "target": TARGET_NAME,
        "target_units": "MPa",
        "metrics": {name: m.as_dict() for name, (_, m) in results.items()},
        "best_metrics": best_metrics.as_dict(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    save_model(best_model, metadata)
    print("\nSaved model/model.joblib and model/metadata.json")


if __name__ == "__main__":
    main()
