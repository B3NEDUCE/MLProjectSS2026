"""Method 1: k-Nearest Neighbors regression (Powerpredict).

Everything lives in one sklearn pipeline:  scaling + encoding + k-NN.
This way the test notebook only needs:  model = load(); model.predict(raw_data).

Runnable directly as a training script:
    python src/method_1/model.py
-> tunes k via cross-validation and saves the best model to models/knn_model.joblib
"""

import os
import sys

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

# preprocessing.py lives in src/ (one directory above this file)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import build_preprocessor, load_data  # noqa: E402

# Location of the final model: <repo>/models/knn_model.joblib
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "knn_model.joblib"
)


def build_model(X, n_neighbors=5, weights="distance", encode_categoricals=True):
    """Build the k-NN pipeline (not yet trained)."""
    pre = build_preprocessor(X, encode_categoricals=encode_categoricals)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    return Pipeline([("pre", pre), ("knn", knn)])


def train(X, y, n_neighbors=5, weights="distance", encode_categoricals=True):
    """Train a pipeline with fixed settings."""
    model = build_model(X, n_neighbors, weights, encode_categoricals)
    model.fit(X, y)
    return model


def tune(X, y, encode_categoricals=True, cv=5):
    """Search the best k (and weights) via cross-validation, scored by MAE."""
    model = build_model(X, encode_categoricals=encode_categoricals)
    grid = {
        "knn__n_neighbors": [3, 5, 11, 25, 51],
        "knn__weights": ["uniform", "distance"],
    }
    search = GridSearchCV(
        model, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
    )
    search.fit(X, y)
    print(f"Best settings: {search.best_params_}")
    print(f"Best CV MAE: {-search.best_score_:.2f}")
    return search.best_estimator_, search


def save(model, path=MODEL_PATH):
    """Save the pipeline (compressed, to stay under 50 MB)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path, compress=3)


def load(path=MODEL_PATH):
    """Load a saved pipeline."""
    return joblib.load(path)


if __name__ == "__main__":
    X, y = load_data()
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    best, _ = tune(X, y)
    save(best)
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"Model saved: {MODEL_PATH} ({size_mb:.1f} MB)")
    if size_mb > 50:
        print("WARNING: model > 50 MB! Try again with encode_categoricals=False.")
