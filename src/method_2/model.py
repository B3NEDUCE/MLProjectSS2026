import os
import sys
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import build_preprocessor, load_data

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "linreg_model.joblib"
)

def build_model(X, alpha=1.0, encode_categoricals=True):
    pre = build_preprocessor(X, encode_categoricals=encode_categoricals)
    linreg = Ridge(alpha=alpha)
    return Pipeline([("pre", pre), ("linreg", linreg)])

def train(X, y, alpha=1.0, encode_categoricals=True):
    model = build_model(X, alpha, encode_categoricals)
    model.fit(X, y)
    return model

def tune(X, y, encode_categoricals=True, cv=5):
    model = build_model(X, encode_categoricals=encode_categoricals)
    
    grid = {
        "linreg__alpha": [np.exp(p) for p in [-15, -10, -5, 0, 5]]
    }
    
    search = GridSearchCV(
        model, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=1, verbose=3
    )
    search.fit(X, y)
    print(f"Best settings: {search.best_params_}")
    print(f"Best CV MAE: {-search.best_score_:.2f}")
    return search.best_estimator_, search

def save(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path, compress=3)

def load(path=MODEL_PATH):
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