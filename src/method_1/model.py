"""Method 1: k-Nearest Neighbors regression (Powerpredict).

Everything lives in ONE sklearn pipeline:  scaling + encoding + k-NN.
Because the preprocessing is baked into the pipeline, the test notebook only needs:
    model = load(); model.predict(raw_data)
i.e. it can feed the raw, unprocessed table straight in.

ML context: k-NN is an *instance-based* (lazy) supervised regressor. It does not
learn parameters during training; it simply stores the training data and, to
predict a new point, averages the targets of its k closest neighbours in feature
space. We use it here as Method 1 for the regression task "predict power
consumption from weather".

Runnable directly as a training script:
    python src/method_1/model.py
-> tunes k via cross-validation and saves the best model to models/knn_model.joblib
"""

import os   # filesystem paths (where to save/load the model)
import sys  # lets us extend the import search path at runtime

import joblib  # efficient save/load of fitted sklearn objects (model persistence)
from sklearn.model_selection import GridSearchCV   # exhaustive hyperparameter search + CV
from sklearn.neighbors import KNeighborsRegressor  # the k-NN regression algorithm itself
from sklearn.pipeline import Pipeline              # chains preprocessing + model into one object

# preprocessing.py lives in src/ (one directory above this file). We add that
# directory to the import path so "from preprocessing import ..." works no matter
# which folder the script is started from.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import build_preprocessor, load_data  # noqa: E402  (import after sys.path tweak)

# Absolute location of the final model file: <repo>/models/knn_model.joblib
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "knn_model.joblib"
)


def build_model(X, n_neighbors=5, weights="distance", encode_categoricals=True):
    """Build the (still untrained) k-NN pipeline.

    What it does: glues the preprocessor and a KNeighborsRegressor into a single
    sklearn Pipeline, so preprocessing and prediction always happen together.
    Precondition: X is a DataFrame matching the training schema; n_neighbors >= 1;
    weights is "uniform" or "distance".
    Postcondition: returns an unfitted Pipeline; calling .fit(X, y) trains it.
    ML context: the Pipeline is the standard way to prevent data leakage — the
    scaler/encoder are fitted only on training folds, never on test data.
    """
    pre = build_preprocessor(X, encode_categoricals=encode_categoricals)  # step 1: scale + encode
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)   # step 2: the k-NN regressor
    return Pipeline([("pre", pre), ("knn", knn)])                         # run step 1 then step 2


def train(X, y, n_neighbors=5, weights="distance", encode_categoricals=True):
    """Train a pipeline with a FIXED set of hyperparameters.

    What it does: builds the pipeline and fits it on (X, y).
    Precondition: X and y have the same number of rows; y is numeric (regression).
    Postcondition: returns a fitted Pipeline ready for .predict().
    ML context: the "model fitting" step. For k-NN, fitting just memorises the
    (preprocessed) training data — there are no weights to optimise.
    """
    model = build_model(X, n_neighbors, weights, encode_categoricals)  # assemble the pipeline
    model.fit(X, y)                                                    # learn from the data
    return model                                                       # hand back the fitted model


def tune(X, y, encode_categoricals=True, cv=5):
    """Find the best hyperparameters via cross-validation (scored by MAE).

    What it does: tries every combination of k and weighting, evaluates each with
    cross-validation, and keeps the combination with the lowest error.
    Precondition: X, y aligned by rows; cv >= 2; the grid below lists valid params.
    Postcondition: returns (best_estimator, search) where best_estimator is the
    pipeline refit on all of (X, y) using the winning hyperparameters.
    ML context: this is *hyperparameter tuning / model selection*. Cross-validation
    estimates how well each setting generalises to UNSEEN data (it splits the data
    into `cv` folds, trains on cv-1 and tests on the held-out fold, rotating). We
    score with MAE because that is the official competition metric.
    """
    model = build_model(X, encode_categoricals=encode_categoricals)  # base pipeline to tune
    grid = {                                          # the hyperparameter search space:
        "knn__n_neighbors": [3, 5, 11, 25, 51],       #   how many neighbours to average over
        "knn__weights": ["uniform", "distance"],      #   equal vote vs. closer = more influence
    }
    # n_jobs=1: k-NN stores all training data, so each parallel worker would hold a
    # full copy of the dataset + model. On JupyterHub's limited RAM that gets the
    # workers killed (TerminatedWorkerError/SIGKILL). Run sequentially to stay safe.
    search = GridSearchCV(
        model, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=1
    )
    search.fit(X, y)                                            # run the full CV search
    print(f"Best settings: {search.best_params_}")             # report the winning hyperparameters
    print(f"Best CV MAE: {-search.best_score_:.2f}")           # report its cross-validated error
    return search.best_estimator_, search                      # best pipeline + full search object


def save(model, path=MODEL_PATH):
    """Persist the fitted pipeline to disk.

    What it does: writes the model to `path`, compressed.
    Precondition: `model` is a fitted estimator; the parent folder is writable.
    Postcondition: a .joblib file exists at `path` that load() can restore.
    ML context: model persistence — we train once and reuse the model at inference
    time (in the test notebook) without retraining. compress=3 keeps it small,
    which matters because of the 50 MB submission limit.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)  # create the models/ folder if missing
    joblib.dump(model, path, compress=3)               # serialise the pipeline (compressed)


def load(path=MODEL_PATH):
    """Load a previously saved pipeline from disk.

    What it does: deserialises and returns the fitted model.
    Precondition: a valid .joblib model file exists at `path`.
    Postcondition: returns a ready-to-use fitted Pipeline.
    ML context: the *inference* side of persistence — used by the test notebook to
    get predictions without retraining.
    """
    return joblib.load(path)  # read the model back into memory


if __name__ == "__main__":
    # This block runs only when the file is executed directly (python model.py),
    # not when it is imported. It is the end-to-end training entry point.
    X, y = load_data()                                  # load features + target
    print(f"Data loaded: X={X.shape}, y={y.shape}")     # sanity-check the data dimensions

    # encode_categoricals=False drops the weather-text one-hot columns and cv=3
    # reduces the number of fits. k-NN keeps the whole dataset in memory and CV
    # makes several copies; one-hot expansion + 5-fold CV exhausts the hub's RAM
    # and the process gets OOM-killed ("Killed"). Numeric features alone stay
    # within memory and still beat the dummy baseline comfortably.
    best, _ = tune(X, y, encode_categoricals=False, cv=3)  # tune + select the best model

    save(best)                                          # persist the winning pipeline
    size_mb = os.path.getsize(MODEL_PATH) / 1e6         # measure the saved file size in MB
    print(f"Model saved: {MODEL_PATH} ({size_mb:.1f} MB)")  # report where + how big
    if size_mb > 50:                                    # enforce the submission size limit
        print("WARNING: model > 50 MB! Try again with encode_categoricals=False.")
