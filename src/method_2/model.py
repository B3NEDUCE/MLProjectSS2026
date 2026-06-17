"""Method 2: Ridge Regression (Powerpredict).

Everything lives in ONE sklearn pipeline:  scaling + encoding + Ridge.
Because the preprocessing is baked into the pipeline, the test notebook only needs:
    model = load(); model.predict(raw_data)
i.e. it can feed the raw, unprocessed table straight in.

ML context: Ridge Regression is a supervised, *parametric* linear model. It learns
a set of weights (one for each feature) during training. To prevent *overfitting* 
(where weights become wildly large to perfectly fit noisy data), it adds an L2 
penalty to the error function. This forces the model to keep weights as small and 
simple as possible. We use it here as Method 2 for the regression task.

Runnable directly as a training script:
    python src/method_2/model.py
-> tunes alpha via cross-validation and saves the best model to models/linreg_model.joblib
"""

import os           # filesystem paths (where to save/load the model)
import sys          # lets us extend the import search path at runtime
import numpy as np  # used to generate the exponential grid for hyperparameter tuning
import joblib       # efficient save/load of fitted sklearn objects (model persistence)

from sklearn.linear_model import Ridge             # the Ridge Regression algorithm itself
from sklearn.model_selection import GridSearchCV   # exhaustive hyperparameter search + CV
from sklearn.pipeline import Pipeline              # chains preprocessing + model into one object

# preprocessing.py lives in src/ (one directory above this file). We add that
# directory to the import path so "from preprocessing import ..." works no matter
# which folder the script is started from.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import build_preprocessor, load_data  # noqa: E402  (import after sys.path tweak)

# Absolute location of the final model file: <repo>/models/linreg_model.joblib
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "linreg_model.joblib"
)


def build_model(X, alpha=1.0, encode_categoricals=True):
    """Build the (still untrained) Ridge Regression pipeline.

    What it does: glues the preprocessor and a Ridge regressor into a single
    sklearn Pipeline, so preprocessing and prediction always happen together.
    Precondition: X is a DataFrame matching the training schema; alpha >= 0.
    Postcondition: returns an unfitted Pipeline; calling .fit(X, y) trains it.
    ML context: the Pipeline prevents data leakage. The scaler/encoder are fitted 
    only on training folds, never on test data. Scaling is particularly critical 
    for Ridge, because the L2 penalty assumes all features are on the same scale.
    """
    pre = build_preprocessor(X, encode_categoricals=encode_categoricals)  # step 1: scale + encode
    linreg = Ridge(alpha=alpha)                                           # step 2: the Ridge regressor
    return Pipeline([("pre", pre), ("linreg", linreg)])                   # run step 1 then step 2


def train(X, y, alpha=1.0, encode_categoricals=True):
    """Train a pipeline with a FIXED set of hyperparameters.

    What it does: builds the pipeline and fits it on (X, y).
    Precondition: X and y have the same number of rows; y is numeric (regression).
    Postcondition: returns a fitted Pipeline ready for .predict().
    ML context: the "model fitting" step. For Ridge, fitting solves the regularized 
    least-squares math equation to find the optimal weights for the weather clues.
    """
    model = build_model(X, alpha, encode_categoricals)  # assemble the pipeline
    model.fit(X, y)                                     # learn the weights from the data
    return model                                        # hand back the fitted model


def tune(X, y, encode_categoricals=True, cv=5):
    """Find the best regularization hyperparameter via cross-validation (scored by MAE).

    What it does: tries different levels of strictness (alpha), evaluates each with
    cross-validation, and keeps the value with the lowest error.
    Precondition: X, y aligned by rows; cv >= 2.
    Postcondition: returns (best_estimator, search) where best_estimator is the
    pipeline refit on all of (X, y) using the winning alpha.
    ML context: this is *hyperparameter tuning*. We search alpha on a logarithmic 
    scale (exponential jumps) because the optimal penalty could be tiny or huge. 
    Cross-validation estimates how well the strictness prevents overfitting on unseen data.
    """
    model = build_model(X, encode_categoricals=encode_categoricals)  # base pipeline to tune
    grid = {                                          
        # the hyperparameter search space (inspired by PS Homework 02):
        # scans values from very loose (e^-15) to very strict (e^5)
        "linreg__alpha": [np.exp(p) for p in [-15, -10, -5, 0, 5]]
    }
    
    # n_jobs=1: Ridge evaluates very fast, but with one-hot encoding, the dataset 
    # becomes very wide. Parallel workers (n_jobs=-1) would duplicate this wide 
    # dataset in RAM, triggering the Linux OOM-killer. We run sequentially to stay safe.
    # verbose=3: prints progress to the terminal so we know it hasn't frozen.
    search = GridSearchCV(
        model, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=1, verbose=3
    )
    search.fit(X, y)                                           # run the full CV search
    print(f"Best settings: {search.best_params_}")             # report the winning alpha
    print(f"Best CV MAE: {-search.best_score_:.2f}")           # report its cross-validated error
    return search.best_estimator_, search                      # best pipeline + full search object


def save(model, path=MODEL_PATH):
    """Persist the fitted pipeline to disk.

    What it does: writes the model to `path`, compressed.
    Precondition: `model` is a fitted estimator; the parent folder is writable.
    Postcondition: a .joblib file exists at `path` that load() can restore.
    ML context: model persistence. Unlike k-NN (which stores the whole dataset), 
    Ridge only saves a tiny list of feature weights. This easily stays under the 
    50 MB submission limit (it will likely be < 0.1 MB).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)  # create the models/ folder if missing
    joblib.dump(model, path, compress=3)               # serialise the pipeline (compressed)


def load(path=MODEL_PATH):
    """Load a previously saved pipeline from disk.

    What it does: deserialises and returns the fitted model.
    Precondition: a valid .joblib model file exists at `path`.
    Postcondition: returns a ready-to-use fitted Pipeline.
    ML context: the *inference* side of persistence — used by the test notebook to
    get predictions without retraining the weights.
    """
    return joblib.load(path)  # read the model back into memory


if __name__ == "__main__":
    # This block runs only when the file is executed directly (python model.py),
    # not when it is imported. It is the end-to-end training entry point.
    X, y = load_data()                                  # load features + target
    print(f"Data loaded: X={X.shape}, y={y.shape}")     # sanity-check the data dimensions
    
    best, _ = tune(X, y)                                # tune + select the best model
    
    save(best)                                          # persist the winning pipeline
    size_mb = os.path.getsize(MODEL_PATH) / 1e6         # measure the saved file size in MB
    print(f"Model saved: {MODEL_PATH} ({size_mb:.1f} MB)")  # report where + how big
    if size_mb > 50:                                    # enforce the submission size limit
        print("WARNING: model > 50 MB! Try again with encode_categoricals=False.")