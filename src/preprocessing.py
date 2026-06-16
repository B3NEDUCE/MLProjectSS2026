"""Shared data preparation for all methods (Powerpredict).

No model is trained in this file. It only does two jobs that EVERY supervised
machine-learning pipeline needs before the actual learning algorithm runs:

  1. Load the raw data and split it into features X and target y -> load_data()
  2. Build a preprocessor that turns the raw table into purely numeric input the
     model can consume (scale numbers + encode text)               -> build_preprocessor()

ML context: this is the "data preprocessing / feature engineering" stage of the
ML workflow. Keeping it in one shared module means Method 1 (k-NN) and Method 2
(Ridge) use *exactly the same* preprocessing, so the comparison between them is
fair and reproducible.
"""

import os  # only used to build/check filesystem paths (locate the dataset)

import pandas as pd  # tabular data handling; gives us the DataFrame type
from sklearn.compose import ColumnTransformer  # apply different transforms per column group
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # the two transforms we need

# Name of the column we want to predict (the regression target / label).
TARGET = "power_consumption"

# Candidate locations of the dataset, tried in order. On JupyterHub the data
# lives under /data/mlproject22 and is only shipped as a .zip (pandas.read_csv
# reads a .zip transparently). Locally it sits in <repo>/data. Listing both the
# unzipped and zipped names makes the code run unchanged in either environment.
_CANDIDATE_PATHS = [
    "/data/mlproject22/powerpredict.csv",      # hub, unzipped (if present)
    "/data/mlproject22/powerpredict.csv.zip",  # hub, zipped (the usual case)
    os.path.join(os.path.dirname(__file__), "..", "data", "powerpredict.csv"),      # local, unzipped
    os.path.join(os.path.dirname(__file__), "..", "data", "powerpredict.csv.zip"),  # local, zipped
]


def _resolve_data_path():
    """Return the first dataset path that actually exists on disk.

    What it does: scans the candidate locations and returns the first one found,
    so the rest of the code does not need to care where it runs.
    Precondition: _CANDIDATE_PATHS is a non-empty list of path strings.
    Postcondition: returns a path string; if none exist, returns the hub-zip path
    as a best-effort default (the later read will then raise a clear error).
    ML context: not an ML step itself, just environment plumbing so the same
    pipeline trains identically on the hub and locally (reproducibility).
    """
    for p in _CANDIDATE_PATHS:   # go through the candidates in priority order
        if os.path.exists(p):    # check whether this file is actually present
            return p             # first match wins -> use it
    return _CANDIDATE_PATHS[1]   # nothing found: fall back to the hub .zip path


# Resolve the path once at import time so every function shares the same default.
DEFAULT_DATA_PATH = _resolve_data_path()


def load_data(path=DEFAULT_DATA_PATH):
    """Read the CSV and split it into features X and target y.

    What it does: loads the raw table and separates the column we want to predict
    (y) from everything used to predict it (X).
    Precondition: `path` points to a readable CSV/zip that contains a column
    named TARGET ("power_consumption").
    Postcondition: returns (X, y) where X is a DataFrame of all feature columns
    and y is a Series with the target values; both have the same number of rows.
    ML context: produces the (X, y) pair that all of supervised learning is built
    on — the model learns the mapping X -> y.
    """
    df = pd.read_csv(path)            # read the whole table into a DataFrame
    X = df.drop(columns=[TARGET])     # features = every column EXCEPT the target
    y = df[TARGET]                    # target = the single column we predict
    return X, y                       # hand both back to the caller


def split_columns(X):
    """Automatically separate numeric columns from text (object) columns.

    What it does: looks at each column's dtype and groups them into "numeric"
    and "categorical/text".
    Precondition: X is a pandas DataFrame.
    Postcondition: returns (numeric, categorical), two lists of column names whose
    union is all columns of X and whose intersection is empty.
    ML context: different feature types need different preprocessing — numbers get
    scaled, text gets encoded. This split tells the ColumnTransformer which
    transform to apply where.
    """
    numeric = X.select_dtypes(exclude="object").columns.tolist()      # everything not text
    categorical = X.select_dtypes(include="object").columns.tolist()  # text columns only
    return numeric, categorical                                       # return the two groups


def build_preprocessor(X, encode_categoricals=True):
    """Build the preprocessing transformer (fitted later inside the pipeline).

    What it does: assembles a ColumnTransformer that
      - scales every numeric column with StandardScaler, and
      - one-hot-encodes the text columns (or drops them, if encode_categoricals=False).
    Precondition: X is a DataFrame with the same columns the model will later see;
    `encode_categoricals` is a bool.
    Postcondition: returns an *unfitted* ColumnTransformer ready to be placed in a
    sklearn Pipeline; it is fitted on training data and then re-applied to test data.
    ML context: this is feature scaling + categorical encoding.
      * StandardScaler is ESSENTIAL for k-NN: k-NN measures distances, and without
        scaling a large-range feature (e.g. pressure) would dominate a small-range
        one (e.g. rain), distorting the neighbour search.
      * OneHotEncoder turns text categories (weather descriptions) into 0/1 columns,
        because k-NN/Ridge can only do arithmetic on numbers.
      * handle_unknown="ignore" prevents a crash if the hidden test set contains a
        weather category never seen during training (it is encoded as all-zeros).
    """
    numeric, categorical = split_columns(X)                   # find the two column groups
    transformers = [("num", StandardScaler(), numeric)]       # always scale the numeric columns
    if encode_categoricals:                                   # optionally also use the text columns
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)  # encode text -> 0/1 columns
        )
    # remainder="drop": any column not listed above is left out of the output.
    return ColumnTransformer(transformers, remainder="drop")  # bundle the transforms into one object
