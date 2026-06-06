"""Shared data preparation for all methods (Powerpredict).

No model is trained here. This file only does two things:
1. Load data            -> load_data()
2. Build the preprocessor (scale numbers + encode weather text) -> build_preprocessor()

Both methods (k-NN and later Method 2) should reuse this.
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Column we want to predict
TARGET = "power_consumption"

# Default path to the file: <repo>/data/powerpredict.csv (relative to this file,
# so it does not matter which folder the script/notebook is started from).
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "powerpredict.csv"
)


def load_data(path=DEFAULT_DATA_PATH):
    """Read the CSV and split it into features X and target y."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def split_columns(X):
    """Automatically separate numeric columns from text columns (object dtype)."""
    numeric = X.select_dtypes(exclude="object").columns.tolist()
    categorical = X.select_dtypes(include="object").columns.tolist()
    return numeric, categorical


def build_preprocessor(X, encode_categoricals=True):
    """Build the preprocessor.

    - numeric columns -> StandardScaler (required for k-NN!)
    - text columns    -> OneHotEncoder if encode_categoricals=True,
                         otherwise they are dropped (experiment: with/without weather text).

    `handle_unknown="ignore"` makes sure that unknown weather categories in the
    (hidden) test set do not raise an error.
    """
    numeric, categorical = split_columns(X)
    transformers = [("num", StandardScaler(), numeric)]
    if encode_categoricals:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        )
    # remainder="drop": anything not listed is left out
    return ColumnTransformer(transformers, remainder="drop")
