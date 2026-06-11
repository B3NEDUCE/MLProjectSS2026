# ML Project SS2026 — Power Consumption Prediction (Powerpredict)

**Regression task:** predict the total `power_consumption` from weather data of 5 cities
(temperature, pressure, humidity, wind, rain, snow, clouds).
Evaluation metric: **Mean Absolute Error (MAE)** — lower is better.

## Methods
- **Method 1 — k-Nearest Neighbors (k-NN)** ✅ implemented (`src/method_1/`)
- **Method 2 — Ridge Regression (Regularized Linear Regression)** ✅ implemented (`src/method_2/`)

> The two methods must come from **different PS topics**.

---

## 📁 Project structure — what goes where?

    MLProjectSS2026/
    ├── README.md               # this file
    ├── .gitignore
    │
    ├── data/                   # dataset – LOCAL, NOT committed (gitignored)
    │   ├── powerpredict.csv         # unzipped dataset (for EDA/training)
    │   └── powerpredict.csv.zip      # zipped (this is what the test notebook reads)
    │
    ├── notebooks/
    │   ├── powerpredict.ipynb       # ❗ TEST NOTEBOOK (submission!) – never rename
    │   ├── training.ipynb           # training & experiments (training happens here)
    │   └── eda.ipynb                # data analysis
    │
    ├── src/
    │   ├── preprocessing.py         # shared data preparation (both methods)
    │   ├── method_1/
    │   │   └── model.py             # k-NN model + training script
    │   └── method_2/
    │       └── model.py             # Ridge regression model + training script
    │
    ├── models/                 # saved models – LOCAL, gitignored, ≤ 50 MB
    │   ├── knn_model.joblib         # produced by method 1 training
    │   └── linreg_model.joblib      # produced by method 2 training
    │
    ├── report/                 # report (graded!)
    │   ├── report-guidelines.pdf
    │   └── latex_template/          # report.tex + ieeeconf.cls
    │
    └── docs/                   # reference material
        ├── project-instructions.pdf
        └── examples/                # example report + model card

**Placement rules:**
- **Code** → `src/` (shared logic in `preprocessing.py`, method-specific in `method_X/`)
- **Notebooks** → `notebooks/`
- **Data** → `data/` (never commit!)
- **Trained models** → `models/` (never commit, ≤ 50 MB)
- **Report** → `report/`  ·  **PDFs / instructions** → `docs/`

---

## 📄 What does each file do?

### `src/preprocessing.py` — shared data preparation
Does **not** train a model, only prepares data. Used by both methods.
- `load_data(path)` → reads the CSV, returns `X` (features) and `y` (`power_consumption`).
- `split_columns(X)` → automatically separates numeric from text columns.
- `build_preprocessor(X, encode_categoricals=True)` → builds the preprocessor:
  numbers → `StandardScaler` (**required for k-NN**), weather text → `OneHotEncoder`
  (or drop it, for the "with/without weather text" experiment).

### `src/method_1/model.py` — k-NN model (Method 1)
Everything in **one sklearn pipeline** (scaling + encoding + k-NN).
- `build_model(...)` → builds the pipeline.
- `train(X, y, ...)` → trains with fixed settings.
- `tune(X, y)` → finds the best `k`/`weights` via cross-validation (metric: MAE).
- `save(model)` / `load()` → saves/loads `models/knn_model.joblib` (compressed).
- **Runnable directly:** `python src/method_1/model.py` tunes and saves the best model.

### `src/method_2/model.py` — Ridge Regression model (Method 2)
Everything in **one sklearn pipeline** (scaling + encoding + Ridge).
- `build_model(...)` → builds the pipeline.
- `train(X, y, ...)` → trains with fixed settings.
- `tune(X, y)` → finds the best regularization parameter `alpha` via cross-validation (metric: MAE).
- `save(model)` / `load()` → saves/loads `models/linreg_model.joblib` (compressed).
- **Runnable directly:** `python src/method_2/model.py` tunes and saves the best model.

### `notebooks/powerpredict.ipynb` — test notebook (submission!) ⚠️
The official notebook from JupyterHub. **Do not rename, do not replace the file.**
- Loads the data and calls `get_score()` (computes the MAE).
- We implemented **only** the function `leader_board_predict_fn(values)`: it loads either
  `knn_model.joblib` or `linreg_model.joblib` and returns `model.predict(values)`. `TRAIN = False` (no training here).
- For submission, your chosen model `.joblib` file must sit **next to the notebook**.

### `notebooks/training.ipynb` — training & experiments
Training happens here (separate from the test notebook). Contains: dummy baseline, `k` and `alpha` tuning
via CV, "encoding on/off" comparison, saving the final model + **size check (≤ 50 MB)**.

### `notebooks/eda.ipynb` — data analysis
Target distribution, missing values, feature scales (motivates the scaling), strongest
correlations. Provides material for the report.

---

## ⚙️ Requirements
This project needs the following Python packages:

| Package | Used for |
|---------|----------|
| `numpy` | numerical arrays |
| `pandas` | loading / handling the dataset |
| `scikit-learn` | preprocessing, k-NN/Ridge models, cross-validation, MAE |
| `joblib` | saving / loading the trained models |
| `matplotlib` | plots in the EDA notebook |
| `jupyter` | running the notebooks |

Install them with:

    pip install numpy pandas scikit-learn joblib matplotlib jupyter

> Note: on **JupyterHub** these are already installed. Locally, create a venv if needed.
> `torch` is **not** required for k-NN or Ridge — only add it if Method 3 becomes a neural network.

## ▶️ How to train the models — step by step

You can train either **locally (VSCode + WSL)** or on **JupyterHub**. Both produce the model files
`models/knn_model.joblib` and `models/linreg_model.joblib`, which the test notebook then loads.

### Variant A — Local (VSCode + WSL terminal)

Open a terminal in VSCode: **Terminal → New Terminal**. Because the project lives in WSL,
this is already a WSL (bash) shell. Run everything from the project root
(`~/MLProjectSS2026`).

**One-time setup** (installs pip support + an isolated environment — only needed once):

    # 1) install venv support (asks for your WSL/Ubuntu password)
    sudo apt update && sudo apt install -y python3.12-venv

    # 2) create a virtual environment inside the project
    python3 -m venv .venv

    # 3) activate it  ->  your prompt now shows (.venv)
    source .venv/bin/activate

    # 4) install the required packages
    pip install numpy==1.26.4 pandas scikit-learn==1.6.1 joblib matplotlib jupyter

**Train the models** (in the same terminal):

    # Train Method 1 (k-NN)
    python src/method_1/model.py

    # Train Method 2 (Ridge Regression)
    python src/method_2/model.py

This tunes the hyperparameters via cross-validation, saves the `.joblib` files to `models/`, and prints their sizes
(must be ≤ 50 MB).

**Test it locally:**

    # the test notebook reads the dataset zip from its OWN folder, so put a copy there:
    cp data/powerpredict.csv.zip notebooks/

Then open `notebooks/powerpredict.ipynb` in VSCode, choose the **`.venv`** kernel (top-right),
and **Run All**. At the bottom, *"Train Dataset Mean Absolute Error"* must be a **number** and
**smaller** than the dummy MAE. *(The hidden "Test Dataset" score only appears on JupyterHub.)*

> Next time you just run `source .venv/bin/activate` — steps 1–4 are one-time.

### Variant B — JupyterHub (browser)

Packages are already installed there, **and** the hidden test data lives only here.

1. Log in to the course **JupyterHub** in your browser.
2. **Upload** `src/`, `notebooks/training.ipynb` and `notebooks/powerpredict.ipynb`.
   Do **not** upload the dataset — it is already at `/data/mlproject22`.
3. **Train:** open a terminal (**New → Terminal**) and run `python src/method_1/model.py` or `python src/method_2/model.py`,
   **or** open `training.ipynb` and **Run All** → creates the saved models.
4. Put the model **next to the notebook**: from the notebook's folder run
   `cp models/knn_model.joblib .` or `cp models/linreg_model.joblib .`
5. Open `powerpredict.ipynb` and **Run All** → now **both** *"Train Dataset MAE"* and
   *"Test Dataset MAE"* (hidden data) appear.

> **Important:** never train inside the test notebook. Training runs in `training.ipynb` /
> `model.py`; the test notebook has `TRAIN = False`.

> Tip: explore the data first with `notebooks/eda.ipynb` (works in both variants).

---

## 📦 Submission (code deadline: 2026-06-24, 12:00)
1. **ZIP on OLAT:** runnable training/eval scripts (`src/`, `notebooks/training.ipynb`).
2. **JupyterHub:** `powerpredict.ipynb` + your best chosen model file (`knn_model.joblib` or `linreg_model.joblib` next to the notebook).
3. **Do NOT upload datasets** (they live on the hub).

## 🗓️ Deadlines
| What | When |
|------|------|
| Code + JupyterHub notebook | 2026-06-24, 12:00 |
| Code presentation | 2026-06-25/26 |
| Report (PDF on OLAT) | 2026-07-09, 23:59 |

## 📏 Model size limit
Saved model files must be **≤ 50 MB**.
