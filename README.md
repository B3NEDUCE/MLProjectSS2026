# ML Project SS2026 — Power Consumption Prediction

Regression task: predict `power_consumption` from multi-city weather data (`powerpredict.csv`).

## Problem
Given weather readings from 5 cities (temperature, pressure, humidity, wind, rain, snow, clouds), predict the total power consumption. This is a **supervised regression** problem.
---
---
---
---
# Preliminary Project idea :
## Methods
- **Method 1** (linear regression / regularized regression)
- **Method 2** (neural network)

## Project Structure
```
MLProjectSS2026/
├── notebooks/
│   ├── eda.ipynb           # Exploratory data analysis
│   └── training.ipynb      # Model training (separate from test notebook!)
├── src/
│   ├── preprocessing.py    # Shared data loading & preprocessing
│   ├── method_1/           # e.g. Linear Regression
│   │   └── model.py
│   └── method_2/           # e.g. Neural Network
│       └── model.py
├── models/                 # Saved model weights (gitignored)
└── report/                 # LaTeX report source
```

## Setup
```bash
pip install numpy pandas scikit-learn matplotlib torch jupyter
```

## Running
1. Start with `notebooks/eda.ipynb` to explore the dataset
2. Train models in `notebooks/training.ipynb`
3. Models are saved to `models/` (not committed to git)

> **Note:** Do not train inside the JupyterHub test notebook. Use a separate training notebook with a `TRAIN = False` flag in the test notebook.

## Deadlines
| What | When |
|------|------|
| Code + JupyterHub notebook | June 24, 2026 12:00 |
| Code presentation | June 25/26, 2026 |
| Report (PDF to OLAT) | July 9, 2026 23:59 |

## Model Size Limit
Saved model files must be **≤ 50 MB**.
