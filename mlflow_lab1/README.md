# MLflow Lab 1 - Complete Guide

MLflow lab demonstrating experiment tracking, model management, and deployment.

## Quick Start

```bash
cd mlflow_lab1
python -m venv mlflow_env
source mlflow_env/bin/activate  # Windows: mlflow_env\Scripts\activate
pip install -r requirements.txt

python linear_regression.py
mlflow ui
# Open: http://localhost:5000
```

## Complete Setup

### 1. Navigate and Create Environment

```bash
cd mlflow_lab1

# Option A: venv (Linux/Mac)
python -m venv mlflow_env
source mlflow_env/bin/activate

# Option B: venv (Windows)
python -m venv mlflow_env
mlflow_env\Scripts\activate

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
mlflow --version  # Verify installation
```

## How to Run Everything

### Run Everything

```bash
# Exercise 1: Basic tracking with artifacts (5 runs)
python linear_regression.py
python linear_regression.py 0.3 0.7
python linear_regression.py 0.1 0.1
python linear_regression.py 0.8 0.9
python linear_regression.py 1.0 0.5

# Exercise 2: Hyperparameter tuning (18 runs: 1 parent + 17 children)
python hyperparameter_tuning.py

# Exercise 3: Autologging demo (5 runs)
python autologging_example.py

# Exercise 4: Model serving and dependencies
python serving.py

# Exercise 5: Model comparison and analysis
python model_comparison.py

# View all results
mlflow ui
```

## What Each Script Does

### `linear_regression.py` - Enhanced Tracking
Trains ElasticNet model with comprehensive logging.

**Usage:**
```bash
python linear_regression.py                # Default: alpha=0.5, l1_ratio=0.5
python linear_regression.py 0.3 0.7        # Custom parameters
```

**Logs:**
- 5 tags (model_type, dataset, developer, version, training_date)
- 6 parameters (alpha, l1_ratio, random_state, etc.)
- 6 metrics (RMSE, MAE, R2, train_size, test_size, n_features)
- 3 plots (predictions, residuals, feature importance)
- 2 files (metrics.json, data_summary.csv)

---

### `hyperparameter_tuning.py` - Nested Runs
Systematic hyperparameter search comparing ElasticNet vs Random Forest.

**Usage:**
```bash
python hyperparameter_tuning.py
```

**Creates:**
- 1 parent run with experiment summary
- 9 ElasticNet child runs (3×3 grid: alpha × l1_ratio)
- 8 Random Forest child runs (various n_estimators × max_depth)
- Console output comparing all models
- Identifies and logs best model

---

### `autologging_example.py` - Automatic Tracking
Demonstrates MLflow autologging with Ridge, Lasso, and Gradient Boosting.

**Usage:**
```bash
python autologging_example.py
```

**Shows:**
- Automatic parameter/metric/model logging
- No explicit `mlflow.log_*` calls needed
- Cross-validation with autologging
- Comparison: autolog vs manual logging

---

### `serving.py` - Dependency Management
Shows different ways to package models with pip requirements.

**Usage:**
```bash
python serving.py
```

**Demonstrates:**
- Default requirements
- Custom requirements
- Extra requirements
- Requirements from file
- Constraints files

---

### `model_comparison.py` - Run Analysis
Programmatically searches, compares, and analyzes all runs.

**Usage:**
```bash
python model_comparison.py  # Run after training scripts
```

**Output:**
- Table of top 10 models by RMSE
- Best model summary
- Artifacts listing
- Query filter examples
- Model Registry guide

## 🖥️ MLflow UI

### Start the UI

```bash
mlflow ui                              # Default: localhost:5000
mlflow ui --port 8080                  # Custom port
mlflow ui --host 0.0.0.0 --port 5000   # Remote access
```

### What to Explore

1. **Experiments Table** - View all runs, sort by metrics
2. **Run Details** - Click any run for full info (params, metrics, tags, artifacts)
3. **Compare Runs** - Select multiple runs, click "Compare" button
4. **Artifacts** - View/download plots and files
5. **Search** - Filter runs: `metrics.rmse < 0.7`, `tags.model_type = "ElasticNet"`

## Expected Results

After running all exercises:

| Metric | Value |
|--------|-------|
| Total Runs | 30+ |
| Model Types | 5 (ElasticNet, RandomForest, Ridge, Lasso, GradientBoosting) |
| Visualizations | 15+ plots |
| Nested Run Hierarchy | 1 parent + 17 children |
| Artifacts | Plots, JSON, CSV files |

## Quick Command Reference

### Training

```bash
# Single run
python linear_regression.py

# Multiple runs with different params
python linear_regression.py 0.1 0.1
python linear_regression.py 0.5 0.5
python linear_regression.py 1.0 0.9

# Hyperparameter tuning
python hyperparameter_tuning.py

# Autologging
python autologging_example.py

# Model comparison
python model_comparison.py
```

### MLflow UI

```bash
mlflow ui                    # Start UI on port 5000
mlflow ui --port 8080       # Start on different port
```

### Model Serving

```bash
# Serve model (replace <RUN_ID> with actual ID from UI)
mlflow models serve -m runs:/<RUN_ID>/model -p 5001

# Test model
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"], "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]}}'
```

### MLflow Server

```bash
# SQLite backend
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000  # Linux/Mac
set MLFLOW_TRACKING_URI=http://localhost:5000     # Windows
```

### Utility Commands

```bash
# Check installation
mlflow --version
python -c "import mlflow; print(mlflow.__version__)"

# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 0

# Download artifacts
mlflow artifacts download --run-id <RUN_ID> --artifact-path model
```

## MLflow Concepts

### Tracking API

```python
import mlflow

with mlflow.start_run():
    # Parameters (hyperparameters)
    mlflow.log_param("alpha", 0.5)
    
    # Metrics (evaluation results)
    mlflow.log_metric("rmse", 0.75)
    
    # Tags (metadata)
    mlflow.set_tag("model_type", "ElasticNet")
    
    # Artifacts (files)
    mlflow.log_artifact("plot.png")
    
    # Models
    mlflow.sklearn.log_model(model, "model")
```

### Nested Runs

```python
with mlflow.start_run(run_name="Hyperparameter_Tuning"):
    for alpha in [0.1, 0.5, 1.0]:
        with mlflow.start_run(nested=True):
            # Train model with alpha
            # Log results
```

### Autologging

```python
mlflow.sklearn.autolog()
model.fit(X, y)  # Everything logged automatically!
```

### Search Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.rmse < 0.7",
    order_by=["metrics.rmse ASC"]
)
```

## Project Structure

```
mlflow_lab1/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── linear_regression.py         # Exercise 1: Basic tracking
├── hyperparameter_tuning.py     # Exercise 2: Nested runs
├── autologging_example.py       # Exercise 3: Autologging
├── serving.py                   # Exercise 4: Model serving
├── model_comparison.py          # Exercise 5: Run analysis
├── *.ipynb                      # Jupyter notebooks
└── mlruns/                      # Tracking data (auto-generated)
```