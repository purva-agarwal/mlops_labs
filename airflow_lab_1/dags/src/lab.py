# src/lab.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import numpy as np
from typing import Tuple, Dict, Any

# Helper: base directories (relative to this file)
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

REQUIRED_COLUMNS = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]

def load_data_file() -> str:
    src_path = os.path.join(DATA_DIR, "file.csv")
    df = pd.read_csv(src_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw CSV: {missing}")
    out_path = os.path.join(DATA_DIR, "loaded.pkl")
    df.to_pickle(out_path)
    return out_path

def clean_data_file(loaded_path: str) -> str:
    df = pd.read_pickle(loaded_path)
    df = df.dropna()
    cleaned = df[REQUIRED_COLUMNS].copy()
    out_path = os.path.join(DATA_DIR, "cleaned.pkl")
    cleaned.to_pickle(out_path)
    return out_path

def validate_loaded_file(loaded_path: str) -> Dict[str, Any]:
    df = pd.read_pickle(loaded_path)
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "sample_head": df.head(1).to_dict(orient="records"),
    }

def scale_data_file(cleaned_path: str) -> str:
    df = pd.read_pickle(cleaned_path)
    arr = MinMaxScaler().fit_transform(df[REQUIRED_COLUMNS].values)
    out_path = os.path.join(DATA_DIR, "scaled.npy")
    np.save(out_path, arr)
    # save scaler
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(MinMaxScaler().fit(df[REQUIRED_COLUMNS].values), f)
    return out_path

def merge_scaled_file(scaled_path: str) -> str:
    return scaled_path

def build_save_model_file(scaled_path: str, filename: str = "model.sav") -> Tuple[list, str]:
    arr = np.load(scaled_path)
    sse = []
    last_model = None
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
        kmeans.fit(arr)
        sse.append(float(kmeans.inertia_))
        last_model = kmeans
    saved_path = os.path.join(MODEL_DIR, filename)
    with open(saved_path, "wb") as f:
        pickle.dump(last_model, f)
    return sse, saved_path

def validate_after_build_file(merged_path: str) -> Dict[str, Any]:
    arr = np.load(merged_path)
    return {"preprocessed_shape": list(arr.shape), "preprocessed_dtype": str(arr.dtype)}

def load_model_elbow_file(filename_or_path: str, sse: list) -> int:
    # Load saved model
    model_path = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(MODEL_DIR, filename_or_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Elbow logging
    try:
        kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
        print(f"Optimal no. of clusters (elbow): {kl.elbow}")
    except Exception:
        pass

    # Read test CSV with required columns only
    test_csv = os.path.join(DATA_DIR, "test.csv")
    df_test = pd.read_csv(test_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df_test.columns]
    if missing:
        raise ValueError(f"Missing required columns in test.csv: {missing}")
    df_test = df_test[REQUIRED_COLUMNS]

    # Predict
    pred = model.predict(df_test.values)[0]
    return int(pred)