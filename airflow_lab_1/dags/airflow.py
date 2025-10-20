# dags/airflow.py
"""
TaskFlow DAG (Option A)
- Split preprocessing: load -> clean + validate (parallel) -> scale -> merge -> build + validate (parallel) -> load/predict
- Uses file paths (strings) for intermediate artifacts => avoids XCom pickling
- Calls functions in src.lab that read/write files (no changes to model logic)
"""

from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.decorators import task

# File-based helpers (implemented in src/lab.py)
from src.lab import (
    load_data_file,
    clean_data_file,
    validate_loaded_file,
    scale_data_file,
    merge_scaled_file,
    build_save_model_file,
    validate_after_build_file,
    load_model_elbow_file,
)

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "your_name",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="Airflow_Lab1_TaskFlow_FileIO",
    default_args=DEFAULT_ARGS,
    description="TaskFlow DAG (file-based intermediates).",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 15),
    catchup=False,
    max_active_runs=1,
    tags=["lab1", "taskflow", "kmeans"],
) as dag:

    dag.doc_md = """
    ## Airflow Lab1 (TaskFlow, File I/O)
    - Preprocessing split into: load -> clean + validate -> scale -> merge -> build + validate -> load/predict
    - Intermediate artifacts are file paths (strings) stored under the `data/` or `model/` directories.
    """

    @task(task_id="load_data_task")
    def t_load_data():
        log.info("Loading data (writes artifact and returns path)")
        return load_data_file()  # returns path string, e.g., "data/loaded.pkl"

    @task(task_id="clean_data_task")
    def t_clean_data(loaded_path: str):
        log.info("Cleaning data at %s", loaded_path)
        return clean_data_file(loaded_path)

    @task(task_id="validate_loaded_task")
    def t_validate_loaded(loaded_path: str):
        log.info("Validating loaded data at %s", loaded_path)
        return validate_loaded_file(loaded_path)

    @task(task_id="scale_data_task")
    def t_scale_data(cleaned_path: str):
        log.info("Scaling cleaned data at %s", cleaned_path)
        return scale_data_file(cleaned_path)

    @task(task_id="merge_scaled_task")
    def t_merge_scaled(scaled_path: str):
        log.info("Merging / preparing scaled data at %s", scaled_path)
        return merge_scaled_file(scaled_path)

    @task(task_id="build_model_task")
    def t_build_model(merged_path: str):
        filename = "model.sav"
        log.info("Building model from %s", merged_path)
        sse, saved_filename = build_save_model_file(merged_path, filename)
        log.info("Model saved as %s", saved_filename)
        return {"sse": sse, "saved_filename": saved_filename}

    @task(task_id="validate_after_build_task")
    def t_validate_after_build(merged_path: str):
        log.info("Running post-preprocess validation on %s", merged_path)
        return validate_after_build_file(merged_path)

    @task(task_id="load_and_predict_task")
    def t_load_and_predict(build_info: dict):
        saved_filename = build_info.get("saved_filename", "model.sav")
        sse = build_info.get("sse")
        log.info("Loading model %s and predicting", saved_filename)
        pred = load_model_elbow_file(saved_filename, sse)
        log.info("Prediction result: %s", pred)
        try:
            return int(pred)
        except Exception:
            return pred

    # DAG wiring (TaskFlow style)
    loaded_path = t_load_data()
    cleaned_path = t_clean_data(loaded_path)
    validated_loaded = t_validate_loaded(loaded_path)
    scaled_path = t_scale_data(cleaned_path)
    merged_path = t_merge_scaled(scaled_path)
    build_info = t_build_model(merged_path)
    validated_after_build = t_validate_after_build(merged_path)
    final_prediction = t_load_and_predict(build_info)

    # explicit dependencies
    loaded_path >> [cleaned_path, validated_loaded]
    cleaned_path >> scaled_path >> merged_path
    merged_path >> [build_info, validated_after_build] >> final_prediction
