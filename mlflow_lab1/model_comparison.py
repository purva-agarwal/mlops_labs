import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd


def search_and_compare_runs(experiment_name="Default"):
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            print(f"Experiment '{experiment_name}' not found. Using default experiment.")
            experiment_id = "0"
    except:
        experiment_id = "0"
    
    print("=" * 80)
    print(f"Comparing Models in Experiment: {experiment_name}")
    print("=" * 80)
    
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=10
    )
    
    if not runs:
        print("No runs found in this experiment.")
        print("\nPlease run one of the training scripts first:")
        print("  python linear_regression.py")
        print("  python hyperparameter_tuning.py")
        print("  python autologging_example.py")
        return
    
    print(f"\nFound {len(runs)} runs. Displaying top 10 by RMSE:\n")
    
    run_data = []
    for idx, run in enumerate(runs, 1):
        run_id = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName", "Unnamed")
        
        params = run.data.params
        metrics = run.data.metrics
        tags = run.data.tags
        
        model_type = tags.get("model_type", tags.get("model_algorithm", "Unknown"))
        
        run_data.append({
            "Rank": idx,
            "Run ID": run_id[:8],
            "Run Name": run_name,
            "Model Type": model_type,
            "RMSE": metrics.get("rmse", "N/A"),
            "MAE": metrics.get("mae", "N/A"),
            "R2": metrics.get("r2", "N/A"),
            "Alpha": params.get("alpha", "N/A"),
            "L1 Ratio": params.get("l1_ratio", "N/A")
        })
    
    df = pd.DataFrame(run_data)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Best Model Summary")
    print("=" * 80)
    
    best_run = runs[0]
    print(f"Run ID: {best_run.info.run_id}")
    print(f"Run Name: {best_run.data.tags.get('mlflow.runName', 'Unnamed')}")
    print(f"Model Type: {best_run.data.tags.get('model_type', 'Unknown')}")
    print(f"RMSE: {best_run.data.metrics.get('rmse', 'N/A'):.4f}")
    print(f"MAE: {best_run.data.metrics.get('mae', 'N/A'):.4f}")
    print(f"R2: {best_run.data.metrics.get('r2', 'N/A'):.4f}")
    
    print("\n" + "=" * 80)
    print("Model Artifacts")
    print("=" * 80)
    
    artifacts = client.list_artifacts(best_run.info.run_id)
    if artifacts:
        print("\nArtifacts logged for best run:")
        for artifact in artifacts:
            print(f"  - {artifact.path}")
    else:
        print("No artifacts found.")
    
    return best_run.info.run_id


def demonstrate_model_registry():
    print("\n" + "=" * 80)
    print("MLflow Model Registry")
    print("=" * 80)
    print("\nThe Model Registry provides:")
    print("  ✓ Centralized model store")
    print("  ✓ Model versioning")
    print("  ✓ Stage transitions (None → Staging → Production → Archived)")
    print("  ✓ Model lineage and annotations")
    
    print("\n" + "=" * 80)
    print("Model Registry Commands")
    print("=" * 80)
    
    print("\n1. Register a model:")
    print("   mlflow.register_model(model_uri='runs:/<run_id>/model', name='MyModel')")
    
    print("\n2. Load a registered model:")
    print("   model = mlflow.pyfunc.load_model('models:/MyModel/1')")
    print("   model = mlflow.pyfunc.load_model('models:/MyModel/Production')")
    
    print("\n3. Transition model stage:")
    print("   client = MlflowClient()")
    print("   client.transition_model_version_stage(")
    print("       name='MyModel',")
    print("       version=1,")
    print("       stage='Production'")
    print("   )")
    
    print("\n4. List registered models:")
    print("   registered_models = client.search_registered_models()")
    
    print("\n5. Add model description:")
    print("   client.update_model_version(")
    print("       name='MyModel',")
    print("       version=1,")
    print("       description='Best performing model on wine quality dataset'")
    print("   )")
    
    print("\n" + "=" * 80)
    print("Setting up Model Registry with Tracking Server")
    print("=" * 80)
    print("\nTo use Model Registry, start MLflow with a database backend:")
    print("\n  1. Install database backend:")
    print("     pip install psycopg2-binary  # For PostgreSQL")
    print("     pip install pymysql  # For MySQL")
    
    print("\n  2. Start tracking server:")
    print("     mlflow server \\")
    print("       --backend-store-uri postgresql://user:password@localhost/mlflow \\")
    print("       --default-artifact-root s3://my-mlflow-bucket/ \\")
    print("       --host 0.0.0.0 \\")
    print("       --port 5000")
    
    print("\n  3. Set tracking URI in your code:")
    print("     mlflow.set_tracking_uri('http://localhost:5000')")
    
    print("\n" + "=" * 80)


def query_runs_by_tag():
    print("\n" + "=" * 80)
    print("Querying Runs by Tags and Filters")
    print("=" * 80)
    
    client = MlflowClient()
    
    print("\nExample 1: Find all ElasticNet models")
    filter_string = "tags.model_type = 'ElasticNet'"
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string=filter_string,
        max_results=5
    )
    print(f"Filter: {filter_string}")
    print(f"Found {len(runs)} runs")
    
    print("\nExample 2: Find runs with R2 > 0.3")
    filter_string = "metrics.r2 > 0.3"
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string=filter_string,
        max_results=5
    )
    print(f"Filter: {filter_string}")
    print(f"Found {len(runs)} runs")
    
    print("\nExample 3: Find runs with specific parameter")
    filter_string = "params.alpha = '0.5'"
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string=filter_string,
        max_results=5
    )
    print(f"Filter: {filter_string}")
    print(f"Found {len(runs)} runs")
    
    print("\n" + "=" * 80)
    print("Useful Filter Examples")
    print("=" * 80)
    print("  • metrics.rmse < 0.7")
    print("  • params.alpha = '0.5'")
    print("  • tags.model_type = 'RandomForest'")
    print("  • attributes.status = 'FINISHED'")
    print("  • metrics.rmse < 0.7 and params.alpha > '0.1'")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MLflow Model Comparison and Registry Demo")
    print("=" * 80)
    
    best_run_id = search_and_compare_runs()
    
    demonstrate_model_registry()
    
    query_runs_by_tag()
    
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("1. View detailed results in MLflow UI: mlflow ui")
    print("2. Access UI at: http://localhost:5000")
    print("3. Try running different training scripts to populate more data")
    print("4. Experiment with different hyperparameters")
    print("5. Compare models visually in the MLflow UI")
