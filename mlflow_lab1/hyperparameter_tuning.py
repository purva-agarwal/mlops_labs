import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_elasticnet(train_x, train_y, test_x, test_y, alpha, l1_ratio):
    with mlflow.start_run(nested=True):
        mlflow.set_tag("model_algorithm", "ElasticNet")
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)
        
        predictions = model.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predictions)
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        signature = infer_signature(train_x, predictions)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        return rmse, mae, r2


def train_random_forest(train_x, train_y, test_x, test_y, n_estimators, max_depth):
    with mlflow.start_run(nested=True):
        mlflow.set_tag("model_algorithm", "RandomForest")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(train_x, train_y)
        
        predictions = model.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predictions)
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        signature = infer_signature(train_x, predictions)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        return rmse, mae, r2


if __name__ == "__main__":
    np.random.seed(42)
    
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    data = pd.read_csv(csv_url, sep=";")
    
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    with mlflow.start_run(run_name="Hyperparameter_Tuning_Experiment"):
        mlflow.set_tag("experiment_type", "hyperparameter_tuning")
        mlflow.set_tag("dataset", "wine-quality")
        
        print("=" * 60)
        print("ElasticNet Hyperparameter Tuning")
        print("=" * 60)
        
        elasticnet_params = [
            (0.1, 0.1),
            (0.1, 0.5),
            (0.1, 0.9),
            (0.5, 0.1),
            (0.5, 0.5),
            (0.5, 0.9),
            (1.0, 0.1),
            (1.0, 0.5),
            (1.0, 0.9),
        ]
        
        best_rmse = float('inf')
        best_params = None
        
        for alpha, l1_ratio in elasticnet_params:
            rmse, mae, r2 = train_elasticnet(train_x, train_y, test_x, test_y, alpha, l1_ratio)
            print(f"Alpha={alpha:.1f}, L1_ratio={l1_ratio:.1f} -> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (alpha, l1_ratio)
        
        mlflow.log_metric("best_elasticnet_rmse", best_rmse)
        mlflow.log_param("best_elasticnet_alpha", best_params[0])
        mlflow.log_param("best_elasticnet_l1_ratio", best_params[1])
        
        print(f"\nBest ElasticNet: Alpha={best_params[0]}, L1_ratio={best_params[1]}, RMSE={best_rmse:.4f}")
        
        print("\n" + "=" * 60)
        print("Random Forest Hyperparameter Tuning")
        print("=" * 60)
        
        rf_params = [
            (50, 5),
            (50, 10),
            (50, None),
            (100, 5),
            (100, 10),
            (100, None),
            (200, 5),
            (200, 10),
        ]
        
        best_rmse_rf = float('inf')
        best_params_rf = None
        
        for n_estimators, max_depth in rf_params:
            rmse, mae, r2 = train_random_forest(train_x, train_y, test_x, test_y, n_estimators, max_depth)
            depth_str = str(max_depth) if max_depth else "None"
            print(f"N_estimators={n_estimators}, Max_depth={depth_str} -> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            
            if rmse < best_rmse_rf:
                best_rmse_rf = rmse
                best_params_rf = (n_estimators, max_depth)
        
        mlflow.log_metric("best_rf_rmse", best_rmse_rf)
        mlflow.log_param("best_rf_n_estimators", best_params_rf[0])
        mlflow.log_param("best_rf_max_depth", best_params_rf[1] if best_params_rf[1] else "None")
        
        print(f"\nBest Random Forest: N_estimators={best_params_rf[0]}, Max_depth={best_params_rf[1]}, RMSE={best_rmse_rf:.4f}")
        
        print("\n" + "=" * 60)
        print("Overall Best Model")
        print("=" * 60)
        if best_rmse < best_rmse_rf:
            print(f"ElasticNet (RMSE={best_rmse:.4f}) outperforms Random Forest (RMSE={best_rmse_rf:.4f})")
            mlflow.set_tag("best_model", "ElasticNet")
        else:
            print(f"Random Forest (RMSE={best_rmse_rf:.4f}) outperforms ElasticNet (RMSE={best_rmse:.4f})")
            mlflow.set_tag("best_model", "RandomForest")
