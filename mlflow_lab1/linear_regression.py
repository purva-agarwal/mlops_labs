# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        mlflow.set_tag("model_type", "ElasticNet")
        mlflow.set_tag("dataset", "wine-quality")
        mlflow.set_tag("developer", "mlops_team")
        mlflow.set_tag("version", "1.0")
        mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l2_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_test_split_ratio", 0.75)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("train_size", len(train_x))
        mlflow.log_metric("test_size", len(test_x))
        mlflow.log_metric("n_features", train_x.shape[1])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(test_y, predicted_qualities, alpha=0.5)
        plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2)
        plt.xlabel('Actual Quality')
        plt.ylabel('Predicted Quality')
        plt.title(f'Actual vs Predicted (R2={r2:.3f})')
        plt.savefig('predictions_plot.png')
        mlflow.log_artifact('predictions_plot.png')
        plt.close()
        
        residuals = test_y.values.flatten() - predicted_qualities
        plt.figure(figsize=(10, 6))
        plt.scatter(predicted_qualities, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Quality')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.savefig('residuals_plot.png')
        mlflow.log_artifact('residuals_plot.png')
        plt.close()
        
        feature_importance = pd.DataFrame({
            'feature': train_x.columns,
            'coefficient': lr.coef_.flatten()
        }).sort_values('coefficient', key=abs, ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['coefficient'])
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance (Coefficients)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        plt.close()
        
        metrics_summary = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'train_size': len(train_x),
            'test_size': len(test_x)
        }
        with open('metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        mlflow.log_artifact('metrics_summary.json')
        
        data.describe().to_csv('data_summary.csv')
        mlflow.log_artifact('data_summary.csv')

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)
