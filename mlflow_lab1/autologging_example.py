import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
import mlflow.sklearn


def load_data():
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    data = pd.read_csv(csv_url, sep=";")
    
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]
    
    return train_x, train_y, test_x, test_y


def train_with_autologging():
    print("=" * 70)
    print("MLflow Autologging Examples")
    print("=" * 70)
    print("\nAutologging automatically logs parameters, metrics, and models")
    print("without requiring explicit log statements.\n")
    
    train_x, train_y, test_x, test_y = load_data()
    
    mlflow.sklearn.autolog()
    
    print("\n" + "=" * 70)
    print("Example 1: Ridge Regression with Autologging")
    print("=" * 70)
    
    with mlflow.start_run(run_name="Ridge_Autolog"):
        mlflow.set_tag("autolog_enabled", "True")
        mlflow.set_tag("model_type", "Ridge")
        
        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(train_x, train_y)
        
        test_score = ridge_model.score(test_x, test_y)
        print(f"Ridge R2 Score: {test_score:.4f}")
        print("✓ Parameters, metrics, and model automatically logged!")
    
    print("\n" + "=" * 70)
    print("Example 2: Lasso Regression with Autologging")
    print("=" * 70)
    
    with mlflow.start_run(run_name="Lasso_Autolog"):
        mlflow.set_tag("autolog_enabled", "True")
        mlflow.set_tag("model_type", "Lasso")
        
        lasso_model = Lasso(alpha=0.1, random_state=42)
        lasso_model.fit(train_x, train_y)
        
        test_score = lasso_model.score(test_x, test_y)
        print(f"Lasso R2 Score: {test_score:.4f}")
        print("✓ Parameters, metrics, and model automatically logged!")
    
    print("\n" + "=" * 70)
    print("Example 3: Gradient Boosting with Autologging")
    print("=" * 70)
    
    with mlflow.start_run(run_name="GradientBoosting_Autolog"):
        mlflow.set_tag("autolog_enabled", "True")
        mlflow.set_tag("model_type", "GradientBoosting")
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        gb_model.fit(train_x, train_y)
        
        test_score = gb_model.score(test_x, test_y)
        print(f"Gradient Boosting R2 Score: {test_score:.4f}")
        print("✓ Parameters, metrics, and model automatically logged!")
        print("✓ Feature importances automatically logged!")
    
    print("\n" + "=" * 70)
    print("Example 4: Cross-Validation with Autologging")
    print("=" * 70)
    
    with mlflow.start_run(run_name="CrossValidation_Autolog"):
        mlflow.set_tag("autolog_enabled", "True")
        mlflow.set_tag("model_type", "Ridge_CV")
        
        ridge_cv = Ridge(alpha=1.0, random_state=42)
        cv_scores = cross_val_score(ridge_cv, train_x, train_y, cv=5)
        
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        mlflow.log_metric("cv_mean_score", cv_scores.mean())
        mlflow.log_metric("cv_std_score", cv_scores.std())
        
        ridge_cv.fit(train_x, train_y)
        print("✓ Cross-validation metrics and final model logged!")
    
    mlflow.sklearn.autolog(disable=True)
    
    print("\n" + "=" * 70)
    print("Example 5: Manual Logging (Autologging Disabled)")
    print("=" * 70)
    
    with mlflow.start_run(run_name="Manual_Logging"):
        mlflow.set_tag("autolog_enabled", "False")
        mlflow.set_tag("model_type", "Ridge_Manual")
        
        ridge_manual = Ridge(alpha=2.0, random_state=42)
        ridge_manual.fit(train_x, train_y)
        
        test_score = ridge_manual.score(test_x, test_y)
        
        mlflow.log_param("alpha", 2.0)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("r2_score", test_score)
        
        mlflow.sklearn.log_model(ridge_manual, "model")
        
        print(f"Ridge (Manual) R2 Score: {test_score:.4f}")
        print("✓ Manually logged parameters, metrics, and model")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Autologging simplifies experiment tracking")
    print("✓ Works with scikit-learn, XGBoost, TensorFlow, PyTorch, and more")
    print("✓ Can be combined with manual logging for custom metrics")
    print("✓ Use mlflow.sklearn.autolog() to enable for scikit-learn")
    print("✓ Use mlflow.sklearn.autolog(disable=True) to disable")


if __name__ == "__main__":
    train_with_autologging()