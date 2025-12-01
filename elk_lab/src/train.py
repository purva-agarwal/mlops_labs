from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
from datetime import datetime

try:
    from src.elk_logger import elk_logger
except ImportError:
    from elk_logger import elk_logger

def train_model():
    elk_logger.log_event("MODEL_INIT", "Starting Iris model training...")
    
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model_params = {
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": None,
        "min_samples_split": 2
    }
    
    elk_logger.log_event(
        "MODEL_CONFIG", 
        "Model configuration set",
        {"parameters": model_params}
    )
    
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    
    elk_logger.log_model_training(
        model_name="RandomForestClassifier",
        accuracy=accuracy,
        params=model_params
    )
    
    elk_logger.log_event(
        "MODEL_READY",
        f"Model trained successfully with accuracy: {accuracy:.4f}",
        {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": iris.feature_names,
            "target_classes": iris.target_names.tolist()
        }
    )

    os.makedirs("logs", exist_ok=True)
    log_file = "logs/training.log"
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} - Model trained with accuracy: {accuracy:.4f}\n")

    return model

if __name__ == "__main__":
    train_model()
