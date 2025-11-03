from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
from datetime import datetime

def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Model trained successfully with accuracy: {acc:.3f}")

    # Log training info
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/training.log"
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} - Model trained with accuracy: {acc:.3f}\n")

    return model
