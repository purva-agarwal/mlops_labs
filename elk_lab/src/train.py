import os
import logging
import logstash
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

LOGSTASH_HOST = os.getenv("LOGSTASH_HOST", "localhost")
LOGSTASH_PORT = int(os.getenv("LOGSTASH_PORT", 5000))

logger = logging.getLogger("iris-training")
logger.setLevel(logging.INFO)

try:
    logstash_handler = logstash.TCPLogstashHandler(LOGSTASH_HOST, LOGSTASH_PORT, version=1)
    logger.addHandler(logstash_handler)
except Exception:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

def train_model():
    logger.info("Starting model training", extra={"type": "training", "event": "start"})
    
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Model trained successfully with accuracy: {acc:.3f}")

    logger.info(
        f"Model training completed",
        extra={
            "type": "training",
            "event": "complete",
            "accuracy": round(acc, 4),
            "n_estimators": 100,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "timestamp": datetime.now().isoformat()
        }
    )

    os.makedirs("logs", exist_ok=True)
    log_file = "logs/training.log"
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} - Model trained with accuracy: {acc:.3f}\n")

    return model