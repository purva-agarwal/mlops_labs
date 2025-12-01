import os
import logging
import logstash
import time
from datetime import datetime
from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.train import train_model

LOGSTASH_HOST = os.getenv("LOGSTASH_HOST", "localhost")
LOGSTASH_PORT = int(os.getenv("LOGSTASH_PORT", 5000))

logger = logging.getLogger("iris-api")
logger.setLevel(logging.INFO)

try:
    logstash_handler = logstash.TCPLogstashHandler(LOGSTASH_HOST, LOGSTASH_PORT, version=1)
    logger.addHandler(logstash_handler)
    logger.info("Logstash handler connected successfully")
except Exception as e:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    logger.warning(f"Could not connect to Logstash, using console: {e}")

IRIS_CLASSES = {0: "setosa", 1: "versicolor", 2: "virginica"}

app = FastAPI(
    title="Iris Model API with ELK",
    description="ML Prediction API with Elasticsearch-Logstash-Kibana logging",
    version="1.0.0"
)

model = train_model()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    species: str
    confidence: float
    timestamp: str

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        "HTTP Request",
        extra={
            "type": "http-request",
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "process_time_ms": round(process_time * 1000, 2),
            "client_host": request.client.host if request.client else "unknown"
        }
    )
    return response

@app.get("/")
def home():
    logger.info("Home endpoint accessed", extra={"type": "api-access", "endpoint": "home"})
    return {
        "message": "Iris Prediction API with ELK Stack",
        "docs": "/docs",
        "kibana": "http://localhost:5601"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))
    species = IRIS_CLASSES.get(int(prediction), "unknown")
    timestamp = datetime.now().isoformat()
    
    logger.info(
        f"Prediction made: {species}",
        extra={
            "type": "iris-prediction",
            "sepal_length": data.sepal_length,
            "sepal_width": data.sepal_width,
            "petal_length": data.petal_length,
            "petal_width": data.petal_width,
            "prediction": int(prediction),
            "species": species,
            "confidence": round(confidence, 4),
            "timestamp": timestamp
        }
    )
    
    return PredictionResponse(
        prediction=int(prediction),
        species=species,
        confidence=round(confidence, 4),
        timestamp=timestamp
    )

@app.get("/stats")
def get_stats():
    logger.info("Stats endpoint accessed", extra={"type": "api-access", "endpoint": "stats"})
    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "classes": IRIS_CLASSES,
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    }
