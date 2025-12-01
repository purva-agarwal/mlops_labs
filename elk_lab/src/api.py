import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.train import train_model
from src.elk_logger import elk_logger

app = FastAPI(
    title="ELK Lab - Iris Prediction API",
    description="ML API with ELK Stack logging integration",
    version="1.0.0"
)

elk_logger.log_event("APP_STARTUP", "FastAPI application starting...")
model = train_model()
elk_logger.log_event("APP_READY", "FastAPI application ready to serve predictions")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    latency_ms: float

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000
    
    elk_logger.log_api_request(
        endpoint=str(request.url.path),
        method=request.method,
        status_code=response.status_code,
        latency_ms=round(latency_ms, 2)
    )
    
    return response

@app.get("/")
def home():
    elk_logger.log_event("HOME_ACCESS", "Home endpoint accessed")
    return {
        "message": "ELK Lab - Iris Prediction API is running!",
        "version": "1.0.0",
        "elk_status": "Logs streaming to Elasticsearch"
    }

@app.get("/health")
def health_check():
    elk_logger.log_event("HEALTH_CHECK", "Health check performed")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "elk_logging": True
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: IrisInput):
    start_time = time.time()
    
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    
    prediction = model.predict(features)[0]
    class_name = IRIS_CLASSES[prediction]
    
    latency_ms = (time.time() - start_time) * 1000
    
    elk_logger.log_prediction(
        input_features=features[0],
        prediction=int(prediction),
        latency_ms=round(latency_ms, 2)
    )
    
    return PredictionResponse(
        prediction=int(prediction),
        class_name=class_name,
        latency_ms=round(latency_ms, 2)
    )

@app.get("/logs/stats")
def log_stats():
    elk_logger.log_event("STATS_REQUEST", "Log statistics requested")
    return {
        "message": "View logs in Kibana at http://localhost:5601",
        "elasticsearch_index": elk_logger.index_name,
        "hint": "Create an index pattern 'ml-logs-*' in Kibana to visualize logs"
    }