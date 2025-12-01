import os
import logging
import json
from datetime import datetime
from elasticsearch import Elasticsearch
from pythonjsonlogger import jsonlogger

class ElkLogger:
    def __init__(self, app_name: str = "ml-api"):
        self.app_name = app_name
        self.es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        self.es_port = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
        self.es_client = None
        self.index_name = f"ml-logs-{datetime.now().strftime('%Y.%m.%d')}"
        self._setup_logger()
        self._connect_elasticsearch()

    def _setup_logger(self):
        self.logger = logging.getLogger(self.app_name)
        self.logger.setLevel(logging.INFO)
        
        log_handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        log_handler.setFormatter(formatter)
        self.logger.addHandler(log_handler)

    def _connect_elasticsearch(self):
        try:
            self.es_client = Elasticsearch([f"http://{self.es_host}:{self.es_port}"])
            if self.es_client.ping():
                self.logger.info("Connected to Elasticsearch successfully!")
            else:
                self.logger.warning("Could not connect to Elasticsearch")
                self.es_client = None
        except Exception as e:
            self.logger.error(f"Elasticsearch connection error: {e}")
            self.es_client = None

    def _send_to_elasticsearch(self, log_data: dict):
        if self.es_client is None:
            return
        
        try:
            self.es_client.index(index=self.index_name, document=log_data)
        except Exception as e:
            self.logger.error(f"Failed to send log to Elasticsearch: {e}")

    def log_event(self, event_type: str, message: str, extra_data: dict = None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "app_name": self.app_name,
            "event_type": event_type,
            "message": message,
            "extra": extra_data or {}
        }
        
        self.logger.info(json.dumps(log_entry))
        self._send_to_elasticsearch(log_entry)

    def log_prediction(self, input_features: list, prediction: int, latency_ms: float):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "app_name": self.app_name,
            "event_type": "PREDICTION",
            "input_features": input_features,
            "prediction": prediction,
            "latency_ms": latency_ms
        }
        
        self.logger.info(json.dumps(log_entry))
        self._send_to_elasticsearch(log_entry)

    def log_model_training(self, model_name: str, accuracy: float, params: dict):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "app_name": self.app_name,
            "event_type": "MODEL_TRAINING",
            "model_name": model_name,
            "accuracy": accuracy,
            "parameters": params
        }
        
        self.logger.info(json.dumps(log_entry))
        self._send_to_elasticsearch(log_entry)

    def log_api_request(self, endpoint: str, method: str, status_code: int, latency_ms: float):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "app_name": self.app_name,
            "event_type": "API_REQUEST",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "latency_ms": latency_ms
        }
        
        self.logger.info(json.dumps(log_entry))
        self._send_to_elasticsearch(log_entry)

elk_logger = ElkLogger()
