from fastapi import APIRouter
from typing import Dict, Any
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import time

router = APIRouter()

# Prometheus metrics
PREDICTION_REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests'
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction request latency in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

MODEL_ERRORS = Counter(
    'model_errors_total',
    'Total number of model errors',
    ['model_name', 'error_type']
)

MODEL_DRIFT_SCORE = Gauge(
    'model_drift_score',
    'Current model drift score',
    ['model_name']
)

FEATURE_DRIFT_SCORE = Gauge(
    'feature_drift_score',
    'Current feature drift score',
    ['model_name', 'feature_name']
)

class MetricsCollector:
    @staticmethod
    def record_prediction_request():
        """Record a prediction request."""
        PREDICTION_REQUEST_COUNT.inc()
    
    @staticmethod
    def record_prediction_latency(start_time: float):
        """Record the latency of a prediction request."""
        PREDICTION_LATENCY.observe(time.time() - start_time)
    
    @staticmethod
    def record_model_error(model_name: str, error_type: str):
        """Record a model error."""
        MODEL_ERRORS.labels(model_name=model_name, error_type=error_type).inc()
    
    @staticmethod
    def update_drift_score(model_name: str, drift_score: float):
        """Update the model drift score."""
        MODEL_DRIFT_SCORE.labels(model_name=model_name).set(drift_score)
    
    @staticmethod
    def update_feature_drift(model_name: str, feature_name: str, drift_score: float):
        """Update the feature drift score."""
        FEATURE_DRIFT_SCORE.labels(
            model_name=model_name,
            feature_name=feature_name
        ).set(drift_score)

@router.get("/metrics")
async def metrics():
    """Endpoint for scraping metrics."""
    return prometheus_client.generate_latest()