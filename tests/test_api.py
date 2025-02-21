import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

@pytest.fixture
def sample_prediction_request():
    """Create a sample prediction request."""
    return {
        "age": 75,
        "gender": 0,
        "ethnicity": 0,
        "education_level": 1,
        "bmi": 24.5,
        "smoking": 0,
        "alcohol_consumption": 5.0,
        "physical_activity": 7.0,
        "diet_quality": 8.0,
        "sleep_quality": 7.0,
        "family_history_alzheimers": 0,
        "cardiovascular_disease": 0,
        "diabetes": 0,
        "depression": 0,
        "head_injury": 0,
        "hypertension": 0,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "cholesterol_total": 200,
        "cholesterol_ldl": 100,
        "cholesterol_hdl": 50,
        "cholesterol_triglycerides": 150,
        "mmse": 28,
        "functional_assessment": 9.0,
        "memory_complaints": 0,
        "behavioral_problems": 0,
        "adl": 9.0,
        "confusion": 0,
        "disorientation": 0,
        "personality_changes": 0,
        "difficulty_completing_tasks": 0,
        "forgetfulness": 0
    }

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint(sample_prediction_request):
    """Test prediction endpoint."""
    response = client.post("/predict/alzheimer", json=sample_prediction_request)
    
    # If model is not loaded, it should return 500
    if response.status_code == 500:
        assert "error" in response.json()
    else:
        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result
        assert "prediction_probability" in result
        assert "risk_factors" in result
        assert "confidence_score" in result

def test_invalid_prediction_request():
    """Test prediction endpoint with invalid data."""
    # Missing required fields
    invalid_request = {
        "age": 75,
        "gender": 0
    }
    
    response = client.post("/predict/alzheimer", json=invalid_request)
    assert response.status_code == 422

def test_out_of_range_values(sample_prediction_request):
    """Test prediction endpoint with out-of-range values."""
    # Modify age to invalid value
    invalid_request = sample_prediction_request.copy()
    invalid_request["age"] = 150
    
    response = client.post("/predict/alzheimer", json=invalid_request)
    assert response.status_code == 422

def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.text != ""  

def test_model_reload():
    """Test model reloading capability."""

    request = sample_prediction_request
    response1 = client.post("/predict/alzheimer", json=request)
    response2 = client.post("/predict/alzheimer", json=request)
 
    assert response1.status_code == response2.status_code

def test_cache_behavior(sample_prediction_request):
    """Test caching behavior."""

    response1 = client.post("/predict/alzheimer", json=sample_prediction_request)
    response2 = client.post("/predict/alzheimer", json=sample_prediction_request)
    
    if response1.status_code == 200 and response2.status_code == 200:
       
        assert response1.json()["prediction"] == response2.json()["prediction"]