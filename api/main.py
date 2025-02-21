from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import mlflow
import pandas as pd
import redis
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="AutoMLOps API",
    description="API for AutoML model training and predictions",
    version="1.0.0"
)

# Initialize Redis
redis_client = redis.Redis(
    host='localhost',  # Change to 'redis' when using docker-compose
    port=6379,
    db=0,
    decode_responses=True
)

# Pydantic models
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_name: str = "best_model"

class PredictionResponse(BaseModel):
    prediction: Any
    prediction_probability: List[float]
    model_name: str
    prediction_time: str
    feature_importance: Dict[str, float]

class TrainingRequest(BaseModel):
    data_path: str
    target_column: str
    model_type: str = "auto"  # auto, random_forest, lightgbm, xgboost

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Check cache
        cache_key = f"prediction:{request.model_name}:{json.dumps(request.features)}"
        cached_prediction = redis_client.get(cache_key)
        
        if cached_prediction:
            return json.loads(cached_prediction)
        
        # Load model
        model = mlflow.sklearn.load_model(
            f"models:/{request.model_name}/Production"
        )
        
        # Prepare features
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        prediction_proba = model.predict_proba(features_df)[0].tolist()
        
        # Get feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(
                features_df.columns,
                model.feature_importances_
            ))
        
        response = PredictionResponse(
            prediction=prediction,
            prediction_probability=prediction_proba,
            model_name=request.model_name,
            prediction_time=datetime.now().isoformat(),
            feature_importance=feature_importance
        )
        
        # Cache response
        redis_client.setex(
            cache_key,
            3600,  # 1 hour cache
            json.dumps(response.dict())
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainingRequest):
    try:
        from automl.trainer import AutoMLTrainer
        
        # Initialize trainer
        trainer = AutoMLTrainer()
        
        # Load data
        data = pd.read_csv(request.data_path)
        X = data.drop(request.target_column, axis=1)
        y = data[request.target_column]
        
        # Train models
        results = trainer.train_and_log(
            X_train=X,
            y_train=y,
            X_test=X,  # In practice, use proper train-test split
            y_test=y
        )
        
        return {
            "status": "success",
            "results": {
                model_name: {
                    "metrics": result["metrics"]
                } for model_name, result in results.items()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}