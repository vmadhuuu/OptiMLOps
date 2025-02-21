# lambda_function.py
import json
import mlflow
import boto3
import os
import pandas as pd
from automl.preprocessor import AlzheimerDataProcessor

def init_model():
    """Initialize MLflow model from S3"""
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    return mlflow.sklearn.load_model(f"models:/alzheimer_model/Production")

def init_preprocessor():
    """Initialize preprocessor from S3"""
    return mlflow.sklearn.load_model(f"models:/preprocessor/Production")

def lambda_handler(event, context):
    """
    AWS Lambda handler for model inference
    """
    try:
        # Parse input
        input_data = json.loads(event['body'])
        
        # Initialize model and preprocessor
        model = init_model()
        preprocessor = init_preprocessor()
        
        # Prepare data
        df = pd.DataFrame([input_data])
        processed_data = preprocessor.preprocess_data(df, is_training=False)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0].tolist()
        
        # Log metrics to CloudWatch
        log_metrics(event, prediction, prediction_proba)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': int(prediction),
                'probability': prediction_proba,
                'model_version': os.environ.get('MODEL_VERSION', 'unknown')
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def log_metrics(event, prediction, prediction_proba):
    """Log metrics to CloudWatch"""
    cloudwatch = boto3.client('cloudwatch')
    
    # Log prediction metrics
    cloudwatch.put_metric_data(
        Namespace='AlzheimerPredictions',
        MetricData=[
            {
                'MetricName': 'PredictionCount',
                'Value': 1,
                'Unit': 'Count'
            },
            {
                'MetricName': 'PredictionConfidence',
                'Value': max(prediction_proba),
                'Unit': 'None'
            }
        ]
    )