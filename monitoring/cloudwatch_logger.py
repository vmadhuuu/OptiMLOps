# monitoring/cloudwatch_logger.py
import boto3
import time
from datetime import datetime

class CloudWatchLogger:
    def __init__(self, namespace="AlzheimerPredictions"):
        self.cloudwatch = boto3.client('cloudwatch')
        self.namespace = namespace

    def log_prediction(self, prediction_data, latency, model_version):
        """
        Log prediction metrics to CloudWatch
        """
        timestamp = datetime.utcnow()
        
        metrics = [
            {
                'MetricName': 'PredictionLatency',
                'Value': latency,
                'Unit': 'Milliseconds',
                'Timestamp': timestamp
            },
            {
                'MetricName': 'ModelVersion',
                'Value': 1,
                'Unit': 'None',
                'Timestamp': timestamp,
                'Dimensions': [
                    {
                        'Name': 'Version',
                        'Value': model_version
                    }
                ]
            }
        ]
        
        if 'prediction_probability' in prediction_data:
            metrics.append({
                'MetricName': 'PredictionConfidence',
                'Value': max(prediction_data['prediction_probability']),
                'Unit': 'None',
                'Timestamp': timestamp
            })

        self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=metrics
        )

    def log_drift_metrics(self, drift_score, feature_importance_drift):
        """
        Log model drift metrics
        """
        timestamp = datetime.utcnow()
        
        self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    'MetricName': 'ModelDriftScore',
                    'Value': drift_score,
                    'Unit': 'None',
                    'Timestamp': timestamp
                },
                {
                    'MetricName': 'FeatureImportanceDrift',
                    'Value': feature_importance_drift,
                    'Unit': 'None',
                    'Timestamp': timestamp
                }
            ]
        )

    def create_alarm(self, metric_name, threshold, evaluation_periods=5):
        """
        Create CloudWatch alarm for a metric
        """
        self.cloudwatch.put_metric_alarm(
            AlarmName=f'{metric_name}_alarm',
            MetricName=metric_name,
            Namespace=self.namespace,
            Statistic='Average',
            Period=300,  # 5 minutes
            EvaluationPeriods=evaluation_periods,
            Threshold=threshold,
            ComparisonOperator='GreaterThanThreshold',
            AlarmActions=['SNS_TOPIC_ARN']  # Replace with your SNS topic ARN
        )