from evidently.metrics import DataDriftTable
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        """Initialize model monitor with reference data."""
        self.reference_data = reference_data
        
    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift between reference and current data."""
        data_drift_report = Report(metrics=[DataDriftTable()])
        data_drift_report.run(reference_data=self.reference_data, 
                            current_data=current_data)
        
        # Create test suite
        drift_test_suite = TestSuite(tests=[
            TestColumnDrift(column_name=column) 
            for column in self.reference_data.columns
        ])
        
        drift_test_suite.run(reference_data=self.reference_data, 
                           current_data=current_data)
        
        return {
            'report': data_drift_report,
            'test_results': drift_test_suite.as_dict()
        }
    
    def calculate_performance_metrics(self, y_true: pd.Series, 
                                   y_pred: pd.Series) -> Dict[str, float]:
        """Calculate model performance metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
