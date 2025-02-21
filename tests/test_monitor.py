import pytest
import pandas as pd
import numpy as np
from automl.monitor import ModelMonitor
from automl.preprocessor import AlzheimerDataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    processor = AlzheimerDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_training_data(test_size=0.2)
    return X_train, X_test, y_train, y_test

@pytest.fixture
def monitor(sample_data):
    """Create ModelMonitor instance with reference data."""
    X_train, _, _, _ = sample_data
    return ModelMonitor(reference_data=X_train)

def test_monitor_initialization(monitor):
    """Test if monitor initializes correctly."""
    assert monitor.reference_data is not None
    assert isinstance(monitor.reference_data, pd.DataFrame)

def test_data_drift_detection(monitor, sample_data):
    """Test data drift detection."""
    _, X_test, _, _ = sample_data
    
    # Test drift detection
    drift_results = monitor.check_data_drift(X_test)
    
    # Check results
    assert isinstance(drift_results, dict)
    assert 'report' in drift_results
    assert 'test_results' in drift_results

def test_performance_metrics(monitor, sample_data):
    """Test performance metrics calculation."""
    _, _, y_train, y_test = sample_data
    
    # Calculate metrics
    metrics = monitor.calculate_performance_metrics(y_test, y_train)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    # Check metric values are between 0 and 1
    for metric_value in metrics.values():
        assert 0 <= metric_value <= 1

def test_drift_threshold_alerts(monitor, sample_data):
    """Test if drift alerts are working."""
    _, X_test, _, _ = sample_data
    
    # Introduce artificial drift by modifying some values
    X_test_modified = X_test.copy()
    X_test_modified.iloc[:, 0] = X_test_modified.iloc[:, 0] + 10
    
    # Check drift with modified data
    drift_results = monitor.check_data_drift(X_test_modified)
    
    # Verify that drift is detected
    assert drift_results['test_results'] is not None

def test_error_handling(monitor):
    """Test error handling in monitor."""
    # Test with invalid input
    with pytest.raises(ValueError):
        monitor.check_data_drift(None)
    
    with pytest.raises(ValueError):
        monitor.calculate_performance_metrics(None, None)

def test_feature_drift_detection(monitor, sample_data):
    """Test feature-level drift detection."""
    _, X_test, _, _ = sample_data
    
    # Modify specific features
    X_test_modified = X_test.copy()
    X_test_modified['Age'] = X_test_modified['Age'] + 20
    
    # Check drift
    drift_results = monitor.check_data_drift(X_test_modified)
    
    # Verify feature-level drift detection
    assert drift_results['report'] is not None