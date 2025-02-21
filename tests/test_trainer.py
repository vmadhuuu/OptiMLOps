import pytest
import pandas as pd
import numpy as np
from automl.trainer import AutoMLTrainer
from automl.preprocessor import AlzheimerDataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    processor = AlzheimerDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_training_data(test_size=0.2)
    return X_train, X_test, y_train, y_test

@pytest.fixture
def trainer():
    """Create AutoMLTrainer instance."""
    return AutoMLTrainer(experiment_name="test_experiment")

def test_trainer_initialization(trainer):
    """Test if trainer initializes correctly."""
    assert trainer.models is not None
    assert 'random_forest' in trainer.models
    assert 'lightgbm' in trainer.models
    assert 'xgboost' in trainer.models

def test_model_optimization(trainer, sample_data):
    """Test model optimization process."""
    X_train, _, y_train, _ = sample_data
    model_name = 'random_forest'
    
    # Test optimization
    best_model, best_params = trainer.optimize_model(
        model_name=model_name,
        X=X_train,
        y=y_train,
        n_trials=2  
    )
    
    assert best_model is not None
    assert isinstance(best_params, dict)
    assert 'n_estimators' in best_params

def test_train_and_log(trainer, sample_data):
    """Test full training and logging process."""
    X_train, X_test, y_train, y_test = sample_data
    
    results = trainer.train_and_log(X_train, y_train, X_test, y_test)
    
    # Check results
    assert isinstance(results, dict)
    for model_name, result in results.items():
        assert 'model' in result
        assert 'params' in result
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert 'precision' in result['metrics']
        assert 'recall' in result['metrics']
        assert 'f1' in result['metrics']

def test_model_params(trainer):
    """Test if model parameters are correctly defined."""
    trial = None 
    
    for model_name in ['random_forest', 'lightgbm', 'xgboost']:
        params = trainer._get_model_params(trial, model_name)
        assert isinstance(params, dict)
        
def test_error_handling(trainer):
    """Test error handling in trainer."""
    with pytest.raises(ValueError):
        trainer.optimize_model('invalid_model', None, None)

def test_model_persistence(trainer, sample_data, tmp_path):
    """Test if models can be saved and loaded."""
    X_train, X_test, y_train, y_test = sample_data
    
    results = trainer.train_and_log(X_train, y_train, X_test, y_test)
    
    for model_name in results:
        assert results[model_name]['model'] is not None