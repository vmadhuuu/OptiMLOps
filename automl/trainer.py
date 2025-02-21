import optuna
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, List, Tuple

class AutoMLTrainer:
    def __init__(self, experiment_name: str = "automl_experiment"):
        """Initialize AutoML trainer with MLflow experiment tracking."""
        mlflow.set_experiment(experiment_name)
        self.models = {
            'random_forest': RandomForestClassifier,
            'lightgbm': LGBMClassifier,
            'xgboost': XGBClassifier
        }
        
    def optimize_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                      n_trials: int = 100) -> Tuple[Any, Dict[str, Any]]:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            params = self._get_model_params(trial, model_name)
            model = self.models[model_name](**params)
            score = cross_val_score(model, X, y, cv=5, scoring='f1')
            return score.mean()
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_model = self.models[model_name](**best_params)
        return best_model, best_params
    
    def _get_model_params(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter search space based on model type."""
        if model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
        elif model_name == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100)
            }
        elif model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7)
            }
    
    def train_and_log(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train all models and log results with MLflow."""
        results = {}
        
        for model_name in self.models.keys():
            with mlflow.start_run(run_name=f"{model_name}_training"):
                # Optimize model
                model, best_params = self.optimize_model(model_name, X_train, y_train)
                
                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                # Log with MLflow
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, f"{model_name}_model")
                
                results[model_name] = {
                    'model': model,
                    'params': best_params,
                    'metrics': metrics
                }
        
        return results