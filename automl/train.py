# automl/train.py
import mlflow
from automl.trainer import AutoMLTrainer
from automl.preprocessor import AlzheimerDataProcessor

def train_models():
    """Train and evaluate models on the Alzheimer's dataset."""
    processor = AlzheimerDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_training_data()
    
    trainer = AutoMLTrainer(experiment_name="alzheimer_prediction")
    results = trainer.train_and_log(X_train, y_train, X_test, y_test)

    best_model_name = max(results.items(), key=lambda x: x[1]['metrics']['f1'])[0]
    with mlflow.start_run():
        mlflow.sklearn.log_model(processor, "preprocessor")
    
    return results

if __name__ == "__main__":
    results = train_models()
    
    for model_name, result in results.items():
        print(f"\nResults for {model_name}:")
        print("Metrics:", result['metrics'])
        print("Best parameters:", result['params'])