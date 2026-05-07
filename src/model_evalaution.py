import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler=logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))
file_handler.setLevel('DEBUG')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path: str):
    """
    Load a trained model from a file.
    
    Args:
        model_path (str): Path to the model file.
    Returns:
        The loaded model.
    """
    try:
        import joblib
        model = joblib.load(model_path)
        logger.debug(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise
def evaluate_model(model, testcsv, target_column='charges'):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: The trained model.
        testcsv (str): Path to the test CSV file.
        target_column (str): Name of the target column.
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    try:
        test_df = pd.read_csv(testcsv)
        logger.debug("Test data loaded successfully for evaluation")
        
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model Evaluation - MSE: {mse}, R2: {r2}")
        
        return {'MSE': mse, 'R2': r2}
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise
def save_evaluation_metrics(metrics: Dict[str, float], metrics_path: str):
    """
    Save evaluation metrics to a file.
    
    Args:
        metrics (Dict[str, float]): Dictionary containing evaluation metrics.
        metrics_path (str): Path to save the metrics file.
    """
    try:
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        logger.debug(f"Evaluation metrics saved successfully at {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation metrics to {metrics_path}: {e}")
        raise
def main():
    try:
        # Paths to model and test data
        model_path = 'models/random_forest_model.pkl'
        testcsv = 'data/transform/test.csv'
        metrics_path = 'reports/evaluation/metrics.txt'
        
        # Load model
        model = load_model(model_path)
        
        # Evaluate model
        metrics = evaluate_model(model, testcsv)
        
        # Save evaluation metrics
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        save_evaluation_metrics(metrics, metrics_path)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
if __name__ == "__main__":
    main()
