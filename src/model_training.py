import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler=logging.FileHandler(os.path.join(log_dir, 'model_training.log'))
file_handler.setLevel('DEBUG')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def train_model(traincsv, testcsv, target_column='charges'):
    try:
        train_df = pd.read_csv(traincsv)
        test_df = pd.read_csv(testcsv)
        logger.debug("Data loaded successfully for model training")
        
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.debug("Model training completed successfully")
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model Evaluation - MSE: {mse}, R2: {r2}")
        
        import joblib
        model_path = 'models/random_forest_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.debug(f"Model saved successfully at {model_path}")
        
        return model, mse, r2
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


def save_model(model):
    try:
        import joblib
        model_path = 'models/random_forest_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.debug(f"Model saved successfully at {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise
def main():
    # Example usage
    traincsv = 'data/transform/train.csv'
    testcsv = 'data/transform/test.csv'
    model, mse, r2 = train_model(traincsv, testcsv)
    
    save_model(model)
    logger.info("Model training pipeline completed successfully")
    # Save processed data
    processed_data_path = 'data/processed/processed_data.pkl'
    with open(processed_data_path, 'wb') as f:
        pickle.dump((traincsv, testcsv), f)
    logger.info(f"Processed data saved successfully at {processed_data_path}")
if __name__ == "__main__":
    main()