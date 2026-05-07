import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler = logging.FileHandler(os.path.join(log_dir, 'data_preprocessing.log'))  # fix log file name
file_handler.setLevel('DEBUG')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(traincsv, testcsv):
    try:
        train_df = pd.read_csv(traincsv)
        test_df = pd.read_csv(testcsv)
        logger.debug("Data loaded successfully for transformation")
        train_df['sex'] = train_df['sex'].map({'male': 0, 'female': 1})
        train_df['smoker'] = train_df['smoker'].map({'yes': 1, 'no': 0})
        test_df['sex'] = test_df['sex'].map({'male': 0, 'female': 1})
        test_df['smoker'] = test_df['smoker'].map({'yes': 1, 'no': 0})
        # Example transformation: Fill missing values
        train_df['region_northwest'] = train_df['region_northwest'].astype('int64')
        train_df['region_southeast'] = train_df['region_southeast'].astype('int64')
        train_df['region_southwest'] = train_df['region_southwest'].astype('int64')
        test_df['region_northwest'] = test_df['region_northwest'].astype('int64')
        test_df['region_southeast'] = test_df['region_southeast'].astype('int64')
        test_df['region_southwest'] = test_df['region_southwest'].astype('int64')
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        train_df[['age','bmi','children']] = sc.fit_transform(train_df[['age','bmi','children']])
        test_df[['age','bmi','children']] = sc.transform(test_df[['age','bmi','children']])
        logger.debug("Data transformation completed successfully")
        
        
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error in data transformation: {e}")
        raise
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, train_path: str, test_path: str):
    """
    Save the training and testing DataFrames to CSV files.
    
    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        train_path (str): Path to save the training CSV file.
        test_path (str): Path to save the testing CSV file.
    """
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True) 
        logger.debug(f"Saving data to {train_path} and {test_path}")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.debug(f"Data saved successfully to {train_path} and {test_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise
def main():
    try:
        # Paths to raw data
        traincsv = 'data/processed/train.csv'
        testcsv = 'data/processed/test.csv'
        # Transform data
        train_df, test_df = transform_text(traincsv, testcsv)
        # Save processed data
        save_data(train_df, test_df, 'data/transform/train.csv', 'data/transform/test.csv')  # fix typo here
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
if __name__ == "__main__":
    main()