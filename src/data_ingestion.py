import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler = logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'))
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.dropna()
        df = pd.get_dummies(df, columns=['region'], drop_first=True)
        logger.debug("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, train_path: str, test_path: str):
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.debug(f"Training data saved to {train_path}")
        logger.debug(f"Testing data saved to {test_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        # Load data
        input_file_path = r'data/insurance.csv'
        data = load_data(input_file_path)
        
        # Preprocess data
        data = preprocess_data(data)
        
        # Split data
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
        
        # Save processed data
        save_data(
            train_df, 
            test_df, 
            'data/processed/train.csv', 
            'data/processed/test.csv'
        )
        logger.debug("Data ingestion process completed successfully")
    except Exception as e:
        logger.error(f"Data ingestion process failed: {e}")
        raise

if __name__ == "__main__":
    main()
