import pandas as pd
from sklearn.model_selection import train_test_split
from model import TextCategorizer
from model_metrics import ModelMetrics
from model_versioning import ModelVersioning
import os
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def train_model(training_file: str, test_size: float = 0.2, random_state: int = 42):
    """
    Train the text categorization model.
    
    Args:
        training_file (str): Path to the training data file
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    """
    try:
        # Check if model already exists
        if os.path.exists('model.joblib'):
            logging.info("Model already exists. Skipping training.")
            return
        
        # Initialize components
        model = TextCategorizer()
        metrics = ModelMetrics()
        versioning = ModelVersioning()
        
        # Load and validate training data
        logging.info(f"Loading training data from {training_file}")
        if not os.path.exists(training_file):
            raise FileNotFoundError(f"Training file not found: {training_file}")
            
        df = pd.read_csv(training_file)
        if df.empty:
            raise ValueError("Training data is empty")
            
        if 'text' not in df.columns or 'category' not in df.columns:
            raise ValueError("Training data must contain 'text' and 'category' columns")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['category'],
            test_size=test_size,
            random_state=random_state
        )
        
        # Train model
        logging.info("Training model...")
        model.train(X_train, y_train)
        
        # Evaluate model
        logging.info("Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        logging.info(f"Model accuracy: {accuracy:.2f}")
        
        # Save metrics
        metrics.save_metrics(
            accuracy=accuracy,
            macro_avg_f1=0.0,  # TODO: Implement F1 score calculation
            confusion_matrix=None  # TODO: Implement confusion matrix
        )
        
        # Save model version
        versioning.save_model_version(
            model_path='model.joblib',
            metadata={
                'training_file': training_file,
                'num_samples': len(df),
                'test_size': test_size,
                'random_state': random_state,
                'accuracy': accuracy
            }
        )
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <training_file>")
        sys.exit(1)
        
    training_file = sys.argv[1]
    train_model(training_file) 