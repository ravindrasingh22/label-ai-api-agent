import pandas as pd
from model import TextCategorizer
import argparse
import sys
from model_metrics import ModelMetrics
from model_versioning import ModelVersioning
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def train_model(input_file):
    """
    Train the model using data from a CSV file
    Expected CSV format: text,category
    """
    try:
        # Initialize metrics and versioning
        metrics = ModelMetrics()
        versioning = ModelVersioning()
        
        # Read the training data
        df = pd.read_csv(input_file)
        
        # Validate the data
        if df.empty:
            logging.error("Error: Training data file is empty")
            sys.exit(1)
            
        if 'text' not in df.columns or 'category' not in df.columns:
            logging.error("Error: CSV file must contain 'text' and 'category' columns")
            sys.exit(1)
            
        # Remove any rows with missing values
        df = df.dropna()
        
        if df.empty:
            logging.error("Error: No valid training data after removing missing values")
            sys.exit(1)
        
        # Initialize and train the model
        categorizer = TextCategorizer()
        categorizer.train(df['text'].values, df['category'].values)
        
        # Get predictions for metrics
        predictions = categorizer.model.predict(df['text'].values)
        
        # Save metrics
        categories = df['category'].unique()
        metrics_summary = metrics.save_metrics(
            df['category'].values,
            predictions,
            categories
        )
        
        # Save model version
        version_info = versioning.save_model_version(
            'model.joblib',
            metadata={
                'training_file': input_file,
                'metrics': metrics_summary,
                'num_samples': len(df),
                'categories': list(categories)
            }
        )
        
        logging.info(f"Model trained and saved successfully!")
        logging.info(f"Model version: {version_info['version']}")
        logging.info(f"Accuracy: {metrics_summary['accuracy']:.4f}")
        
        # Generate metrics plots
        metrics.plot_metrics_history(metric='accuracy')
        metrics.plot_metrics_history(metric='macro_avg_f1')
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the text categorization model')
    parser.add_argument('input_file', help='Path to the CSV file containing training data')
    args = parser.parse_args()
    
    train_model(args.input_file) 