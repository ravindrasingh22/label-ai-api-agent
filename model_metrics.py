import json
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import logging

class ModelMetrics:
    def __init__(self, metrics_dir='data/metrics'):
        self.metrics_dir = metrics_dir
        self.ensure_metrics_dir()
        
    def ensure_metrics_dir(self):
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)
            
    def save_metrics(self, y_true, y_pred, categories, model_name='model'):
        """
        Save model metrics and generate visualizations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            categories: List of category names
            model_name: Name of the model
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_dir = os.path.join(self.metrics_dir, timestamp)
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Calculate metrics
            report = classification_report(y_true, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            # Convert numpy arrays to lists
            y_true = y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true)
            y_pred = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
            conf_matrix = conf_matrix.tolist()
            
            # Save metrics data
            metrics_data = {
                'timestamp': timestamp,
                'model_name': model_name,
                'accuracy': float(report['accuracy']),  # Convert numpy float to Python float
                'macro_avg_precision': float(report['macro avg']['precision']),
                'macro_avg_recall': float(report['macro avg']['recall']),
                'macro_avg_f1': float(report['macro avg']['f1-score']),
                'categories': categories.tolist() if hasattr(categories, 'tolist') else list(categories),
                'confusion_matrix': conf_matrix,
                'category_metrics': {
                    cat: {
                        'precision': float(report.get(cat, {}).get('precision', 0.0)),
                        'recall': float(report.get(cat, {}).get('recall', 0.0)),
                        'f1-score': float(report.get(cat, {}).get('f1-score', 0.0))
                    }
                    for cat in categories
                }
            }
            
            # Save metrics to JSON
            metrics_file = os.path.join(metrics_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Generate and save plots
            self.plot_confusion_matrix(conf_matrix, categories, metrics_dir)
            self.plot_metrics_history(metrics_data, metrics_dir)
            
            return metrics_data
            
        except Exception as e:
            logging.error(f"Error saving metrics: {str(e)}")
            raise
            
    def get_latest_metrics(self):
        """Get the most recent metrics."""
        try:
            if not os.path.exists(self.metrics_dir):
                return None
                
            metrics_dirs = sorted(os.listdir(self.metrics_dir), reverse=True)
            if not metrics_dirs:
                return None
                
            latest_dir = metrics_dirs[0]
            metrics_file = os.path.join(self.metrics_dir, latest_dir, 'metrics.json')
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            return None
            
        except Exception as e:
            logging.error(f"Error getting latest metrics: {str(e)}")
            return None
    
    def plot_confusion_matrix(self, conf_matrix, categories, metrics_dir):
        """
        Plot and save confusion matrix.
        
        Args:
            conf_matrix: Confusion matrix array
            categories: List of category names
            metrics_dir: Directory to save the plot
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=categories, yticklabels=categories)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            plot_path = os.path.join(metrics_dir, 'confusion_matrix.png')
            plt.savefig(plot_path)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def plot_metrics_history(self, metrics_data, metrics_dir):
        """
        Plot and save metrics history.
        
        Args:
            metrics_data: Current metrics data
            metrics_dir: Directory to save the plot
        """
        try:
            # Get all metrics history
            metrics_history = []
            for timestamp_dir in sorted(os.listdir(self.metrics_dir)):
                metrics_file = os.path.join(self.metrics_dir, timestamp_dir, 'metrics.json')
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            data = json.load(f)
                            metrics_history.append({
                                'timestamp': data['timestamp'],
                                'accuracy': data['accuracy'],
                                'macro_avg_f1': data['macro_avg_f1']
                            })
                    except json.JSONDecodeError as e:
                        logging.warning(f"Could not read metrics file {metrics_file}: {str(e)}")
                        continue
                    except KeyError as e:
                        logging.warning(f"Missing required field in metrics file {metrics_file}: {str(e)}")
                        continue
            
            # Add current metrics to history
            metrics_history.append({
                'timestamp': metrics_data['timestamp'],
                'accuracy': metrics_data['accuracy'],
                'macro_avg_f1': metrics_data['macro_avg_f1']
            })
            
            if not metrics_history:
                logging.warning("No metrics history available for plotting")
                return
            
            # Plot accuracy history
            plt.figure(figsize=(10, 6))
            timestamps = [m['timestamp'] for m in metrics_history]
            accuracy_values = [m['accuracy'] for m in metrics_history]
            
            plt.plot(timestamps, accuracy_values, marker='o', label='Accuracy')
            plt.title('Model Accuracy Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            
            accuracy_plot_path = os.path.join(metrics_dir, 'accuracy_history.png')
            plt.savefig(accuracy_plot_path)
            plt.close()
            
            # Plot F1 score history
            plt.figure(figsize=(10, 6))
            f1_values = [m['macro_avg_f1'] for m in metrics_history]
            
            plt.plot(timestamps, f1_values, marker='o', label='F1 Score')
            plt.title('Model F1 Score Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('F1 Score')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            
            f1_plot_path = os.path.join(metrics_dir, 'f1_history.png')
            plt.savefig(f1_plot_path)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting metrics history: {str(e)}")
            # Don't raise the error, just log it and continue
            return 