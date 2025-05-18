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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Save classification report
        report_path = os.path.join(self.metrics_dir, f'{model_name}_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories, yticklabels=categories)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        matrix_path = os.path.join(self.metrics_dir, f'{model_name}_confusion_matrix_{timestamp}.png')
        plt.savefig(matrix_path)
        plt.close()
        
        # Save metrics summary
        metrics_summary = {
            'timestamp': timestamp,
            'model_name': model_name,
            'accuracy': report['accuracy'],
            'macro_avg_precision': report['macro avg']['precision'],
            'macro_avg_recall': report['macro avg']['recall'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'categories': categories,
            'confusion_matrix_path': matrix_path,
            'report_path': report_path
        }
        
        summary_path = os.path.join(self.metrics_dir, f'{model_name}_summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
            
        return metrics_summary
    
    def get_latest_metrics(self, model_name='model'):
        """Get the latest metrics for a model"""
        pattern = os.path.join(self.metrics_dir, f'{model_name}_summary_*.json')
        files = glob.glob(pattern)
        if not files:
            return None
            
        latest_file = max(files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            return json.load(f)
            
    def plot_metrics_history(self, model_name='model', metric='accuracy'):
        """Plot the history of a specific metric"""
        pattern = os.path.join(self.metrics_dir, f'{model_name}_summary_*.json')
        files = glob.glob(pattern)
        
        if not files:
            return None
            
        metrics_history = []
        for file in sorted(files):
            with open(file, 'r') as f:
                data = json.load(f)
                metrics_history.append({
                    'timestamp': data['timestamp'],
                    metric: data[metric]
                })
                
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamp'], df[metric])
        plt.title(f'{metric.capitalize()} Over Time')
        plt.xlabel('Time')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = os.path.join(self.metrics_dir, f'{model_name}_{metric}_history.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def plot_confusion_matrix(self, y_true=None, y_pred=None, labels=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array-like, optional): True labels
            y_pred (array-like, optional): Predicted labels
            labels (list, optional): Label names
        """
        try:
            # Get latest metrics if no data provided
            if y_true is None or y_pred is None:
                latest_metrics = self.get_latest_metrics()
                if latest_metrics and 'confusion_matrix' in latest_metrics:
                    cm = np.array(latest_metrics['confusion_matrix'])
                else:
                    logging.warning("No confusion matrix data available")
                    return
            else:
                cm = confusion_matrix(y_true, y_pred)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_dir = os.path.join(self.metrics_dir, timestamp)
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def plot_metrics_history(self, metric='accuracy'):
        """
        Plot metrics history over time.
        
        Args:
            metric (str): Metric to plot ('accuracy' or 'macro_avg_f1')
        """
        try:
            # Get all metrics
            metrics_data = []
            for timestamp_dir in sorted(os.listdir(self.metrics_dir)):
                metrics_file = os.path.join(self.metrics_dir, timestamp_dir, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        metrics_data.append({
                            'timestamp': data['timestamp'],
                            'value': data[metric]
                        })
            
            if not metrics_data:
                logging.warning(f"No {metric} data available")
                return
            
            # Create plot
            plt.figure(figsize=(10, 6))
            timestamps = [m['timestamp'] for m in metrics_data]
            values = [m['value'] for m in metrics_data]
            
            plt.plot(timestamps, values, marker='o')
            plt.title(f'{metric.capitalize()} Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel(metric.capitalize())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_dir = os.path.join(self.metrics_dir, timestamp)
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, 'metrics_history.png'))
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting metrics history: {str(e)}")
            raise 