import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from train import train_model
import glob
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class TrainingDataHandler(FileSystemEventHandler):
    def __init__(self, training_dir):
        self.training_dir = training_dir
        self.last_processed = set()
        self.process_training_files()

    def process_training_files(self):
        """Process all CSV files in the training directory"""
        csv_files = glob.glob(os.path.join(self.training_dir, "*.csv"))
        for file_path in csv_files:
            if file_path not in self.last_processed:
                try:
                    logging.info(f"Processing new training file: {file_path}")
                    train_model(file_path)
                    self.last_processed.add(file_path)
                    logging.info(f"Successfully trained model with {file_path}")
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.csv'):
            logging.info(f"New training file detected: {event.src_path}")
            self.process_training_files()

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.csv'):
            logging.info(f"Training file modified: {event.src_path}")
            self.process_training_files()

def watch_training_directory(training_dir='data/training'):
    """Watch the training directory for changes"""
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
        logging.info(f"Created training directory: {training_dir}")

    event_handler = TrainingDataHandler(training_dir)
    observer = Observer()
    observer.schedule(event_handler, training_dir, recursive=False)
    observer.start()

    logging.info(f"Started watching {training_dir} for training data changes")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Stopped watching training directory")
    
    observer.join()

if __name__ == "__main__":
    watch_training_directory() 