import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import logging
import pandas as pd

class TextCategorizer:
    def __init__(self, model_dir='data/models'):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = MultinomialNB()
        self.model_dir = model_dir
        self.ensure_model_dir()
        
    def ensure_model_dir(self):
        """Ensure the model directory exists."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
    def train(self, texts, labels):
        """
        Train the text categorization model.
        
        Args:
            texts (array-like): List of text samples
            labels (array-like): List of corresponding labels
        """
        if isinstance(texts, pd.Series) and texts.empty or isinstance(labels, pd.Series) and labels.empty:
            raise ValueError("Training data and labels cannot be empty")
            
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
            
        # Convert inputs to numpy arrays for consistent handling
        texts = np.array(texts)
        labels = np.array(labels)
        
        # Transform texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        
        # Train the classifier
        self.classifier.fit(X, labels)
        
        # Save the model
        self.save_model()
        
    def predict(self, texts):
        """
        Predict categories for given texts.
        
        Args:
            texts (array-like): List of texts to categorize
            
        Returns:
            list: List of dictionaries containing predictions
        """
        try:
            if not hasattr(self, 'classifier') or not hasattr(self, 'vectorizer'):
                self.load_model()
                
            if not texts:
                raise ValueError("No texts provided for prediction")
                
            # Convert input to numpy array for consistent handling
            texts = np.array(texts)
            
            # Transform texts to TF-IDF features
            X = self.vectorizer.transform(texts)
            
            # Get predictions
            predictions = self.classifier.predict(X)
            
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(X)
            
            # Format results
            results = []
            for text, pred, probs in zip(texts, predictions, probabilities):
                results.append({
                    'text': text,
                    'predicted_category': pred,
                    'confidence': float(max(probs))  # Convert numpy float to Python float
                })
                
            return results
            
        except Exception as e:
            logging.error(f"Error in predict method: {str(e)}")
            raise
        
    def save_model(self):
        """Save the trained model and vectorizer."""
        try:
            model_path = os.path.join(self.model_dir, 'model.joblib')
            vectorizer_path = os.path.join(self.model_dir, 'vectorizer.joblib')
            
            joblib.dump(self.classifier, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            
            logging.info(f"Model saved successfully to {self.model_dir}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
        
    def load_model(self):
        """Load the trained model and vectorizer."""
        try:
            model_path = os.path.join(self.model_dir, 'model.joblib')
            vectorizer_path = os.path.join(self.model_dir, 'vectorizer.joblib')
            
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                raise ValueError("Model files not found. Please train the model first.")
                
            self.classifier = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            logging.info(f"Model loaded successfully from {self.model_dir}")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def evaluate(self, texts, labels):
        """
        Evaluate model performance
        """
        try:
            if not hasattr(self, 'classifier') or not hasattr(self, 'vectorizer'):
                self.load_model()
                
            from sklearn.metrics import classification_report
            predictions = self.classifier.predict(self.vectorizer.transform(texts))
            return classification_report(labels, predictions)
            
        except Exception as e:
            logging.error(f"Error in evaluate method: {str(e)}")
            raise 