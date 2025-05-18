import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import numpy as np

class TextCategorizer:
    def __init__(self, model_path='model.joblib'):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
            except Exception as e:
                print(f"Error loading model: {e}")
                self._initialize_model()
        else:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize a new model"""
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
        self.is_trained = False

    def train(self, texts, labels):
        """
        Train the model with new data
        """
        if isinstance(texts, np.ndarray):
            if texts.size == 0:
                raise ValueError("Training data cannot be empty")
        elif not texts:
            raise ValueError("Training data cannot be empty")
            
        if isinstance(labels, np.ndarray):
            if labels.size == 0:
                raise ValueError("Labels cannot be empty")
        elif not labels:
            raise ValueError("Labels cannot be empty")
        
        # Initialize model if not already done
        if not self.model:
            self._initialize_model()
            
        # Fit the model
        self.model.fit(texts, labels)
        self.is_trained = True
        self.save_model()

    def predict(self, texts):
        """
        Predict categories for given texts
        """
        if not self.is_trained:
            raise Exception("Model is not trained. Please train the model first.")
            
        if not texts:
            raise ValueError("Input texts cannot be empty")
            
        predictions = self.model.predict(texts)
        probabilities = self.model.predict_proba(texts)
        max_probs = probabilities.max(axis=1)
        
        results = []
        for text, pred, prob in zip(texts, predictions, max_probs):
            results.append({
                'text': text,
                'category': pred,
                'confidence': float(prob)
            })
        return results

    def save_model(self):
        """
        Save the trained model to disk
        """
        if self.is_trained:
            joblib.dump(self.model, self.model_path)
        else:
            raise Exception("Cannot save untrained model")

    def evaluate(self, texts, labels):
        """
        Evaluate model performance
        """
        if not self.is_trained:
            raise Exception("Model is not trained. Please train the model first.")
            
        from sklearn.metrics import classification_report
        predictions = self.model.predict(texts)
        return classification_report(labels, predictions) 