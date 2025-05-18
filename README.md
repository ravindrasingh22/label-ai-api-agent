# Text Categorization AI - API

This project provides a text categorization service that can classify text inputs into predefined categories. It uses a machine learning model based on TF-IDF vectorization and Naive Bayes classification.

## Features

- Text categorization using machine learning
- REST API endpoint for predictions
- Automatic model training when new data is added
- Docker containerization for easy deployment
- Hot-reload support for development

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)

## Quick Start

1. Clone the repository
2. Place your training data CSV files in the `data/training` directory
3. Run the application using Docker Compose:
   ```bash
   docker-compose up --build
   ```
4. The API will be available at `http://localhost:3040`

## Training Data Structure

Place your training data CSV files in the `data/training` directory. The system will automatically:
- Train the model when new CSV files are added
- Retrain the model when existing files are modified
- Process all CSV files in the directory on startup

Expected CSV format:
```csv
text,category
"This is a sample text",category1
"Another example text",category2
```

## API Endpoints

### Predict Categories
- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "texts": ["text to categorize", "another text to categorize"]
  }
  ```
- **Response**:
  ```json
  {
    "predictions": [
      {
        "text": "text to categorize",
        "category": "predicted_category",
        "confidence": 0.95
      }
    ]
  }
  ```

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**: `{"status": "healthy", "model_status": "trained"}`

## Development

The application is configured with hot-reload support. Any changes to the Python files will automatically trigger a reload of the application.

## Project Structure

- `app.py`: FastAPI application and endpoints
- `model.py`: Text categorization model implementation
- `train.py`: Script for training the model
- `watch_and_train.py`: File watcher for automatic training
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container configuration
- `docker-compose.yml`: Docker Compose configuration
- `data/training/`: Directory for training data files

## Model Details

The model uses:
- TF-IDF vectorization for text feature extraction
- Multinomial Naive Bayes for classification
- Maximum features: 5000
- Model persistence using joblib

## Adding New Training Data

To add new training data:

1. Place your CSV file in the `data/training` directory
2. The system will automatically:
   - Detect the new file
   - Train the model with the new data
   - Update the model for predictions

## Notes

- The model is automatically loaded when the API starts
- Training data should be representative of the categories you want to predict
- More training data generally leads to better accuracy
- The confidence score indicates the model's certainty in its prediction
- The system will automatically retrain when training data is modified 