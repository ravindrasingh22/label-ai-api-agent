from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from model import TextCategorizer
from model_metrics import ModelMetrics
import uvicorn
import os
import json
from datetime import datetime
from fastapi.responses import JSONResponse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="Text Categorization API")
categorizer = TextCategorizer()
metrics = ModelMetrics()

# Load the model when the app starts
try:
    categorizer.load_model()
    logging.info("Model loaded successfully on startup")
except Exception as e:
    logging.warning(f"Could not load model on startup: {str(e)}")

class PredictionRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    predictions: List[dict]

class MetricsResponse(BaseModel):
    accuracy: float
    macro_avg_f1: float
    category_metrics: Dict[str, Dict[str, float]]

class MetricsPlotResponse(BaseModel):
    confusion_matrix: str
    accuracy_history: str
    f1_history: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict categories for the given texts."""
    try:
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="No texts provided in the request"
            )
            
        model_path = os.path.join('data/models', 'model.joblib')
        vectorizer_path = os.path.join('data/models', 'vectorizer.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise HTTPException(
                status_code=400,
                detail="Model not trained yet. Please train the model first using train.py"
            )
            
        try:
            predictions = categorizer.predict(request.texts)
            logging.info(f"Successfully predicted categories for {len(request.texts)} texts")
            return PredictionResponse(predictions=predictions)
        except ValueError as e:
            logging.error(f"Value error during prediction: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    model_path = os.path.join('data/models', 'model.joblib')
    model_status = "trained" if os.path.exists(model_path) else "not trained"
    return {
        "status": "healthy",
        "model_status": model_status
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get the latest model metrics."""
    try:
        metrics = ModelMetrics()
        latest_metrics = metrics.get_latest_metrics()
        
        if not latest_metrics:
            raise HTTPException(
                status_code=404,
                detail="No metrics available. Train the model first."
            )
            
        return {
            "accuracy": latest_metrics["accuracy"],
            "macro_avg_f1": latest_metrics["macro_avg_f1"],
            "category_metrics": latest_metrics["category_metrics"]
        }
    except Exception as e:
        logging.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics: {str(e)}"
        )

@app.get("/metrics/plot", response_model=MetricsPlotResponse)
async def get_metrics_plot():
    """Get the latest metrics visualization plots."""
    try:
        metrics = ModelMetrics()
        latest_metrics = metrics.get_latest_metrics()
        
        if not latest_metrics:
            raise HTTPException(
                status_code=404,
                detail="No metrics available. Train the model first."
            )
            
        # Get the directory containing the latest metrics
        metrics_dirs = sorted(os.listdir(metrics.metrics_dir), reverse=True)
        if not metrics_dirs:
            raise HTTPException(
                status_code=404,
                detail="No metrics plots available."
            )
            
        latest_dir = metrics_dirs[0]
        metrics_dir = os.path.join(metrics.metrics_dir, latest_dir)
        
        # Check for plot files
        confusion_matrix_path = os.path.join(metrics_dir, 'confusion_matrix.png')
        accuracy_history_path = os.path.join(metrics_dir, 'accuracy_history.png')
        f1_history_path = os.path.join(metrics_dir, 'f1_history.png')
        
        if not all(os.path.exists(p) for p in [confusion_matrix_path, accuracy_history_path, f1_history_path]):
            raise HTTPException(
                status_code=404,
                detail="Some metrics plots are missing."
            )
            
        return {
            "confusion_matrix": confusion_matrix_path,
            "accuracy_history": accuracy_history_path,
            "f1_history": f1_history_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting metrics plots: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics plots: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3040, reload=True) 