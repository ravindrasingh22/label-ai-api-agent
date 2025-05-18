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

app = FastAPI(title="Text Categorization API")
categorizer = TextCategorizer()
metrics = ModelMetrics()

class PredictionRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    predictions: List[dict]

class MetricsResponse(BaseModel):
    latest: Dict[str, Any]
    history: List[Dict[str, Any]]

class MetricsPlotResponse(BaseModel):
    confusion_matrix: str
    metrics_history: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="No texts provided in the request"
            )
            
        if not os.path.exists('model.joblib'):
            raise HTTPException(
                status_code=400,
                detail="Model not trained yet. Please train the model first using train.py"
            )
            
        predictions = categorizer.predict(request.texts)
        return PredictionResponse(predictions=predictions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    model_status = "trained" if os.path.exists('model.joblib') else "not trained"
    return {
        "status": "healthy",
        "model_status": model_status
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    try:
        # Get latest metrics
        latest_metrics = metrics.get_latest_metrics()
        
        # Get metrics history
        metrics_dir = 'data/metrics'
        history = []
        
        if os.path.exists(metrics_dir):
            for timestamp_dir in sorted(os.listdir(metrics_dir), reverse=True):
                metrics_file = os.path.join(metrics_dir, timestamp_dir, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        history.append({
                            'timestamp': timestamp_dir,
                            'metrics': json.load(f)
                        })
        
        # Get confusion matrix if available
        confusion_matrix = None
        if latest_metrics and 'confusion_matrix' in latest_metrics:
            confusion_matrix = latest_metrics['confusion_matrix']
        
        # Get category-wise metrics if available
        category_metrics = None
        if latest_metrics and 'category_metrics' in latest_metrics:
            category_metrics = latest_metrics['category_metrics']
        
        response = {
            'latest': {
                'accuracy': latest_metrics.get('accuracy', 0.0),
                'macro_avg_f1': latest_metrics.get('macro_avg_f1', 0.0),
                'timestamp': latest_metrics.get('timestamp', ''),
                'confusion_matrix': confusion_matrix,
                'category_metrics': category_metrics
            },
            'history': history
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/plot", response_model=MetricsPlotResponse)
async def get_metrics_plot():
    try:
        # Generate and save plots
        metrics.plot_confusion_matrix()
        metrics.plot_metrics_history()
        
        # Get the paths to the generated plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        confusion_matrix_path = f'data/metrics/{timestamp}/confusion_matrix.png'
        metrics_history_path = f'data/metrics/{timestamp}/metrics_history.png'
        
        # Check if plots were generated
        if not os.path.exists(confusion_matrix_path) or not os.path.exists(metrics_history_path):
            raise HTTPException(
                status_code=404,
                detail="Metrics plots not available"
            )
        
        # Return the paths to the plots
        return {
            'confusion_matrix': confusion_matrix_path,
            'metrics_history': metrics_history_path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3040, reload=True) 