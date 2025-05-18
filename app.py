from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model import TextCategorizer
import uvicorn
import os

app = FastAPI(title="Text Categorization API")
categorizer = TextCategorizer()

class PredictionRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    predictions: List[dict]

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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3040, reload=True) 