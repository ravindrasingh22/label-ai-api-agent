# API Reference

## Endpoints

### 1. Predict Category
```http
POST /predict
Content-Type: application/json

{
    "description": "string"
}
```

#### Response
```json
{
    "category": "string",
    "confidence": float
}
```

### 2. Train Model
```http
POST /train
Content-Type: application/json

{
    "training_file": "string"
}
```

#### Response
```json
{
    "status": "success",
    "message": "string",
    "metrics": {
        "accuracy": float,
        "macro_avg_f1": float
    }
}
```

### 3. Get Model Metrics
```http
GET /metrics
```

#### Response
```json
{
    "latest": {
        "accuracy": float,
        "macro_avg_f1": float,
        "timestamp": "string"
    },
    "history": [
        {
            "accuracy": float,
            "macro_avg_f1": float,
            "timestamp": "string"
        }
    ]
}
```

### 4. Get Model Versions
```http
GET /versions
```

#### Response
```json
{
    "versions": [
        {
            "timestamp": "string",
            "model_hash": "string",
            "metrics": {
                "accuracy": float,
                "macro_avg_f1": float
            }
        }
    ]
}
```

## Error Responses

### 1. Bad Request
```json
{
    "error": "Bad Request",
    "message": "string"
}
```

### 2. Not Found
```json
{
    "error": "Not Found",
    "message": "string"
}
```

### 3. Internal Server Error
```json
{
    "error": "Internal Server Error",
    "message": "string"
}
```

## Status Codes

- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

## Rate Limiting

- 100 requests per minute
- 1000 requests per hour

## Authentication

Currently, the API does not require authentication.

## CORS

Cross-Origin Resource Sharing is enabled for all origins.

## Response Headers

```
Content-Type: application/json
X-Request-ID: string
X-Response-Time: number
```

## Request Headers

```
Content-Type: application/json
Accept: application/json
```

## Examples

### Python
```python
import requests

# Predict category
response = requests.post(
    'http://localhost:5000/predict',
    json={'description': 'Invoice for services rendered'}
)
print(response.json())

# Train model
response = requests.post(
    'http://localhost:5000/train',
    json={'training_file': 'data/training/invoice_training_data.csv'}
)
print(response.json())
```

### cURL
```bash
# Predict category
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"description": "Invoice for services rendered"}'

# Train model
curl -X POST http://localhost:5000/train \
    -H "Content-Type: application/json" \
    -d '{"training_file": "data/training/invoice_training_data.csv"}'
```

## WebSocket Events

### 1. Training Progress
```json
{
    "event": "training_progress",
    "data": {
        "progress": float,
        "status": "string"
    }
}
```

### 2. Model Update
```json
{
    "event": "model_update",
    "data": {
        "version": "string",
        "metrics": {
            "accuracy": float,
            "macro_avg_f1": float
        }
    }
}
```

## WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:5000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
``` 