# Usage Examples

## Basic Usage

### 1. Training the Model
```python
from train import train_model

# Train with default settings
train_model('data/training/invoice_training_data.csv')

# Train with custom settings
train_model(
    'data/training/invoice_training_data.csv',
    test_size=0.2,
    random_state=42
)
```

### 2. Making Predictions
```python
from model import TextCategorizer

# Initialize model
model = TextCategorizer()

# Load trained model
model.load_model('data/models/latest/model.joblib')

# Make prediction
prediction = model.predict("Invoice for consulting services")
print(f"Category: {prediction['category']}")
print(f"Confidence: {prediction['confidence']:.2f}")
```

### 3. Using the API
```python
import requests

# Make prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={'description': 'Invoice for consulting services'}
)
print(response.json())

# Train model
response = requests.post(
    'http://localhost:5000/train',
    json={'training_file': 'data/training/invoice_training_data.csv'}
)
print(response.json())
```

## Advanced Usage

### 1. Model Metrics
```python
from model_metrics import ModelMetrics

# Initialize metrics
metrics = ModelMetrics()

# Save metrics after training
metrics.save_metrics(
    accuracy=0.95,
    macro_avg_f1=0.94,
    confusion_matrix=[[10, 2], [1, 15]]
)

# Get latest metrics
latest = metrics.get_latest_metrics()
print(f"Latest accuracy: {latest['accuracy']:.2f}")

# Plot confusion matrix
metrics.plot_confusion_matrix()
```

### 2. Model Versioning
```python
from model_versioning import ModelVersioning

# Initialize versioning
versioning = ModelVersioning()

# Save new version
version_info = versioning.save_model_version(
    model_path='model.joblib',
    metadata={
        'training_file': 'training_data.csv',
        'metrics': metrics_summary,
        'num_samples': 1000
    }
)

# Get specific version
version = versioning.get_model_version('20240101_120000')

# Compare versions
comparison = versioning.compare_versions(
    '20240101_120000',
    '20240102_120000'
)
```

### 3. File Watcher
```python
from watch_and_train import watch_training_directory

# Start watching directory
watch_training_directory(
    training_dir='data/training',
    model_dir='data/models',
    metrics_dir='data/metrics'
)
```

## Docker Usage

### 1. Running with Docker
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### 2. Training with Docker
```bash
# Train model
docker-compose exec app python train.py

# Train with specific file
docker-compose exec app python train.py data/training/invoice_training_data.csv
```

### 3. API with Docker
```bash
# Make prediction
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"description": "Invoice for consulting services"}'

# Train model
curl -X POST http://localhost:5000/train \
    -H "Content-Type: application/json" \
    -d '{"training_file": "data/training/invoice_training_data.csv"}'
```

## WebSocket Usage

### 1. Python Client
```python
import websocket
import json

# Connect to WebSocket
ws = websocket.WebSocketApp("ws://localhost:5000/ws")

# Handle messages
def on_message(ws, message):
    data = json.loads(message)
    print(f"Received: {data}")

ws.on_message = on_message
ws.run_forever()
```

### 2. JavaScript Client
```javascript
const ws = new WebSocket('ws://localhost:5000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

ws.onopen = () => {
    console.log('Connected to WebSocket');
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Disconnected from WebSocket');
};
```

## Error Handling

### 1. Model Errors
```python
from model import TextCategorizer, ModelNotTrainedError

try:
    model = TextCategorizer()
    prediction = model.predict("Some text")
except ModelNotTrainedError:
    print("Model needs to be trained first")
except Exception as e:
    print(f"Error: {str(e)}")
```

### 2. API Errors
```python
import requests
from requests.exceptions import RequestException

try:
    response = requests.post(
        'http://localhost:5000/predict',
        json={'description': 'Some text'}
    )
    response.raise_for_status()
    print(response.json())
except RequestException as e:
    print(f"API Error: {str(e)}")
```

### 3. File Watcher Errors
```python
from watch_and_train import watch_training_directory
from watchdog.events import FileSystemError

try:
    watch_training_directory(
        training_dir='data/training',
        model_dir='data/models',
        metrics_dir='data/metrics'
    )
except FileSystemError as e:
    print(f"File system error: {str(e)}")
except Exception as e:
    print(f"Error: {str(e)}")
```

## Best Practices

### 1. Model Training
- Use sufficient training data
- Validate data before training
- Monitor training metrics
- Save model versions

### 2. Predictions
- Handle untrained model cases
- Validate input data
- Check prediction confidence
- Log predictions

### 3. API Usage
- Use proper error handling
- Implement retry logic
- Monitor API health
- Cache responses

### 4. File Management
- Regular cleanup
- Backup important files
- Monitor disk space
- Archive old data 