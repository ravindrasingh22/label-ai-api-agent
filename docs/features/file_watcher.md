# File Watcher System

The File Watcher system provides automatic model retraining when new training data is added.

## Features

### 1. Automatic Monitoring
- Real-time directory monitoring
- File change detection
- Automatic retraining triggers
- Event-based processing

### 2. Training Management
- Automatic model retraining
- Training status tracking
- Error handling and logging
- Training history maintenance

### 3. File System Integration
- Directory structure monitoring
- File change detection
- Automatic retraining
- Logging and notifications

## Usage

### Basic Usage
```python
from watch_and_train import watch_training_directory

# Start watching directory
watch_training_directory(
    training_dir='data/training',
    model_dir='data/models',
    metrics_dir='data/metrics'
)
```

### Event Handling
```python
# File creation event
def on_created(event):
    if event.is_directory:
        return
    if event.src_path.endswith('.csv'):
        retrain_model()

# File modification event
def on_modified(event):
    if event.is_directory:
        return
    if event.src_path.endswith('.csv'):
        retrain_model()
```

## File Structure
```
data/
├── training/           # Training data directory
│   └── *.csv          # Training data files
├── models/            # Model versions
│   └── version_*/     # Version-specific directories
└── metrics/           # Training metrics
    └── TIMESTAMP/     # Training-specific metrics
```

## Event Types

### 1. File Events
- File creation
- File modification
- File deletion
- Directory changes

### 2. Training Events
- Training start
- Training completion
- Training failure
- Model update

## Best Practices

1. **Directory Structure**
   - Maintain clear organization
   - Use consistent naming
   - Separate training data
   - Organize model versions

2. **File Management**
   - Regular cleanup
   - Backup important files
   - Monitor disk space
   - Archive old data

3. **Training Process**
   - Validate new data
   - Track training history
   - Monitor performance
   - Handle errors gracefully

4. **System Maintenance**
   - Regular log review
   - Performance monitoring
   - Error tracking
   - System health checks

## Error Handling

### 1. File System Errors
- Directory not found
- Permission issues
- Disk space problems
- File access errors

### 2. Training Errors
- Invalid data format
- Training failures
- Model saving issues
- Resource constraints

### 3. System Errors
- Memory issues
- CPU overload
- Network problems
- Service interruptions

## Logging and Monitoring

### 1. Event Logging
- File changes
- Training events
- Error messages
- System status

### 2. Performance Monitoring
- Training duration
- Resource usage
- Model performance
- System health

### 3. Error Tracking
- Error types
- Error frequency
- Error resolution
- System recovery 