# Model Versioning System

The Model Versioning system provides robust version control and management for trained models.

## Features

### 1. Version Control
- Automatic versioning of models
- Timestamp-based versioning
- Hash-based model identification
- Metadata tracking

### 2. Model Management
- Version comparison
- Version history
- Latest version tracking
- Model metadata storage

### 3. Storage and Organization
- Structured directory organization
- Metadata JSON storage
- Model file management
- Version-specific directories

## Usage

### Basic Usage
```python
from model_versioning import ModelVersioning

# Initialize versioning
versioning = ModelVersioning()

# Save new model version
version_info = versioning.save_model_version(
    model_path='model.joblib',
    metadata={
        'training_file': 'training_data.csv',
        'metrics': metrics_summary,
        'num_samples': 1000
    }
)
```

### Version Management
```python
# Get specific version
version = versioning.get_model_version('20240101_120000')

# List all versions
versions = versioning.list_versions()

# Get latest version
latest = versioning.get_latest_version()

# Compare versions
comparison = versioning.compare_versions('20240101_120000', '20240102_120000')
```

## File Structure
```
data/models/
└── version_TIMESTAMP/
    ├── model.joblib        # Model file
    └── metadata.json       # Version metadata
```

## Version Details

### Metadata Structure
```json
{
    "timestamp": "20240101_120000",
    "model_hash": "sha256_hash",
    "original_path": "path/to/original",
    "training_file": "training_data.csv",
    "metrics": {
        "accuracy": 0.95,
        "macro_avg_f1": 0.94
    },
    "num_samples": 1000,
    "categories": ["category1", "category2"]
}
```

### Version Comparison
- Time difference between versions
- Model hash comparison
- Metadata comparison
- Performance metrics comparison

## Best Practices

1. **Version Management**
   - Regular version creation
   - Meaningful metadata
   - Proper version documentation
   - Cleanup of old versions

2. **Metadata Usage**
   - Include relevant training information
   - Track performance metrics
   - Document model changes
   - Maintain version history

3. **Storage Organization**
   - Regular cleanup of old versions
   - Archive important versions
   - Maintain clear directory structure
   - Backup critical versions

4. **Version Control**
   - Use meaningful timestamps
   - Track model changes
   - Document improvements
   - Maintain version history 