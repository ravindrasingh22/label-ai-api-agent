# Development Setup

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git

## Local Development Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd cat-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Directory Structure
```bash
mkdir -p data/{training,models,metrics}
```

### 5. Run Development Server
```bash
python app.py
```

## Docker Development

### 1. Build Docker Image
```bash
docker-compose build
```

### 2. Run Services
```bash
docker-compose up
```

### 3. Run in Background
```bash
docker-compose up -d
```

### 4. View Logs
```bash
docker-compose logs -f
```

## Development Workflow

### 1. Code Structure
```
cat-ai/
├── app.py              # Main application
├── model.py            # Model implementation
├── train.py            # Training script
├── model_metrics.py    # Metrics tracking
├── model_versioning.py # Version control
├── watch_and_train.py  # File watcher
├── requirements.txt    # Dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker services
└── data/              # Data directory
    ├── training/      # Training data
    ├── models/        # Model versions
    └── metrics/       # Performance metrics
```

### 2. Testing
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=.
```

### 3. Linting
```bash
# Run linter
flake8

# Run formatter
black .
```

### 4. Type Checking
```bash
mypy .
```

## Development Tools

### 1. VS Code Extensions
- Python
- Docker
- GitLens
- Pylance
- Black Formatter

### 2. Recommended Settings
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

## Debugging

### 1. Local Debugging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Docker Debugging
```bash
# Attach to container
docker-compose exec app bash

# View logs
docker-compose logs -f app
```

## Performance Monitoring

### 1. Local Monitoring
```bash
# CPU usage
top

# Memory usage
free -m

# Disk usage
df -h
```

### 2. Docker Monitoring
```bash
# Container stats
docker stats

# Resource usage
docker-compose top
```

## Security

### 1. Local Development
- Use virtual environment
- Don't commit sensitive data
- Use environment variables
- Regular dependency updates

### 2. Docker Security
- Use non-root user
- Regular image updates
- Scan for vulnerabilities
- Limit container resources

## Best Practices

### 1. Code Quality
- Follow PEP 8
- Write unit tests
- Use type hints
- Document code

### 2. Git Workflow
- Use feature branches
- Write meaningful commits
- Review code changes
- Keep history clean

### 3. Docker Practices
- Use multi-stage builds
- Optimize image size
- Cache dependencies
- Use .dockerignore

### 4. Testing
- Write unit tests
- Use test fixtures
- Mock external services
- Test edge cases

## Troubleshooting

### 1. Common Issues
- Port conflicts
- Permission issues
- Memory problems
- Network issues

### 2. Solutions
- Check port availability
- Verify permissions
- Monitor resources
- Check network config

## Contributing

### 1. Pull Request Process
- Create feature branch
- Write tests
- Update documentation
- Submit PR

### 2. Code Review
- Review changes
- Check tests
- Verify documentation
- Approve changes 