# Telecom AI Platform v2.0

A comprehensive, refactored AI platform for telecom network anomaly detection and analysis with conversational AI capabilities.

## Overview

The Telecom AI Platform is a modernized, well-structured system designed for analyzing telecom network performance data, detecting anomalies, and providing insights through natural language interactions. This refactored version follows clean coding principles, modular architecture, and provides a user-friendly interface.

## Key Features

- **Advanced Anomaly Detection**: KPI-specific machine learning models for accurate anomaly detection
- **Conversational AI**: Natural language interface powered by Google Gemini LLM
- **RESTful API**: Complete FastAPI-based web service
- **Rich Visualizations**: Interactive charts and dashboards
- **Performance Monitoring**: Real-time KPI tracking and analysis
- **Modular Architecture**: Clean, maintainable codebase
- **Comprehensive Logging**: Detailed logging and monitoring

## Architecture

```
telecom_ai_platform/
├── core/                   # Core functionality
│   ├── config.py          # Configuration management
│   └── data_processor.py  # Data processing pipeline
├── models/                 # Machine learning models
│   ├── anomaly_detector.py # Anomaly detection models
│   └── trainer.py         # Model training pipeline
├── agents/                 # AI agents
│   └── conversational_ai.py # Conversational AI agent
├── server/                 # Web API server
│   └── api.py             # FastAPI application
├── utils/                  # Utilities
│   ├── logger.py          # Logging utilities
│   └── visualizer.py      # Visualization tools
├── ui/                     # User interface components
├── data/                   # Data storage
├── notebooks/              # Jupyter notebooks
└── tests/                  # Test suites
```

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd telecom_ai_platform
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Create .env file with your API keys
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

### Basic Usage

#### 1. Command Line Interface

**Train Models**:
```bash
python -m telecom_ai_platform.main train data/AD_data_10KPI.csv
```

**Start Interactive Chat**:
```bash
python -m telecom_ai_platform.main chat --load-models
```

**Start API Server**:
```bash
python -m telecom_ai_platform.main server --host 0.0.0.0 --port 8000
```

**Check Status**:
```bash
python -m telecom_ai_platform.main status
```

#### 2. Python API

```python
from telecom_ai_platform import TelecomAIPlatform

# Initialize platform
platform = TelecomAIPlatform()

# Train models
training_result = platform.train_models('data/AD_data_10KPI.csv')

# Load pre-trained models
platform.load_models()

# Chat with AI agent
response = platform.chat("Show me anomalies in RSRP for the last week")
print(response.message)

# Start web server
platform.start_server(host='localhost', port=8000)
```

#### 3. Web API Usage

Once the server is running, access:
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

**Example API calls**:

```bash
# Upload data
curl -X POST "http://localhost:8000/upload-data" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/AD_data_10KPI.csv"

# Chat with AI
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Analyze RSRP trends for site 001"}'

# Detect anomalies
curl -X POST "http://localhost:8000/anomaly-detection" \
     -H "Content-Type: application/json" \
     -d '{"kpi_name": "RSRP", "site_id": "001"}'
```

## Data Format

The platform expects CSV data with the following structure:

| Column | Description | Example |
|--------|-------------|---------|
| Date | Timestamp | 2024-01-01 |
| Site_ID | Site identifier | SITE_001 |
| Sector_ID | Sector identifier (optional) | SEC_A |
| RSRP | Signal strength | -95.5 |
| SINR | Signal quality | 15.2 |
| DL_Throughput | Download throughput | 45.6 |
| UL_Throughput | Upload throughput | 12.3 |
| ... | Other KPIs | ... |

## Conversational AI Features

The AI agent can understand and respond to various queries:

- **Data Analysis**: "Show me the performance trends for site 001"
- **Anomaly Detection**: "Find anomalies in RSRP for the last week"
- **Site Comparison**: "Compare throughput between sites 001 and 002"
- **Performance Insights**: "What's the average SINR across all sites?"

## Configuration

### Environment Variables

Create a `.env` file with:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key

# Directories (optional - will use defaults)
DATA_DIR=./data
MODELS_DIR=./models
LOGS_DIR=./logs

# Model Parameters (optional)
CONTAMINATION_RATE=0.1
AUTOENCODER_EPOCHS=100
```

### Configuration Options

```python
from telecom_ai_platform.core.config import TelecomConfig

config = TelecomConfig(
    # Data configuration
    data_dir="./data",
    models_dir="./models",
    
    # Model parameters
    contamination_rate=0.1,
    
    # AI Agent configuration
    gemini_api_key="your_key",
    model_name="gemini-pro",
    temperature=0.7
)
```

## Anomaly Detection Models

The platform uses different algorithms optimized for each KPI type:

| KPI Type | Algorithm | Reasoning |
|----------|-----------|-----------|
| RSRP (Signal Strength) | Isolation Forest | Clear outlier detection |
| SINR (Signal Quality) | AutoEncoder | Temporal pattern learning |
| Throughput | Isolation Forest | Performance outliers |
| CPU Utilization | One-Class SVM | Non-linear boundaries |
| Active Users | Gaussian Mixture | Multi-modal distribution |

## API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | API information |
| `/status` | GET | System health status |
| `/chat` | POST | Conversational AI |
| `/anomaly-detection` | POST | Detect anomalies |
| `/upload-data` | POST | Upload and process data |
| `/train` | POST | Train models |
| `/models/summary` | GET | Model information |
| `/data/summary` | GET | Data summary |

## Logging

The platform provides comprehensive logging:

- **Console Output**: Real-time feedback
- **File Logging**: Detailed logs in `logs/` directory
- **Function Tracking**: Automatic function call logging
- **Performance Metrics**: Execution time tracking

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test categories:

```bash
# Test data processing
pytest tests/test_data_processor.py

# Test anomaly detection
pytest tests/test_anomaly_detector.py

# Test API endpoints
pytest tests/test_api.py
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY telecom_ai_platform/ ./telecom_ai_platform/
COPY data/ ./data/

EXPOSE 8000
CMD ["python", "-m", "telecom_ai_platform.main", "server"]
```

### Production Deployment

```bash
# Using Gunicorn
pip install gunicorn
gunicorn telecom_ai_platform.server.api:create_app --workers 4 --host 0.0.0.0 --port 8000
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes following the coding standards
4. Add tests for new functionality
5. Submit a pull request

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for function parameters and returns
- Add docstrings for all classes and functions
- Write unit tests for new features
- Use meaningful variable and function names

## Documentation

- **API Documentation**: Available at `/docs` when server is running
- **Code Documentation**: Inline docstrings and comments
- **Architecture Guide**: See `docs/architecture.md`
- **Development Guide**: See `docs/development.md`

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure GEMINI_API_KEY is set in environment
   - Check API key validity

2. **Data Loading Issues**:
   - Verify CSV format matches expected schema
   - Check file permissions and paths

3. **Model Training Failures**:
   - Ensure sufficient data for training
   - Check data quality and missing values

4. **Server Connection Issues**:
   - Verify port availability
   - Check firewall settings

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Acknowledgments

- Built with modern Python best practices
- Powered by Google Gemini LLM
- Uses industry-standard ML libraries
- Follows clean architecture principles

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the troubleshooting guide

---

**Version**: 2.0.0  
**Last Updated**: 2024-01-30  
**Python**: 3.9+
