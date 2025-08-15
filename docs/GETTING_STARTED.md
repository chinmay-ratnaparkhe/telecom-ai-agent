# Getting Started with Telecom AI Agent

## Quick Setup Guide

### 1. Environment Setup

Create and activate a Python virtual environment:

`ash
python -m venv telecom-ai-env
source telecom-ai-env/bin/activate  # On Windows: telecom-ai-env\Scripts\activate
`

### 2. Install Dependencies

`ash
pip install -r requirements.txt
`

### 3. Configuration

Copy the environment template and configure your settings:

`ash
cp .env.example .env
`

Edit .env file with your configurations:
- Add your Google API key for Gemini
- Configure data paths
- Set ML model parameters

### 4. Data Preparation

Place your telecom KPI data in the data/ directory. Expected format:
- CSV file with columns: Date, Site_ID, Sector_ID, KPI columns
- Sample data is provided in data/AD_data_10KPI.csv

### 5. Start the MCP Server

`ash
cd mcp_server
python updated_mcp_server.py
`

The server will start on http://localhost:8000

### 6. Launch the Application

In a new terminal:

`ash
cd src/telecom_ai_platform
streamlit run streamlit_ui/app.py
`

### 7. Access the Application

Open your browser and navigate to the Streamlit URL (typically http://localhost:8501)

## Usage Examples

### Basic Anomaly Detection

`python
# Natural language query
"Show me RSRP anomalies for site 001 in the last week"

# Expected response: Interactive visualization with anomaly highlights
`

### Multi-site Comparison

`python
# Query for comparison
"Compare downlink throughput between sites 010, 020, and 030"

# Result: Comparative charts and statistical analysis
`

### Trend Analysis

`python
# Trend query
"What are the KPI trends for sector A over the past month?"

# Output: Time-series analysis with trend indicators
`

## Advanced Features

### Custom Model Training

To train your own anomaly detection models:

1. Prepare your dataset in the expected format
2. Run the training notebook: 
otebooks/Enhanced_AutoEncoder_Training.ipynb
3. Models will be saved to mcp_server/models/

### API Integration

Direct API access to MCP tools:

`python
import requests

response = requests.post("http://localhost:8000/call_tool", json={
    "tool": "detect_anomalies",
    "parameters": {
        "site_id": "SITE_001",
        "kpi_name": "DL_Throughput",
        "date_range": "last_week"
    }
})

anomalies = response.json()
`

### Batch Processing

For large-scale analysis:

`python
from src.telecom_ai_platform.core.data_processor import DataProcessor
from src.telecom_ai_platform.models.anomaly_detector import AnomalyDetector

processor = DataProcessor()
detector = AnomalyDetector()

# Process multiple sites
sites = ["SITE_001", "SITE_002", "SITE_003"]
results = detector.batch_detect(sites, kpis=["RSRP", "DL_Throughput"])
`

## Troubleshooting

### Common Issues

1. **MCP Server Not Starting**
   - Check port 8000 is available
   - Verify Python dependencies are installed
   - Review server logs for specific errors

2. **Model Loading Errors**
   - Ensure model files exist in mcp_server/models/
   - Check PyTorch version compatibility
   - Verify CUDA setup for GPU models

3. **API Connection Issues**
   - Confirm MCP server is running
   - Check firewall settings
   - Verify environment configuration

### Performance Optimization

1. **GPU Acceleration**
   - Install CUDA-compatible PyTorch
   - Set DEVICE=cuda in environment
   - Monitor GPU memory usage

2. **Memory Management**
   - Adjust batch sizes for your hardware
   - Use data streaming for large datasets
   - Monitor system resources

### Support

For technical questions or issues:
1. Check the documentation in docs/
2. Review example implementations in examples/
3. Examine the Jupyter notebooks for detailed workflows

## Next Steps

1. **Explore the Notebooks**: Start with 
otebooks/Exploratory Data Analysis.ipynb
2. **Customize Models**: Modify anomaly detection parameters in the configuration
3. **Extend Functionality**: Add new KPIs or analysis methods
4. **Scale Deployment**: Consider containerization for production use

---

This guide provides the essential steps to get started with the Telecom AI Agent platform. The system is designed to be modular and extensible for various telecommunications analytics use cases.
