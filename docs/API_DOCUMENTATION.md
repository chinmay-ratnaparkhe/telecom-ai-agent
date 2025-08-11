# API Documentation

## MCP Server Endpoints

The Model Context Protocol server provides the following tools for telecom analytics:

### Base URL
`
http://localhost:8000
`

### Available Tools

#### 1. Detect Anomalies
**Endpoint**: POST /call_tool

**Description**: Performs anomaly detection on specified KPIs using statistical and ML methods.

**Request Body**:
`json
{
  "tool": "detect_anomalies",
  "parameters": {
    "site_id": "SITE_001",
    "kpi_name": "DL_Throughput",
    "date_range": "last_week",
    "method": "statistical",
    "contamination": 0.05,
    "text": "optional natural language description"
  }
}
`

**Response**:
`json
{
  "success": true,
  "anomalies": [
    {
      "timestamp": "2025-08-01T10:30:00",
      "site_id": "SITE_001",
      "kpi_name": "DL_Throughput",
      "value": 45.2,
      "anomaly_score": 0.85,
      "severity": "high",
      "is_anomaly": true
    }
  ],
  "summary": {
    "total_samples": 1000,
    "anomalies_found": 15,
    "anomaly_rate": 0.015
  }
}
`

#### 2. Analyze KPI Trends
**Endpoint**: POST /call_tool

**Request Body**:
`json
{
  "tool": "analyze_kpi_trends",
  "parameters": {
    "kpi_name": "RSRP",
    "site_id": "SITE_002",
    "date_range": "last_month",
    "aggregation": "mean"
  }
}
`

#### 3. Get Site Summary
**Endpoint**: POST /call_tool

**Request Body**:
`json
{
  "tool": "get_site_summary",
  "parameters": {
    "site_id": "SITE_003",
    "date_range": "last_week"
  }
}
`

#### 4. Compare Sites
**Endpoint**: POST /call_tool

**Request Body**:
`json
{
  "tool": "compare_sites",
  "parameters": {
    "site_ids": ["SITE_001", "SITE_002", "SITE_003"],
    "kpi_name": "UL_Throughput",
    "date_range": "last_week"
  }
}
`

#### 5. List Available Data
**Endpoint**: POST /call_tool

**Request Body**:
`json
{
  "tool": "list_available_data",
  "parameters": {}
}
`

#### 6. Visualize Anomalies
**Endpoint**: POST /call_tool

**Request Body**:
`json
{
  "tool": "visualize_anomalies",
  "parameters": {
    "site_id": "SITE_001",
    "kpi_name": "RTT",
    "date_range": "last_week",
    "chart_type": "time_series"
  }
}
`

#### 7. Governance Check
**Endpoint**: POST /call_tool

**Request Body**:
`json
{
  "tool": "governance_check",
  "parameters": {
    "query": "Check compliance for data access patterns",
    "audit_trail": true
  }
}
`

### Error Handling

All endpoints return standardized error responses:

`json
{
  "success": false,
  "error": "Error description",
  "error_code": "ERROR_CODE",
  "details": {
    "additional": "error information"
  }
}
`

### Common Error Codes

- INVALID_TOOL: Unknown tool name
- MISSING_PARAMETERS: Required parameters not provided
- DATA_NOT_FOUND: Requested data not available
- PROCESSING_ERROR: Error during analysis
- TIMEOUT_ERROR: Request timeout exceeded

### Rate Limiting

- Maximum 100 requests per minute per client
- Concurrent request limit: 10
- Timeout: 120 seconds per request

### Authentication

Currently using basic API key authentication. Include in headers:

`
Authorization: Bearer YOUR_API_KEY
`

## Conversational AI API

### Natural Language Processing

**Endpoint**: POST /api/chat

**Request Body**:
`json
{
  "message": "Show me RSRP anomalies for site 001",
  "conversation_id": "optional_conversation_id",
  "context": {
    "user_role": "network_analyst",
    "preferences": {}
  }
}
`

**Response**:
`json
{
  "response": "I found 3 RSRP anomalies for site 001...",
  "tool_calls": [
    {
      "tool": "detect_anomalies",
      "parameters": {...},
      "result": {...}
    }
  ],
  "conversation_id": "conv_12345",
  "follow_up_suggestions": [
    "Would you like to see the trend analysis?",
    "Should I compare with neighboring sites?"
  ]
}
`

### Conversation Management

**Endpoint**: GET /api/conversations/{conversation_id}

Returns conversation history and context.

**Endpoint**: DELETE /api/conversations/{conversation_id}

Clears conversation history.

## Data Models

### KPI Data Schema

`python
{
  "Date": "2025-08-01T10:00:00Z",
  "Site_ID": "SITE_001",
  "Sector_ID": "SECTOR_A",
  "RSRP": -85.5,
  "DL_Throughput": 125.3,
  "UL_Throughput": 45.7,
  "RTT": 15.2,
  "SINR": 12.8,
  "Call_Drop_Rate": 0.02,
  "Handover_Success_Rate": 0.98,
  "Active_Users": 156,
  "CPU_Utilization": 0.75,
  "Packet_Loss": 0.001
}
`

### Anomaly Result Schema

`python
{
  "timestamp": "2025-08-01T10:00:00Z",
  "site_id": "SITE_001",
  "kpi_name": "RSRP",
  "value": -95.2,
  "anomaly_score": 0.92,
  "severity": "high",  # low, medium, high
  "is_anomaly": true,
  "confidence": 0.87,
  "threshold": -90.0,
  "method": "autoencoder"
}
`

## SDK Usage Examples

### Python SDK

`python
from telecom_ai_agent import TelecomAI

# Initialize client
client = TelecomAI(api_key="your_api_key")

# Detect anomalies
anomalies = client.detect_anomalies(
    site_id="SITE_001",
    kpi_name="DL_Throughput",
    date_range="last_week"
)

# Natural language query
response = client.chat("Show me performance issues in sector A")

# Batch processing
results = client.batch_analyze(
    sites=["SITE_001", "SITE_002"],
    kpis=["RSRP", "DL_Throughput"],
    methods=["autoencoder", "isolation_forest"]
)
`

### JavaScript SDK

`javascript
import { TelecomAI } from 'telecom-ai-agent';

const client = new TelecomAI({ apiKey: 'your_api_key' });

// Async anomaly detection
const anomalies = await client.detectAnomalies({
  siteId: 'SITE_001',
  kpiName: 'RSRP',
  dateRange: 'last_week'
});

// Chat interface
const response = await client.chat('Compare sites 001 and 002');
`

## Deployment

### Docker Deployment

`dockerfile
FROM python:3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8000 8501

CMD ["uvicorn", "mcp_server.updated_mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
`

### Kubernetes Deployment

`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: telecom-ai-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: telecom-ai-agent
  template:
    spec:
      containers:
      - name: mcp-server
        image: telecom-ai-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: google-api-key
`

---

This API documentation provides comprehensive information for integrating with the Telecom AI Agent platform. All endpoints support standard HTTP methods and return JSON responses.
