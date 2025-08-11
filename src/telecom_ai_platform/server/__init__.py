"""
Server Module

This module contains the FastAPI server components for the telecom AI platform,
providing RESTful API access to all platform capabilities.

Available Components:
- TelecomAPIServer: Main API server
- API request/response models
- Server configuration and deployment utilities
"""

from .api import (
    TelecomAPIServer,
    create_api_server,
    create_app,
    ChatRequest,
    ChatResponse,
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    DataUploadResponse,
    TrainingRequest,
    TrainingResponse,
    SystemStatusResponse
)

__all__ = [
    'TelecomAPIServer',
    'create_api_server',
    'create_app',
    'ChatRequest',
    'ChatResponse',
    'AnomalyDetectionRequest',
    'AnomalyDetectionResponse',
    'DataUploadResponse',
    'TrainingRequest',
    'TrainingResponse',
    'SystemStatusResponse'
]
