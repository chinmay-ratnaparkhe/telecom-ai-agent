"""
Telecom AI Platform - Production-Ready Conversational AI for Network Analytics

This package provides a comprehensive solution for telecom network monitoring
and analysis through conversational AI, combining KPI-specific anomaly detection
with natural language interfaces.

Author: Chinmay Ratnaparkhe
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Chinmay Ratnaparkhe"
__email__ = "your.email@example.com"

# Package-level imports for easy access
from .core.config import TelecomConfig
from .core.data_processor import TelecomDataProcessor
from .models.anomaly_detector import KPIAnomalyDetector
from .agents.conversational_ai import TelecomConversationalAgent
from .utils.logger import setup_logger

__all__ = [
    "TelecomConfig",
    "TelecomDataProcessor", 
    "KPIAnomalyDetector",
    "TelecomConversationalAgent",
    "setup_logger"
]
