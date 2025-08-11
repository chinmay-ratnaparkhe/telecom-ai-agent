"""
Models Module

This module contains all machine learning models and training components
for the telecom AI platform.

Available Components:
- KPIAnomalyDetector: Main anomaly detection system
- KPISpecificDetector: Individual KPI anomaly detectors
- AutoEncoder: Neural network for pattern learning
- ModelTrainer: Training pipeline orchestrator
- ModelPerformanceTracker: Performance monitoring
"""

from .anomaly_detector import (
    AnomalyResult,
    AutoEncoder,
    KPISpecificDetector,
    KPIAnomalyDetector
)

from .trainer import (
    ModelPerformanceTracker,
    ModelTrainer,
    create_training_pipeline
)

__all__ = [
    # Anomaly Detection
    'AnomalyResult',
    'AutoEncoder',
    'KPISpecificDetector',
    'KPIAnomalyDetector',
    
    # Training and Performance
    'ModelPerformanceTracker',
    'ModelTrainer',
    'create_training_pipeline'
]
