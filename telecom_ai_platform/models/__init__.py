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

# General AutoEncoder (cross-KPI)
try:
    from .enhanced_autoencoder import (
        GeneralAutoEncoderDetector,
        engineer_features,
        build_feature_subset
    )
except Exception:  # pragma: no cover - import may fail lazily; UI has diagnostics
    GeneralAutoEncoderDetector = None  # type: ignore
    engineer_features = None  # type: ignore
    build_feature_subset = None  # type: ignore

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
    , 'GeneralAutoEncoderDetector', 'engineer_features', 'build_feature_subset'
]
