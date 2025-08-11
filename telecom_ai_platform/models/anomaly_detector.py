"""
KPI-Specific Anomaly Detection Models

This module implements various anomaly detection algorithms optimized for different
types of telecom KPIs. Each KPI type gets the most suitable algorithm based on
its characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import pickle
import json
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin, log_function_call
from ..core.config import TelecomConfig


def get_device():
    """
    Get the best available device for PyTorch operations.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU for computations")
    return device


@dataclass
class AnomalyResult:
    """Structured result for anomaly detection"""
    timestamp: str
    site_id: str
    sector_id: Optional[str]
    kpi_name: str
    value: float
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    method: str
    severity: str  # 'low', 'medium', 'high'
    threshold: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class AutoEncoder(nn.Module):
    """
    PyTorch AutoEncoder for anomaly detection with GPU support.
    
    Learns to reconstruct normal patterns and flags high reconstruction
    error samples as anomalies.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        """
        Initialize AutoEncoder.
        
        Args:
            input_dim: Input feature dimension
            encoding_dim: Encoded representation dimension
        """
        super(AutoEncoder, self).__init__()
        
        self.device = get_device()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim)
        )
        
        # Move model to device
        self.to(self.device)
    
    def forward(self, x):
        """Forward pass through encoder-decoder"""
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class KPISpecificDetector(LoggerMixin):
    """
    Individual detector for a specific KPI type.
    
    Automatically selects and trains the most appropriate anomaly detection
    algorithm based on KPI characteristics.
    """
    
    # KPI to algorithm mapping based on domain knowledge
    KPI_ALGORITHM_MAP = {
        'RSRP': 'isolation_forest',      # Signal strength - clear outliers
        'SINR': 'autoencoder',           # Signal quality - temporal patterns
        'DL_Throughput': 'isolation_forest',  # Performance metric - outliers
        'UL_Throughput': 'isolation_forest',  # Performance metric - outliers
        'CPU_Utilization': 'one_class_svm',   # Resource usage - non-linear
        'Active_Users': 'gaussian_mixture',   # User count - multi-modal
        'RTT': 'isolation_forest',       # Latency - clear outliers
        'Packet_Loss': 'one_class_svm',  # Loss rate - threshold behavior
        'Call_Drop_Rate': 'one_class_svm',    # Drop rate - threshold behavior
        'Handover_Success_Rate': 'gaussian_mixture'  # Success rate - bimodal
    }
    
    def __init__(self, kpi_name: str, config: TelecomConfig):
        """
        Initialize KPI-specific detector.
        
        Args:
            kpi_name: Name of the KPI
            config: Configuration object
        """
        self.kpi_name = kpi_name
        self.config = config
        self.device = get_device()
        self.algorithm = self.KPI_ALGORITHM_MAP.get(kpi_name, 'isolation_forest')
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        
        self.logger.info(f"Initialized {self.algorithm} detector for {kpi_name} on {self.device}")
    
    @log_function_call
    def _create_model(self):
        """Create the appropriate model based on algorithm choice"""
        if self.algorithm == 'isolation_forest':
            return IsolationForest(**self.config.model.isolation_forest_params)
        
        elif self.algorithm == 'one_class_svm':
            return OneClassSVM(**self.config.model.one_class_svm_params)
        
        elif self.algorithm == 'gaussian_mixture':
            return GaussianMixture(n_components=2, random_state=42)
        
        elif self.algorithm == 'local_outlier_factor':
            return LocalOutlierFactor(contamination=self.config.model.contamination_rate)
        
        elif self.algorithm == 'autoencoder':
            # AutoEncoder requires special handling
            return None  # Will be created during fit
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    @log_function_call
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'KPISpecificDetector':
        """
        Train the anomaly detection model.
        
        Args:
            X: Training data
            y: Not used (unsupervised learning)
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Training {self.algorithm} on {X.shape[0]} samples")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        if self.algorithm == 'autoencoder':
            self._fit_autoencoder(X_scaled)
        else:
            # Standard sklearn-like models
            self.model = self._create_model()
            self.model.fit(X_scaled)
            self.is_fitted = True  # Set before threshold calculation
        
        # Calculate threshold for anomaly scoring
        self._calculate_threshold(X_scaled)
        
        self.is_fitted = True  # Ensure it's set for autoencoder too
        self.logger.info(f"Training completed for {self.kpi_name}")
        return self
    
    def _fit_autoencoder(self, X: np.ndarray):
        """Train AutoEncoder model with GPU support"""
        input_dim = X.shape[1]
        encoding_dim = min(self.config.model.autoencoder_params['encoding_dim'], input_dim // 2)
        
        self.model = AutoEncoder(input_dim, encoding_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.model.autoencoder_params['learning_rate']
        )
        
        # Convert to tensor and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Training loop
        self.model.train()
        epochs = self.config.model.autoencoder_params['epochs']
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        # Set fitted after training
        self.is_fitted = True
    
    def _calculate_threshold(self, X: np.ndarray):
        """Calculate threshold for anomaly detection"""
        scores = self._get_anomaly_scores(X)
        
        if self.algorithm in ['isolation_forest', 'one_class_svm']:
            # For these algorithms, -1 indicates anomaly, 1 indicates normal
            # We'll use the contamination rate to set threshold
            self.threshold = np.percentile(scores, (1 - self.config.model.contamination_rate) * 100)
        else:
            # For other algorithms, use statistical threshold
            self.threshold = np.mean(scores) + 2 * np.std(scores)
    
    def _get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores for samples"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if self.algorithm == 'autoencoder':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                reconstructed = self.model(X_tensor)
                scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        elif self.algorithm == 'isolation_forest':
            scores = -self.model.score_samples(X)  # Negative for consistency
        
        elif self.algorithm == 'one_class_svm':
            scores = -self.model.score_samples(X)  # Negative for consistency
        
        elif self.algorithm == 'gaussian_mixture':
            scores = -self.model.score_samples(X)  # Negative log-likelihood
        
        elif self.algorithm == 'local_outlier_factor':
            scores = -self.model.negative_outlier_factor_
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return scores
    
    @log_function_call
    def predict(self, X: np.ndarray) -> List[AnomalyResult]:
        """
        Predict anomalies in new data.
        
        Args:
            X: Input data
            
        Returns:
            List of AnomalyResult objects
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        scores = self._get_anomaly_scores(X_scaled)
        
        # Determine anomalies
        is_anomaly = scores > self.threshold
        
        # Create results
        results = []
        for i, (score, anomaly) in enumerate(zip(scores, is_anomaly)):
            # Calculate confidence and severity
            confidence = min(abs(score - self.threshold) / (self.threshold + 1e-8), 1.0)
            
            if anomaly:
                if score > self.threshold * 2:
                    severity = 'high'
                elif score > self.threshold * 1.5:
                    severity = 'medium'
                else:
                    severity = 'low'
            else:
                severity = 'normal'
            
            result = AnomalyResult(
                timestamp=f"sample_{i}",  # Will be updated with actual timestamp
                site_id="unknown",        # Will be updated with actual site
                sector_id=None,
                kpi_name=self.kpi_name,
                value=float(X[i, 0]) if X.shape[1] > 0 else 0.0,
                is_anomaly=bool(anomaly),
                anomaly_score=float(score),
                confidence=float(confidence),
                method=self.algorithm,
                severity=severity,
                threshold=float(self.threshold)
            )
            results.append(result)
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'kpi_name': self.kpi_name,
            'algorithm': self.algorithm,
            'threshold': self.threshold,
            'scaler': self.scaler
        }
        
        if self.algorithm == 'autoencoder':
            # Save PyTorch model state
            model_data['model_state'] = self.model.state_dict()
            model_data['input_dim'] = self.model.encoder[0].in_features
            model_data['encoding_dim'] = self.model.encoder[2].in_features
        else:
            model_data['model'] = self.model
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.kpi_name = model_data['kpi_name']
        self.algorithm = model_data['algorithm']
        self.threshold = model_data['threshold']
        self.scaler = model_data['scaler']
        
        if self.algorithm == 'autoencoder':
            # Reconstruct PyTorch model
            input_dim = model_data['input_dim']
            encoding_dim = model_data['encoding_dim']
            self.model = AutoEncoder(input_dim, encoding_dim)
            self.model.load_state_dict(model_data['model_state'])
            self.model.to(self.device)  # Move to appropriate device
            self.model.eval()
        else:
            self.model = model_data['model']
        
        self.is_fitted = True
        self.logger.info(f"Model loaded from {filepath}")


class KPIAnomalyDetector(LoggerMixin):
    """
    Main anomaly detector that manages multiple KPI-specific detectors.
    
    This class orchestrates training and inference across all KPIs,
    providing a unified interface for anomaly detection.
    """
    
    def __init__(self, config: TelecomConfig):
        """
        Initialize the multi-KPI anomaly detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = get_device()
        self.detectors: Dict[str, KPISpecificDetector] = {}
        self.is_fitted = False
        self.logger.info(f"Initialized KPIAnomalyDetector on {self.device}")
    
    @log_function_call
    def fit(self, data: pd.DataFrame) -> 'KPIAnomalyDetector':
        """
        Train anomaly detectors for all KPIs.
        
        Args:
            data: Processed DataFrame with KPI data
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Starting training for all KPI detectors")
        
        available_kpis = [kpi for kpi in self.config.data.kpi_columns if kpi in data.columns]
        
        for kpi in available_kpis:
            self.logger.info(f"Training detector for {kpi}")
            
            # Extract KPI data (and related features)
            kpi_features = [col for col in data.columns if kpi in col and data[col].dtype in ['int64', 'float64']]
            if not kpi_features:
                kpi_features = [kpi]  # Fallback to just the KPI itself
            
            X = data[kpi_features].values
            
            # Create and train detector
            detector = KPISpecificDetector(kpi, self.config)
            detector.fit(X)
            self.detectors[kpi] = detector
        
        self.is_fitted = True
        self.logger.info(f"Training completed for {len(self.detectors)} KPI detectors")
        return self
    
    @log_function_call
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        kpi_name: Optional[str] = None,
        site_id: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in the provided data.
        
        Args:
            data: Input DataFrame
            kpi_name: Optional specific KPI to analyze
            site_id: Optional specific site to analyze
            date_range: Optional date range tuple (start, end)
            
        Returns:
            List of anomaly results
        """
        if not self.is_fitted:
            raise ValueError("Detectors not fitted yet")
        
        # Filter data if needed
        filtered_data = data.copy()
        
        if site_id and 'Site_ID' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Site_ID'] == site_id]
        
        if date_range and 'Date' in filtered_data.columns:
            start_date, end_date = date_range
            filtered_data = filtered_data[
                (filtered_data['Date'] >= start_date) & (filtered_data['Date'] <= end_date)
            ]
        
        # Determine which KPIs to analyze
        kpis_to_analyze = [kpi_name] if kpi_name else list(self.detectors.keys())
        kpis_to_analyze = [kpi for kpi in kpis_to_analyze if kpi in filtered_data.columns]
        
        all_results = []
        
        for kpi in kpis_to_analyze:
            detector = self.detectors[kpi]
            
            # Prepare data for this KPI
            kpi_features = [col for col in filtered_data.columns if kpi in col and filtered_data[col].dtype in ['int64', 'float64']]
            if not kpi_features:
                kpi_features = [kpi]
            
            X = filtered_data[kpi_features].values
            
            if len(X) == 0:
                continue
            
            # Get predictions
            results = detector.predict(X)
            
            # Update results with actual metadata
            for i, result in enumerate(results):
                row = filtered_data.iloc[i]
                result.timestamp = str(row.get('Date', f'row_{i}'))
                result.site_id = str(row.get('Site_ID', 'unknown'))
                result.sector_id = str(row.get('Sector_ID', None)) if 'Sector_ID' in row else None
                result.value = float(row[kpi])
            
            all_results.extend(results)
        
        # Sort results by anomaly score (highest first)
        all_results.sort(key=lambda x: x.anomaly_score, reverse=True)
        
        self.logger.info(f"Detected {sum(1 for r in all_results if r.is_anomaly)} anomalies out of {len(all_results)} samples")
        
        return all_results
    
    def get_model_summary(self) -> Dict:
        """Get summary of all trained models"""
        if not self.is_fitted:
            return {"error": "Models not fitted yet"}
        
        summary = {}
        for kpi, detector in self.detectors.items():
            summary[kpi] = {
                'algorithm': detector.algorithm,
                'threshold': float(detector.threshold),
                'is_fitted': detector.is_fitted
            }
        
        return summary
    
    def save_all_models(self, models_dir: Optional[str] = None):
        """Save all trained models"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted models")
        
        if models_dir is None:
            models_dir = self.config.models_dir
        
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual detectors
        for kpi, detector in self.detectors.items():
            filepath = models_dir / f"{kpi}_detector.pkl"
            detector.save_model(str(filepath))
        
        # Save master configuration
        config_path = models_dir / "detectors_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.get_model_summary(), f, indent=2)
        
        self.logger.info(f"All models saved to {models_dir}")
    
    def load_all_models(self, models_dir: Optional[str] = None):
        """Load all trained models"""
        if models_dir is None:
            models_dir = self.config.models_dir
        
        models_dir = Path(models_dir)
        
        # Load master configuration
        config_path = models_dir / "detectors_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = {}
        
        # Load individual detectors
        for kpi in self.config.data.kpi_columns:
            filepath = models_dir / f"{kpi}_detector.pkl"
            if filepath.exists():
                detector = KPISpecificDetector(kpi, self.config)
                detector.load_model(str(filepath))
                self.detectors[kpi] = detector
        
        self.is_fitted = len(self.detectors) > 0
        self.logger.info(f"Loaded {len(self.detectors)} model detectors")
