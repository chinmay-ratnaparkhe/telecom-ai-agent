"""
Model Training and Management Module

This module handles training, validation, and management of machine learning models
for the telecom AI platform. It provides automated training pipelines and
model performance tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from .anomaly_detector import KPIAnomalyDetector, AnomalyResult
from ..core.data_processor import TelecomDataProcessor
from ..core.config import TelecomConfig
from ..utils.logger import LoggerMixin, log_function_call


class ModelPerformanceTracker(LoggerMixin):
    """
    Tracks and evaluates model performance over time.
    
    Provides metrics for anomaly detection performance, training history,
    and model comparison capabilities.
    """
    
    def __init__(self, config: TelecomConfig):
        """
        Initialize performance tracker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.performance_history = []
        self.model_metadata = {}
    
    @log_function_call
    def evaluate_anomaly_detector(
        self,
        detector: KPIAnomalyDetector,
        test_data: pd.DataFrame,
        true_anomalies: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Evaluate anomaly detection performance.
        
        Args:
            detector: Trained anomaly detector
            test_data: Test dataset
            true_anomalies: Ground truth anomalies (if available)
            
        Returns:
            Performance metrics dictionary
        """
        self.logger.info("Evaluating anomaly detector performance")
        
        # Get predictions
        results = detector.detect_anomalies(test_data)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(test_data),
            'anomalies_detected': sum(1 for r in results if r.is_anomaly),
            'anomaly_rate': sum(1 for r in results if r.is_anomaly) / len(results) if results else 0,
            'average_anomaly_score': np.mean([r.anomaly_score for r in results]),
            'confidence_distribution': self._analyze_confidence_distribution(results),
            'severity_distribution': self._analyze_severity_distribution(results),
            'kpi_performance': self._analyze_kpi_performance(results)
        }
        
        # If ground truth is available, calculate additional metrics
        if true_anomalies is not None:
            supervised_metrics = self._calculate_supervised_metrics(results, true_anomalies)
            metrics.update(supervised_metrics)
        
        self.performance_history.append(metrics)
        self.logger.info(f"Detection completed: {metrics['anomalies_detected']} anomalies found")
        
        return metrics
    
    def _analyze_confidence_distribution(self, results: List[AnomalyResult]) -> Dict:
        """Analyze confidence score distribution"""
        confidences = [r.confidence for r in results]
        return {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'q25': np.percentile(confidences, 25),
            'q50': np.percentile(confidences, 50),
            'q75': np.percentile(confidences, 75)
        }
    
    def _analyze_severity_distribution(self, results: List[AnomalyResult]) -> Dict:
        """Analyze severity distribution"""
        severities = [r.severity for r in results]
        distribution = {}
        for severity in ['normal', 'low', 'medium', 'high']:
            distribution[severity] = sum(1 for s in severities if s == severity)
        return distribution
    
    def _analyze_kpi_performance(self, results: List[AnomalyResult]) -> Dict:
        """Analyze performance per KPI"""
        kpi_stats = {}
        for result in results:
            kpi = result.kpi_name
            if kpi not in kpi_stats:
                kpi_stats[kpi] = {
                    'total_samples': 0,
                    'anomalies': 0,
                    'avg_score': 0,
                    'scores': []
                }
            
            kpi_stats[kpi]['total_samples'] += 1
            kpi_stats[kpi]['scores'].append(result.anomaly_score)
            if result.is_anomaly:
                kpi_stats[kpi]['anomalies'] += 1
        
        # Calculate final statistics
        for kpi in kpi_stats:
            scores = kpi_stats[kpi]['scores']
            kpi_stats[kpi]['avg_score'] = np.mean(scores)
            kpi_stats[kpi]['anomaly_rate'] = (
                kpi_stats[kpi]['anomalies'] / kpi_stats[kpi]['total_samples']
            )
            del kpi_stats[kpi]['scores']  # Remove raw scores to save space
        
        return kpi_stats
    
    def _calculate_supervised_metrics(
        self,
        results: List[AnomalyResult],
        true_anomalies: pd.DataFrame
    ) -> Dict:
        """Calculate supervised learning metrics if ground truth is available"""
        # This would require matching results with ground truth
        # Implementation depends on the format of true_anomalies
        return {
            'supervised_evaluation': 'Not implemented - requires ground truth format specification'
        }
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.performance_history:
            return "No performance data available"
        
        latest = self.performance_history[-1]
        
        report = f"""
ANOMALY DETECTION PERFORMANCE REPORT
=====================================
Generated: {latest['timestamp']}

OVERVIEW:
- Total Samples Analyzed: {latest['total_samples']:,}
- Anomalies Detected: {latest['anomalies_detected']:,}
- Anomaly Rate: {latest['anomaly_rate']:.2%}
- Average Anomaly Score: {latest['average_anomaly_score']:.4f}

CONFIDENCE ANALYSIS:
- Mean Confidence: {latest['confidence_distribution']['mean']:.3f}
- Confidence Range: {latest['confidence_distribution']['min']:.3f} - {latest['confidence_distribution']['max']:.3f}
- Median Confidence: {latest['confidence_distribution']['q50']:.3f}

SEVERITY BREAKDOWN:
- Normal: {latest['severity_distribution']['normal']:,}
- Low Severity: {latest['severity_distribution']['low']:,}
- Medium Severity: {latest['severity_distribution']['medium']:,}
- High Severity: {latest['severity_distribution']['high']:,}

KPI-SPECIFIC PERFORMANCE:
"""
        
        for kpi, stats in latest['kpi_performance'].items():
            report += f"""
{kpi}:
  - Samples: {stats['total_samples']:,}
  - Anomalies: {stats['anomalies']:,} ({stats['anomaly_rate']:.2%})
  - Avg Score: {stats['avg_score']:.4f}
"""
        
        return report
    
    def save_performance_history(self, filepath: Optional[str] = None):
        """Save performance history to file"""
        if filepath is None:
            filepath = self.config.models_dir / "performance_history.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        self.logger.info(f"Performance history saved to {filepath}")


class ModelTrainer(LoggerMixin):
    """
    Main training orchestrator for the telecom AI platform.
    
    Handles data preparation, model training, validation, and deployment
    workflows for anomaly detection models.
    """
    
    def __init__(self, config: TelecomConfig):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_processor = TelecomDataProcessor(config)
        self.performance_tracker = ModelPerformanceTracker(config)
        self.trained_models = {}
    
    @log_function_call
    def prepare_training_data(
        self,
        data_path: str,
        validation_split: float = 0.2,
        test_split: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training, validation, and testing.
        
        Args:
            data_path: Path to raw data file
            validation_split: Fraction for validation set
            test_split: Fraction for test set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        self.logger.info(f"Preparing training data from {data_path}")
        
        # Load and process data
        raw_data = self.data_processor.load_data(data_path)
        processed_data = self.data_processor.process_pipeline(raw_data)
        
        # Time-based split (important for time series data)
        if 'Date' in processed_data.columns:
            processed_data = processed_data.sort_values('Date')
            n_samples = len(processed_data)
            
            train_end = int(n_samples * (1 - validation_split - test_split))
            val_end = int(n_samples * (1 - test_split))
            
            train_data = processed_data.iloc[:train_end].copy()
            val_data = processed_data.iloc[train_end:val_end].copy()
            test_data = processed_data.iloc[val_end:].copy()
        else:
            # Random split if no date column
            train_val, test_data = train_test_split(
                processed_data, test_size=test_split, random_state=42
            )
            train_data, val_data = train_test_split(
                train_val, test_size=validation_split/(1-test_split), random_state=42
            )
        
        self.logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    @log_function_call
    def train_anomaly_detector(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        save_model: bool = True
    ) -> KPIAnomalyDetector:
        """
        Train the main anomaly detection model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset (optional)
            save_model: Whether to save the trained model
            
        Returns:
            Trained anomaly detector
        """
        self.logger.info("Starting anomaly detector training")
        
        # Initialize and train detector
        detector = KPIAnomalyDetector(self.config)
        detector.fit(train_data)
        
        # Validate if validation data is provided
        if val_data is not None:
            self.logger.info("Evaluating on validation data")
            val_metrics = self.performance_tracker.evaluate_anomaly_detector(
                detector, val_data
            )
            self.logger.info(f"Validation anomaly rate: {val_metrics['anomaly_rate']:.2%}")
        
        # Save model if requested
        if save_model:
            detector.save_all_models()
            self.logger.info("Model saved successfully")
        
        self.trained_models['anomaly_detector'] = detector
        return detector
    
    @log_function_call
    def run_full_training_pipeline(
        self,
        data_path: str,
        validation_split: float = 0.2,
        test_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to training data
            validation_split: Validation set fraction
            test_split: Test set fraction
            
        Returns:
            Training results and metrics
        """
        self.logger.info("Starting full training pipeline")
        
        # Prepare data
        train_data, val_data, test_data = self.prepare_training_data(
            data_path, validation_split, test_split
        )
        
        # Train models
        detector = self.train_anomaly_detector(train_data, val_data)
        
        # Final evaluation on test set
        test_metrics = self.performance_tracker.evaluate_anomaly_detector(
            detector, test_data
        )
        
        # Generate training summary
        training_summary = {
            'training_timestamp': datetime.now().isoformat(),
            'data_path': data_path,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'model_summary': detector.get_model_summary(),
            'test_performance': test_metrics,
            'config': {
                'validation_split': validation_split,
                'test_split': test_split
            }
        }
        
        # Save training summary
        summary_path = self.config.models_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        # Save performance history
        self.performance_tracker.save_performance_history()
        
        self.logger.info("Training pipeline completed successfully")
        return training_summary
    
    @log_function_call
    def load_trained_models(self) -> Dict[str, Any]:
        """
        Load previously trained models.
        
        Returns:
            Dictionary of loaded models
        """
        self.logger.info("Loading trained models")
        
        # Load anomaly detector
        detector = KPIAnomalyDetector(self.config)
        try:
            detector.load_all_models()
            self.trained_models['anomaly_detector'] = detector
            self.logger.info("Anomaly detector loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load anomaly detector: {e}")
        
        return self.trained_models
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and available models"""
        models_dir = Path(self.config.models_dir)
        
        status = {
            'models_directory': str(models_dir),
            'models_available': models_dir.exists(),
            'trained_models': list(self.trained_models.keys()),
            'model_files': []
        }
        
        if models_dir.exists():
            status['model_files'] = [f.name for f in models_dir.glob('*.pkl')]
            
            # Check for training summary
            summary_path = models_dir / "training_summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    status['last_training'] = json.load(f)
        
        return status
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        status = self.get_training_status()
        
        report = f"""
MODEL TRAINING STATUS REPORT
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODELS DIRECTORY: {status['models_directory']}
Models Available: {'Yes' if status['models_available'] else 'No'}

LOADED MODELS:
"""
        
        for model_name in status['trained_models']:
            report += f"- {model_name}\n"
        
        if not status['trained_models']:
            report += "- No models currently loaded\n"
        
        report += f"\nMODEL FILES ON DISK:\n"
        for file_name in status['model_files']:
            report += f"- {file_name}\n"
        
        if not status['model_files']:
            report += "- No model files found\n"
        
        if 'last_training' in status:
            last_training = status['last_training']
            report += f"""
LAST TRAINING SESSION:
- Date: {last_training['training_timestamp']}
- Training Samples: {last_training['train_samples']:,}
- Validation Samples: {last_training['val_samples']:,}
- Test Samples: {last_training['test_samples']:,}
- Models Trained: {len(last_training['model_summary'])}
- Test Anomaly Rate: {last_training['test_performance']['anomaly_rate']:.2%}
"""
        else:
            report += "\nLAST TRAINING SESSION: No training history found\n"
        
        report += "\n" + self.performance_tracker.generate_performance_report()
        
        return report


def create_training_pipeline(config: TelecomConfig) -> ModelTrainer:
    """
    Factory function to create a configured training pipeline.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured ModelTrainer instance
    """
    return ModelTrainer(config)
