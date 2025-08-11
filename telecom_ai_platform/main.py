"""
Main Application Entry Point

This module provides the main entry point for the telecom AI platform,
offering easy-to-use interfaces for all platform capabilities.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core.config import TelecomConfig
from .models import create_training_pipeline
from .agents import create_telecom_agent
from .server import create_api_server
from .utils.logger import setup_logger


class TelecomAIPlatform:
    """
    Main platform class that orchestrates all components.
    
    This class provides a unified interface to access all platform
    capabilities including data processing, anomaly detection,
    conversational AI, and the web API.
    """
    
    def __init__(self, config: Optional[TelecomConfig] = None):
        """
        Initialize the telecom AI platform.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or TelecomConfig()
        self.logger = setup_logger("TelecomAIPlatform")
        
        # Initialize components
        self.trainer = create_training_pipeline(self.config)
        self.agent = create_telecom_agent(self.config)
        self.api_server = create_api_server(self.config)
        
        self.logger.info("Telecom AI Platform initialized successfully")
    
    def train_models(self, data_path: str, **kwargs):
        """
        Train anomaly detection models.
        
        Args:
            data_path: Path to training data
            **kwargs: Additional training parameters
        """
        self.logger.info(f"Starting model training with data from {data_path}")
        return self.trainer.run_full_training_pipeline(data_path, **kwargs)
    
    def train_with_dataframe(self, df, **kwargs):
        """
        Train anomaly detection models with DataFrame.
        
        Args:
            df: Training DataFrame
            **kwargs: Additional training parameters
        """
        self.logger.info(f"Starting model training with DataFrame of shape {df.shape}")
        
        # Process data first if needed
        if not hasattr(self.trainer.data_processor, 'processed_data') or self.trainer.data_processor.processed_data is None:
            df = self.trainer.data_processor.process_dataframe(df)
        
        # Train anomaly detector directly
        detector = self.trainer.train_anomaly_detector(df)
        
        # Update the agent's detector with the trained one
        self.agent.anomaly_detector = detector
        
        return detector
    
    def detect_anomalies(self, df):
        """
        Detect anomalies in the provided DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of anomaly results
        """
        # Process data if needed
        if not hasattr(self.trainer.data_processor, 'processed_data') or self.trainer.data_processor.processed_data is None:
            df = self.trainer.data_processor.process_dataframe(df)
        
        # Use the agent's anomaly detector
        return self.agent.anomaly_detector.detect_anomalies(df)
    
    def chat(self, message: str, **kwargs):
        """
        Interact with the conversational agent.
        
        Args:
            message: User message
            **kwargs: Additional chat parameters
        """
        return self.agent.chat(message, **kwargs)
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """
        Start the web API server.
        
        Args:
            host: Server host
            port: Server port
            **kwargs: Additional server parameters
        """
        self.api_server.run_server(host=host, port=port, **kwargs)
    
    def load_models(self):
        """Load pre-trained models"""
        self.agent.load_models()
        self.logger.info("Models loaded successfully")
    
    def save_autoencoder_model(self, model_name: str = "autoencoder[1]"):
        """
        Save the SINR AutoEncoder model with a specific name.
        
        Args:
            model_name: Name for the saved model
        """
        # Check if we have a SINR detector with AutoEncoder
        if hasattr(self.agent, 'anomaly_detector') and self.agent.anomaly_detector.is_fitted:
            if 'SINR' in self.agent.anomaly_detector.detectors:
                sinr_detector = self.agent.anomaly_detector.detectors['SINR']
                if hasattr(sinr_detector, 'model') and hasattr(sinr_detector.model, 'encoder'):
                    # This is an AutoEncoder
                    model_path = self.config.models_dir / f"{model_name}.pkl"
                    sinr_detector.save_model(str(model_path))
                    self.logger.info(f"AutoEncoder model saved as {model_name}")
                    return True
        
        self.logger.warning("No AutoEncoder model found for SINR")
        return False
        
    def get_autoencoder_info(self):
        """Get information about the AutoEncoder model."""
        if hasattr(self.agent, 'anomaly_detector') and self.agent.anomaly_detector.is_fitted:
            if 'SINR' in self.agent.anomaly_detector.detectors:
                sinr_detector = self.agent.anomaly_detector.detectors['SINR']
                if hasattr(sinr_detector, 'algorithm') and sinr_detector.algorithm == 'autoencoder':
                    return {
                        'kpi': 'SINR',
                        'algorithm': 'autoencoder',
                        'device': str(sinr_detector.device),
                        'is_fitted': sinr_detector.is_fitted
                    }
        return None
    
    def get_status(self):
        """Get platform status"""
        return {
            'config': {
                'data_dir': str(self.config.data_dir),
                'models_dir': str(self.config.models_dir),
                'logs_dir': str(self.config.logs_dir)
            },
            'models_loaded': self.agent.anomaly_detector.is_fitted,
            'agent_ready': True,
            'server_ready': True
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Telecom AI Platform")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train anomaly detection models')
    train_parser.add_argument('data_path', help='Path to training data')
    train_parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    train_parser.add_argument('--test-split', type=float, default=0.1, help='Test split ratio')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat session')
    chat_parser.add_argument('--load-models', action='store_true', help='Load pre-trained models')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show platform status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize platform
    platform = TelecomAIPlatform()
    
    if args.command == 'train':
        print(f"Training models with data from: {args.data_path}")
        result = platform.train_models(
            args.data_path,
            validation_split=args.validation_split,
            test_split=args.test_split
        )
        print("Training completed successfully!")
        print(f"Models saved to: {platform.config.models_dir}")
    
    elif args.command == 'chat':
        if args.load_models:
            print("Loading pre-trained models...")
            platform.load_models()
        
        print("Starting interactive chat session...")
        print("Type 'quit' to exit")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                response = platform.chat(user_input)
                print(f"\nAgent: {response.message}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Chat session ended.")
    
    elif args.command == 'server':
        print(f"Starting API server on {args.host}:{args.port}")
        platform.start_server(
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    
    elif args.command == 'status':
        status = platform.get_status()
        print("=== Telecom AI Platform Status ===")
        for key, value in status.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
