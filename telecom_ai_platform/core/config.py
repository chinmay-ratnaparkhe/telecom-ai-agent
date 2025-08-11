"""
Core Configuration Module

Centralized configuration management for the Telecom AI Platform.
Handles API keys, model parameters, and system settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    # Anomaly detection thresholds
    contamination_rate: float = 0.05
    confidence_threshold: float = 0.8
    
    # Model-specific parameters
    isolation_forest_params: Dict = field(default_factory=lambda: {
        'contamination': 0.05,
        'random_state': 42,
        'n_estimators': 100
    })
    
    autoencoder_params: Dict = field(default_factory=lambda: {
        'encoding_dim': 32,
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001
    })
    
    one_class_svm_params: Dict = field(default_factory=lambda: {
        'nu': 0.05,
        'kernel': 'rbf',
        'gamma': 'scale'
    })


@dataclass
class AIAgentConfig:
    """Configuration for AI Agent"""
    # LLM Settings
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Conversation settings
    memory_window: int = 10
    max_clarification_attempts: int = 3
    
    # API endpoints
    mcp_server_url: str = "http://localhost:8000"
    search_enabled: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing"""
    # File paths
    data_file: str = "AD_data_10KPI.csv"
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    # KPI definitions
    kpi_columns: List[str] = field(default_factory=lambda: [
        'RSRP', 'SINR', 'DL_Throughput', 'UL_Throughput',
        'CPU_Utilization', 'Active_Users', 'RTT', 'Packet_Loss',
        'Call_Drop_Rate', 'Handover_Success_Rate'
    ])
    
    # Data processing parameters
    rolling_window: int = 7
    lag_periods: List[int] = field(default_factory=lambda: [1, 2])
    scaling_method: str = "standard"


@dataclass
class TelecomConfig:
    """Main configuration class"""
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AIAgentConfig = field(default_factory=AIAgentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # API Keys (loaded from environment)
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    langchain_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LANGCHAIN_API_KEY"))
    
    # System settings
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = "INFO"
    
    # Directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    def __post_init__(self):
        """Create necessary directories and validate configuration"""
        # Create directories if they don't exist
        for directory in [self.data_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate API keys in non-debug mode
        if not self.debug_mode and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TelecomConfig':
        """Load configuration from file"""
        # Implementation for loading from JSON/YAML file
        # For now, return default config
        return cls()
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        return asdict(self)


# Global configuration instance
config = TelecomConfig()
