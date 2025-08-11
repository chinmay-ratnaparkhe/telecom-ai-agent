"""
FastAPI Server for Telecom AI Platform

This module provides a RESTful API server for the telecom AI platform,
enabling web-based access to anomaly detection, data analysis, and
conversational AI capabilities.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import uvicorn

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.config import TelecomConfig
from ..models import KPIAnomalyDetector, ModelTrainer, create_training_pipeline
from ..agents import TelecomConversationalAgent, create_telecom_agent, AgentResponse
from ..core.data_processor import TelecomDataProcessor
from ..utils.logger import LoggerMixin, log_function_call


# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    context: Optional[Dict] = Field(None, description="Optional context")
    session_id: Optional[str] = Field(None, description="Session identifier")


class ChatResponse(BaseModel):
    message: str
    data: Optional[Dict] = None
    actions_taken: Optional[List[str]] = None
    confidence: float
    timestamp: str
    session_id: Optional[str] = None


class AnomalyDetectionRequest(BaseModel):
    kpi_name: Optional[str] = Field(None, description="Specific KPI to analyze")
    site_id: Optional[str] = Field(None, description="Specific site to analyze")
    date_range: Optional[List[str]] = Field(None, description="Date range [start, end]")


class AnomalyDetectionResponse(BaseModel):
    total_samples: int
    anomalies_detected: int
    anomaly_rate: float
    results: List[Dict]
    timestamp: str


class DataUploadResponse(BaseModel):
    filename: str
    size: int
    processed_records: int
    kpis_available: List[str]
    date_range: Optional[List[str]] = None
    message: str


class TrainingRequest(BaseModel):
    data_path: str
    validation_split: float = Field(0.2, ge=0.1, le=0.4)
    test_split: float = Field(0.1, ge=0.05, le=0.3)


class TrainingResponse(BaseModel):
    training_id: str
    status: str
    message: str
    training_summary: Optional[Dict] = None


class SystemStatusResponse(BaseModel):
    status: str
    models_loaded: bool
    data_loaded: bool
    agent_ready: bool
    uptime: str
    version: str


class TelecomAPIServer(LoggerMixin):
    """
    Main API server for the telecom AI platform.
    
    Provides RESTful endpoints for data analysis, anomaly detection,
    model training, and conversational AI interaction.
    """
    
    def __init__(self, config: TelecomConfig):
        """
        Initialize the API server.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.app = FastAPI(
            title="Telecom AI Platform API",
            description="RESTful API for telecom network analysis and anomaly detection",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.data_processor = TelecomDataProcessor(config)
        self.anomaly_detector = KPIAnomalyDetector(config)
        self.conversational_agent = create_telecom_agent(config)
        self.model_trainer = create_training_pipeline(config)
        
        # Server state
        self.server_start_time = datetime.now()
        self.current_data = None
        self.training_tasks = {}
        self.chat_sessions = {}
        
        # Setup FastAPI app
        self._setup_middleware()
        self._setup_routes()
        
        # Load models on startup
        self._load_models()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            self.anomaly_detector.load_all_models()
            self.conversational_agent.load_models()
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load models: {e}")
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/", response_model=Dict)
        async def root():
            """Root endpoint with API information"""
            return {
                "name": "Telecom AI Platform API",
                "version": "2.0.0",
                "status": "running",
                "endpoints": {
                    "chat": "/chat",
                    "anomaly_detection": "/anomaly-detection",
                    "data_upload": "/upload-data",
                    "training": "/train",
                    "status": "/status",
                    "docs": "/docs"
                }
            }
        
        @self.app.get("/status", response_model=SystemStatusResponse)
        async def get_status():
            """Get system status and health information"""
            uptime = datetime.now() - self.server_start_time
            
            return SystemStatusResponse(
                status="healthy",
                models_loaded=self.anomaly_detector.is_fitted,
                data_loaded=self.current_data is not None,
                agent_ready=True,
                uptime=str(uptime),
                version="2.0.0"
            )
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
            """Conversational AI endpoint"""
            try:
                # Get or create session
                session_id = request.session_id or f"session_{datetime.now().timestamp()}"
                
                # Process message
                response = self.conversational_agent.chat(request.message, request.context)
                
                # Convert to API response
                api_response = ChatResponse(
                    message=response.message,
                    data=response.data,
                    actions_taken=response.actions_taken,
                    confidence=response.confidence,
                    timestamp=response.timestamp,
                    session_id=session_id
                )
                
                return api_response
                
            except Exception as e:
                self.logger.error(f"Chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")
        
        @self.app.post("/anomaly-detection", response_model=AnomalyDetectionResponse)
        async def detect_anomalies(request: AnomalyDetectionRequest):
            """Anomaly detection endpoint"""
            try:
                if not self.anomaly_detector.is_fitted:
                    raise HTTPException(status_code=400, detail="Anomaly detection models not loaded")
                
                if self.current_data is None:
                    raise HTTPException(status_code=400, detail="No data loaded for analysis")
                
                # Convert date range if provided
                date_range = None
                if request.date_range and len(request.date_range) == 2:
                    date_range = (request.date_range[0], request.date_range[1])
                
                # Detect anomalies
                results = self.anomaly_detector.detect_anomalies(
                    self.current_data,
                    kpi_name=request.kpi_name,
                    site_id=request.site_id,
                    date_range=date_range
                )
                
                # Convert results to dictionaries
                results_dicts = [result.to_dict() for result in results]
                anomalies = [r for r in results if r.is_anomaly]
                
                response = AnomalyDetectionResponse(
                    total_samples=len(results),
                    anomalies_detected=len(anomalies),
                    anomaly_rate=len(anomalies) / len(results) if results else 0,
                    results=results_dicts,
                    timestamp=datetime.now().isoformat()
                )
                
                return response
                
            except Exception as e:
                self.logger.error(f"Anomaly detection error: {e}")
                raise HTTPException(status_code=500, detail=f"Anomaly detection error: {str(e)}")
        
        @self.app.post("/upload-data", response_model=DataUploadResponse)
        async def upload_data(file: UploadFile = File(...)):
            """Data upload and processing endpoint"""
            try:
                # Save uploaded file
                upload_dir = Path(self.config.data_dir)
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = upload_dir / file.filename
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Process the data
                raw_data = self.data_processor.load_data(str(file_path))
                self.current_data = self.data_processor.process_pipeline(raw_data)
                
                # Get data summary
                kpis_available = [col for col in self.config.data.kpi_columns if col in self.current_data.columns]
                date_range = None
                if 'Date' in self.current_data.columns:
                    date_range = [
                        str(self.current_data['Date'].min()),
                        str(self.current_data['Date'].max())
                    ]
                
                response = DataUploadResponse(
                    filename=file.filename,
                    size=len(content),
                    processed_records=len(self.current_data),
                    kpis_available=kpis_available,
                    date_range=date_range,
                    message=f"Successfully processed {len(self.current_data)} records"
                )
                
                self.logger.info(f"Data uploaded and processed: {file.filename}")
                return response
                
            except Exception as e:
                self.logger.error(f"Data upload error: {e}")
                raise HTTPException(status_code=500, detail=f"Data upload error: {str(e)}")
        
        @self.app.post("/train", response_model=TrainingResponse)
        async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
            """Model training endpoint"""
            try:
                # Generate training ID
                training_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Add training task to background
                background_tasks.add_task(
                    self._run_training_pipeline,
                    training_id,
                    request.data_path,
                    request.validation_split,
                    request.test_split
                )
                
                self.training_tasks[training_id] = {
                    'status': 'started',
                    'start_time': datetime.now().isoformat(),
                    'data_path': request.data_path
                }
                
                response = TrainingResponse(
                    training_id=training_id,
                    status="started",
                    message=f"Training pipeline started with ID: {training_id}"
                )
                
                return response
                
            except Exception as e:
                self.logger.error(f"Training initiation error: {e}")
                raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")
        
        @self.app.get("/training/{training_id}")
        async def get_training_status(training_id: str):
            """Get training status"""
            if training_id not in self.training_tasks:
                raise HTTPException(status_code=404, detail="Training ID not found")
            
            return self.training_tasks[training_id]
        
        @self.app.get("/models/summary")
        async def get_models_summary():
            """Get summary of available models"""
            try:
                summary = self.anomaly_detector.get_model_summary()
                training_status = self.model_trainer.get_training_status()
                
                return {
                    "models": summary,
                    "training_status": training_status,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.error(f"Models summary error: {e}")
                raise HTTPException(status_code=500, detail=f"Models summary error: {str(e)}")
        
        @self.app.get("/data/summary")
        async def get_data_summary():
            """Get summary of currently loaded data"""
            if self.current_data is None:
                return {"message": "No data currently loaded"}
            
            try:
                summary = {
                    "total_records": len(self.current_data),
                    "columns": list(self.current_data.columns),
                    "kpis_available": [col for col in self.config.data.kpi_columns if col in self.current_data.columns],
                    "date_range": None,
                    "sites": self.current_data['Site_ID'].nunique() if 'Site_ID' in self.current_data.columns else 0,
                    "data_quality": {
                        "missing_values": self.current_data.isnull().sum().to_dict(),
                        "duplicates": self.current_data.duplicated().sum()
                    }
                }
                
                if 'Date' in self.current_data.columns:
                    summary["date_range"] = [
                        str(self.current_data['Date'].min()),
                        str(self.current_data['Date'].max())
                    ]
                
                return summary
                
            except Exception as e:
                self.logger.error(f"Data summary error: {e}")
                raise HTTPException(status_code=500, detail=f"Data summary error: {str(e)}")
        
        @self.app.delete("/reset")
        async def reset_system():
            """Reset system state"""
            try:
                self.current_data = None
                self.conversational_agent.reset_conversation()
                self.training_tasks.clear()
                self.chat_sessions.clear()
                
                return {"message": "System reset successfully"}
                
            except Exception as e:
                self.logger.error(f"System reset error: {e}")
                raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")
    
    async def _run_training_pipeline(
        self,
        training_id: str,
        data_path: str,
        validation_split: float,
        test_split: float
    ):
        """Run training pipeline in background"""
        try:
            self.training_tasks[training_id]['status'] = 'running'
            
            # Run training
            summary = self.model_trainer.run_full_training_pipeline(
                data_path, validation_split, test_split
            )
            
            # Update task status
            self.training_tasks[training_id].update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'summary': summary
            })
            
            # Reload models
            self._load_models()
            
        except Exception as e:
            self.training_tasks[training_id].update({
                'status': 'failed',
                'end_time': datetime.now().isoformat(),
                'error': str(e)
            })
            self.logger.error(f"Training pipeline failed: {e}")
    
    @log_function_call
    def run_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False
    ):
        """
        Run the FastAPI server.
        
        Args:
            host: Host address
            port: Port number
            reload: Enable auto-reload for development
        """
        self.logger.info(f"Starting Telecom AI Platform API server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )


def create_api_server(config: TelecomConfig) -> TelecomAPIServer:
    """
    Factory function to create a configured API server.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured TelecomAPIServer instance
    """
    return TelecomAPIServer(config)


# FastAPI app instance for ASGI deployment
def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create FastAPI app instance for ASGI deployment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        FastAPI application instance
    """
    if config_path:
        # Load custom configuration
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        # Implement config loading logic as needed
        config = TelecomConfig()
    else:
        config = TelecomConfig()
    
    server = create_api_server(config)
    return server.app
