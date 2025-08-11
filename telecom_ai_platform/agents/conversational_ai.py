"""
Conversational AI Agent for Telecom Network Management

This module implements an intelligent conversational agent that can analyze
telecom network data, detect anomalies, and provide insights through natural
language interactions using Google's Gemini LLM.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime, timedelta
import re
from dataclasses import dataclass, asdict

import google.generativeai as genai
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..models.anomaly_detector import KPIAnomalyDetector, AnomalyResult
from ..core.data_processor import TelecomDataProcessor
from ..core.config import TelecomConfig
from ..utils.logger import LoggerMixin, log_function_call


@dataclass
class AgentResponse:
    """Structured response from the conversational agent"""
    message: str
    data: Optional[Dict] = None
    visualizations: Optional[List[str]] = None
    actions_taken: Optional[List[str]] = None
    confidence: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class NetworkAnalysisTools:
    """
    Collection of tools available to the conversational agent for network analysis.
    
    These tools provide the agent with capabilities to analyze network data,
    detect anomalies, and generate insights.
    """
    
    def __init__(self, config: TelecomConfig, data_processor: TelecomDataProcessor, anomaly_detector: KPIAnomalyDetector):
        """
        Initialize network analysis tools.
        
        Args:
            config: Configuration object
            data_processor: Data processing component
            anomaly_detector: Anomaly detection component
        """
        self.config = config
        self.data_processor = data_processor
        self.anomaly_detector = anomaly_detector
        self.current_data = None
    
    def load_network_data(self, data_path: str) -> str:
        """Load and process network data"""
        try:
            raw_data = self.data_processor.load_data(data_path)
            self.current_data = self.data_processor.process_pipeline(raw_data)
            
            return f"""Successfully loaded network data with {len(self.current_data)} records.
            
Data Summary:
- Date Range: {self.current_data['Date'].min()} to {self.current_data['Date'].max()}
- Sites: {self.current_data['Site_ID'].nunique()} unique sites
- KPIs Available: {', '.join([col for col in self.config.data.kpi_columns if col in self.current_data.columns])}
- Total Records: {len(self.current_data):,}
"""
        except Exception as e:
            return f"Error loading data: {str(e)}"
    
    def analyze_kpi_trends(self, kpi_name: str, site_id: Optional[str] = None, days: int = 7) -> str:
        """Analyze trends for a specific KPI"""
        if self.current_data is None:
            return "No data loaded. Please load network data first."
        
        try:
            # Filter data
            data = self.current_data.copy()
            
            if site_id:
                data = data[data['Site_ID'] == site_id]
                if data.empty:
                    return f"No data found for site {site_id}"
            
            # Get recent data
            if 'Date' in data.columns:
                end_date = data['Date'].max()
                start_date = end_date - timedelta(days=days)
                data = data[data['Date'] >= start_date]
            
            if kpi_name not in data.columns:
                return f"KPI '{kpi_name}' not found in data. Available KPIs: {', '.join([col for col in self.config.data.kpi_columns if col in data.columns])}"
            
            # Calculate statistics
            kpi_values = data[kpi_name].dropna()
            if kpi_values.empty:
                return f"No valid data found for KPI '{kpi_name}'"
            
            stats = {
                'mean': kpi_values.mean(),
                'std': kpi_values.std(),
                'min': kpi_values.min(),
                'max': kpi_values.max(),
                'median': kpi_values.median(),
                'trend': 'improving' if kpi_values.iloc[-10:].mean() > kpi_values.iloc[:10].mean() else 'declining'
            }
            
            return f"""KPI Analysis for {kpi_name} (Last {days} days):

Statistics:
- Average: {stats['mean']:.2f}
- Standard Deviation: {stats['std']:.2f}
- Range: {stats['min']:.2f} - {stats['max']:.2f}
- Median: {stats['median']:.2f}
- Recent Trend: {stats['trend']}

Data Quality:
- Valid Samples: {len(kpi_values):,}
- Coverage: {len(kpi_values)/len(data)*100:.1f}%
"""
        except Exception as e:
            return f"Error analyzing KPI trends: {str(e)}"
    
    def detect_network_anomalies(self, kpi_name: Optional[str] = None, site_id: Optional[str] = None) -> str:
        """Detect anomalies in network data"""
        if self.current_data is None:
            return "No data loaded. Please load network data first."
        
        try:
            # Detect anomalies
            results = self.anomaly_detector.detect_anomalies(
                self.current_data,
                kpi_name=kpi_name,
                site_id=site_id
            )
            
            # Filter for actual anomalies
            anomalies = [r for r in results if r.is_anomaly]
            
            if not anomalies:
                return "No anomalies detected in the current data."
            
            # Categorize by severity
            high_severity = [a for a in anomalies if a.severity == 'high']
            medium_severity = [a for a in anomalies if a.severity == 'medium']
            low_severity = [a for a in anomalies if a.severity == 'low']
            
            report = f"""Anomaly Detection Results:

SUMMARY:
- Total Anomalies: {len(anomalies)}
- High Severity: {len(high_severity)}
- Medium Severity: {len(medium_severity)}
- Low Severity: {len(low_severity)}

TOP ANOMALIES (by severity):
"""
            
            # Show top anomalies
            top_anomalies = sorted(anomalies, key=lambda x: x.anomaly_score, reverse=True)[:5]
            
            for i, anomaly in enumerate(top_anomalies, 1):
                report += f"""
{i}. {anomaly.kpi_name} at Site {anomaly.site_id}
   - Value: {anomaly.value:.2f}
   - Severity: {anomaly.severity.upper()}
   - Confidence: {anomaly.confidence:.1%}
   - Method: {anomaly.method}
"""
            
            return report
            
        except Exception as e:
            return f"Error detecting anomalies: {str(e)}"
    
    def get_site_overview(self, site_id: str) -> str:
        """Get comprehensive overview of a specific site"""
        if self.current_data is None:
            return "No data loaded. Please load network data first."
        
        try:
            site_data = self.current_data[self.current_data['Site_ID'] == site_id]
            
            if site_data.empty:
                return f"No data found for site {site_id}"
            
            # Calculate KPI summaries
            kpi_summary = {}
            for kpi in self.config.data.kpi_columns:
                if kpi in site_data.columns:
                    values = site_data[kpi].dropna()
                    if not values.empty:
                        kpi_summary[kpi] = {
                            'avg': values.mean(),
                            'latest': values.iloc[-1] if not values.empty else None,
                            'samples': len(values)
                        }
            
            # Detect site-specific anomalies
            site_anomalies = self.anomaly_detector.detect_anomalies(site_data)
            anomaly_count = sum(1 for a in site_anomalies if a.is_anomaly)
            
            report = f"""Site Overview: {site_id}

DATA COVERAGE:
- Total Records: {len(site_data):,}
- Date Range: {site_data['Date'].min()} to {site_data['Date'].max()}
- Active Anomalies: {anomaly_count}

KPI PERFORMANCE:
"""
            
            for kpi, stats in kpi_summary.items():
                report += f"- {kpi}: Avg={stats['avg']:.2f}, Latest={stats['latest']:.2f} ({stats['samples']} samples)\n"
            
            return report
            
        except Exception as e:
            return f"Error getting site overview: {str(e)}"
    
    def compare_sites(self, site_ids: List[str], kpi_name: str) -> str:
        """Compare KPI performance across multiple sites"""
        if self.current_data is None:
            return "No data loaded. Please load network data first."
        
        try:
            if kpi_name not in self.current_data.columns:
                return f"KPI '{kpi_name}' not found in data."
            
            comparison = {}
            for site_id in site_ids:
                site_data = self.current_data[self.current_data['Site_ID'] == site_id]
                if not site_data.empty:
                    kpi_values = site_data[kpi_name].dropna()
                    if not kpi_values.empty:
                        comparison[site_id] = {
                            'mean': kpi_values.mean(),
                            'std': kpi_values.std(),
                            'latest': kpi_values.iloc[-1],
                            'samples': len(kpi_values)
                        }
            
            if not comparison:
                return f"No valid data found for sites: {', '.join(site_ids)}"
            
            report = f"Site Comparison for {kpi_name}:\n\n"
            
            # Sort by performance (mean value)
            sorted_sites = sorted(comparison.items(), key=lambda x: x[1]['mean'], reverse=True)
            
            for i, (site_id, stats) in enumerate(sorted_sites, 1):
                report += f"{i}. Site {site_id}:\n"
                report += f"   - Average: {stats['mean']:.2f}\n"
                report += f"   - Latest: {stats['latest']:.2f}\n"
                report += f"   - Stability: {stats['std']:.2f} (std dev)\n"
                report += f"   - Samples: {stats['samples']}\n\n"
            
            return report
            
        except Exception as e:
            return f"Error comparing sites: {str(e)}"


class TelecomConversationalAgent(LoggerMixin):
    """
    Main conversational agent for telecom network management.
    
    This agent can understand natural language queries about network performance,
    detect anomalies, analyze trends, and provide insights through conversation.
    """
    
    def __init__(self, config: TelecomConfig):
        """
        Initialize the conversational agent.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_processor = TelecomDataProcessor(config)
        self.anomaly_detector = KPIAnomalyDetector(config)
        self.tools_handler = NetworkAnalysisTools(config, self.data_processor, self.anomaly_detector)
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize LLM
        self._initialize_llm()
        self._setup_tools()
        self._create_agent()
        
        self.conversation_history = []
    
    def _initialize_llm(self):
        """Initialize the Google Gemini LLM"""
        try:
            genai.configure(api_key=self.config.gemini_api_key)
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.agent.model_name,
                temperature=self.config.agent.temperature,
                google_api_key=self.config.gemini_api_key
            )
            
            self.logger.info("Gemini LLM initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _setup_tools(self):
        """Setup tools available to the agent"""
        self.tools = [
            Tool(
                name="LoadNetworkData",
                description="Load and process telecom network data from a file path. Use this when asked to analyze data from a specific file.",
                func=self.tools_handler.load_network_data
            ),
            Tool(
                name="AnalyzeKPITrends",
                description="Analyze trends for a specific KPI. Format: 'kpi_name,site_id,days' (site_id and days are optional).",
                func=lambda query: self._parse_and_call(query, self.tools_handler.analyze_kpi_trends, ['kpi_name', 'site_id', 'days'])
            ),
            Tool(
                name="DetectAnomalies",
                description="Detect network anomalies. Format: 'kpi_name,site_id' (both optional).",
                func=lambda query: self._parse_and_call(query, self.tools_handler.detect_network_anomalies, ['kpi_name', 'site_id'])
            ),
            Tool(
                name="GetSiteOverview",
                description="Get comprehensive overview of a specific site. Provide site_id.",
                func=self.tools_handler.get_site_overview
            ),
            Tool(
                name="CompareSites",
                description="Compare KPI performance across sites. Format: 'site1,site2,site3|kpi_name'.",
                func=lambda query: self._parse_sites_comparison(query)
            )
        ]
    
    def _create_agent(self):
        """Create the LangChain agent with tools"""
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
    
    def _parse_and_call(self, query: str, func, param_names: List[str]):
        """Parse query parameters and call function"""
        try:
            parts = [p.strip() for p in query.split(',')]
            kwargs = {}
            
            for i, param in enumerate(param_names):
                if i < len(parts) and parts[i]:
                    if param == 'days':
                        kwargs[param] = int(parts[i])
                    else:
                        kwargs[param] = parts[i]
            
            return func(**kwargs)
        except Exception as e:
            return f"Error parsing query '{query}': {str(e)}"
    
    def _parse_sites_comparison(self, query: str) -> str:
        """Parse sites comparison query"""
        try:
            if '|' not in query:
                return "Invalid format. Use: 'site1,site2,site3|kpi_name'"
            
            sites_part, kpi_name = query.split('|', 1)
            site_ids = [s.strip() for s in sites_part.split(',')]
            
            return self.tools_handler.compare_sites(site_ids, kpi_name.strip())
        except Exception as e:
            return f"Error parsing sites comparison: {str(e)}"
    
    @log_function_call
    def load_models(self):
        """Load pre-trained anomaly detection models"""
        try:
            self.anomaly_detector.load_all_models()
            self.logger.info("Anomaly detection models loaded successfully")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load models: {e}")
            return False
    
    @log_function_call
    def chat(self, message: str, context: Optional[Dict] = None) -> AgentResponse:
        """
        Process a chat message and return response.
        
        Args:
            message: User message
            context: Optional context information
            
        Returns:
            AgentResponse object
        """
        try:
            self.logger.info(f"Processing message: {message[:100]}...")
            
            # Add system context to the message
            system_context = self._build_system_context(context)
            full_message = f"{system_context}\n\nUser: {message}"
            
            # Get agent response
            response = self.agent.run(full_message)
            
            # Create structured response
            agent_response = AgentResponse(
                message=response,
                confidence=0.8,  # Default confidence
                actions_taken=["analyze_message", "generate_response"]
            )
            
            # Store in conversation history
            self.conversation_history.append({
                'user_message': message,
                'agent_response': agent_response.to_dict(),
                'timestamp': agent_response.timestamp
            })
            
            self.logger.info("Message processed successfully")
            return agent_response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            
            error_response = AgentResponse(
                message=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                confidence=0.0,
                actions_taken=["error_handling"]
            )
            
            return error_response
    
    def _build_system_context(self, context: Optional[Dict] = None) -> str:
        """Build system context for the LLM"""
        base_context = f"""You are a Telecom Network AI Assistant specialized in analyzing network performance data and detecting anomalies.

Available KPIs: {', '.join(self.config.data.kpi_columns)}

Current Status:
- Anomaly Detection Models: {'Loaded' if self.anomaly_detector.is_fitted else 'Not Loaded'}
- Data Loaded: {'Yes' if self.tools_handler.current_data is not None else 'No'}

Capabilities:
1. Load and analyze network data
2. Detect anomalies in KPIs
3. Analyze performance trends
4. Compare sites and sectors
5. Provide insights and recommendations

Instructions:
- Use tools to access data and perform analysis
- Provide clear, actionable insights
- Explain technical concepts in accessible language
- Always consider network context in your analysis
"""
        
        if context:
            base_context += f"\n\nAdditional Context: {json.dumps(context, indent=2)}"
        
        return base_context
    
    def get_conversation_summary(self) -> str:
        """Get summary of recent conversation"""
        if not self.conversation_history:
            return "No conversation history available."
        
        recent_conversations = self.conversation_history[-5:]  # Last 5 exchanges
        
        summary = "Recent Conversation Summary:\n\n"
        for i, conv in enumerate(recent_conversations, 1):
            summary += f"{i}. User: {conv['user_message'][:100]}...\n"
            summary += f"   Agent: {conv['agent_response']['message'][:100]}...\n\n"
        
        return summary
    
    def reset_conversation(self):
        """Reset conversation memory and history"""
        self.memory.clear()
        self.conversation_history = []
        self.logger.info("Conversation reset")
    
    def save_conversation_history(self, filepath: Optional[str] = None):
        """Save conversation history to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"conversation_history_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        self.logger.info(f"Conversation history saved to {filepath}")


def create_telecom_agent(config: TelecomConfig) -> TelecomConversationalAgent:
    """
    Factory function to create a configured telecom conversational agent.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured TelecomConversationalAgent instance
    """
    return TelecomConversationalAgent(config)
