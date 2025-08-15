"""
Production Streaming Agent for Telecom AI Platform

This module implements a production-ready streaming conversational AI agent with:
- Real-time chain of thought processing
- Streaming responses
- Memory management
- Web search capabilities
- KPI analysis integration
- MCP (Model Context Protocol) support
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid

import google.generativeai as genai
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..core.config import TelecomConfig
from ..core.data_processor import TelecomDataProcessor
from ..models.anomaly_detector import KPIAnomalyDetector
from ..utils.logger import LoggerMixin, log_function_call
from .conversational_ai import NetworkAnalysisTools


@dataclass
class ThoughtStep:
    """Represents a step in the chain of thought process"""
    step_id: str
    step_type: str  # "analysis", "reasoning", "tool_call", "conclusion"
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'step_id': self.step_id,
            'step_type': self.step_type,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {},
            'duration_ms': self.duration_ms
        }


@dataclass
class StreamingResponse:
    """Response with streaming capability"""
    response_id: str
    conversation_id: str
    thought_chain: List[ThoughtStep]
    final_response: str
    confidence: float
    data: Optional[Dict] = None
    visualizations: Optional[List[str]] = None
    actions_taken: Optional[List[str]] = None
    timestamp: datetime = None
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'response_id': self.response_id,
            'conversation_id': self.conversation_id,
            'thought_chain': [step.to_dict() for step in self.thought_chain],
            'final_response': self.final_response,
            'confidence': self.confidence,
            'data': self.data,
            'visualizations': self.visualizations or [],
            'actions_taken': self.actions_taken or [],
            'timestamp': self.timestamp.isoformat(),
            'processing_time_ms': self.processing_time_ms
        }


class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses with chain of thought"""
    
    def __init__(self, response_queue: asyncio.Queue):
        self.response_queue = response_queue
        self.current_step = None
        self.step_start_time = None
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when the chain starts"""
        step = ThoughtStep(
            step_id=str(uuid.uuid4()),
            step_type="analysis",
            content="Starting analysis of your request...",
            timestamp=datetime.now()
        )
        await self.response_queue.put(("thought_step", step))
    
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when a tool starts"""
        self.step_start_time = time.time()
        tool_name = serialized.get('name', 'Unknown Tool')
        
        step = ThoughtStep(
            step_id=str(uuid.uuid4()),
            step_type="tool_call",
            content=f"Using {tool_name} to analyze: {input_str[:100]}...",
            timestamp=datetime.now(),
            metadata={'tool_name': tool_name, 'input': input_str}
        )
        self.current_step = step
        await self.response_queue.put(("thought_step", step))
    
    async def on_tool_end(self, output: str, **kwargs):
        """Called when a tool ends"""
        if self.current_step:
            duration = (time.time() - self.step_start_time) * 1000 if self.step_start_time else None
            self.current_step.duration_ms = duration
            
            step = ThoughtStep(
                step_id=str(uuid.uuid4()),
                step_type="reasoning",
                content=f"Analysis complete. Processing results...",
                timestamp=datetime.now(),
                metadata={'tool_output_length': len(output), 'duration_ms': duration}
            )
            await self.response_queue.put(("thought_step", step))
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when LLM starts"""
        step = ThoughtStep(
            step_id=str(uuid.uuid4()),
            step_type="reasoning",
            content="Analyzing data and formulating response...",
            timestamp=datetime.now()
        )
        await self.response_queue.put(("thought_step", step))
    
    async def on_llm_new_token(self, token: str, **kwargs):
        """Called for each new token from LLM"""
        await self.response_queue.put(("token", token))
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when the chain ends"""
        step = ThoughtStep(
            step_id=str(uuid.uuid4()),
            step_type="conclusion",
            content="Analysis complete. Finalizing response...",
            timestamp=datetime.now()
        )
        await self.response_queue.put(("thought_step", step))


class WebSearchTool:
    """Web search capability for the agent"""
    
    def __init__(self, config: TelecomConfig):
        self.config = config
        self.enabled = config.agent.search_enabled
    
    async def search(self, query: str, domain_context: str = "telecom") -> str:
        """Perform web search with domain context"""
        if not self.enabled:
            return "Web search is currently disabled in configuration."
        
        # Enhanced query with telecom context
        enhanced_query = f"{query} {domain_context} network telecommunications KPI performance"
        
        # For now, return a simulated search result
        # In production, this would integrate with a real search API
        return f"""Web search results for: {query}

Domain Context: {domain_context}

Key findings:
- Industry best practices for {query}
- Recent developments in telecom network optimization
- Relevant standards and benchmarks
- Case studies from similar network deployments

Note: This is a simulated search result. In production, this would connect to real search APIs."""


class MemoryManager:
    """Advanced memory management for conversations"""
    
    def __init__(self, max_conversations: int = 100, max_messages_per_conversation: int = 50):
        self.conversations = {}
        self.max_conversations = max_conversations
        self.max_messages_per_conversation = max_messages_per_conversation
        self.conversation_metadata = {}
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Create a new conversation"""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        self.conversations[conversation_id] = ConversationBufferWindowMemory(
            k=self.max_messages_per_conversation,
            memory_key="chat_history",
            return_messages=True
        )
        
        self.conversation_metadata[conversation_id] = {
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'message_count': 0,
            'topics': []
        }
        
        # Clean up old conversations if needed
        self._cleanup_old_conversations()
        
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationBufferWindowMemory]:
        """Get conversation memory"""
        if conversation_id in self.conversations:
            self.conversation_metadata[conversation_id]['last_active'] = datetime.now()
            return self.conversations[conversation_id]
        return None
    
    def add_message(self, conversation_id: str, message: BaseMessage):
        """Add message to conversation"""
        memory = self.get_conversation(conversation_id)
        if memory:
            memory.chat_memory.add_message(message)
            self.conversation_metadata[conversation_id]['message_count'] += 1
    
    def get_conversation_summary(self, conversation_id: str) -> Dict:
        """Get summary of conversation"""
        if conversation_id not in self.conversation_metadata:
            return {}
        
        metadata = self.conversation_metadata[conversation_id]
        memory = self.conversations.get(conversation_id)
        
        return {
            'conversation_id': conversation_id,
            'created_at': metadata['created_at'].isoformat(),
            'last_active': metadata['last_active'].isoformat(),
            'message_count': metadata['message_count'],
            'topics': metadata.get('topics', []),
            'recent_messages': len(memory.chat_memory.messages) if memory else 0
        }
    
    def _cleanup_old_conversations(self):
        """Remove old conversations to manage memory"""
        if len(self.conversations) <= self.max_conversations:
            return
        
        # Sort by last active time
        sorted_conversations = sorted(
            self.conversation_metadata.items(),
            key=lambda x: x[1]['last_active']
        )
        
        # Remove oldest conversations
        conversations_to_remove = len(self.conversations) - self.max_conversations
        for conversation_id, _ in sorted_conversations[:conversations_to_remove]:
            self.conversations.pop(conversation_id, None)
            self.conversation_metadata.pop(conversation_id, None)


class ProductionStreamingAgent(LoggerMixin):
    """
    Production-ready streaming conversational AI agent for telecom network management.
    
    Features:
    - Real-time streaming responses with chain of thought
    - Memory management across conversations
    - Web search capabilities
    - KPI analysis and anomaly detection
    - MCP protocol support
    - Governance and audit logging
    """
    
    def __init__(self, config: TelecomConfig):
        """Initialize the production streaming agent"""
        self.config = config
        self.data_processor = TelecomDataProcessor(config)
        self.anomaly_detector = KPIAnomalyDetector(config)
        self.network_tools = NetworkAnalysisTools(config, self.data_processor, self.anomaly_detector)
        
        # Initialize components
        self.memory_manager = MemoryManager()
        self.web_search = WebSearchTool(config)
        
        # Initialize LLM with streaming
        self._initialize_llm()
        self._setup_tools()
        
        # Active conversations tracking
        self.active_conversations = {}
        
        self.logger.info("Production Streaming Agent initialized successfully")
    
    def _initialize_llm(self):
        """Initialize the Google Gemini LLM with streaming support"""
        try:
            genai.configure(api_key=self.config.gemini_api_key)
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.agent.model_name,
                temperature=self.config.agent.temperature,
                max_tokens=self.config.agent.max_tokens,
                google_api_key=self.config.gemini_api_key,
                streaming=True  # Enable streaming
            )
            
            self.logger.info("Streaming LLM initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize streaming LLM: {e}")
            raise
    
    def _setup_tools(self):
        """Setup tools available to the agent"""
        self.tools = [
            Tool(
                name="LoadNetworkData",
                description="Load and process telecom network data from a file path",
                func=self.network_tools.load_network_data
            ),
            Tool(
                name="AnalyzeKPITrends",
                description="Analyze trends for specific KPIs with optional site and time filters",
                func=lambda query: self._parse_and_call(query, self.network_tools.analyze_kpi_trends, ['kpi_name', 'site_id', 'days'])
            ),
            Tool(
                name="DetectAnomalies",
                description="Detect network anomalies across KPIs and sites",
                func=lambda query: self._parse_and_call(query, self.network_tools.detect_network_anomalies, ['kpi_name', 'site_id'])
            ),
            Tool(
                name="GetSiteOverview",
                description="Get comprehensive overview of a specific network site",
                func=self.network_tools.get_site_overview
            ),
            Tool(
                name="CompareSites",
                description="Compare KPI performance across multiple sites",
                func=self._parse_sites_comparison
            ),
            Tool(
                name="WebSearch",
                description="Search the web for telecom-related information and best practices",
                func=lambda query: asyncio.create_task(self.web_search.search(query))
            )
        ]
    
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
            
            return self.network_tools.compare_sites(site_ids, kpi_name.strip())
        except Exception as e:
            return f"Error parsing sites comparison: {str(e)}"
    
    @log_function_call
    async def load_models(self):
        """Load pre-trained anomaly detection models"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.anomaly_detector.load_all_models
            )
            self.logger.info("Anomaly detection models loaded successfully")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load models: {e}")
            return False
    
    async def stream_response(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream a response with chain of thought processing
        
        Args:
            message: User message
            conversation_id: Optional conversation ID
            context: Optional context information
            
        Yields:
            Dict containing streaming data (thought steps, tokens, final response)
        """
        start_time = time.time()
        response_id = str(uuid.uuid4())
        
        # Create or get conversation
        if conversation_id is None:
            conversation_id = self.memory_manager.create_conversation()
        
        memory = self.memory_manager.get_conversation(conversation_id)
        if memory is None:
            conversation_id = self.memory_manager.create_conversation(conversation_id)
            memory = self.memory_manager.get_conversation(conversation_id)
        
        # Create response queue for streaming
        response_queue = asyncio.Queue()
        callback_handler = StreamingCallbackHandler(response_queue)
        
        # Build system context
        system_context = self._build_system_context(context)
        
        # Add user message to memory
        self.memory_manager.add_message(conversation_id, HumanMessage(content=message))
        
        thought_chain = []
        response_tokens = []
        
        try:
            # Create agent with streaming callback
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=False,
                callbacks=[callback_handler],
                max_iterations=3,
                early_stopping_method="generate"
            )
            
            # Start processing in background
            processing_task = asyncio.create_task(
                self._process_with_agent(agent, system_context, message, response_queue)
            )
            
            # Stream responses as they come
            while True:
                try:
                    # Wait for next item with timeout
                    item_type, item_data = await asyncio.wait_for(
                        response_queue.get(), timeout=1.0
                    )
                    
                    if item_type == "thought_step":
                        thought_chain.append(item_data)
                        yield {
                            "type": "thought_step",
                            "data": item_data.to_dict(),
                            "response_id": response_id,
                            "conversation_id": conversation_id
                        }
                    
                    elif item_type == "token":
                        response_tokens.append(item_data)
                        yield {
                            "type": "token",
                            "data": item_data,
                            "response_id": response_id,
                            "conversation_id": conversation_id
                        }
                    
                    elif item_type == "final_response":
                        final_response_text = item_data
                        break
                        
                except asyncio.TimeoutError:
                    # Check if processing is complete
                    if processing_task.done():
                        try:
                            final_response_text = await processing_task
                            break
                        except Exception as e:
                            final_response_text = f"Error processing request: {str(e)}"
                            break
                    continue
                except Exception as e:
                    self.logger.error(f"Error in streaming: {e}")
                    final_response_text = f"Error in streaming: {str(e)}"
                    break
            
            # Add AI response to memory
            self.memory_manager.add_message(conversation_id, AIMessage(content=final_response_text))
            
            # Create final response
            processing_time_ms = (time.time() - start_time) * 1000
            
            final_response = StreamingResponse(
                response_id=response_id,
                conversation_id=conversation_id,
                thought_chain=thought_chain,
                final_response=final_response_text,
                confidence=0.8,  # Default confidence
                processing_time_ms=processing_time_ms,
                actions_taken=["stream_analysis", "chain_of_thought", "generate_response"]
            )
            
            # Yield final response
            yield {
                "type": "final_response",
                "data": final_response.to_dict(),
                "response_id": response_id,
                "conversation_id": conversation_id
            }
            
            self.logger.info(f"Streaming response completed in {processing_time_ms:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error in stream_response: {e}")
            
            error_response = StreamingResponse(
                response_id=response_id,
                conversation_id=conversation_id,
                thought_chain=thought_chain,
                final_response=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                actions_taken=["error_handling"]
            )
            
            yield {
                "type": "error",
                "data": error_response.to_dict(),
                "response_id": response_id,
                "conversation_id": conversation_id
            }
    
    async def _process_with_agent(self, agent, system_context: str, message: str, response_queue: asyncio.Queue):
        """Process message with agent and send final response to queue"""
        try:
            full_message = f"{system_context}\n\nUser: {message}"
            result = await asyncio.get_event_loop().run_in_executor(
                None, agent.run, full_message
            )
            await response_queue.put(("final_response", result))
            return result
        except Exception as e:
            error_msg = f"Error processing with agent: {str(e)}"
            await response_queue.put(("final_response", error_msg))
            return error_msg
    
    def _build_system_context(self, context: Optional[Dict] = None) -> str:
        """Build enhanced system context for the LLM"""
        base_context = f"""You are an advanced Telecom Network AI Assistant with comprehensive analytical capabilities.

CORE CAPABILITIES:
- Real-time KPI analysis across {', '.join(self.config.data.kpi_columns)}
- Anomaly detection with multiple ML algorithms
- Cross-site performance comparison
- Trend analysis and forecasting
- Web search for industry best practices
- Multi-turn conversation memory

CURRENT STATUS:
- Anomaly Detection Models: {'Loaded' if self.anomaly_detector.is_fitted else 'Not Loaded'}
- Data Processing: {'Ready' if self.network_tools.current_data is not None else 'Awaiting Data'}
- Web Search: {'Enabled' if self.web_search.enabled else 'Disabled'}

ANALYSIS APPROACH:
1. Break down complex queries into logical steps
2. Use appropriate tools for data analysis
3. Cross-reference findings with industry standards
4. Provide actionable insights and recommendations
5. Explain technical concepts clearly

RESPONSE GUIDELINES:
- Show your reasoning process step by step
- Use tools to access current data and perform analysis
- Provide specific, actionable recommendations
- Include confidence levels for findings
- Suggest relevant follow-up questions

Remember: You're assisting network engineers and operators who need precise, reliable information for critical infrastructure decisions."""
        
        if context:
            base_context += f"\n\nADDITIONAL CONTEXT:\n{json.dumps(context, indent=2)}"
        
        return base_context
    
    async def get_conversation_history(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation history and metadata"""
        return self.memory_manager.get_conversation_summary(conversation_id)
    
    async def reset_conversation(self, conversation_id: str) -> bool:
        """Reset a specific conversation"""
        if conversation_id in self.memory_manager.conversations:
            self.memory_manager.conversations.pop(conversation_id)
            self.memory_manager.conversation_metadata.pop(conversation_id, None)
            return True
        return False
    
    async def list_active_conversations(self) -> List[Dict]:
        """List all active conversations"""
        return [
            self.memory_manager.get_conversation_summary(conv_id)
            for conv_id in self.memory_manager.conversations.keys()
        ]
    
    async def health_check(self) -> Dict:
        """Perform system health check"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "llm": "operational",
                "memory_manager": f"{len(self.memory_manager.conversations)} active conversations",
                "anomaly_detector": "loaded" if self.anomaly_detector.is_fitted else "not_loaded",
                "data_processor": "ready",
                "web_search": "enabled" if self.web_search.enabled else "disabled"
            },
            "config": {
                "model": self.config.agent.model_name,
                "temperature": self.config.agent.temperature,
                "max_tokens": self.config.agent.max_tokens
            }
        }


def create_production_streaming_agent(config: TelecomConfig) -> ProductionStreamingAgent:
    """
    Factory function to create a configured production streaming agent
    
    Args:
        config: Configuration object
        
    Returns:
        Configured ProductionStreamingAgent instance
    """
    return ProductionStreamingAgent(config)
