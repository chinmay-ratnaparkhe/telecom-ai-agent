"""
Model Context Protocol (MCP) Bridge for Telecom AI Platform

This module provides integration with MCP servers to enable tool invocation,
resource access, and structured interactions with external systems.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import httpx

from ..core.config import TelecomConfig
from ..utils.logger import LoggerMixin, log_function_call


@dataclass
class MCPTool:
    """Represents an MCP tool definition"""
    name: str
    description: str
    input_schema: Dict
    output_schema: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MCPResource:
    """Represents an MCP resource"""
    uri: str
    name: str
    description: str
    mime_type: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MCPToolCall:
    """Represents an MCP tool call and its result"""
    tool_name: str
    arguments: Dict
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'tool_name': self.tool_name,
            'arguments': self.arguments,
            'result': self.result,
            'error': self.error,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp.isoformat()
        }


class MCPClient(LoggerMixin):
    """
    Client for communicating with MCP servers
    
    Handles tool discovery, resource access, and tool invocation
    following the Model Context Protocol specification.
    """
    
    def __init__(self, server_url: str, timeout: float = 30.0):
        """
        Initialize MCP client
        
        Args:
            server_url: URL of the MCP server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.available_tools = {}
        self.available_resources = {}
        self.connected = False
        
        self.logger.info(f"MCP Client initialized for server: {server_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Connect to the MCP server and discover capabilities"""
        try:
            self.session = httpx.AsyncClient(timeout=self.timeout)
            
            # Test connection
            response = await self.session.get(f"{self.server_url}/health")
            response.raise_for_status()
            
            # Discover tools
            await self._discover_tools()
            
            # Discover resources
            await self._discover_resources()
            
            self.connected = True
            self.logger.info(f"Connected to MCP server successfully")
            self.logger.info(f"Discovered {len(self.available_tools)} tools and {len(self.available_resources)} resources")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            if self.session:
                await self.session.aclose()
                self.session = None
            raise
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session:
            await self.session.aclose()
            self.session = None
        self.connected = False
        self.logger.info("Disconnected from MCP server")
    
    async def _discover_tools(self):
        """Discover available tools from the MCP server"""
        try:
            response = await self.session.get(f"{self.server_url}/tools")
            response.raise_for_status()
            
            tools_data = response.json()
            self.available_tools = {}
            
            for tool_data in tools_data.get('tools', []):
                tool = MCPTool(
                    name=tool_data['name'],
                    description=tool_data['description'],
                    input_schema=tool_data.get('inputSchema', {}),
                    output_schema=tool_data.get('outputSchema')
                )
                self.available_tools[tool.name] = tool
                
        except Exception as e:
            self.logger.error(f"Failed to discover tools: {e}")
            self.available_tools = {}
    
    async def _discover_resources(self):
        """Discover available resources from the MCP server"""
        try:
            response = await self.session.get(f"{self.server_url}/resources")
            response.raise_for_status()
            
            resources_data = response.json()
            self.available_resources = {}
            
            for resource_data in resources_data.get('resources', []):
                resource = MCPResource(
                    uri=resource_data['uri'],
                    name=resource_data['name'],
                    description=resource_data['description'],
                    mime_type=resource_data.get('mimeType', 'text/plain')
                )
                self.available_resources[resource.uri] = resource
                
        except Exception as e:
            self.logger.error(f"Failed to discover resources: {e}")
            self.available_resources = {}
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> MCPToolCall:
        """
        Call an MCP tool
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            MCPToolCall with result or error
        """
        start_time = time.time()
        
        if not self.connected:
            return MCPToolCall(
                tool_name=tool_name,
                arguments=arguments,
                error="Not connected to MCP server"
            )
        
        if tool_name not in self.available_tools:
            return MCPToolCall(
                tool_name=tool_name,
                arguments=arguments,
                error=f"Tool '{tool_name}' not available. Available tools: {list(self.available_tools.keys())}"
            )
        
        try:
            # Validate arguments against schema if available
            tool = self.available_tools[tool_name]
            if tool.input_schema:
                # Basic validation - in production, use jsonschema
                self._validate_arguments(arguments, tool.input_schema)
            
            # Make the tool call
            payload = {
                "name": tool_name,
                "arguments": arguments
            }
            
            response = await self.session.post(
                f"{self.server_url}/tools/call",
                json=payload
            )
            response.raise_for_status()
            
            result_data = response.json()
            execution_time_ms = (time.time() - start_time) * 1000
            
            return MCPToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=result_data.get('result'),
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Error calling tool '{tool_name}': {e}")
            
            return MCPToolCall(
                tool_name=tool_name,
                arguments=arguments,
                error=str(e),
                execution_time_ms=execution_time_ms
            )
    
    async def get_resource(self, resource_uri: str) -> Optional[str]:
        """
        Get resource content from the MCP server
        
        Args:
            resource_uri: URI of the resource to retrieve
            
        Returns:
            Resource content as string, or None if error
        """
        if not self.connected:
            self.logger.error("Not connected to MCP server")
            return None
        
        try:
            response = await self.session.get(
                f"{self.server_url}/resources/read",
                params={"uri": resource_uri}
            )
            response.raise_for_status()
            
            resource_data = response.json()
            return resource_data.get('contents', {}).get('text')
            
        except Exception as e:
            self.logger.error(f"Error getting resource '{resource_uri}': {e}")
            return None
    
    def _validate_arguments(self, arguments: Dict, schema: Dict):
        """Basic argument validation against schema"""
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in arguments:
                raise ValueError(f"Required field '{field}' missing from arguments")
    
    def get_available_tools(self) -> List[MCPTool]:
        """Get list of available tools"""
        return list(self.available_tools.values())
    
    def get_available_resources(self) -> List[MCPResource]:
        """Get list of available resources"""
        return list(self.available_resources.values())


class MCPBridge(LoggerMixin):
    """
    Bridge between Telecom AI Platform and MCP servers
    
    Provides high-level interface for MCP integration with
    telecom-specific functionality and error handling.
    """
    
    def __init__(self, config: TelecomConfig):
        """Initialize MCP Bridge"""
        self.config = config
        self.clients = {}
        self.default_client = None
        
        # Tool mappings for telecom-specific functionality
        self.telecom_tool_mappings = {
            'analyze_kpi': 'analyze_kpi_trends',
            'detect_anomalies': 'detect_anomalies',
            'get_site_data': 'site_data_retriever',
            'generate_report': 'report_generator',
            'predict_trends': 'trend_predictor'
        }
        
        self.logger.info("MCP Bridge initialized")
    
    async def connect_to_server(self, server_url: str, alias: str = "default") -> bool:
        """
        Connect to an MCP server
        
        Args:
            server_url: URL of the MCP server
            alias: Alias for the server connection
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            client = MCPClient(server_url)
            await client.connect()
            
            self.clients[alias] = client
            if alias == "default" or self.default_client is None:
                self.default_client = client
            
            self.logger.info(f"Connected to MCP server '{alias}' at {server_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server '{alias}': {e}")
            return False
    
    async def disconnect_from_server(self, alias: str = "default"):
        """Disconnect from an MCP server"""
        if alias in self.clients:
            await self.clients[alias].disconnect()
            client = self.clients.pop(alias)
            
            if self.default_client == client:
                self.default_client = list(self.clients.values())[0] if self.clients else None
            
            self.logger.info(f"Disconnected from MCP server '{alias}'")
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for alias in list(self.clients.keys()):
            await self.disconnect_from_server(alias)
    
    @log_function_call
    async def analyze_kpi_with_mcp(
        self, 
        kpi_name: str, 
        site_id: Optional[str] = None,
        date_range: Optional[str] = None,
        time_range: Optional[str] = None,
        server_alias: str = "default"
    ) -> Dict:
        """
        Analyze KPI using MCP tools
        
        Args:
            kpi_name: Name of the KPI to analyze
            site_id: Optional site ID filter
            time_range: Optional time range (e.g., "7d", "1h")
            server_alias: MCP server to use
            
        Returns:
            Analysis result dictionary
        """
        client = self.clients.get(server_alias)
        if not client:
            return {'error': f"No connection to MCP server '{server_alias}'"}
        
        # Map to MCP tool
        mcp_tool_name = self.telecom_tool_mappings.get('analyze_kpi', 'analyze_kpi')
        
        # Server expects 'date_range' not 'time_range'
        arguments = {
            'kpi_name': kpi_name,
            'site_id': site_id,
            'date_range': date_range or (time_range if time_range else 'last_week')
        }
        
        result = await client.call_tool(mcp_tool_name, arguments)
        
        if result.error:
            self.logger.error(f"MCP KPI analysis failed: {result.error}")
            return {'error': result.error}
        
        return {
            'kpi_name': kpi_name,
            'analysis': result.result,
            'execution_time_ms': result.execution_time_ms,
            'timestamp': result.timestamp.isoformat()
        }
    
    @log_function_call
    async def detect_anomalies_with_mcp(
        self,
        kpi_filters: Optional[List[str]] = None,
        site_filters: Optional[List[str]] = None,
        severity_threshold: str = "medium",
        kpi_name: Optional[str] = None,
        site_id: Optional[str] = None,
        date_range: Optional[str] = None,
        server_alias: str = "default"
    ) -> Dict:
        """
        Detect anomalies using MCP tools
        
        Args:
            kpi_filters: Optional list of KPIs to analyze
            site_filters: Optional list of sites to analyze
            severity_threshold: Minimum severity level ("low", "medium", "high")
            server_alias: MCP server to use
            
        Returns:
            Anomaly detection results
        """
        client = self.clients.get(server_alias)
        if not client:
            return {'error': f"No connection to MCP server '{server_alias}'"}
        
        mcp_tool_name = self.telecom_tool_mappings.get('detect_anomalies', 'detect_anomalies')
        
        # Map to server-compatible args
        if not kpi_name and kpi_filters:
            kpi_name = kpi_filters[0] if isinstance(kpi_filters, list) and kpi_filters else None
        if not site_id and site_filters:
            site_id = site_filters[0] if isinstance(site_filters, list) and site_filters else None
        arguments = {
            'kpi_name': kpi_name or (self.config.data.kpi_columns[0] if hasattr(self.config.data, 'kpi_columns') and self.config.data.kpi_columns else None),
            'site_id': site_id,
            'date_range': date_range or 'last_week'
        }
        
        result = await client.call_tool(mcp_tool_name, arguments)
        
        if result.error:
            self.logger.error(f"MCP anomaly detection failed: {result.error}")
            return {'error': result.error}
        
        return {
            'anomalies': result.result,
            'execution_time_ms': result.execution_time_ms,
            'timestamp': result.timestamp.isoformat()
        }
    
    async def get_telecom_resources(self, server_alias: str = "default") -> List[MCPResource]:
        """Get telecom-specific resources from MCP server"""
        client = self.clients.get(server_alias)
        if not client:
            return []
        
        all_resources = client.get_available_resources()
        
        # Filter for telecom-relevant resources
        telecom_keywords = ['kpi', 'network', 'site', 'anomaly', 'performance', 'telecom']
        telecom_resources = []
        
        for resource in all_resources:
            if any(keyword in resource.description.lower() or keyword in resource.name.lower() 
                   for keyword in telecom_keywords):
                telecom_resources.append(resource)
        
        return telecom_resources
    
    async def get_server_status(self, server_alias: str = "default") -> Dict:
        """Get status of MCP server"""
        client = self.clients.get(server_alias)
        if not client:
            return {'status': 'disconnected', 'error': f"No connection to server '{server_alias}'"}
        
        return {
            'status': 'connected' if client.connected else 'disconnected',
            'server_url': client.server_url,
            'tools_count': len(client.available_tools),
            'resources_count': len(client.available_resources),
            'available_tools': [tool.name for tool in client.get_available_tools()]
        }
    
    async def execute_telecom_workflow(
        self,
        workflow_steps: List[Dict],
        server_alias: str = "default"
    ) -> List[MCPToolCall]:
        """
        Execute a multi-step telecom analysis workflow using MCP tools
        
        Args:
            workflow_steps: List of workflow steps with tool and arguments
            server_alias: MCP server to use
            
        Returns:
            List of tool call results
        """
        client = self.clients.get(server_alias)
        if not client:
            return [MCPToolCall("workflow", {}, error=f"No connection to server '{server_alias}'")]
        
        results = []
        
        for i, step in enumerate(workflow_steps):
            tool_name = step.get('tool')
            arguments = step.get('arguments', {})
            
            if not tool_name:
                results.append(MCPToolCall(f"step_{i}", arguments, error="No tool specified"))
                continue
            
            # Map telecom tool names to MCP tools if needed
            mcp_tool_name = self.telecom_tool_mappings.get(tool_name, tool_name)
            
            result = await client.call_tool(mcp_tool_name, arguments)
            results.append(result)
            
            # Stop on error if specified
            if result.error and step.get('stop_on_error', True):
                break
        
        return results
    
    def create_langchain_tools(self, server_alias: str = "default") -> List:
        """
        Create LangChain tools from MCP tools
        
        Args:
            server_alias: MCP server to use
            
        Returns:
            List of LangChain Tool objects
        """
        from langchain.agents import Tool
        
        client = self.clients.get(server_alias)
        if not client:
            return []
        
        langchain_tools = []
        
        for mcp_tool in client.get_available_tools():
            
            async def tool_func(query: str, tool_name=mcp_tool.name):
                """Wrapper function for MCP tool call"""
                try:
                    # Parse query as JSON if possible, otherwise use as single argument
                    try:
                        arguments = json.loads(query)
                    except json.JSONDecodeError:
                        arguments = {'query': query}
                    
                    result = await client.call_tool(tool_name, arguments)
                    return result.result if not result.error else f"Error: {result.error}"
                except Exception as e:
                    return f"Error calling MCP tool: {str(e)}"
            
            # Create LangChain tool
            langchain_tool = Tool(
                name=f"MCP_{mcp_tool.name}",
                description=f"MCP Tool: {mcp_tool.description}",
                func=lambda q, tn=mcp_tool.name: asyncio.create_task(tool_func(q, tn))
            )
            
            langchain_tools.append(langchain_tool)
        
        return langchain_tools


def create_mcp_bridge(config: TelecomConfig) -> MCPBridge:
    """
    Factory function to create a configured MCP bridge
    
    Args:
        config: Configuration object
        
    Returns:
        Configured MCPBridge instance
    """
    return MCPBridge(config)
