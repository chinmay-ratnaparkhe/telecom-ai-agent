#!/usr/bin/env python3
"""
Telecom AI Agent - Production Ready Version

A comprehensive AI-powered telecom network analysis agent that provides:
- Real-time anomaly detection using statistical methods
- Site performance comparison and analysis
- KPI explanation and technical consultation
- Google AI integration for enhanced insights
- Web search integration for additional context

Author: AI Assistant
Created: August 2025
Version: 2.0 - Production Ready
"""

import asyncio
import json
import re
import os
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Google AI integration (optional dependency)
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    logging.warning("Google Generative AI not available. Install with: pip install google-generativeai")

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging with proper formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('telecom_agent.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    """
    Data class to store query processing context and extracted information.
    
    Attributes:
        intent: The detected intent of the query (anomaly_detection, comparison, etc.)
        site_ids: List of extracted site identifiers from the query
        kpi_names: List of extracted KPI names from the query
        raw_query: The original user query string
    """
    intent: str
    site_ids: List[str]
    kpi_names: List[str]
    raw_query: str

class SimpleQueryProcessor:
    """
    Natural Language Query Processor for Telecom Network Analysis
    
    This class handles the parsing and interpretation of user queries to extract
    relevant information such as site IDs, KPI names, and query intent.
    """
    
    def __init__(self):
        """Initialize the query processor with KPI mapping dictionary."""
        # Mapping of common terms to standardized KPI names
        self.kpi_mapping = {
            'sinr': 'SINR',
            'signal': 'SINR',
            'rsrp': 'RSRP',
            'cpu': 'CPU_Utilization',
            'processor': 'CPU_Utilization',
            'utilization': 'CPU_Utilization',
            'throughput': 'DL_Throughput',
            'dl_throughput': 'DL_Throughput',
            'download': 'DL_Throughput',
            'ul_throughput': 'UL_Throughput',
            'upload': 'UL_Throughput',
            'rtt': 'RTT',
            'latency': 'RTT',
            'delay': 'RTT',
            'packet': 'Packet_Loss',
            'loss': 'Packet_Loss',
            'call': 'Call_Drop_Rate',
            'drop': 'Call_Drop_Rate',
            'call drop': 'Call_Drop_Rate',
            'call_drop': 'Call_Drop_Rate',
            'call drop rate': 'Call_Drop_Rate',
            'call_drop_rate': 'Call_Drop_Rate',
            'handover': 'Handover_Success_Rate',
            'handover success': 'Handover_Success_Rate',
            'handover_success': 'Handover_Success_Rate',
            'active': 'Active_Users',
            'users': 'Active_Users',
            'active users': 'Active_Users',
            'active_users': 'Active_Users'
        }
    
    def extract_sites(self, query: str) -> List[str]:
        """
        Extract site identifiers from user query using regex patterns.
        Handles various formats: SITE_001, site 2, location 5, etc.
        
        Args:
            query: User input string
            
        Returns:
            List of standardized site IDs (e.g., ['SITE_001', 'SITE_005'])
        """
        sites = []
        patterns = [
            r'SITE_(\d+)',              # SITE_001, SITE_2
            r'site[_\s]*(\d+)',         # site_2, site 2, site2
            r'location[_\s]*(\d+)',     # location_5, location 5
            r'cell[_\s]*(\d+)',         # cell_3, cell 3
            r'base[_\s]*station[_\s]*(\d+)',  # base station 4
            r'bs[_\s]*(\d+)',           # bs_1, bs 1
            r'tower[_\s]*(\d+)'         # tower_6, tower 6
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                site_num = match.zfill(3)  # Pad with zeros to 3 digits
                sites.append(f"SITE_{site_num}")
        
        return list(set(sites))  # Remove duplicates
    
    def extract_kpis(self, query: str) -> List[str]:
        """
        Extract Key Performance Indicator names from user query.
        Prioritizes longer matches (e.g., "call drop rate" over "call").
        
        Args:
            query: User input string
            
        Returns:
            List of standardized KPI names
        """
        query_lower = query.lower()
        found_kpis = []
        
        # Sort keys by length (descending) to match longer phrases first
        sorted_keys = sorted(self.kpi_mapping.keys(), key=len, reverse=True)
        
        for key in sorted_keys:
            if key in query_lower:
                found_kpis.append(self.kpi_mapping[key])
                # Remove the matched key from query to avoid partial matches
                query_lower = query_lower.replace(key, ' ')
        
        return list(set(found_kpis))  # Remove duplicates
    
    def detect_intent(self, query: str) -> str:
        """
        Classify the user's intent based on keywords in the query.
        
        Args:
            query: User input string
            
        Returns:
            String representing the detected intent
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['anomal', 'find', 'detect', 'unusual', 'outlier']):
            return 'anomaly_detection'
        elif any(word in query_lower for word in ['compare', 'comparison']):
            return 'comparison'
        elif any(word in query_lower for word in ['visualize', 'plot', 'chart', 'show']):
            return 'visualization'
        elif any(word in query_lower for word in ['trend', 'analyze', 'analysis']):
            return 'analysis'
        else:
            return 'general_query'
    
    def process_query(self, query: str) -> QueryContext:
        """
        Main query processing method that combines all extraction methods.
        
        Args:
            query: User input string
            
        Returns:
            QueryContext object containing all extracted information
        """
        return QueryContext(
            intent=self.detect_intent(query),
            site_ids=self.extract_sites(query),
            kpi_names=self.extract_kpis(query),
            raw_query=query
        )

class GoogleSearchIntegration:
    """
    Web Search Integration for Additional Telecom Knowledge
    
    This class provides web search capabilities using DuckDuckGo API to find
    additional information about telecom concepts and KPIs.
    """
    
    def __init__(self):
        """Initialize the search integration with DuckDuckGo API endpoint."""
        self.base_url = "https://api.duckduckgo.com/"
        
    def search_telecom_info(self, query: str) -> str:
        """
        Search for telecom-specific information using DuckDuckGo API.
        
        Args:
            query: Search query string
            
        Returns:
            Formatted search results or empty string if no results found
        """
        try:
            # Enhanced query with telecom context
            search_query = f"{query} telecom network KPI 3GPP standard definition"
            
            params = {
                'q': search_query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant information
                results = []
                if 'AbstractText' in data and data['AbstractText']:
                    results.append(data['AbstractText'])
                
                if 'RelatedTopics' in data:
                    for topic in data['RelatedTopics'][:2]:  # Limit to top 2
                        if 'Text' in topic:
                            results.append(topic['Text'])
                
                if results:
                    search_result = " ".join(results)
                    return f"Additional Information from Search:\n{search_result}\n"
                else:
                    return ""
            else:
                return ""
        
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ""

class GoogleAIIntegration:
    """
    Google Gemini AI Integration for Enhanced Telecom Analysis
    
    This class integrates with Google's Gemini AI to provide expert-level
    telecom insights and enhanced response generation.
    """
    
    def __init__(self):
        """Initialize Google AI integration with API key validation."""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.enabled = False
        
        if self.api_key and GOOGLE_AI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.enabled = True
                logger.info("Google Gemini AI initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google AI: {e}")
                self.enabled = False
        else:
            logger.info("Google API key not found or package unavailable - AI enhancement disabled")
    
    def enhance_response(self, query: str, data_result: str) -> str:
        """
        Enhance data analysis results with AI-generated insights.
        
        Args:
            query: Original user query
            data_result: Basic data analysis result
            
        Returns:
            Enhanced response with AI insights or original result if AI unavailable
        """
        if not self.enabled:
            return data_result
            
        try:
            # Create a prompt for enhancing the response
            prompt = f"""
You are a telecom network expert. A user asked: "{query}"

I have this data analysis result:
{data_result}

Please enhance this response by:
1. Adding professional telecom insights
2. Explaining technical terms if needed
3. Providing context about network performance standards
4. Suggesting next steps or related analysis

Keep the response concise and professional. Do not repeat the data - just add valuable insights.
"""
            
            response = self.model.generate_content(prompt)
            if response and response.text:
                enhanced_text = response.text.strip()
                return f"{data_result}\n\nAI Expert Insights:\n{enhanced_text}"
            else:
                return data_result
                
        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            return data_result
    
    def answer_general_question(self, query: str) -> str:
        """
        Answer general telecom questions using Google AI.
        
        Args:
            query: User question about telecommunications
            
        Returns:
            AI-generated expert answer or empty string if unavailable
        """
        if not self.enabled:
            return ""
            
        try:
            prompt = f"""
You are a telecom network expert. Answer this question about telecommunications:
"{query}"

Important instructions:
- If the question asks about multiple terms (e.g., "What is CPU Utilization and SINR?"), provide separate explanations for each term
- Be specific and accurate for each individual concept mentioned
- Provide comprehensive but concise answers focusing on:
  * Technical accuracy
  * Practical applications in telecom networks
  * Industry standards (3GPP, ITU, etc.)
  * Best practices
  * Specific values/thresholds where relevant

Use professional telecom terminology and provide distinct explanations for each concept if multiple are mentioned.
"""
            
            response = self.model.generate_content(prompt)
            if response and response.text:
                return f"AI Expert Answer:\n{response.text.strip()}"
            else:
                return ""
                
        except Exception as e:
            logger.error(f"AI question answering failed: {e}")
            return ""

class StreamlitTelecomAgent:
    """
    Production-Ready Telecom AI Agent for Streamlit Integration
    
    This class provides a comprehensive telecom network analysis platform with:
    - Statistical anomaly detection
    - Site performance comparison
    - AI-powered insights and explanations
    - Web search integration for additional context
    
    The agent is designed to be synchronous and compatible with Streamlit applications.
    """
    
    def __init__(self):
        """
        Initialize the Telecom AI Agent with all necessary components.
        
        This method sets up:
        - Query processor for natural language understanding
        - Google AI integration for enhanced insights
        - Web search capabilities
        - Telecom data loading and validation
        """
        # Initialize core components
        self.query_processor = SimpleQueryProcessor()
        self.google_search = GoogleSearchIntegration()
        self.google_ai = GoogleAIIntegration()
        
        # Load telecom dataset
        self._load_data()
        
        logger.info("Telecom AI Agent initialized successfully")
    
    def _load_data(self) -> None:
        """
        Load and validate the telecom network dataset.
        
        Attempts to load the main dataset file and performs basic validation.
        Sets self.data to None if loading fails, which triggers fallback behavior.
        """
        try:
            data_file = "AD_data_10KPI.csv"
            if os.path.exists(data_file):
                self.data = pd.read_csv(data_file)
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                
                # Log dataset statistics
                num_records = len(self.data)
                num_sites = len(self.data['Site_ID'].unique())
                logger.info(f"Successfully loaded dataset: {num_records} records from {num_sites} sites")
            else:
                logger.warning(f"Dataset file {data_file} not found")
                self.data = None
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            self.data = None
    
    def process_query(self, query: str) -> str:
        """
        Main query processing method that routes requests to appropriate handlers.
        
        This method:
        1. Parses the user query to extract intent and parameters
        2. Routes to specialized handlers based on detected intent
        3. Returns formatted results with optional AI enhancement
        
        Args:
            query: Natural language query from the user
            
        Returns:
            Formatted response string with analysis results or error message
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Parse the query to extract intent and parameters
            context = self.query_processor.process_query(query)
            logger.debug(f"Detected intent: {context.intent}, Sites: {context.site_ids}, KPIs: {context.kpi_names}")
            
            # Route to appropriate handler based on detected intent
            if context.intent == 'anomaly_detection':
                return self._detect_anomalies(context)
            elif context.intent == 'comparison':
                return self._compare_sites(context)
            elif context.intent == 'general_query':
                return self._handle_general_query(query)
            else:
                return self._provide_help()
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return f"Error processing query: {str(e)}. Please try rephrasing your question."
    
    def _detect_anomalies(self, context: QueryContext) -> str:
        """
        Detect anomalies in telecom network data using statistical methods.
        
        This method implements a 2-sigma statistical anomaly detection algorithm:
        1. Filters data by specified site and KPI
        2. Calculates mean and standard deviation
        3. Identifies values outside 2 standard deviations
        4. Enhances results with AI insights if available
        
        Args:
            context: QueryContext containing site IDs and KPI names
            
        Returns:
            Formatted anomaly detection results with statistics and AI insights
        """
        if self.data is None:
            return "Error: No telecom dataset available for analysis."
        
        # Extract parameters from context
        site_id = context.site_ids[0] if context.site_ids else None
        kpi_name = context.kpi_names[0] if context.kpi_names else None
        
        # Validate input parameters
        if not site_id:
            return "Please specify a site ID (e.g., SITE_005, SITE_001). Available sites: SITE_001 to SITE_100"
        
        if not kpi_name:
            available_kpis = [col for col in self.data.columns if col not in ['Date', 'Site_ID', 'Sector_ID']]
            return f"Please specify a KPI. Available KPIs: {', '.join(available_kpis)}"
        
        # Filter dataset by site
        site_data = self.data[self.data['Site_ID'] == site_id]
        
        if site_data.empty:
            return f"No data found for site {site_id}."
        
        # Validate KPI exists in dataset
        if kpi_name not in site_data.columns:
            available_kpis = [col for col in site_data.columns if col not in ['Date', 'Site_ID', 'Sector_ID']]
            return f"KPI {kpi_name} not found. Available KPIs: {', '.join(available_kpis)}"
        
        # Perform anomaly detection
        kpi_values = site_data[kpi_name].dropna()
        if kpi_values.empty:
            return f"No valid data for {kpi_name} in {site_id}"
        
        # Statistical anomaly detection using 2-sigma threshold
        mean_val = kpi_values.mean()
        std_val = kpi_values.std()
        threshold = 2 * std_val
        
        # Identify anomalous records
        anomaly_mask = abs(site_data[kpi_name] - mean_val) > threshold
        anomalies = site_data[anomaly_mask & site_data[kpi_name].notna()]
        
        # Format results
        if anomalies.empty:
            basic_result = (f"No anomalies detected for {kpi_name} in {site_id}. "
                          f"Analyzed {len(site_data)} records with mean {mean_val:.2f}.")
        else:
            # Format anomaly information
            anomaly_dates = anomalies['Date'].dt.strftime('%Y-%m-%d').tolist()
            anomaly_values = anomalies[kpi_name].tolist()
            
            anomaly_percentage = len(anomalies) / len(site_data) * 100
            
            basic_result = (f"ANOMALY ALERT: Detected {len(anomalies)} anomalies ({anomaly_percentage:.1f}%) "
                          f"for {kpi_name} in {site_id}.\n\n"
                          f"Anomalous dates and values:\n")
            
            # Show first 5 anomalies with details
            for date, value in zip(anomaly_dates[:5], anomaly_values[:5]):
                basic_result += f"• {date}: {value:.2f}\n"
            
            if len(anomalies) > 5:
                basic_result += f"... and {len(anomalies) - 5} more anomalies\n"
            
            basic_result += (f"\nStatistical Analysis:\n"
                           f"• Mean: {mean_val:.2f}\n"
                           f"• Standard Deviation: {std_val:.2f}\n"
                           f"• Anomaly Threshold: ±{threshold:.2f}")
        
        # Enhance with AI insights if available
        enhanced_result = self.google_ai.enhance_response(context.raw_query, basic_result)
        
        return enhanced_result
    
    def _compare_sites(self, context: QueryContext) -> str:
        """
        Compare performance metrics across multiple telecom sites.
        
        This method:
        1. Validates that multiple sites are specified
        2. Calculates statistical summaries for each site
        3. Presents comparative analysis results
        
        Args:
            context: QueryContext containing site IDs and KPI names
            
        Returns:
            Formatted comparison results showing site performance metrics
        """
        if self.data is None:
            return "Error: No telecom dataset available for comparison."
        
        sites = context.site_ids
        kpi_name = context.kpi_names[0] if context.kpi_names else None
        
        # Validate input parameters
        if len(sites) < 2:
            return "Please specify at least 2 sites to compare (e.g., 'compare SITE_001 and SITE_005')."
        
        if not kpi_name:
            return "Please specify a KPI to compare."
        
        # Collect performance data for each site
        comparison_data = {}
        for site_id in sites:
            site_data = self.data[self.data['Site_ID'] == site_id]
            if not site_data.empty and kpi_name in site_data.columns:
                kpi_values = site_data[kpi_name].dropna()
                if not kpi_values.empty:
                    comparison_data[site_id] = {
                        'mean': kpi_values.mean(),
                        'std': kpi_values.std(),
                        'latest': kpi_values.iloc[-1],
                        'count': len(kpi_values),
                        'min': kpi_values.min(),
                        'max': kpi_values.max()
                    }
        
        if not comparison_data:
            return f"No valid data found for {kpi_name} in the specified sites."
        
        # Format comparison results
        result = f"SITE PERFORMANCE COMPARISON - {kpi_name}\n\n"
        
        for site_id, stats in comparison_data.items():
            result += (f"{site_id}:\n"
                      f"  • Mean: {stats['mean']:.2f}\n"
                      f"  • Latest Value: {stats['latest']:.2f}\n"
                      f"  • Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
                      f"  • Records: {stats['count']}\n\n")
        
        return result
    
    def _handle_general_query(self, query: str) -> str:
        """
        Handle general telecom questions and knowledge requests.
        
        This method prioritizes AI-powered responses over hardcoded fallbacks:
        1. Attempts to use Google AI for comprehensive answers
        2. Supplements with web search results when available
        3. Falls back to basic responses only if AI is unavailable
        
        Args:
            query: User's general question about telecom concepts
            
        Returns:
            AI-enhanced response with additional search context when available
        """
        # Always prioritize Google AI for comprehensive and accurate responses
        if self.google_ai.enabled:
            ai_response = self.google_ai.answer_general_question(query)
            if ai_response:
                # Supplement with web search results for additional context
                search_result = self.google_search.search_telecom_info(query)
                if search_result:
                    return f"{ai_response}\n\n{search_result}"
                return ai_response
        
        # Minimal fallback responses when AI is unavailable
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['help', 'what can you do', 'capabilities']):
            return self._provide_help()
        
        # Basic fallback for KPI-related queries (minimal to encourage AI setup)
        if any(word in query_lower for word in ['kpi', 'kpis', 'metrics', 'indicators']):
            return ("Available KPIs: SINR, RSRP, DL_Throughput, UL_Throughput, CPU_Utilization, "
                   "Active_Users, RTT, Packet_Loss, Call_Drop_Rate, Handover_Success_Rate.\n\n"
                   "For detailed explanations and technical insights, please configure Google AI "
                   "integration by setting the GOOGLE_API_KEY environment variable.")
        
        # Generic response encouraging proper AI setup
        return ("I can analyze telecom network data and provide technical explanations. "
               "For comprehensive answers to your question, please ensure Google AI integration "
               "is enabled by setting the GOOGLE_API_KEY environment variable. "
               "You can also ask for help to see available analysis options.")
    
    def _provide_help(self) -> str:
        """
        Provide comprehensive help information about agent capabilities.
        
        Returns:
            Formatted help text with examples and feature status information
        """
        help_text = """TELECOM AI AGENT - HELP GUIDE

ANOMALY DETECTION:
• "Find anomalies in SINR for SITE_005"
• "Detect anomalies in CPU utilization for SITE_001"

SITE COMPARISON:
• "Compare SITE_001 and SITE_005 for throughput"
• "Compare SITE_010 and SITE_020 for packet loss"

KPI EXPLANATIONS:
• "What is SINR?"
• "Explain CPU utilization"
• "How does packet loss affect performance?"

ADVANCED TELECOM QUESTIONS:
• "What are 5G network optimization techniques?"
• "How to improve network latency?"
• "What are 3GPP standards for network performance?"

AVAILABLE KPIS:
SINR, RSRP, DL_Throughput, UL_Throughput, CPU_Utilization, 
Active_Users, RTT, Packet_Loss, Call_Drop_Rate, Handover_Success_Rate

AVAILABLE SITES:
SITE_001 to SITE_100 (100 sites total)"""
        
        # Add feature status information
        if self.google_ai.enabled:
            help_text += ("\n\nENHANCED FEATURES ACTIVE:\n"
                         "• Google AI integration for expert telecom insights\n"
                         "• Web search for additional technical information\n"
                         "• Advanced analysis with industry best practices")
        else:
            help_text += ("\n\nBASIC MODE:\n"
                         "• Data analysis capabilities available\n"
                         "• For enhanced AI features, set GOOGLE_API_KEY environment variable")
        
        return help_text


def test_telecom_agent():
    """
    Comprehensive test suite for the Telecom AI Agent.
    
    This function tests:
    - Agent initialization
    - Query processing capabilities
    - Different types of analysis requests
    - Error handling
    """
    print("Initializing Telecom AI Agent...")
    agent = StreamlitTelecomAgent()
    
    # Test cases covering different functionalities
    test_queries = [
        "Find anomalies in SINR for SITE_005",
        "Compare SITE_001 and SITE_003 for CPU utilization", 
        "What is packet loss in telecom networks?",
        "Help"
    ]
    
    print("\nRunning Test Cases:")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)
        
        try:
            response = agent.process_query(query)
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        
        print()
    
    print("Test suite completed.")


if __name__ == "__main__":
    test_telecom_agent()
