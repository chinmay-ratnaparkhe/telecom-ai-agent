#!/usr/bin/env python3
"""
Working Telecom AI Agent
Simplified, direct approach without complex LangChain agent issues
"""

import asyncio
import json
import re
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# LangChain imports for LLM only
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    """Simple context for query processing"""
    intent: str
    site_ids: List[str]
    kpi_names: List[str]
    raw_query: str

class SimpleQueryProcessor:
    """Simple query processor without complex patterns"""
    
    def __init__(self):
        self.kpi_mapping = {
            'sinr': 'SINR',
            'signal': 'SINR',
            'rsrp': 'RSRP',
            'cpu': 'CPU_Utilization',
            'processor': 'CPU_Utilization',
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
            'handover': 'Handover_Success_Rate',
            'active': 'Active_Users',
            'users': 'Active_Users'
        }
    
    def extract_sites(self, query: str) -> List[str]:
        """Extract site IDs from query"""
        sites = []
        patterns = [
            r'SITE_(\d+)',
            r'site[_\s]*(\d+)',
            r'location[_\s]*(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                site_num = match.zfill(3)
                sites.append(f"SITE_{site_num}")
        
        return list(set(sites))
    
    def extract_kpis(self, query: str) -> List[str]:
        """Extract KPI names from query"""
        query_lower = query.lower()
        found_kpis = []
        
        for key, value in self.kpi_mapping.items():
            if key in query_lower:
                found_kpis.append(value)
        
        return list(set(found_kpis))
    
    def detect_intent(self, query: str) -> str:
        """Simple intent detection"""
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
        """Process query and return context"""
        return QueryContext(
            intent=self.detect_intent(query),
            site_ids=self.extract_sites(query),
            kpi_names=self.extract_kpis(query),
            raw_query=query
        )

class WorkingTelecomAgent:
    """Working Telecom AI Agent with direct processing"""
    
    def __init__(self, api_key: str = None):
        # Load API key
        if not api_key:
            api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Initialize components
        self.query_processor = SimpleQueryProcessor()
        
        # Initialize LLM for explanations
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        
        # Load real data
        self._load_data()
        
        logger.info("Working Telecom Agent initialized successfully")
    
    def _load_data(self):
        """Load the actual telecom data"""
        try:
            data_file = "AD_data_10KPI.csv"
            if os.path.exists(data_file):
                self.data = pd.read_csv(data_file)
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                logger.info(f"Loaded data: {len(self.data)} records from {len(self.data['Site_ID'].unique())} sites")
                print(f"Available KPIs: {[col for col in self.data.columns if col not in ['Date', 'Site_ID']]}")
            else:
                logger.warning(f"Data file {data_file} not found")
                self.data = None
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.data = None
    
    async def process_query(self, query: str) -> str:
        """Process user query with direct approach"""
        try:
            logger.info(f"Processing: {query}")
            
            # Process query
            context = self.query_processor.process_query(query)
            
            # Route to appropriate handler
            if context.intent == 'anomaly_detection':
                return self._detect_anomalies(context)
            elif context.intent == 'comparison':
                return self._compare_sites(context)
            elif context.intent == 'general_query':
                return await self._handle_general_query(query)
            else:
                return self._provide_help()
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return f"I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def _detect_anomalies(self, context: QueryContext) -> str:
        """Detect anomalies in the data"""
        if self.data is None:
            return "No data available for analysis."
        
        # Get site and KPI
        site_id = context.site_ids[0] if context.site_ids else None
        kpi_name = context.kpi_names[0] if context.kpi_names else None
        
        if not site_id:
            return "Please specify a site ID (e.g., SITE_003, SITE_001). Available sites: SITE_001 to SITE_100"
        
        if not kpi_name:
            available_kpis = [col for col in self.data.columns if col not in ['Date', 'Site_ID']]
            return f"Please specify a KPI. Available KPIs: {', '.join(available_kpis)}"
        
        # Filter data
        site_data = self.data[self.data['Site_ID'] == site_id]
        
        if site_data.empty:
            return f"No data found for {site_id}."
        
        if kpi_name not in site_data.columns:
            available_kpis = [col for col in site_data.columns if col not in ['Date', 'Site_ID']]
            return f"KPI {kpi_name} not found. Available: {', '.join(available_kpis)}"
        
        # Anomaly detection
        kpi_values = site_data[kpi_name].dropna()
        if kpi_values.empty:
            return f"No valid data for {kpi_name} in {site_id}"
        
        # Statistical anomaly detection
        mean_val = kpi_values.mean()
        std_val = kpi_values.std()
        threshold = 2 * std_val
        
        # Find anomalies
        anomaly_mask = abs(site_data[kpi_name] - mean_val) > threshold
        anomalies = site_data[anomaly_mask & site_data[kpi_name].notna()]
        
        if anomalies.empty:
            return f"No anomalies detected for {kpi_name} in {site_id}. Analyzed {len(site_data)} records."
        
        # Format response
        anomaly_dates = anomalies['Date'].dt.strftime('%Y-%m-%d').tolist()
        anomaly_values = anomalies[kpi_name].tolist()
        
        result = f"Detected {len(anomalies)} anomalies ({len(anomalies)/len(site_data)*100:.1f}%) for {kpi_name} in {site_id}.\n\n"
        result += "Anomalous dates and values:\n"
        
        for date, value in zip(anomaly_dates[:5], anomaly_values[:5]):  # Show first 5
            result += f"- {date}: {value:.2f}\n"
        
        if len(anomalies) > 5:
            result += f"... and {len(anomalies) - 5} more anomalies"
        
        return result
    
    def _compare_sites(self, context: QueryContext) -> str:
        """Compare multiple sites"""
        if self.data is None:
            return "No data available for comparison."
        
        sites = context.site_ids
        kpi_name = context.kpi_names[0] if context.kpi_names else None
        
        if len(sites) < 2:
            return "Please specify at least 2 sites to compare (e.g., 'compare SITE_001 and SITE_003')."
        
        if not kpi_name:
            return "Please specify a KPI to compare."
        
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
                        'count': len(kpi_values)
                    }
        
        if not comparison_data:
            return f"No valid data found for {kpi_name} in the specified sites."
        
        result = f"Site Comparison for {kpi_name}:\n\n"
        for site_id, stats in comparison_data.items():
            result += f"{site_id}: Mean={stats['mean']:.2f}, Latest={stats['latest']:.2f}, Records={stats['count']}\n"
        
        return result
    
    async def _handle_general_query(self, query: str) -> str:
        """Handle general questions"""
        try:
            # Use LLM for general telecom questions
            prompt = f"Answer this telecom question concisely in 2-3 sentences: {query}"
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            return "I can help with telecom data analysis. Try asking about anomalies or comparisons."
    
    def _provide_help(self) -> str:
        """Provide help information"""
        return """I can help you analyze telecom network data! Try these examples:

• "Find anomalies in SINR for SITE_003"
• "Find anomalies in CPU utilization for SITE_001"  
• "Compare SITE_001 and SITE_005 for throughput"
• "What is SINR?"

Available KPIs: SINR, RSRP, DL_Throughput, UL_Throughput, CPU_Utilization, Active_Users, RTT, Packet_Loss, Call_Drop_Rate, Handover_Success_Rate

Available Sites: SITE_001 to SITE_100"""

async def main():
    """Test the working agent"""
    try:
        agent = WorkingTelecomAgent()
        
        # Test queries
        test_queries = [
            "Find anomalies in SINR for SITE_003",
            "Find anomalies in CPU utilization for SITE_001",
            "What is SINR?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("=" * 50)
            response = await agent.process_query(query)
            print(f"Response: {response}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
