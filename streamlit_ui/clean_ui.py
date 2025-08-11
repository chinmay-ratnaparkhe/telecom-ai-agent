import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import subprocess
import sys
import re
import os
import requests
import json
import base64
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the streamlit-compatible agent
from streamlit_telecom_agent import StreamlitTelecomAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="Telecom AI Agent",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base Styling */
    .main > div {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        color: #333;
        font-size: 1.2rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    /* Chat Styling */
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .agent-message {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    /* Enhanced AI Response Styling */
    .ai-enhanced {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f5e8 100%);
        border: 1px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Feature Status Styling */
    .feature-active {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    .feature-basic {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'selected_site' not in st.session_state:
    st.session_state.selected_site = None
if 'selected_kpi' not in st.session_state:
    st.session_state.selected_kpi = None

# --- Data Loading Functions ---
@st.cache_data
def load_telecom_data():
    """Load and cache telecom data"""
    try:
        data = pd.read_csv("AD_data_10KPI.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except FileNotFoundError:
        # Generate sample data if file not found
        return generate_sample_data()

def generate_sample_data():
    """Generate sample telecom data for demo"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    sites = [f"SITE_{i:03d}" for i in range(1, 101)]
    sectors = ['A', 'B', 'C', 'D']
    
    data = []
    for date in dates:
        for site in sites[:20]:  # Limit to 20 sites for demo
            for sector in sectors[:3]:  # 3 sectors per site
                row = {
                    'Date': date,
                    'Site_ID': site,
                    'Sector_ID': f"{site}_SECTOR_{sector}",
                    'SINR': np.random.normal(25, 5),
                    'DL_Throughput': np.random.normal(50, 10),
                    'UL_Throughput': np.random.normal(20, 5),
                    'CPU_Utilization': np.random.normal(45, 15),
                    'Call_Drop_Rate': np.random.exponential(0.02),
                    'Handover_Success_Rate': np.random.normal(0.95, 0.05),
                    'RTT': np.random.exponential(20),
                    'RSRP': np.random.normal(-80, 10),
                    'Active_Users': np.random.poisson(100),
                    'Packet_Loss': np.random.exponential(0.01)
                }
                data.append(row)
    
    return pd.DataFrame(data)

# --- Analysis Functions ---
def detect_anomalies(data, site_id, kpi, threshold=2):
    """Simple anomaly detection using statistical methods"""
    site_data = data[data['Site_ID'] == site_id]
    if site_data.empty or kpi not in site_data.columns:
        return pd.DataFrame()
    
    kpi_values = site_data[kpi].dropna()
    mean_val = kpi_values.mean()
    std_val = kpi_values.std()
    
    anomalies = site_data[abs(site_data[kpi] - mean_val) > threshold * std_val]
    return anomalies

def create_kpi_visualization(data, site_id, kpi):
    """Create interactive visualization for KPI data"""
    site_data = data[data['Site_ID'] == site_id].copy()
    
    if site_data.empty:
        return None
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'{kpi} Trends', 'Anomaly Detection'],
        vertical_spacing=0.1
    )
    
    # Main trend line
    fig.add_trace(
        go.Scatter(
            x=site_data['Date'],
            y=site_data[kpi],
            mode='lines+markers',
            name=f'{kpi} Values',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Anomaly detection
    anomalies = detect_anomalies(data, site_id, kpi)
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies['Date'],
                y=anomalies[kpi],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ),
            row=1, col=1
        )
    
    # Statistical thresholds
    mean_val = site_data[kpi].mean()
    std_val = site_data[kpi].std()
    
    fig.add_hline(
        y=mean_val, line_dash="dash", line_color="green",
        annotation_text="Mean", row=1, col=1
    )
    fig.add_hline(
        y=mean_val + 2*std_val, line_dash="dash", line_color="orange",
        annotation_text="Upper Threshold", row=1, col=1
    )
    fig.add_hline(
        y=mean_val - 2*std_val, line_dash="dash", line_color="orange",
        annotation_text="Lower Threshold", row=1, col=1
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=site_data[kpi],
            name='Distribution',
            nbinsx=30,
            marker_color='lightblue',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{kpi} Analysis for {site_id}',
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

# --- Chat Interface Functions ---
@st.cache_resource
def get_telecom_agent():
    """Initialize and cache the telecom agent"""
    try:
        return StreamlitTelecomAgent()
    except Exception as e:
        st.error(f"Failed to initialize AI agent: {e}")
        return None

def process_chat_query(query, data):
    """Process chat queries using the streamlit-compatible agent"""
    agent = get_telecom_agent()
    if agent is None:
        return "AI agent is not available. Please check your configuration."
    
    try:
        # Use the synchronous method for Streamlit
        response = agent.process_query(query)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

def process_anomaly_query(query, data):
    """Process anomaly detection queries"""
    # Extract site and KPI from query (simplified)
    sites = data['Site_ID'].unique()
    kpis = ['SINR', 'DL_Throughput', 'UL_Throughput', 'CPU_Utilization', 
            'Call_Drop_Rate', 'Handover_Success_Rate', 'RTT', 'RSRP', 
            'Active_Users', 'Packet_Loss']
    
    found_site = None
    found_kpi = None
    
    for site in sites:
        if site.lower() in query.lower():
            found_site = site
            break
    
    for kpi in kpis:
        if kpi.lower().replace('_', ' ') in query.lower():
            found_kpi = kpi
            break
    
    if not found_site:
        found_site = sites[0]  # Default to first site
    if not found_kpi:
        found_kpi = 'SINR'  # Default to SINR
    
    anomalies = detect_anomalies(data, found_site, found_kpi)
    
    if anomalies.empty:
        return f"No anomalies detected for {found_kpi} in {found_site} using 2-sigma threshold."
    else:
        count = len(anomalies)
        percentage = (count / len(data[data['Site_ID'] == found_site])) * 100
        return f"Detected {count} anomalies ({percentage:.1f}%) for {found_kpi} in {found_site}. Anomalous dates: {', '.join(anomalies['Date'].dt.strftime('%Y-%m-%d').tolist()[:5])}{'...' if count > 5 else ''}"

def process_visualization_query(query, data):
    """Process visualization requests"""
    return "I can create visualizations for KPI data. Please use the sidebar controls to select a site and KPI, then view the generated chart."

def process_definition_query(query):
    """Process definition/explanation queries"""
    definitions = {
        'sinr': "SINR (Signal-to-Interference-plus-Noise Ratio) measures the quality of a wireless signal by comparing the signal power to the interference and noise power. Higher values indicate better signal quality.",
        'throughput': "Throughput refers to the actual data transfer rate achieved in a network. DL (Downlink) is from tower to device, UL (Uplink) is from device to tower.",
        'cpu': "CPU Utilization measures the percentage of processing capacity being used by network equipment. High utilization may indicate system stress.",
        'rtt': "Round Trip Time (RTT) measures the time for a signal to travel from source to destination and back. Lower values indicate better responsiveness.",
        'rsrp': "RSRP (Reference Signal Received Power) measures the power of LTE reference signals. It indicates signal strength and coverage quality."
    }
    
    query_lower = query.lower()
    for term, definition in definitions.items():
        if term in query_lower:
            return definition
    
    return "I can explain various telecom KPIs including SINR, Throughput, CPU Utilization, RTT, and RSRP. What would you like to know about?"

def process_general_query(query, data):
    """Process general queries"""
    total_sites = data['Site_ID'].nunique()
    total_records = len(data)
    date_range = f"{data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}"
    
    return f"I'm analyzing telecom network data with {total_records:,} records from {total_sites} sites covering the period {date_range}. I can help with anomaly detection, visualizations, and KPI explanations. What would you like to explore?"

# --- Main Application ---
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Telecom AI Agent with Google Integration</h1>
        <p>Advanced Network Monitoring, Analysis & AI-Powered Insights Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading telecom data...'):
        data = load_telecom_data()
        st.session_state.data_loaded = True
    
    # Sidebar
    with st.sidebar:
        st.header("Network Analysis Controls")
        
        # Site selection
        sites = sorted(data['Site_ID'].unique())
        selected_site = st.selectbox(
            "Select Site",
            sites,
            index=0 if sites else None
        )
        st.session_state.selected_site = selected_site
        
        # KPI selection
        kpis = ['SINR', 'DL_Throughput', 'UL_Throughput', 'CPU_Utilization', 
                'Call_Drop_Rate', 'Handover_Success_Rate', 'RTT', 'RSRP', 
                'Active_Users', 'Packet_Loss']
        selected_kpi = st.selectbox("Select KPI", kpis)
        st.session_state.selected_kpi = selected_kpi
        
        # Date range
        if not data.empty:
            date_range = st.date_input(
                "Date Range",
                value=[data['Date'].min().date(), data['Date'].max().date()],
                min_value=data['Date'].min().date(),
                max_value=data['Date'].max().date()
            )
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("Quick Actions")
        if st.button("Detect Anomalies"):
            anomalies = detect_anomalies(data, selected_site, selected_kpi)
            count = len(anomalies)
            st.write(f"Found {count} anomalies")
        
        if st.button("Generate Report"):
            st.write("Report generation feature coming soon!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Data visualization
        st.subheader(f"KPI Analysis: {selected_kpi}")
        
        if selected_site and selected_kpi:
            fig = create_kpi_visualization(data, selected_site, selected_kpi)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for {selected_site}")
        
        # Data overview
        st.subheader("Data Overview")
        
        if not data.empty:
            site_data = data[data['Site_ID'] == selected_site]
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric(
                    "Records",
                    len(site_data),
                    delta=None
                )
            
            with metrics_col2:
                if selected_kpi in site_data.columns:
                    avg_val = site_data[selected_kpi].mean()
                    st.metric(
                        f"Avg {selected_kpi}",
                        f"{avg_val:.2f}",
                        delta=None
                    )
            
            with metrics_col3:
                anomaly_count = len(detect_anomalies(data, selected_site, selected_kpi))
                st.metric(
                    "Anomalies",
                    anomaly_count,
                    delta=None
                )
            
            with metrics_col4:
                if selected_kpi in site_data.columns:
                    std_val = site_data[selected_kpi].std()
                    st.metric(
                        f"Std Dev",
                        f"{std_val:.2f}",
                        delta=None
                    )
    
    with col2:
        # Chat interface
        st.subheader("AI Assistant with Google Integration")
        
        # Enhanced features info
        agent = get_telecom_agent()
        if agent and hasattr(agent, 'google_ai') and agent.google_ai.enabled:
            st.info("✨ **Enhanced Features Active:**\n- Google Gemini AI for expert insights\n- Web search for additional information\n- Advanced telecom analysis")
        else:
            st.warning("**Basic Mode:** Set GOOGLE_API_KEY for enhanced AI features")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="agent-message">
                        <strong>AI Agent:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        user_query = st.text_input(
            "Ask about network performance, anomalies, telecom technology, or industry best practices:",
            placeholder="e.g., 'What is call drop rate for site 2?' or 'Find anomalies in tower 5 CPU utilization'"
        )
        
        if st.button("Send") and user_query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Process query
            with st.spinner("Processing with AI enhancement..."):
                response = process_chat_query(user_query, data)
            
            # Add agent response
            st.session_state.messages.append({"role": "agent", "content": response})
            
            # Rerun to update chat display
            st.rerun()
        
        # Enhanced sample questions
        st.markdown("**Try These Enhanced Queries:**")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.markdown("**Data Analysis:**")
            analysis_questions = [
                "What is call drop rate for site 2",
                "Find anomalies in SINR for location 7",
                "Compare site 1 and site 5 throughput",
                "Show CPU utilization for tower 3"
            ]
            
            for question in analysis_questions:
                if st.button(question, key=f"analysis_{question}"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    response = process_chat_query(question, data)
                    st.session_state.messages.append({"role": "agent", "content": response})
                    st.rerun()
        
        with col2b:
            st.markdown("**AI-Enhanced Questions:**")
            ai_questions = [
                "What are 5G network optimization techniques?",
                "How does MIMO improve performance?",
                "What are 3GPP standards for 5G?"
            ]
            
            for question in ai_questions:
                if st.button(question, key=f"ai_{question}"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    response = process_chat_query(question, data)
                    st.session_state.messages.append({"role": "agent", "content": response})
                    st.rerun()

if __name__ == "__main__":
    main()
