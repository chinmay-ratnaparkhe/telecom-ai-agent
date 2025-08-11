"""
Streamlit Web UI for Telecom AI Platform

This module provides a user    elif page == "Data Analysis":
        show_data_analysis_page()
    elif page == "Anomaly Detection":
        show_anomaly_detection_page()
    elif page == "AI Chat":ndly web interface for the telecom AI platform
using Streamlit, allowing users to upload data, train models, detect anomalies,
and interact with the conversational AI agent.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import io
import sys
from pathlib import Path

# Add platform to path
sys.path.append('..')

from telecom_ai_platform.main import TelecomAIPlatform
from telecom_ai_platform.core.config import TelecomConfig
from telecom_ai_platform.utils.visualizer import TelecomVisualizer


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'platform' not in st.session_state:
        st.session_state.platform = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'anomaly_results' not in st.session_state:
        st.session_state.anomaly_results = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def setup_platform():
    """Initialize the telecom AI platform"""
    if st.session_state.platform is None:
        config = TelecomConfig()
        st.session_state.platform = TelecomAIPlatform(config)
        st.success("Platform initialized successfully!")


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Telecom AI Platform",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("Telecom AI Platform v2.0")
    st.markdown("### Intelligent Network Analysis & Anomaly Detection")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Data Analysis", "Anomaly Detection", "AI Chat", "Settings"]
    )
    
    # Initialize platform
    setup_platform()
    
    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Data Analysis":
        show_data_analysis_page()
    elif page == "Anomaly Detection":
        show_anomaly_detection_page()
    elif page == "AI Chat":
        show_chat_page()
    elif page == "Settings":
        show_settings_page()


def show_home_page():
    """Display the home page"""
    st.header("Welcome to Telecom AI Platform v2.0")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Platform Features")
        features = [
            "Advanced Anomaly Detection",
            "Conversational AI Interface", 
            "Interactive Visualizations",
            "RESTful API Server",
            "Real-time Performance Monitoring",
            "Modular Architecture"
        ]
        for feature in features:
            st.write(f"- {feature}")
    
    with col2:
        st.subheader("Quick Start")
        st.write("1. Upload your telecom data")
        st.write("2. Train anomaly detection models")
        st.write("3. Analyze network performance")
        st.write("4. Chat with AI for insights")
        
        if st.button("View Platform Status"):
            if st.session_state.platform:
                status = st.session_state.platform.get_status()
                st.json(status)
    
    # Platform statistics
    st.subheader("Platform Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Loaded", "Yes" if st.session_state.data is not None else "No")
    
    with col2:
        data_count = len(st.session_state.data) if st.session_state.data is not None else 0
        st.metric("Records", f"{data_count:,}")
    
    with col3:
        anomaly_count = len([r for r in st.session_state.anomaly_results if r.is_anomaly]) if st.session_state.anomaly_results else 0
        st.metric("Anomalies", f"{anomaly_count:,}")
    
    with col4:
        models_loaded = "Yes" if st.session_state.platform and st.session_state.platform.agent.anomaly_detector.is_fitted else "No"
        st.metric("Models Trained", models_loaded)


def show_data_analysis_page():
    """Display data analysis page"""
    st.header("Data Analysis")
    
    # File upload
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your telecom network data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Read the data
            data = pd.read_csv(uploaded_file)
            
            # Process through platform
            if st.session_state.platform:
                processed_data = st.session_state.platform.trainer.data_processor.process_pipeline(data)
                st.session_state.data = processed_data
                
                st.success(f"Data loaded successfully! {len(processed_data):,} records processed.")
                
                # Display data summary
                st.subheader("Data Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"- Records: {len(processed_data):,}")
                    st.write(f"- Columns: {len(processed_data.columns)}")
                    st.write(f"- Sites: {processed_data['Site_ID'].nunique() if 'Site_ID' in processed_data.columns else 'N/A'}")
                    
                    if 'Date' in processed_data.columns:
                        st.write(f"- Date Range: {processed_data['Date'].min()} to {processed_data['Date'].max()}")
                
                with col2:
                    st.write("**Available KPIs:**")
                    config = st.session_state.platform.config
                    available_kpis = [kpi for kpi in config.data.kpi_columns if kpi in processed_data.columns]
                    for kpi in available_kpis:
                        st.write(f"- {kpi}")
                
                # Display sample data
                st.subheader("Data Preview")
                st.dataframe(processed_data.head(10))
                
                # Data quality analysis
                st.subheader("🔬 Data Quality")
                quality_report = st.session_state.platform.trainer.data_processor.analyze_data_quality(data)
                
                quality_col1, quality_col2 = st.columns(2)
                with quality_col1:
                    for key, value in list(quality_report.items())[:len(quality_report)//2]:
                        st.metric(key.replace('_', ' ').title(), value)
                
                with quality_col2:
                    for key, value in list(quality_report.items())[len(quality_report)//2:]:
                        st.metric(key.replace('_', ' ').title(), value)
        
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    # Data visualization
    if st.session_state.data is not None:
        st.subheader("📈 Data Visualization")
        
        # KPI selection
        config = st.session_state.platform.config
        available_kpis = [kpi for kpi in config.data.kpi_columns if kpi in st.session_state.data.columns]
        
        selected_kpi = st.selectbox("Select KPI to visualize:", available_kpis)
        
        if selected_kpi:
            # Create trend plot
            fig = go.Figure()
            
            y_data = st.session_state.data[selected_kpi]
            
            if 'Date' in st.session_state.data.columns:
                x_data = pd.to_datetime(st.session_state.data['Date'])
            else:
                x_data = st.session_state.data.index
            
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name=selected_kpi,
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title=f'{selected_kpi} Trends',
                xaxis_title='Time' if 'Date' in st.session_state.data.columns else 'Sample Index',
                yaxis_title=selected_kpi,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader("Statistics")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("Mean", f"{y_data.mean():.2f}")
            with stats_col2:
                st.metric("Std Dev", f"{y_data.std():.2f}")
            with stats_col3:
                st.metric("Min", f"{y_data.min():.2f}")
            with stats_col4:
                st.metric("Max", f"{y_data.max():.2f}")


def show_anomaly_detection_page():
    """Display anomaly detection page"""
    st.header("Anomaly Detection")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload and process data first in the Data Analysis page.")
        return
    
    # Training section
    st.subheader("Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Train Anomaly Detection Models"):
            with st.spinner("Training models..."):
                try:
                    # Train models
                    detector = st.session_state.platform.trainer.train_anomaly_detector(st.session_state.data)
                    st.success("Models trained successfully!")
                    
                    # Show model summary
                    model_summary = detector.get_model_summary()
                    st.write("**Trained Models:**")
                    for kpi, details in model_summary.items():
                        st.write(f"- {kpi}: {details['algorithm']} (threshold: {details['threshold']:.4f})")
                
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    with col2:
        if st.button("📂 Load Pre-trained Models"):
            try:
                st.session_state.platform.load_models()
                st.success("Pre-trained models loaded!")
            except Exception as e:
                st.error(f"❌ Failed to load models: {str(e)}")
    
    # Anomaly detection section
    st.subheader("🔎 Detect Anomalies")
    
    if st.session_state.platform.agent.anomaly_detector.is_fitted:
        detection_col1, detection_col2 = st.columns(2)
        
        with detection_col1:
            # KPI selection for detection
            config = st.session_state.platform.config
            available_kpis = [kpi for kpi in config.data.kpi_columns if kpi in st.session_state.data.columns]
            selected_kpi = st.selectbox("Select KPI for anomaly detection:", ["All KPIs"] + available_kpis)
            
        with detection_col2:
            # Site selection
            if 'Site_ID' in st.session_state.data.columns:
                available_sites = st.session_state.data['Site_ID'].unique().tolist()
                selected_site = st.selectbox("Select Site:", ["All Sites"] + available_sites)
            else:
                selected_site = "All Sites"
        
        if st.button("Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                try:
                    # Configure detection parameters
                    kpi_param = None if selected_kpi == "All KPIs" else selected_kpi
                    site_param = None if selected_site == "All Sites" else selected_site
                    
                    # Detect anomalies
                    anomaly_results = st.session_state.platform.agent.anomaly_detector.detect_anomalies(
                        st.session_state.data,
                        kpi_name=kpi_param,
                        site_id=site_param
                    )
                    
                    st.session_state.anomaly_results = anomaly_results
                    
                    # Show results summary
                    total_samples = len(anomaly_results)
                    anomalies = [r for r in anomaly_results if r.is_anomaly]
                    
                    st.success(f"Detection completed!")
                    
                    # Results metrics
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    with result_col1:
                        st.metric("Total Samples", f"{total_samples:,}")
                    with result_col2:
                        st.metric("Anomalies", f"{len(anomalies):,}")
                    with result_col3:
                        st.metric("Anomaly Rate", f"{len(anomalies)/total_samples:.2%}")
                    with result_col4:
                        high_severity = sum(1 for a in anomalies if a.severity == 'high')
                        st.metric("High Severity", f"{high_severity:,}")
                
                except Exception as e:
                    st.error(f"❌ Detection failed: {str(e)}")
    
    else:
        st.warning("Please train models first before detecting anomalies.")
    
    # Results visualization
    if st.session_state.anomaly_results:
        st.subheader("Anomaly Results")
        
        # Severity distribution
        anomalies = [r for r in st.session_state.anomaly_results if r.is_anomaly]
        
        if anomalies:
            severity_counts = {}
            for severity in ['low', 'medium', 'high']:
                severity_counts[severity] = sum(1 for a in anomalies if a.severity == severity)
            
            # Pie chart for severity distribution
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                hole=0.3
            )])
            
            fig_pie.update_layout(title="Anomaly Severity Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Top anomalies table
            st.subheader("🔥 Top Anomalies")
            top_anomalies = sorted(anomalies, key=lambda x: x.anomaly_score, reverse=True)[:10]
            
            anomaly_data = []
            for anomaly in top_anomalies:
                anomaly_data.append({
                    'KPI': anomaly.kpi_name,
                    'Site': anomaly.site_id,
                    'Value': f"{anomaly.value:.2f}",
                    'Score': f"{anomaly.anomaly_score:.3f}",
                    'Severity': anomaly.severity.upper(),
                    'Confidence': f"{anomaly.confidence:.1%}"
                })
            
            st.dataframe(pd.DataFrame(anomaly_data))


def show_chat_page():
    """Display conversational AI chat page"""
    st.header("🤖 AI Chat Assistant")
    
    # Chat interface
    st.subheader("💬 Chat with the Telecom AI Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])
    
    # Chat input
    user_message = st.chat_input("Ask me about your telecom network data...")
    
    if user_message:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now()
        })
        
        # Display user message
        st.chat_message("user").write(user_message)
        
        # Generate AI response (simulated for demo)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Simulate AI response based on common queries
                response = generate_demo_response(user_message)
                st.write(response)
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
    
    # Example queries
    st.subheader("💡 Example Queries")
    example_queries = [
        "Show me RSRP trends for the last week",
        "Find anomalies in throughput data",
        "Compare performance between sites",
        "What's the average SINR across all sites?",
        "Analyze data quality issues"
    ]
    
    for query in example_queries:
        if st.button(f"💬 {query}", key=f"example_{query}"):
            # Add example query to chat
            st.session_state.chat_history.append({
                'role': 'user',
                'content': query,
                'timestamp': datetime.now()
            })
            
            response = generate_demo_response(query)
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            })
            
            st.rerun()


def generate_demo_response(user_message):
    """Generate a demo response for the chat interface"""
    message_lower = user_message.lower()
    
    if 'rsrp' in message_lower or 'signal' in message_lower:
        return """📡 **RSRP Analysis Results:**

I've analyzed the RSRP (Reference Signal Received Power) data in your network. Here's what I found:

**Key Findings:**
- Average RSRP: -95.2 dBm across all sites
- Signal quality is generally good with few coverage issues
- Detected 8 anomalies with RSRP values below -110 dBm

**Recommendations:**
- Sites with poor RSRP may need antenna optimization
- Consider cell planning review for coverage gaps
- Monitor trend patterns for proactive maintenance

Would you like me to create a detailed visualization or analyze specific sites?"""
    
    elif 'anomal' in message_lower:
        return """**Anomaly Detection Summary:**

I've performed comprehensive anomaly detection across your network data:

**Anomalies Detected:**
- Total: 45 anomalies found
- High Severity: 12 cases requiring immediate attention
- Medium Severity: 23 cases for monitoring
- Low Severity: 10 statistical outliers

**Top Issues:**
1. RSRP drops at Site_003 (signal quality)
2. CPU utilization spikes at Site_001 (resource constraint)
3. Throughput degradation during peak hours

📈 **Next Steps:**
- Review high-severity anomalies first
- Check equipment status at affected sites
- Consider capacity planning for peak periods

Would you like detailed analysis of any specific anomaly type?"""
    
    elif 'compare' in message_lower or 'sites' in message_lower:
        return """**Site Comparison Analysis:**

I've compared performance across your network sites:

**Best Performing Sites:**
- Site_001: Excellent throughput and signal quality
- Site_005: Consistent performance with low variability
- Site_007: Strong SINR values across all sectors

**Sites Needing Attention:**
- Site_003: Below-average RSRP and throughput
- Site_008: High CPU utilization and packet loss
- Site_012: Inconsistent performance patterns

**Performance Metrics:**
- Average DL Throughput: 42.3 Mbps
- Average UL Throughput: 15.7 Mbps
- Network Availability: 99.2%

Would you like a detailed comparison chart or specific site analysis?"""
    
    elif 'quality' in message_lower or 'data' in message_lower:
        return """🔬 **Data Quality Assessment:**

I've analyzed the quality of your telecom data:

**Data Health Status:**
- Completeness: 96.8% (good)
- Missing Values: 3.2% mostly in optional fields
- Duplicates: 0.1% (minimal)
- Date Range Coverage: Complete for analysis period

**KPI Coverage:**
- RSRP: 99.5% complete
- SINR: 98.9% complete  
- Throughput: 97.2% complete
- CPU Utilization: 95.8% complete

**Recommendations:**
- Data quality is sufficient for reliable analysis
- Consider improving throughput data collection
- Monitor missing value patterns for trends

The data is ready for anomaly detection and performance analysis!"""
    
    else:
        return f"""**AI Assistant Response:**

I understand you're asking about: "{user_message}"

I'm here to help you analyze your telecom network data! I can assist with:

**Data Analysis:**
- KPI trend analysis and performance monitoring
- Site comparison and benchmarking
- Data quality assessment and validation

**Anomaly Detection:**
- Identify performance issues and outliers
- Severity classification and prioritization
- Root cause analysis suggestions

**Insights & Reporting:**
- Generate comprehensive network reports
- Performance trend identification
- Capacity planning recommendations

Please ask me specific questions about your network data, and I'll provide detailed analysis and actionable insights!

**Try asking:** "Show me throughput trends" or "Find CPU anomalies"
"""


def show_settings_page():
    """Display settings and configuration page"""
    st.header("Settings & Configuration")
    
    if st.session_state.platform:
        config = st.session_state.platform.config
        
        # Configuration display
        st.subheader("🔧 Current Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write("**Directories:**")
            st.code(f"Data: {config.data_dir}")
            st.code(f"Models: {config.models_dir}")
            st.code(f"Logs: {config.logs_dir}")
            
            st.write("**Model Parameters:**")
            st.code(f"Contamination Rate: {config.model.contamination_rate}")
            st.code(f"AutoEncoder Epochs: {config.model.autoencoder_params['epochs']}")
        
        with config_col2:
            st.write("**KPI Columns:**")
            for kpi in config.data.kpi_columns:
                st.code(kpi)
            
            st.write("**AI Configuration:**")
            st.code(f"Model: {config.ai_agent.model_name}")
            st.code(f"Temperature: {config.ai_agent.temperature}")
        
        # System information
        st.subheader("ℹ️ System Information")
        
        system_info = {
            "Python Version": sys.version.split()[0],
            "Platform Status": "Ready" if st.session_state.platform else "Not Initialized",
            "Data Loaded": "Yes" if st.session_state.data is not None else "No",
            "Models Trained": "Yes" if st.session_state.platform.agent.anomaly_detector.is_fitted else "No",
            "Chat History": f"{len(st.session_state.chat_history)} messages"
        }
        
        for key, value in system_info.items():
            st.metric(key, value)
        
        # Reset options
        st.subheader("🔄 Reset Options")
        
        reset_col1, reset_col2, reset_col3 = st.columns(3)
        
        with reset_col1:
            if st.button("🗑️ Clear Data"):
                st.session_state.data = None
                st.session_state.anomaly_results = None
                st.success("Data cleared!")
        
        with reset_col2:
            if st.button("💬 Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
        
        with reset_col3:
            if st.button("🔄 Reset All"):
                st.session_state.data = None
                st.session_state.anomaly_results = None
                st.session_state.chat_history = []
                st.session_state.platform = None
                st.success("All data reset!")
                st.rerun()


if __name__ == "__main__":
    main()
