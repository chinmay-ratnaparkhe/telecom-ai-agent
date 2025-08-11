#!/usr/bin/env python3
"""
Fixed MCP Server with Complete Visualization Support and Governance
Properly structured with all functions implemented
"""


# CRITICAL: Environment setup MUST be first
import asyncio
import json
import os

import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict
os.environ['MPLBACKEND'] = 'Agg'  # Force non-GUI backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF warnings

# Threading configuration for matplotlib
import matplotlib
matplotlib.use('Agg')  # Ensure Agg backend is used

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle
import os
import sys
from fastapi import HTTPException, Request
import math
import re

# Memory and system monitoring imports
import gc
import psutil
import threading
import time

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from scipy import stats
import io
import base64

# Logging and error handling
import logging
import traceback
from functools import wraps

# Configure logging with better format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('telecom_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Memory monitoring with better thresholds
def monitor_memory():
    """Improved memory monitoring function"""
    while True:
        try:
            process = psutil.Process()
            memory_percent = process.memory_percent()
            if memory_percent > 75:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")
        time.sleep(60)

# Start memory monitor
memory_thread = threading.Thread(target=monitor_memory, daemon=True)
memory_thread.start()

def robust_error_handler(func):
    """Enhanced error handler with better error reporting"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Function args: {args[:2]}...")
            logger.error(f"Function kwargs keys: {list(kwargs.keys())}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return json.dumps({
                "success": False,
                "error": f"Function {func.__name__} failed",
                "details": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            })
    return wrapper

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Simple autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(input_dim // 2, 1), max(input_dim // 4, 1)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(max(input_dim // 4, 1), max(input_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(input_dim // 2, 1), input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Global variables
df = None
autoencoder_model = None
scaler = None
model_input_dim = None

# Utility functions
def safe_float_conversion(value):
    """Safely convert values to JSON-serializable format"""
    if pd.isna(value) or value is None:
        return None
    elif math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    elif math.isnan(value):
        return None
    else:
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

def json_serial(obj):
    """JSON serializer for complex objects"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif hasattr(obj, 'date'):
        return str(obj.date())
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return safe_float_conversion(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (float, int)) and (math.isinf(obj) or math.isnan(obj)):
        return safe_float_conversion(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def clean_kpi_data(data, kpi_name):
    """Clean and prepare KPI data"""
    try:
        if kpi_name not in data.columns:
            raise ValueError(f"KPI {kpi_name} not found in data")
        
        kpi_data = data[kpi_name].copy()
        kpi_data = kpi_data.replace([np.inf, -np.inf], np.nan)
        kpi_data = kpi_data.dropna()
        
        if len(kpi_data) > 10:
            mean_val = kpi_data.mean()
            std_val = kpi_data.std()
            if std_val > 0:
                mask = abs(kpi_data - mean_val) <= 5 * std_val
                kpi_data = kpi_data[mask]
        
        return kpi_data
    except Exception as e:
        logger.error(f"Data cleaning error for {kpi_name}: {e}")
        return pd.Series([], dtype=float)

def parse_date_range(date_range: str) -> tuple:
    """Parse date range with better error handling"""
    try:
        if date_range.lower() == "last_week":
            end_date = pd.to_datetime("2024-02-29")
            start_date = end_date - timedelta(days=7)
        elif date_range.lower() == "last_month":
            end_date = pd.to_datetime("2024-02-29")
            start_date = end_date - timedelta(days=30)
        elif ":" in date_range:
            start_str, end_str = date_range.split(":")
            start_date = pd.to_datetime(start_str.strip())
            end_date = pd.to_datetime(end_str.strip())
        else:
            start_date = pd.to_datetime("2024-01-01")
            end_date = pd.to_datetime("2024-02-29")
        
        return start_date, end_date
    except Exception as e:
        logger.error(f"Date parsing error: {e}")
        return pd.to_datetime("2024-01-01"), pd.to_datetime("2024-02-29")

def extract_kpi_from_text(text: str) -> Optional[str]:
    """Extract KPI name from text"""
    if not text:
        return None
    
    text_lower = text.lower()
    kpi_mappings = {
        'sinr': 'SINR',
        'signal': 'SINR',
        'throughput': 'DL_Throughput',
        'download': 'DL_Throughput',
        'dl_throughput': 'DL_Throughput',
        'cpu': 'CPU_Utilization',
        'active_users': 'Active_Users',
        'users': 'Active_Users',
        'latency': 'RTT',
        'rtt': 'RTT',
        'packet_loss': 'Packet_Loss',
        'rsrp': 'RSRP',
        'call_drop': 'Call_Drop_Rate',
        'handover': 'Handover_Success_Rate',
        'ul_throughput': 'UL_Throughput',
        'upload': 'UL_Throughput'
    }
    
    for key, kpi_name in kpi_mappings.items():
        if key in text_lower:
            return kpi_name
    
    return None

def extract_site_from_text(text: str) -> Optional[str]:
    """Extract site ID from text"""
    patterns = [
        r'site[_\s]*(\d+)',
        r'SITE_(\d+)',
        r'Site_(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            site_num = int(match.group(1))
            site_id = f"SITE_{site_num:03d}"
            return site_id
    
    return None

def load_data_and_models():
    """Simplified and robust data loading"""
    global df, autoencoder_model, scaler, model_input_dim

    try:
        logger.info("Starting simplified data and model loading...")
        
        # Check if data file exists
        data_path = "telecom_mcp_server/data/processed_kpi_data.csv"
        if not os.path.exists(data_path):
            alternative_paths = [
                "data/processed_kpi_data.csv",
                "../data/processed_kpi_data.csv",
                "processed_kpi_data.csv"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    data_path = alt_path
                    break
            else:
                logger.error("No data file found. Creating sample data...")
                create_sample_data()
                data_path = "sample_telecom_data.csv"

        # Load dataset
        logger.info(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Optimize memory usage
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Available columns: {df.columns.tolist()}")

        autoencoder_model = None
        scaler = None
        model_input_dim = None
        
        logger.info("Basic data loading completed successfully")
        gc.collect()

    except Exception as e:
        logger.error(f"Critical error loading data: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        create_sample_data()

def create_sample_data():
    """Create sample telecom data if none exists"""
    try:
        logger.info("Creating sample telecom data...")
        
        dates = pd.date_range('2024-01-01', '2024-02-29', freq='H')
        sites = [f'SITE_{i:03d}' for i in range(1, 11)]
        
        data = []
        for date in dates[:1000]:
            for site in sites[:5]:
                record = {
                    'Date': date,
                    'Site_ID': site,
                    'Sector_ID': f'{site}_S1',
                    'RSRP': np.random.normal(-85, 10),
                    'SINR': np.random.normal(15, 5),
                    'DL_Throughput': np.random.gamma(2, 10),
                    'UL_Throughput': np.random.gamma(1.5, 5),
                    'Call_Drop_Rate': np.random.beta(1, 20) * 100,
                    'RTT': np.random.gamma(2, 5),
                    'CPU_Utilization': np.random.beta(3, 2) * 100,
                    'Active_Users': np.random.poisson(50),
                    'Handover_Success_Rate': np.random.beta(20, 2) * 100,
                    'Packet_Loss': np.random.beta(1, 50) * 100
                }
                data.append(record)
        
        global df
        df = pd.DataFrame(data)
        df.to_csv('sample_telecom_data.csv', index=False)
        logger.info(f"Created sample data: {df.shape}")
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")

# API Models
class ToolRequest(BaseModel):
    tool: str
    parameters: Dict[str, Any] = {}

# Visualization helper functions
def get_kpi_unit(kpi_name):
    """Get unit for KPI"""
    units = {
        'SINR': 'SINR (dB)',
        'DL_Throughput': 'Download Throughput (Mbps)',
        'UL_Throughput': 'Upload Throughput (Mbps)',
        'CPU_Utilization': 'CPU Utilization (%)',
        'Active_Users': 'Active Users',
        'RTT': 'RTT (ms)',
        'Packet_Loss': 'Packet Loss (%)',
        'RSRP': 'RSRP (dBm)',
        'Call_Drop_Rate': 'Call Drop Rate (%)',
        'Handover_Success_Rate': 'Handover Success Rate (%)'
    }
    return units.get(kpi_name, kpi_name)

def get_chart_description(viz_type, kpi_name, site_id, anomaly_count):
    """Generate description for the chart"""
    descriptions = {
        'time_series': f"Time series plot showing {kpi_name} values over time" + 
                      (f" for {site_id}" if site_id else "") + 
                      f". Found {anomaly_count} anomalies highlighted by severity.",
        'distribution': f"Distribution analysis of {kpi_name} showing histogram with KDE and box plot. " +
                       f"Anomalies ({anomaly_count} found) are marked in red.",
        'heatmap': f"Anomaly heatmap for {kpi_name} across all sites over time. " +
                   "Darker colors indicate more anomalous behavior.",
        'comparison': f"Site comparison for {kpi_name} showing top performing sites " +
                     "with time series and bar chart comparisons.",
        'correlation': "Correlation matrix showing relationships between all KPIs. " +
                      "Red indicates positive correlation, blue indicates negative."
    }
    return descriptions.get(viz_type, f"Visualization of {kpi_name} data")

# Also modify the visualization function to ensure proper cleanup
# In your create_time_series_plot and other plotting functions, add:
def create_time_series_plot(data, kpi_name, anomalies, site_id):
    """Create enhanced time series plot with anomalies highlighted"""
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Sort data by date
    plot_data = data.sort_values('Date')
    
    # Plot main line
    ax.plot(plot_data['Date'], plot_data[kpi_name], 'b-', alpha=0.7, linewidth=2, label='Normal')
    
    # Add rolling average for trend
    if len(plot_data) > 7:
        rolling_mean = plot_data[kpi_name].rolling(window=7, center=True).mean()
        ax.plot(plot_data['Date'], rolling_mean, 'g--', alpha=0.8, linewidth=2, label='7-day Average')
    
    # Highlight anomalies with different colors by severity
    if anomalies:
        anomaly_dates = [pd.to_datetime(a['timestamp']) for a in anomalies]
        anomaly_values = [a['value'] for a in anomalies]
        anomaly_severities = [a.get('severity', 'medium') for a in anomalies]
        
        # Color mapping
        colors = {'low': '#FFD700', 'medium': '#FF8C00', 'high': '#FF0000'}
        sizes = {'low': 80, 'medium': 100, 'high': 120}
        
        for severity in ['low', 'medium', 'high']:
            severity_mask = [s == severity for s in anomaly_severities]
            if any(severity_mask):
                severity_dates = [d for d, m in zip(anomaly_dates, severity_mask) if m]
                severity_values = [v for v, m in zip(anomaly_values, severity_mask) if m]
                ax.scatter(severity_dates, severity_values, c=colors[severity], 
                          s=sizes[severity], label=f'{severity.capitalize()} severity',
                          edgecolors='black', linewidth=1, zorder=5, alpha=0.8)
    
    # Formatting
    ax.set_title(f'{kpi_name} Time Series Analysis' + (f' - {site_id}' if site_id else ''), 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(get_kpi_unit(kpi_name), fontsize=12)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=45, ha='right')
    
    # Add statistics box
    stats_text = f'Mean: {plot_data[kpi_name].mean():.2f}\n'
    stats_text += f'Std: {plot_data[kpi_name].std():.2f}\n'
    stats_text += f'Anomalies: {len(anomalies)}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def create_distribution_plot(data, kpi_name, anomalies):
    """Create distribution plot with anomalies"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    kpi_values = data[kpi_name].dropna()
    
    # Histogram with KDE
    ax1.hist(kpi_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # Add KDE
    kde = stats.gaussian_kde(kpi_values)
    x_range = np.linspace(kpi_values.min(), kpi_values.max(), 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # Mark anomalies on histogram
    if anomalies:
        anomaly_values = [a['value'] for a in anomalies]
        for val in anomaly_values:
            ax1.axvline(x=val, color='red', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel(get_kpi_unit(kpi_name), fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'{kpi_name} Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot with anomalies
    box_data = [kpi_values]
    bp = ax2.boxplot(box_data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('darkblue')
    bp['medians'][0].set_linewidth(2)
    
    # Add anomaly points
    if anomalies:
        anomaly_values = [a['value'] for a in anomalies]
        anomaly_severities = [a.get('severity', 'medium') for a in anomalies]
        
        colors = {'low': '#FFD700', 'medium': '#FF8C00', 'high': '#FF0000'}
        for val, sev in zip(anomaly_values, anomaly_severities):
            ax2.scatter([1], [val], c=colors[sev], s=100, zorder=5,
                       edgecolors='black', linewidth=1)
    
    ax2.set_ylabel(get_kpi_unit(kpi_name), fontsize=12)
    ax2.set_title(f'{kpi_name} Box Plot with Anomalies', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([kpi_name])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add anomaly legend
    if anomalies:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FFD700', label='Low'),
                          Patch(facecolor='#FF8C00', label='Medium'),
                          Patch(facecolor='#FF0000', label='High')]
        ax2.legend(handles=legend_elements, title='Anomaly Severity', loc='upper right')
    
    plt.tight_layout()
    return fig

def create_anomaly_heatmap(data, kpi_name, start_date, end_date):
    """Create heatmap showing anomaly patterns across sites and time"""
    # Filter data
    mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
    filtered_data = data[mask].copy()
    
    # Create pivot table
    filtered_data['Date_str'] = filtered_data['Date'].dt.strftime('%Y-%m-%d')
    pivot = filtered_data.pivot_table(
        values=kpi_name,
        index='Site_ID',
        columns='Date_str',
        aggfunc='mean'
    )
    
    # Calculate z-scores for anomaly detection
    z_scores = np.abs(stats.zscore(pivot.fillna(pivot.mean()).values, axis=1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create heatmap
    im = ax.imshow(z_scores, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Anomaly Score (Z-score)', fontsize=12)
    
    # Add title
    ax.set_title(f'Anomaly Heatmap for {kpi_name}\n(Darker = More Anomalous)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Site ID', fontsize=12)
    
    # Add grid
    ax.set_xticks(np.arange(len(pivot.columns))-.5, minor=True)
    ax.set_yticks(np.arange(len(pivot.index))-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    return fig

def create_site_comparison_plot(data, kpi_name, date_range):
    """Create comparison plot for multiple sites"""
    start_date, end_date = parse_date_range(date_range)
    mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
    filtered_data = data[mask].copy()
    
    # Get top 10 sites by mean KPI value
    site_stats = filtered_data.groupby('Site_ID')[kpi_name].agg(['mean', 'std', 'count'])
    top_sites = site_stats.nlargest(10, 'mean').index.tolist()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series comparison for top sites
    for site in top_sites[:5]:  # Show top 5 for clarity
        site_data = filtered_data[filtered_data['Site_ID'] == site].sort_values('Date')
        ax1.plot(site_data['Date'], site_data[kpi_name], linewidth=2, 
                label=site, alpha=0.8)
    
    ax1.set_title(f'Top 5 Sites - {kpi_name} Time Series', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel(get_kpi_unit(kpi_name), fontsize=12)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Bar chart comparison
    site_means = site_stats.loc[top_sites, 'mean']
    site_stds = site_stats.loc[top_sites, 'std']
    
    x_pos = np.arange(len(top_sites))
    bars = ax2.bar(x_pos, site_means, yerr=site_stds, capsize=5,
                   color='skyblue', edgecolor='darkblue', linewidth=1)
    
    # Color bars by performance
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(top_sites, rotation=45, ha='right')
    ax2.set_ylabel(f'Mean {get_kpi_unit(kpi_name)}', fontsize=12)
    ax2.set_title(f'Top 10 Sites - Mean {kpi_name} Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, site_means)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + site_stds.iloc[i],
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_correlation_plot(data):
    """Create correlation matrix for all KPIs"""
    # Select only numeric KPI columns
    kpi_columns = ['SINR', 'DL_Throughput', 'UL_Throughput', 'CPU_Utilization',
                   'Active_Users', 'RTT', 'Packet_Loss', 'RSRP',
                   'Call_Drop_Rate', 'Handover_Success_Rate']
    
    available_kpis = [col for col in kpi_columns if col in data.columns]
    correlation_data = data[available_kpis].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))
    sns.heatmap(correlation_data, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": .8},
                annot=True, fmt='.2f', ax=ax)
    
    ax.set_title('KPI Correlation Matrix', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main tool functions with robust error handling
@robust_error_handler
def detect_anomalies(
    site_id: str = None,
    kpi_name: str = None,
    date_range: str = "last_week",
    method: str = "statistical",
    contamination: float = 0.05,
    text: str = ""
) -> str:
    """Simplified anomaly detection with statistical methods only"""
    
    try:
        logger.info(f"Starting anomaly detection: site={site_id}, kpi={kpi_name}")
        
        if df is None:
            return json.dumps({"error": "Dataset not loaded", "success": False})
        
        # Extract parameters from text
        if text:
            if not kpi_name:
                kpi_name = extract_kpi_from_text(text)
            if not site_id:
                site_id = extract_site_from_text(text)
        
        if not kpi_name:
            kpi_name = 'DL_Throughput'
        
        # Parse date range
        start_date, end_date = parse_date_range(date_range)
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Filter data
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        if site_id:
            mask &= (df['Site_ID'] == site_id)
        
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            return json.dumps({
                "error": "No data found for specified criteria",
                "success": False,
                "criteria": {"site_id": site_id, "kpi_name": kpi_name, "date_range": date_range}
            })
        
        logger.info(f"Processing {len(filtered_df)} rows")
        
        # Validate KPI exists
        if kpi_name not in filtered_df.columns:
            available_kpis = [col for col in filtered_df.columns 
                            if col not in ['Date', 'Site_ID', 'Sector_ID']]
            return json.dumps({
                "error": f"KPI '{kpi_name}' not found",
                "success": False,
                "available_kpis": available_kpis
            })
        
        # Clean data
        kpi_data = clean_kpi_data(filtered_df, kpi_name)
        
        if len(kpi_data) < 10:
            return json.dumps({
                "error": "Insufficient clean data for analysis",
                "success": False,
                "samples_found": len(kpi_data)
            })
        
        # Statistical anomaly detection using IQR method
        results = []
        try:
            Q1 = kpi_data.quantile(0.25)
            Q3 = kpi_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find anomalies
            anomaly_mask = (kpi_data < lower_bound) | (kpi_data > upper_bound)
            anomalies = kpi_data[anomaly_mask]
            
            # Create results
            for idx in anomalies.index:
                if len(results) >= 50:  # Limit results
                    break
                
                row = filtered_df.loc[idx]
                anomaly_value = float(kpi_data.loc[idx])
                
                # Calculate severity
                distance = min(abs(anomaly_value - lower_bound), abs(anomaly_value - upper_bound))
                severity = "high" if distance > 2 * IQR else "medium" if distance > IQR else "low"
                
                results.append({
                    "timestamp": row['Date'].isoformat(),
                    "site_id": str(row['Site_ID']),
                    "kpi_name": kpi_name,
                    "value": anomaly_value,
                    "anomaly_score": float(distance / IQR) if IQR > 0 else 1.0,
                    "is_anomaly": True,
                    "severity": severity,
                    "method": "IQR",
                    "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                })
            
            summary = {
                "success": True,
                "total_samples": len(filtered_df),
                "clean_samples": len(kpi_data),
                "anomalies_detected": len(results),
                "anomaly_rate": len(results) / len(kpi_data) if len(kpi_data) > 0 else 0,
                "method": "IQR_statistical",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "kpi_analyzed": kpi_name,
                "site_analyzed": site_id,
                "statistics": {
                    "mean": safe_float_conversion(kpi_data.mean()),
                    "median": safe_float_conversion(kpi_data.median()),
                    "std": safe_float_conversion(kpi_data.std()),
                    "Q1": safe_float_conversion(Q1),
                    "Q3": safe_float_conversion(Q3),
                    "IQR": safe_float_conversion(IQR)
                },
                "anomalies": results
            }
            
            logger.info(f"Anomaly detection completed: {len(results)} anomalies found")
            return json.dumps(summary, indent=2, default=json_serial)
            
        except Exception as detection_error:
            logger.error(f"Detection algorithm error: {detection_error}")
            return json.dumps({
                "error": "Detection algorithm failed",
                "success": False,
                "details": str(detection_error)
            })
    
    except Exception as e:
        logger.error(f"Top-level error in detect_anomalies: {e}")
        return json.dumps({
            "error": "Anomaly detection failed",
            "success": False,
            "details": str(e)
        })

@robust_error_handler
def analyze_kpi_trends(
    kpi_name: str = None,
    site_id: str = None,
    date_range: str = "last_week",
    aggregation: str = "mean",
    text: str = ""
) -> str:
    """Analyze KPI trends with improved error handling"""
    
    try:
        logger.info(f"Starting KPI trends analysis: kpi={kpi_name}, site={site_id}")

        if df is None:
            return json.dumps({"error": "Dataset not loaded", "success": False})

        # Extract parameters from text
        if text:
            if not kpi_name:
                kpi_name = extract_kpi_from_text(text)
            if not site_id:
                site_id = extract_site_from_text(text)

        if not kpi_name:
            kpi_name = 'DL_Throughput'

        # Parse date range
        start_date, end_date = parse_date_range(date_range)

        # Filter data
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        if site_id:
            mask &= (df['Site_ID'] == site_id)

        filtered_df = df[mask].copy()

        if filtered_df.empty:
            return json.dumps({
                "error": "No data found for specified criteria",
                "success": False
            })

        if kpi_name not in filtered_df.columns:
            available_kpis = [col for col in filtered_df.columns 
                            if col not in ['Date', 'Site_ID', 'Sector_ID']]
            return json.dumps({
                "error": f"KPI '{kpi_name}' not found",
                "success": False,
                "available_kpis": available_kpis[:10]
            })

        # Clean data
        kpi_data = clean_kpi_data(filtered_df, kpi_name)

        if len(kpi_data) == 0:
            return json.dumps({
                "error": "No clean data available for analysis",
                "success": False
            })

        # Multi-site KPI comparison logic
        if site_id is None:
            # Multi-site analysis: group by Site_ID and aggregate KPI
            grouped = filtered_df.groupby('Site_ID')[kpi_name].agg([
                'mean', 'max', 'min', 'std', 'count'
            ]).fillna(0)

            grouped_safe = {}
            for site, stats in grouped.iterrows():
                grouped_safe[site] = {
                    'mean': safe_float_conversion(stats['mean']),
                    'max': safe_float_conversion(stats['max']),
                    'min': safe_float_conversion(stats['min']),
                    'std': safe_float_conversion(stats['std']),
                    'count': int(stats['count'])
                }

            # Sort by aggregation method
            if aggregation in ['mean', 'max', 'min', 'std']:
                try:
                    sorted_sites = sorted(
                        grouped_safe.items(),
                        key=lambda x: x[1][aggregation] if x[1][aggregation] is not None else -float('inf'),
                        reverse=True
                    )
                    grouped_safe = dict(sorted_sites[:10])
                except Exception as sort_error:
                    logger.warning(f"Sorting error: {sort_error}")

            # Add explicit top site for agent consumption
            top_site = None
            if grouped_safe:
                top_site = next(iter(grouped_safe))

            results = {
                "success": True,
                "kpi_name": kpi_name,
                "analysis_type": "multi_site",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "aggregation": aggregation,
                "site_rankings": grouped_safe,
                "top_site": top_site,
                "overall_stats": {
                    "mean": safe_float_conversion(kpi_data.mean()),
                    "max": safe_float_conversion(kpi_data.max()),
                    "min": safe_float_conversion(kpi_data.min()),
                    "std": safe_float_conversion(kpi_data.std()),
                    "total_samples": len(kpi_data)
                }
            }
        else:
            # Single site analysis
            try:
                daily_trends = filtered_df.groupby(filtered_df['Date'].dt.date)[kpi_name].agg([
                    'mean', 'max', 'min', 'count'
                ]).fillna(0)

                daily_trends_safe = {}
                for date, stats in daily_trends.iterrows():
                    daily_trends_safe[str(date)] = {
                        'mean': safe_float_conversion(stats['mean']),
                        'max': safe_float_conversion(stats['max']),
                        'min': safe_float_conversion(stats['min']),
                        'count': int(stats['count'])
                    }

                results = {
                    "success": True,
                    "kpi_name": kpi_name,
                    "site_id": site_id,
                    "analysis_type": "single_site",
                    "date_range": f"{start_date.date()} to {end_date.date()}",
                    "daily_trends": daily_trends_safe,
                    "overall_stats": {
                        "mean": safe_float_conversion(kpi_data.mean()),
                        "max": safe_float_conversion(kpi_data.max()),
                        "min": safe_float_conversion(kpi_data.min()),
                        "std": safe_float_conversion(kpi_data.std()),
                        "total_samples": len(kpi_data)
                    }
                }
            except Exception as trend_error:
                logger.error(f"Trend calculation error: {trend_error}")
                return json.dumps({
                    "error": "Trend calculation failed",
                    "success": False,
                    "details": str(trend_error)
                })

        logger.info("KPI trends analysis completed successfully")
        return json.dumps(results, indent=2, default=json_serial)

    except Exception as e:
        logger.error(f"KPI trends analysis failed: {e}")
        return json.dumps({
            "error": "KPI trends analysis failed",
            "success": False,
            "details": str(e)
        })

@robust_error_handler
def get_site_summary(site_id: str = None, date_range: str = "last_week", text: str = "") -> str:
    """Get site summary with simplified processing"""
    
    try:
        if df is None:
            return json.dumps({"error": "Dataset not loaded", "success": False})
        
        if text and not site_id:
            site_id = extract_site_from_text(text)
        
        if not site_id:
            return json.dumps({"error": "No site ID specified", "success": False})
        
        start_date, end_date = parse_date_range(date_range)
        
        site_data = df[(df['Site_ID'] == site_id) &
                      (df['Date'] >= start_date) &
                      (df['Date'] <= end_date)].copy()
        
        if site_data.empty:
            return json.dumps({
                "error": f"No data found for site '{site_id}'",
                "success": False
            })
        
        # Calculate KPI statistics
        kpi_columns = [col for col in site_data.columns 
                      if col not in ['Date', 'Site_ID', 'Sector_ID']]
        
        kpi_stats = {}
        for kpi in kpi_columns:
            kpi_data = clean_kpi_data(site_data, kpi)
            if len(kpi_data) > 0:
                kpi_stats[kpi] = {
                    "mean": safe_float_conversion(kpi_data.mean()),
                    "max": safe_float_conversion(kpi_data.max()),
                    "min": safe_float_conversion(kpi_data.min()),
                    "std": safe_float_conversion(kpi_data.std()),
                    "samples": len(kpi_data)
                }
        
        results = {
            "success": True,
            "site_id": site_id,
            "date_range": f"{start_date.date()} to {end_date.date()}",
            "total_samples": len(site_data),
            "sectors": list(site_data['Sector_ID'].unique()) if 'Sector_ID' in site_data.columns else [],
            "kpi_statistics": kpi_stats
        }
        
        return json.dumps(results, indent=2, default=json_serial)
        
    except Exception as e:
        logger.error(f"Site summary failed: {e}")
        return json.dumps({
            "error": "Site summary failed",
            "success": False,
            "details": str(e)
        })

@robust_error_handler
def compare_sites(
    site_ids: str = None,
    kpi_name: str = None,
    date_range: str = "last_week",
    text: str = ""
) -> str:
    """Compare multiple sites"""
    
    try:
        if df is None:
            return json.dumps({"error": "Dataset not loaded", "success": False})
        
        # Extract parameters from text
        if text:
            if not kpi_name:
                kpi_name = extract_kpi_from_text(text)
        
        if not site_ids:
            return json.dumps({"error": "No site IDs specified", "success": False})
        
        if not kpi_name:
            kpi_name = 'DL_Throughput'
        
        # Parse site IDs
        sites = [s.strip() for s in site_ids.split(',')]
        start_date, end_date = parse_date_range(date_range)
        
        comparisons = {}
        for site in sites:
            site_data = df[(df['Site_ID'] == site) &
                          (df['Date'] >= start_date) &
                          (df['Date'] <= end_date)].copy()
            
            if not site_data.empty and kpi_name in site_data.columns:
                kpi_data = clean_kpi_data(site_data, kpi_name)
                
                if len(kpi_data) > 0:
                    comparisons[site] = {
                        "mean": safe_float_conversion(kpi_data.mean()),
                        "max": safe_float_conversion(kpi_data.max()),
                        "min": safe_float_conversion(kpi_data.min()),
                        "std": safe_float_conversion(kpi_data.std()),
                        "samples": len(kpi_data)
                    }
        
        if comparisons:
            # Rank sites by mean value
            ranked_sites = sorted(comparisons.items(), 
                                key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else -float('inf'), 
                                reverse=True)
            
            results = {
                "success": True,
                "kpi_name": kpi_name,
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "sites_compared": len(comparisons),
                "rankings": [{"site_id": site, "stats": stats} for site, stats in ranked_sites],
                "best_site": {
                    "site_id": ranked_sites[0][0],
                    "mean_value": ranked_sites[0][1]['mean']
                } if ranked_sites else None,
                "worst_site": {
                    "site_id": ranked_sites[-1][0],
                    "mean_value": ranked_sites[-1][1]['mean']
                } if ranked_sites else None
            }
        else:
            results = {
                "error": f"No data found for any sites with KPI '{kpi_name}'",
                "success": False
            }
        
        return json.dumps(results, indent=2, default=json_serial)
        
    except Exception as e:
        logger.error(f"Site comparison failed: {e}")
        return json.dumps({
            "error": "Site comparison failed",
            "success": False,
            "details": str(e)
        })


@robust_error_handler
def visualize_anomalies(
    site_id: str = None,
    kpi_name: str = None,
    date_range: str = "last_week",
    visualization_type: str = "time_series",
    text: str = ""
) -> str:
    """Enhanced visualization with multiple chart types"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Ensure Agg backend
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        if df is None:
            return json.dumps({"error": "Dataset not loaded", "success": False})
        
        # Extract parameters
        if text:
            if not kpi_name:
                kpi_name = extract_kpi_from_text(text)
            if not site_id:
                site_id = extract_site_from_text(text)
        
        if not kpi_name:
            kpi_name = 'DL_Throughput'
        
        # Parse date range
        start_date, end_date = parse_date_range(date_range)
        
        # Filter data
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        if site_id and visualization_type != "comparison":
            mask &= (df['Site_ID'] == site_id)
        
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            return json.dumps({"error": "No data for visualization", "success": False})
        
        # Detect anomalies for visualization
        anomaly_params = {
            "site_id": site_id,
            "kpi_name": kpi_name,
            "date_range": date_range,
            "method": "statistical",
            "contamination": 0.05,
            "text": text
        }
        anomaly_result = json.loads(detect_anomalies(**anomaly_params))
        anomalies = anomaly_result.get('anomalies', []) if anomaly_result.get('success') else []
        
        # Create appropriate visualization
        if visualization_type == "time_series":
            fig = create_time_series_plot(filtered_df, kpi_name, anomalies, site_id)
        elif visualization_type == "distribution":
            fig = create_distribution_plot(filtered_df, kpi_name, anomalies)
        elif visualization_type == "heatmap":
            fig = create_anomaly_heatmap(df, kpi_name, start_date, end_date)
        elif visualization_type == "comparison":
            fig = create_site_comparison_plot(df, kpi_name, date_range)
        elif visualization_type == "correlation":
            fig = create_correlation_plot(filtered_df)
        else:
            fig = create_time_series_plot(filtered_df, kpi_name, anomalies, site_id)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close('all')  # Important: close all figures
        plt.clf()  # Clear the current figure
        plt.cla()  # Clear the current axes
        gc.collect()  # Force garbage collection
        
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return json.dumps({
            "success": True,
            "image_base64": img_b64,
            "kpi_name": kpi_name,
            "site_id": site_id,
            "date_range": f"{start_date.date()} to {end_date.date()}",
            "visualization_type": visualization_type,
            "anomaly_count": len(anomalies),
            "chart_description": get_chart_description(visualization_type, kpi_name, site_id, len(anomalies))
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return json.dumps({"error": "Visualization failed", "success": False, "details": str(e)})

@robust_error_handler
def governance_check(
    query: str = "",
    user_id: str = None,
    action: str = "query"
) -> str:
    """Enhanced governance and compliance checking with audit trail"""
    try:
        governance_log = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id or "anonymous",
            "action": action,
            "query": query[:200],  # Limit query length for logging
            "query_hash": hash(query),
            "checks_performed": [],
            "approved": True,
            "warnings": [],
            "recommendations": [],
            "risk_level": "low"
        }
        
        # Check 1: Sensitive data patterns
        sensitive_patterns = [
            (r'password|credential|secret|private.*key', 'credentials'),
            (r'personal.*info|pii|privacy', 'personal_data'),
            (r'financial|payment|credit', 'financial_data'),
            (r'delete.*all|drop.*table|truncate', 'destructive_operation')
        ]
        
        for pattern, category in sensitive_patterns:
            if re.search(pattern, query.lower()):
                governance_log["warnings"].append(f"Potential {category} request detected")
                governance_log["risk_level"] = "high"
                governance_log["checks_performed"].append(f"sensitive_data_check_{category}")
        
        # Check 2: Bulk operations
        bulk_patterns = [
            (r'all.*sites.*all.*kpis', 'full_export'),
            (r'export.*everything|download.*complete', 'bulk_export'),
            (r'no.*limit|unlimited|entire.*dataset', 'unlimited_query')
        ]
        
        for pattern, op_type in bulk_patterns:
            if re.search(pattern, query.lower()):
                governance_log["warnings"].append(f"Bulk operation detected: {op_type}")
                governance_log["recommendations"].append("Consider limiting scope to specific sites/KPIs")
                governance_log["recommendations"].append("Use date ranges to reduce data volume")
                governance_log["risk_level"] = "medium"
                governance_log["checks_performed"].append(f"bulk_operation_check_{op_type}")
        
        # Check 3: Rate limiting
        if action in ["anomaly_detection", "complex_analysis"]:
            governance_log["checks_performed"].append("rate_limit_check")
            governance_log["recommendations"].append("Rate limits: 100 queries/hour, 1000 queries/day")
            
            if action == "complex_analysis":
                governance_log["recommendations"].append("Complex queries may be queued during peak hours (9-5 EST)")
        
        # Check 4: Data modification attempts
        modification_patterns = [
            (r'update.*set|alter.*table', 'schema_modification'),
            (r'delete.*from|remove.*data', 'data_deletion'),
            (r'insert.*into|add.*data', 'data_insertion')
        ]
        
        for pattern, mod_type in modification_patterns:
            if re.search(pattern, query.lower()):
                governance_log["approved"] = False
                governance_log["warnings"].append(f"Data modification attempted: {mod_type}")
                governance_log["risk_level"] = "critical"
                governance_log["checks_performed"].append(f"modification_check_{mod_type}")
        
        # Check 5: Query complexity and resource usage
        complexity_indicators = {
            'high_complexity': len(query) > 500,
            'multiple_operations': query.count('and') + query.count('or') > 5,
            'date_range_full': '2024-01-01:2024-02-29' in query
        }
        
        complexity_score = sum(1 for indicator, present in complexity_indicators.items() if present)
        
        if complexity_score >= 2:
            governance_log["recommendations"].append("Query complexity is high - consider breaking into smaller queries")
            governance_log["checks_performed"].append("complexity_check")
            if governance_log["risk_level"] == "low":
                governance_log["risk_level"] = "medium"
        
        # Check 6: Compliance with data retention
        retention_keywords = ['archive', 'historical', 'backup', '2023', '2022']
        if any(keyword in query.lower() for keyword in retention_keywords):
            governance_log["warnings"].append("Query may involve archived data - verify retention policy")
            governance_log["checks_performed"].append("retention_policy_check")
        
        # Check 7: Geographic restrictions
        geo_restricted = ['eu', 'gdpr', 'california', 'ccpa']
        if any(term in query.lower() for term in geo_restricted):
            governance_log["recommendations"].append("Geographic data restrictions may apply")
            governance_log["checks_performed"].append("geographic_restriction_check")
        
        # Generate governance ID for tracking
        gov_id = f"GOV-{datetime.now().strftime('%Y%m%d%H%M%S')}-{abs(hash(query)) % 10000:04d}"
        
        # Log for audit
        logger.info(f"Governance check {gov_id}: Risk={governance_log['risk_level']}, " +
                   f"Approved={governance_log['approved']}, Checks={len(governance_log['checks_performed'])}")
        
        # Prepare response
        response = {
            "success": True,
            "governance_id": gov_id,
            "approved": governance_log["approved"],
            "risk_level": governance_log["risk_level"],
            "warnings": governance_log["warnings"],
            "recommendations": governance_log["recommendations"],
            "checks_performed": len(governance_log["checks_performed"]),
            "message": "Query approved" if governance_log["approved"] else "Query requires review"
        }
        
        # Add specific messages based on risk level
        if governance_log["risk_level"] == "critical":
            response["message"] = "Query blocked due to critical risk. Please contact admin."
            response["admin_contact"] = "admin@telecom-analytics.com"
        elif governance_log["risk_level"] == "high":
            response["message"] = "Query flagged for high risk. Proceeding with caution."
            response["audit_required"] = True
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Governance check error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return json.dumps({
            "success": False,
            "error": "Governance check failed",
            "details": str(e),
            "approved": True,  # Fail open for availability
            "risk_level": "unknown",
            "message": "Governance check encountered an error - proceeding with standard policies"
        })

@robust_error_handler
def list_available_data() -> str:
    """List available data with validation"""
    try:
        if df is None:
            return json.dumps({
                "error": "Dataset not loaded",
                "success": False,
                "suggestion": "Check if data file exists and server initialization completed"
            })
        
        # Get basic info
        kpi_columns = [col for col in df.columns 
                      if col not in ['Date', 'Site_ID', 'Sector_ID']]
        
        results = {
            "success": True,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "date_range": {
                "start": df['Date'].min().isoformat(),
                "end": df['Date'].max().isoformat()
            },
            "sites": {
                "total_sites": df['Site_ID'].nunique(),
                "sample_sites": df['Site_ID'].unique()[:10].tolist(),
                "site_format": "SITE_XXX (e.g., SITE_001, SITE_002)"
            },
            "kpis": {
                "total_kpis": len(kpi_columns),
                "available_kpis": kpi_columns,
                "description": {
                    "RSRP": "Reference Signal Received Power (-140 to -40 dBm)",
                    "DL_Throughput": "Download Throughput (0+ Mbps)",
                    "CPU_Utilization": "CPU Utilization percentage (0-200%)",
                    "Active_Users": "Number of active users (0+)",
                    "SINR": "Signal to Interference plus Noise Ratio (-50 to +50 dB)",
                    "RTT": "Round Trip Time/Latency (0+ ms)",
                    "Packet_Loss": "Packet Loss Rate (0-100%)",
                    "Call_Drop_Rate": "Call Drop Rate (0-100%)",
                    "Handover_Success_Rate": "Handover Success Rate (0-100%)",
                    "UL_Throughput": "Upload Throughput (0+ Mbps)"
                }
            },
            "sectors": df['Sector_ID'].unique().tolist() if 'Sector_ID' in df.columns else [],
            "server_status": {
                "data_loaded": True,
                "memory_usage": f"{psutil.Process().memory_percent():.1f}%",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return json.dumps(results, indent=2, default=json_serial)
        
    except Exception as e:
        logger.error(f"Failed to get data info: {e}")
        return json.dumps({
            "error": "Failed to get data info",
            "success": False,
            "details": str(e)
        })

# FastAPI Application Setup
app = FastAPI(
    title="Enhanced Telecom KPI Analysis Server",
    description="MCP server with visualization support and governance",
    version="3.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "Enhanced Telecom KPI Analysis Server", 
        "status": "healthy",
        "version": "3.0.0",
        "features": ["anomaly_detection", "visualization", "governance"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    try:
        memory_usage = psutil.Process().memory_percent()
        
        # Test basic data access
        data_test = "passed"
        if df is not None:
            try:
                _ = len(df)
                _ = df.columns.tolist()
            except Exception:
                data_test = "failed"
        else:
            data_test = "not_loaded"
        
        status = {
            "status": "healthy" if data_test == "passed" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "memory_usage_percent": memory_usage,
            "data_status": data_test,
            "data_shape": df.shape if df is not None else None,
            "available_endpoints": [
                "/", "/health", "/call_tool"
            ],
            "available_tools": [
                "detect_anomalies", "analyze_kpi_trends", 
                "get_site_summary", "compare_sites", "list_available_data",
                "visualize_anomalies", "governance_check"
            ]
        }
        return status
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Enhanced global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(f"Headers: {dict(request.headers)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Check server logs for detailed error information"
        }
    )

@app.post("/call_tool")
async def call_tool_endpoint(request: ToolRequest):
    """Enhanced tool endpoint with comprehensive error handling"""
    try:
        logger.info(f"Received tool request: {request.tool}")
        logger.debug(f"Parameters: {request.parameters}")
        
        # Validate tool name
        valid_tools = [
            "detect_anomalies", "analyze_kpi_trends", 
            "get_site_summary", "compare_sites", "list_available_data",
            "visualize_anomalies", "governance_check"
        ]
        
        if request.tool not in valid_tools:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tool: {request.tool}. Valid tools: {valid_tools}"
            )
        
        # Check if data is loaded for data-dependent tools
        if request.tool not in ["list_available_data", "governance_check"] and df is None:
            logger.error("Data not loaded for data-dependent tool")
            raise HTTPException(
                status_code=503,
                detail="Server data not loaded. Please check server initialization."
            )
        
        # Execute tool with timeout
        try:
            result = await asyncio.wait_for(
                call_tool_async(request.tool, request.parameters),
                timeout=120.0  # 2 minute timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Tool {request.tool} timed out")
            raise HTTPException(
                status_code=504,
                detail=f"Tool {request.tool} timed out. Try reducing data range or parameters."
            )
        
        # Parse and validate result
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
                logger.info(f"Tool {request.tool} completed successfully")
                return parsed_result
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing error: {json_err}")
                return {
                    "error": "Result parsing failed",
                    "raw_result": result[:500],  # First 500 chars
                    "json_error": str(json_err)
                }
        else:
            return result
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Tool execution failed",
                "tool": request.tool,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

async def call_tool_async(tool_name: str, parameters: dict):
    """Async tool dispatcher"""
    try:
        if tool_name == "detect_anomalies":
            return detect_anomalies(**parameters)
        elif tool_name == "analyze_kpi_trends":
            return analyze_kpi_trends(**parameters)
        elif tool_name == "get_site_summary":
            return get_site_summary(**parameters)
        elif tool_name == "compare_sites":
            return compare_sites(**parameters)
        elif tool_name == "list_available_data":
            return list_available_data()
        elif tool_name == "visualize_anomalies":
            return visualize_anomalies(**parameters)
        elif tool_name == "governance_check":
            return governance_check(**parameters)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    except Exception as e:
        logger.error(f"Tool {tool_name} execution error: {e}")
        raise

# Server startup
if __name__ == "__main__":
    print("Starting Enhanced Telecom KPI MCP Server...")
    print("Version 3.0 with Visualization and Governance")
    print("New Features: Advanced visualizations, governance checks")
    
    try:
        print(" Loading data and models...")
        load_data_and_models()
        print(" Server initialization completed!")
    except Exception as e:
        print(f"  Initialization warning: {e}")
        print(" Starting server in limited mode...")
    
    print("\n Starting HTTP server on localhost:8000...")
    print(" Available tools:")
    print("   - detect_anomalies: Statistical anomaly detection")
    print("   - analyze_kpi_trends: KPI trend analysis")
    print("   - get_site_summary: Site performance summary")
    print("   - compare_sites: Multi-site comparison")
    print("   - list_available_data: Data inventory")
    print("   - visualize_anomalies: Generate charts and graphs")
    print("   - governance_check: Query compliance checking")
    
    # Production-ready server configuration
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        timeout_keep_alive=60,
        timeout_graceful_shutdown=30,
        limit_concurrency=10,  # Increased for visualization support
        limit_max_requests=1000,  # Increased limit
        workers=1,  # Single worker for memory management
        reload=False,  # Disable reload in production
        server_header=False,  # Security
        date_header=False   # Security
    )
    
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n Server shutdown requested")
    except Exception as e:
        print(f" Server error: {e}")
        logger.error(f"Server startup failed: {e}")
    finally:
        print(" Server stopped")