import asyncio
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from telecom_ai_agent import clean_kpi_data

def clean_kpi_data(df, kpi_name):
    """
    Cleans KPI data by dropping NaNs and filtering out extreme outliers (simple implementation).
    Replace this with your actual cleaning logic if needed.
    """
    kpi_series = df[kpi_name].dropna()
    # Remove outliers beyond 3 standard deviations
    if len(kpi_series) > 0:
        mean = kpi_series.mean()
        std = kpi_series.std()
        kpi_series = kpi_series[(kpi_series > mean - 3*std) & (kpi_series < mean + 3*std)]
    return kpi_series


# Global data and models (loaded on startup)
df = None
autoencoder_model = None
scaler = None
model_input_dim = None

class AnomalyResult(BaseModel):
    timestamp: str
    site_id: str
    kpi_name: str
    anomaly_score: float
    is_anomaly: bool
    severity: str

class KPIAnalysis(BaseModel):
    kpi_name: str
    site_id: Optional[str]
    statistics: Dict[str, float]
    trend: str
    anomaly_count: int

class ToolRequest(BaseModel):
    tool: str
    parameters: Dict[str, Any] = {}

# Define the Autoencoder architecture (EXACT match to your training code)
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_data_and_models():
    """Load dataset and trained models on server startup"""
    global df, autoencoder_model, scaler, model_input_dim

    try:
        # Load your processed dataset
        df = pd.read_csv("telecom_mcp_server/data/processed_kpi_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])

        # Determine input dimension from the saved model FIRST
        try:
            # Load the state dict to determine the correct input dimension
            temp_state_dict = torch.load("telecom_mcp_server/models/autoencoder_model.pth", map_location='cpu')
            # The first layer weight shape tells us the actual input dimension
            first_layer_weight = temp_state_dict['encoder.0.weight']
            input_dim = first_layer_weight.shape[1]
            print(f"✅ Detected input_dim from saved model: {input_dim}")

            # Initialize autoencoder model with the CORRECT input dimension
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            autoencoder_model = Autoencoder(input_dim=input_dim)

            # Load the state dict - this should work now
            autoencoder_model.load_state_dict(temp_state_dict)
            autoencoder_model.to(device)
            autoencoder_model.eval()
            print(f"✅ Loaded autoencoder model on {device} with input_dim={input_dim}")

        except Exception as model_error:
            print(f"⚠️ Could not load autoencoder model: {model_error}")
            print(" Continuing without autoencoder (statistical methods only)")
            autoencoder_model = None
            input_dim = None

        # Load scaler
        try:
            with open("telecom_mcp_server/models/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            print("✅ Loaded scaler")
        except Exception as scaler_error:
            print(f"⚠️ Could not load scaler: {scaler_error}")
            print(" Continuing without scaler")
            scaler = None

        print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✅ Available sites: {df['Site_ID'].nunique()}")
        print(f"✅ Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Print available KPI columns for debugging
        kpi_columns = [col for col in df.columns if col not in ['Date', 'Site_ID', 'Sector_ID'] and not col.endswith('_scaled')]
        print(f"✅ Available KPI columns: {kpi_columns[:10]}...")

        # Store the input_dim globally for use in anomaly detection
        model_input_dim = input_dim

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif hasattr(obj, 'date'):
        return str(obj.date())
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def parse_date_range(date_range: str) -> tuple:
    """Parse date range string into start and end dates - Fixed for your data"""
    if date_range.lower() == "last_week":
        # Use your actual data range instead of current date
        end_date = pd.to_datetime("2024-02-29")
        start_date = end_date - timedelta(days=7)
    elif date_range.lower() == "last_month":
        end_date = pd.to_datetime("2024-02-29")
        start_date = end_date - timedelta(days=30)
    elif ":" in date_range:
        start_str, end_str = date_range.split(":")
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
    else:
        # Default to your actual data range
        start_date = pd.to_datetime("2024-01-01")
        end_date = pd.to_datetime("2024-02-29")

    return start_date, end_date

def extract_kpi_from_text(text: str) -> Optional[str]:
    """Extract KPI name from text input - Updated with your actual data structure"""
    global df
    if df is None:
        return None
    
    # Get available KPI columns
    kpi_columns = [col for col in df.columns if col not in ['Date', 'Site_ID', 'Sector_ID'] and not col.endswith('_scaled')]
    
    text_lower = text.lower()
    
    # KPI mappings based on your actual data structure
    kpi_mappings = {
        'sinr': 'SINR',
        'signal': 'SINR',
        'throughput': 'DL_Throughput',
        'download': 'DL_Throughput',
        'dl_throughput': 'DL_Throughput',
        'cpu': 'CPU_Utilization',
        'cpu_utilization': 'CPU_Utilization',
        'active_users': 'Active_Users',
        'users': 'Active_Users',
        'latency': 'RTT',  # RTT is your latency metric
        'rtt': 'RTT',
        'round_trip': 'RTT',
        'packet_loss': 'Packet_Loss',
        'packet': 'Packet_Loss',
        'loss': 'Packet_Loss',
        'rsrp': 'RSRP',
        'reference_signal': 'RSRP',
        'call_drop': 'Call_Drop_Rate',
        'drop_rate': 'Call_Drop_Rate',
        'handover': 'Handover_Success_Rate',
        'ul_throughput': 'UL_Throughput',
        'upload': 'UL_Throughput'
    }
    
    # Try to find matching KPI
    for key, kpi_name in kpi_mappings.items():
        if key in text_lower:
            return kpi_name
    
    # If no mapping found, try direct match
    for kpi in kpi_columns:
        if kpi.lower() in text_lower or any(word in kpi.lower() for word in text_lower.split()):
            return kpi
    
    return None

def extract_site_from_text(text: str) -> Optional[str]:
    """Extract site ID from text input - Updated for SITE_XXX format"""
    import re
    
    # Look for patterns like "site 1", "SITE_001", etc.
    patterns = [
        r'site[_\s]*(\d+)',
        r'SITE_(\d+)',
        r'Site_(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            site_num = int(match.group(1))
            # Your data uses SITE_XXX format
            site_id = f"SITE_{site_num:03d}"
            
            # Verify this site exists in data
            if df is not None and site_id in df['Site_ID'].unique():
                return site_id
    
    return None

def extract_multiple_sites_from_text(text: str) -> List[str]:
    """Extract multiple site IDs from text"""
    import re
    
    site_ids = []
    
    # Look for patterns like "sites 1, 2, and 3"
    numbers = re.findall(r'\b(\d+)\b', text)
    
    for num in numbers:
        site_id = f"SITE_{int(num):03d}"
        if df is not None and site_id in df['Site_ID'].unique():
            site_ids.append(site_id)
    
    return site_ids

# Tool functions with improved parameter parsing
def detect_anomalies(
    site_id: str = None,
    kpi_name: str = None,
    date_range: str = "last_week",
    method: str = "statistical",
    contamination: float = 0.05,
    text: str = ""  # Add text parameter for parsing
) -> str:
    """Detect anomalies in telecom KPI data using trained models."""
    try:
        # Extract parameters from text if not provided
        if text:
            if not kpi_name:
                kpi_name = extract_kpi_from_text(text)
            if not site_id:
                site_id = extract_site_from_text(text)
        
        # Parse date range
        start_date, end_date = parse_date_range(date_range)

        # Filter data
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        if site_id:
            mask &= (df['Site_ID'] == site_id)

        filtered_df = df[mask].copy()

        if filtered_df.empty:
            return json.dumps({"error": "No data found for specified criteria"})

        results = []

        if method == "autoencoder" and autoencoder_model is not None:
            # Get the original KPI columns (no rolling, lag, or scaled versions)
            original_kpis = ['RSRP', 'DL_Throughput', 'Call_Drop_Rate', 'RTT', 'CPU_Utilization', 
                           'Active_Users', 'SINR', 'UL_Throughput', 'Handover_Success_Rate', 'Packet_Loss']
            
            # Use the model's actual input dimension
            expected_input_dim = model_input_dim
            analysis_kpis = original_kpis[:expected_input_dim] if len(original_kpis) >= expected_input_dim else original_kpis

            # Prepare data exactly like in training (fillna(0))
            X = filtered_df[analysis_kpis].fillna(0).values

            # Ensure input shape matches model expectation
            if X.shape[1] < expected_input_dim:
                pad_width = expected_input_dim - X.shape[1]
                X = np.pad(X, ((0,0),(0,pad_width)), mode='constant')
            elif X.shape[1] > expected_input_dim:
                X = X[:, :expected_input_dim]

            print(f"🔍 Processing {X.shape[0]} samples with {X.shape[1]} features for autoencoder")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

            with torch.no_grad():
                reconstructed = autoencoder_model(X_tensor)
                reconstruction_errors = torch.mean((reconstructed - X_tensor)**2, dim=1).cpu().numpy()

            # Determine anomaly threshold
            threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
            anomalies = reconstruction_errors > threshold

            print(f"🎯 Threshold: {threshold:.6f}, Anomalies found: {sum(anomalies)}/{len(anomalies)}")

            # Create results
            for i, is_anomaly in enumerate(anomalies):
                if is_anomaly:
                    row = filtered_df.iloc[i]
                    severity = "high" if reconstruction_errors[i] > threshold * 1.5 else "medium"

                    results.append(AnomalyResult(
                        timestamp=row['Date'].isoformat(),
                        site_id=str(row['Site_ID']),
                        kpi_name=kpi_name or "multiple_kpis",
                        anomaly_score=float(reconstruction_errors[i]),
                        is_anomaly=True,
                        severity=severity
                    ).dict())

        else:  # Statistical method
            from sklearn.ensemble import IsolationForest

            # Use the extracted or specified KPI
            if not kpi_name:
                # Default to first available KPI
                original_kpis = ['RSRP', 'DL_Throughput', 'Call_Drop_Rate', 'RTT', 'CPU_Utilization', 
                               'Active_Users', 'SINR', 'UL_Throughput', 'Handover_Success_Rate', 'Packet_Loss']
                kpi_name = original_kpis[0]

            if kpi_name not in filtered_df.columns:
                return json.dumps({"error": f"KPI '{kpi_name}' not found in data"})

            kpi_data = clean_kpi_data(filtered_df, kpi_name)
            if len(kpi_data) < 10:
                return json.dumps({"error": "Insufficient data for statistical analysis"})

            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(kpi_data.values.reshape(-1, 1))
            anomaly_scores = iso_forest.decision_function(kpi_data.values.reshape(-1, 1))

            # Create results
            for i, (is_anomaly, score) in enumerate(zip(anomaly_labels == -1, anomaly_scores)):
                if is_anomaly:
                    idx = kpi_data.index[i]
                    row = filtered_df.loc[idx]

                    results.append(AnomalyResult(
                        timestamp=row['Date'].isoformat(),
                        site_id=str(row['Site_ID']),
                        kpi_name=kpi_name,
                        anomaly_score=float(abs(score)),
                        is_anomaly=True,
                        severity="medium"
                    ).dict())

        summary = {
            "total_samples": len(filtered_df),
            "anomalies_detected": len(results),
            "anomaly_rate": len(results) / len(filtered_df) if len(filtered_df) > 0 else 0,
            "method": method,
            "date_range": f"{start_date.date()} to {end_date.date()}",
            "kpi_analyzed": kpi_name,
            "site_analyzed": site_id,
            "anomalies": results[:50]
        }

        return json.dumps(summary, indent=2, default=json_serial)

    except Exception as e:
        return json.dumps({"error": f"Anomaly detection failed: {str(e)}"}, default=json_serial)

def analyze_kpi_trends(
    kpi_name: str = None,
    site_id: str = None,
    date_range: str = "last_week",
    aggregation: str = "mean",
    text: str = ""  # Add text parameter for parsing
) -> str:
    """Analyze KPI trends and statistics over time."""
    try:
        # Extract parameters from text if not provided
        if text:
            if not kpi_name:
                kpi_name = extract_kpi_from_text(text)
            if not site_id:
                site_id = extract_site_from_text(text)
        
        # Default to DL_Throughput if no KPI specified
        if not kpi_name:
            kpi_name = 'DL_Throughput'
        
        # Parse date range - handle explicit ranges
        if ":" in date_range:
            start_str, end_str = date_range.split(":")
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
        else:
            start_date, end_date = parse_date_range(date_range)

        print(f"🔍 Analyzing {kpi_name} from {start_date} to {end_date}, site: {site_id}")

        # Filter data
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        if site_id:
            mask &= (df['Site_ID'] == site_id)

        filtered_df = df[mask].copy()
        
        print(f"📊 Found {len(filtered_df)} rows after filtering")

        if filtered_df.empty:
            return json.dumps({"error": f"No data found for the specified criteria. Date range: {start_date} to {end_date}, Site: {site_id}"})

        if kpi_name not in filtered_df.columns:
            available_kpis = [col for col in filtered_df.columns if col not in ['Date', 'Site_ID', 'Sector_ID'] and not col.endswith('_scaled')]
            return json.dumps({"error": f"KPI '{kpi_name}' not found in data. Available KPIs: {available_kpis[:10]}"})

        # Calculate statistics
        kpi_data = clean_kpi_data(filtered_df, kpi_name)
        
        print(f"📈 Analyzing {len(kpi_data)} data points for {kpi_name}")

        # Group by site if analyzing multiple sites
        if site_id is None:
            grouped = filtered_df.groupby('Site_ID')[kpi_name].agg([
                'mean', 'max', 'min', 'std', 'count'
            ]).round(4)

            # Sort by the specified aggregation
            if aggregation in grouped.columns:
                grouped = grouped.sort_values(aggregation, ascending=False)

            results = {
                "kpi_name": kpi_name,
                "analysis_type": "multi_site",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "aggregation": aggregation,
                "site_rankings": grouped.head(10).to_dict('index'),
                "overall_stats": {
                    "mean": float(kpi_data.mean()),
                    "max": float(kpi_data.max()),
                    "min": float(kpi_data.min()),
                    "std": float(kpi_data.std()),
                    "total_samples": len(kpi_data)
                }
            }
        else:
            # Single site analysis
            daily_trends = filtered_df.groupby(filtered_df['Date'].dt.date)[kpi_name].agg([
                'mean', 'max', 'min', 'count'
            ]).round(4)

            # Convert date index to strings for JSON serialization
            daily_trends.index = daily_trends.index.astype(str)

            results = {
                "kpi_name": kpi_name,
                "site_id": site_id,
                "analysis_type": "single_site",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "daily_trends": daily_trends.to_dict('index'),
                "overall_stats": {
                    "mean": float(kpi_data.mean()),
                    "max": float(kpi_data.max()),
                    "min": float(kpi_data.min()),
                    "std": float(kpi_data.std()),
                    "total_samples": len(kpi_data)
                }
            }

        return json.dumps(results, indent=2, default=json_serial)

    except Exception as e:
        print(f"❌ Error in analyze_kpi_trends: {e}")
        return json.dumps({"error": f"KPI analysis failed: {str(e)}"}, default=json_serial)

def get_site_summary(site_id: str = None, date_range: str = "last_week", text: str = "") -> str:
    """Get comprehensive summary for a specific site."""
    try:
        # Extract site ID from text if not provided
        if text and not site_id:
            site_id = extract_site_from_text(text)
        
        if not site_id:
            return json.dumps({"error": "No site ID specified"})
        
        # Parse date range
        start_date, end_date = parse_date_range(date_range)

        # Filter data for the site
        site_data = df[(df['Site_ID'] == site_id) &
                      (df['Date'] >= start_date) &
                      (df['Date'] <= end_date)].copy()

        if site_data.empty:
            return json.dumps({"error": f"No data found for site '{site_id}'"})

        # Get original KPI columns (not derived features)
        original_kpis = ['RSRP', 'DL_Throughput', 'Call_Drop_Rate', 'RTT', 'CPU_Utilization', 
                        'Active_Users', 'SINR', 'UL_Throughput', 'Handover_Success_Rate', 'Packet_Loss']

        # Calculate KPI statistics
        kpi_stats = {}
        for kpi in original_kpis:
            if kpi in site_data.columns:
                kpi_data = site_data[kpi].dropna()
                if len(kpi_data) > 0:
                    kpi_stats[kpi] = {
                        "mean": float(kpi_data.mean()),
                        "max": float(kpi_data.max()),
                        "min": float(kpi_data.min()),
                        "std": float(kpi_data.std()),
                        "samples": len(kpi_data)
                    }

        # Run anomaly detection for this site
        anomaly_result = detect_anomalies(site_id=site_id, date_range=date_range)
        anomaly_data = json.loads(anomaly_result)

        results = {
            "site_id": site_id,
            "date_range": f"{start_date.date()} to {end_date.date()}",
            "total_samples": len(site_data),
            "sectors": list(site_data['Sector_ID'].unique()) if 'Sector_ID' in site_data.columns else [],
            "kpi_statistics": kpi_stats,
            "anomaly_summary": {
                "total_anomalies": len(anomaly_data.get('anomalies', [])),
                "anomaly_rate": anomaly_data.get('anomaly_rate', 0),
                "recent_anomalies": anomaly_data.get('anomalies', [])[:5]
            }
        }

        return json.dumps(results, indent=2, default=json_serial)

    except Exception as e:
        return json.dumps({"error": f"Site summary failed: {str(e)}"}, default=json_serial)

def compare_sites(
    site_ids: str = None,
    kpi_name: str = None,
    date_range: str = "last_week",
    text: str = ""
) -> str:
    """Compare multiple sites for a specific KPI."""
    try:
        # Extract parameters from text if not provided
        if text:
            if not site_ids:
                extracted_sites = extract_multiple_sites_from_text(text)
                if extracted_sites:
                    site_ids = ','.join(extracted_sites)
            if not kpi_name:
                kpi_name = extract_kpi_from_text(text)
        
        if not site_ids:
            return json.dumps({"error": "No site IDs specified for comparison"})
        
        if not kpi_name:
            kpi_name = 'DL_Throughput'  # Default KPI
        
        # Parse site IDs
        sites = [s.strip() for s in site_ids.split(',')]

        # Parse date range
        start_date, end_date = parse_date_range(date_range)

        # Compare sites
        comparisons = {}

        for site in sites:
            site_data = df[(df['Site_ID'] == site) &
                          (df['Date'] >= start_date) &
                          (df['Date'] <= end_date)].copy()

            if not site_data.empty and kpi_name in site_data.columns:
                kpi_data = site_data[kpi_name].dropna()

                if len(kpi_data) > 0:
                    comparisons[site] = {
                        "mean": float(kpi_data.mean()),
                        "max": float(kpi_data.max()),
                        "min": float(kpi_data.min()),
                        "std": float(kpi_data.std()),
                        "samples": len(kpi_data)
                    }

        # Rank sites by mean value
        if comparisons:
            ranked_sites = sorted(comparisons.items(), key=lambda x: x[1]['mean'], reverse=True)

            results = {
                "kpi_name": kpi_name,
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "sites_compared": len(comparisons),
                "rankings": [{"site_id": site, "stats": stats} for site, stats in ranked_sites],
                "best_site": ranked_sites[0][0] if ranked_sites else None,
                "worst_site": ranked_sites[-1][0] if ranked_sites else None
            }
        else:
            results = {"error": f"No data found for any sites with KPI '{kpi_name}'"}

        return json.dumps(results, indent=2, default=json_serial)

    except Exception as e:
        return json.dumps({"error": f"Site comparison failed: {str(e)}"}, default=json_serial)

def list_available_data() -> str:
    """List available sites, KPIs, and date ranges in the dataset."""
    try:
        # Original KPI columns only
        original_kpis = ['RSRP', 'DL_Throughput', 'Call_Drop_Rate', 'RTT', 'CPU_Utilization', 
                        'Active_Users', 'SINR', 'UL_Throughput', 'Handover_Success_Rate', 'Packet_Loss']

        results = {
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
                "total_original_kpis": len(original_kpis),
                "available_kpis": original_kpis,
                "description": {
                    "RSRP": "Reference Signal Received Power",
                    "DL_Throughput": "Download Throughput",
                    "CPU_Utilization": "CPU Utilization percentage",
                    "Active_Users": "Number of active users",
                    "SINR": "Signal to Interference plus Noise Ratio",
                    "RTT": "Round Trip Time (Latency)",
                    "Packet_Loss": "Packet Loss Rate",
                    "Call_Drop_Rate": "Call Drop Rate",
                    "Handover_Success_Rate": "Handover Success Rate",
                    "UL_Throughput": "Upload Throughput"
                }
            },
            "sectors": df['Sector_ID'].unique().tolist() if 'Sector_ID' in df.columns else [],
            "models_loaded": {
                "autoencoder": autoencoder_model is not None,
                "scaler": scaler is not None
            }
        }

        return json.dumps(results, indent=2, default=json_serial)

    except Exception as e:
        return json.dumps({"error": f"Failed to get data info: {str(e)}"}, default=json_serial)

# Create FastAPI app
app = FastAPI(title="Telecom KPI Analysis Server")

@app.get("/")
async def root():
    return {"message": "Telecom KPI Analysis Server is running", "status": "healthy"}

@app.post("/call_tool")
async def call_tool_endpoint(request: ToolRequest):
    """HTTP endpoint to call MCP tools with improved parameter parsing"""
    try:
        tool_name = request.tool
        parameters = request.parameters
        
        # Call the appropriate tool function directly
        if tool_name == "detect_anomalies":
            result = detect_anomalies(**parameters)
        elif tool_name == "analyze_kpi_trends":
            result = analyze_kpi_trends(**parameters)
        elif tool_name == "get_site_summary":
            result = get_site_summary(**parameters)
        elif tool_name == "compare_sites":
            result = compare_sites(**parameters)
        elif tool_name == "list_available_data":
            result = list_available_data()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

        # Parse JSON result if it's a string
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                result = {"raw_result": result}

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

# Server startup
if __name__ == "__main__":
    print("🚀 Starting Telecom KPI MCP Server...")
    load_data_and_models()
    print("✅ Server ready!")
    print("🌐 Starting HTTP server on localhost:8000...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )