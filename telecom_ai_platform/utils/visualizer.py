"""
Visualization Utilities

This module provides visualization capabilities for the telecom AI platform,
including charts for KPI trends, anomaly detection results, and network
performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..models.anomaly_detector import AnomalyResult
from ..core.config import TelecomConfig
from ..utils.logger import LoggerMixin, log_function_call


class TelecomVisualizer(LoggerMixin):
    """
    Main visualization class for telecom network data.
    
    Provides various plotting capabilities for KPI analysis, anomaly detection,
    and network performance monitoring.
    """
    
    def __init__(self, config: TelecomConfig):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.output_dir = Path(config.data_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style preferences
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    @log_function_call
    def plot_kpi_trends(
        self,
        data: pd.DataFrame,
        kpi_name: str,
        site_id: Optional[str] = None,
        interactive: bool = True,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot KPI trends over time.
        
        Args:
            data: Input DataFrame
            kpi_name: KPI to plot
            site_id: Optional specific site
            interactive: Whether to create interactive plot
            save_path: Optional save path
            
        Returns:
            Path to saved plot
        """
        if kpi_name not in data.columns:
            raise ValueError(f"KPI '{kpi_name}' not found in data")
        
        # Filter data if site specified
        plot_data = data.copy()
        title_suffix = ""
        
        if site_id and 'Site_ID' in data.columns:
            plot_data = plot_data[plot_data['Site_ID'] == site_id]
            title_suffix = f" - Site {site_id}"
            
            if plot_data.empty:
                raise ValueError(f"No data found for site {site_id}")
        
        # Ensure date column is datetime
        if 'Date' in plot_data.columns:
            plot_data['Date'] = pd.to_datetime(plot_data['Date'])
            plot_data = plot_data.sort_values('Date')
        
        if interactive:
            return self._create_interactive_trend_plot(plot_data, kpi_name, title_suffix, save_path)
        else:
            return self._create_static_trend_plot(plot_data, kpi_name, title_suffix, save_path)
    
    def _create_interactive_trend_plot(
        self,
        data: pd.DataFrame,
        kpi_name: str,
        title_suffix: str,
        save_path: Optional[str]
    ) -> str:
        """Create interactive trend plot using Plotly"""
        fig = go.Figure()
        
        # Add main trend line
        if 'Date' in data.columns:
            x_data = data['Date']
            x_label = 'Date'
        else:
            x_data = data.index
            x_label = 'Sample Index'
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=data[kpi_name],
            mode='lines+markers',
            name=kpi_name,
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add moving average if enough data points
        if len(data) >= 7:
            moving_avg = data[kpi_name].rolling(window=7, center=True).mean()
            fig.add_trace(go.Scatter(
                x=x_data,
                y=moving_avg,
                mode='lines',
                name='7-day Moving Average',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        # Customize layout
        fig.update_layout(
            title=f'{kpi_name} Trends{title_suffix}',
            xaxis_title=x_label,
            yaxis_title=kpi_name,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"{kpi_name}_trends_{timestamp}.html"
        
        pyo.plot(fig, filename=str(save_path), auto_open=False)
        self.logger.info(f"Interactive trend plot saved to {save_path}")
        
        return str(save_path)
    
    def _create_static_trend_plot(
        self,
        data: pd.DataFrame,
        kpi_name: str,
        title_suffix: str,
        save_path: Optional[str]
    ) -> str:
        """Create static trend plot using matplotlib"""
        plt.figure(figsize=(12, 6))
        
        # Plot main trend
        if 'Date' in data.columns:
            x_data = data['Date']
            plt.xlabel('Date')
        else:
            x_data = data.index
            plt.xlabel('Sample Index')
        
        plt.plot(x_data, data[kpi_name], 'b-', alpha=0.7, linewidth=1, label=kpi_name)
        
        # Add moving average if enough data
        if len(data) >= 7:
            moving_avg = data[kpi_name].rolling(window=7, center=True).mean()
            plt.plot(x_data, moving_avg, 'r--', linewidth=2, label='7-day Moving Average')
        
        plt.title(f'{kpi_name} Trends{title_suffix}')
        plt.ylabel(kpi_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"{kpi_name}_trends_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Static trend plot saved to {save_path}")
        return str(save_path)
    
    @log_function_call
    def plot_anomaly_detection_results(
        self,
        data: pd.DataFrame,
        anomaly_results: List[AnomalyResult],
        kpi_name: str,
        interactive: bool = True,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot anomaly detection results.
        
        Args:
            data: Original data
            anomaly_results: Anomaly detection results
            kpi_name: KPI name
            interactive: Whether to create interactive plot
            save_path: Optional save path
            
        Returns:
            Path to saved plot
        """
        # Filter results for the specific KPI
        kpi_results = [r for r in anomaly_results if r.kpi_name == kpi_name]
        
        if not kpi_results:
            raise ValueError(f"No anomaly results found for KPI '{kpi_name}'")
        
        if interactive:
            return self._create_interactive_anomaly_plot(data, kpi_results, kpi_name, save_path)
        else:
            return self._create_static_anomaly_plot(data, kpi_results, kpi_name, save_path)
    
    def _create_interactive_anomaly_plot(
        self,
        data: pd.DataFrame,
        anomaly_results: List[AnomalyResult],
        kpi_name: str,
        save_path: Optional[str]
    ) -> str:
        """Create interactive anomaly detection plot"""
        fig = go.Figure()
        
        # Prepare data
        if 'Date' in data.columns:
            x_data = pd.to_datetime(data['Date'])
            x_label = 'Date'
        else:
            x_data = data.index
            x_label = 'Sample Index'
        
        # Plot normal data points
        normal_mask = [not r.is_anomaly for r in anomaly_results]
        anomaly_mask = [r.is_anomaly for r in anomaly_results]
        
        # Normal points
        fig.add_trace(go.Scatter(
            x=x_data[normal_mask],
            y=data[kpi_name][normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Anomaly points
        if any(anomaly_mask):
            anomaly_colors = ['red' if r.severity == 'high' else 'orange' if r.severity == 'medium' else 'yellow' 
                             for r in anomaly_results if r.is_anomaly]
            
            fig.add_trace(go.Scatter(
                x=x_data[anomaly_mask],
                y=data[kpi_name][anomaly_mask],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color=anomaly_colors,
                    size=10,
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                customdata=[[r.severity, r.confidence, r.anomaly_score] for r in anomaly_results if r.is_anomaly],
                hovertemplate='<b>Anomaly</b><br>' +
                            'Value: %{y}<br>' +
                            'Severity: %{customdata[0]}<br>' +
                            'Confidence: %{customdata[1]:.1%}<br>' +
                            'Score: %{customdata[2]:.3f}<extra></extra>'
            ))
        
        # Add threshold line if available
        if anomaly_results:
            threshold = anomaly_results[0].threshold
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Threshold: {threshold:.3f}"
            )
        
        # Customize layout
        fig.update_layout(
            title=f'Anomaly Detection Results - {kpi_name}',
            xaxis_title=x_label,
            yaxis_title=kpi_name,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"{kpi_name}_anomalies_{timestamp}.html"
        
        pyo.plot(fig, filename=str(save_path), auto_open=False)
        self.logger.info(f"Interactive anomaly plot saved to {save_path}")
        
        return str(save_path)
    
    def _create_static_anomaly_plot(
        self,
        data: pd.DataFrame,
        anomaly_results: List[AnomalyResult],
        kpi_name: str,
        save_path: Optional[str]
    ) -> str:
        """Create static anomaly detection plot"""
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        if 'Date' in data.columns:
            x_data = pd.to_datetime(data['Date'])
            plt.xlabel('Date')
        else:
            x_data = data.index
            plt.xlabel('Sample Index')
        
        # Plot all data points
        plt.scatter(x_data, data[kpi_name], c='blue', alpha=0.6, s=30, label='Normal')
        
        # Highlight anomalies
        anomalies = [r for r in anomaly_results if r.is_anomaly]
        if anomalies:
            anomaly_indices = [i for i, r in enumerate(anomaly_results) if r.is_anomaly]
            colors = ['red' if r.severity == 'high' else 'orange' if r.severity == 'medium' else 'yellow' 
                     for r in anomalies]
            
            plt.scatter(
                x_data.iloc[anomaly_indices],
                data[kpi_name].iloc[anomaly_indices],
                c=colors,
                s=100,
                marker='D',
                edgecolors='black',
                linewidth=1,
                label='Anomalies',
                alpha=0.8
            )
        
        # Add threshold line
        if anomaly_results:
            threshold = anomaly_results[0].threshold
            plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7, label=f'Threshold: {threshold:.3f}')
        
        plt.title(f'Anomaly Detection Results - {kpi_name}')
        plt.ylabel(kpi_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"{kpi_name}_anomalies_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Static anomaly plot saved to {save_path}")
        return str(save_path)
    
    @log_function_call
    def create_dashboard(
        self,
        data: pd.DataFrame,
        anomaly_results: List[AnomalyResult],
        kpis: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            data: Input data
            anomaly_results: Anomaly detection results
            kpis: List of KPIs to include
            save_path: Optional save path
            
        Returns:
            Path to saved dashboard
        """
        if kpis is None:
            kpis = [col for col in self.config.data.kpi_columns if col in data.columns][:4]  # Limit to 4 for layout
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{kpi} Trends' for kpi in kpis[:4]],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, kpi in enumerate(kpis[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            if kpi not in data.columns:
                continue
            
            # Prepare data for this KPI
            if 'Date' in data.columns:
                x_data = pd.to_datetime(data['Date'])
            else:
                x_data = data.index
            
            # Plot normal trend
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=data[kpi],
                    mode='lines+markers',
                    name=f'{kpi} Trend',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=3)
                ),
                row=row, col=col
            )
            
            # Add anomalies for this KPI
            kpi_anomalies = [r for r in anomaly_results if r.kpi_name == kpi and r.is_anomaly]
            if kpi_anomalies:
                anomaly_indices = [i for i, r in enumerate(anomaly_results) 
                                 if r.kpi_name == kpi and r.is_anomaly]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_data.iloc[anomaly_indices],
                        y=data[kpi].iloc[anomaly_indices],
                        mode='markers',
                        name=f'{kpi} Anomalies',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='diamond',
                            line=dict(color='black', width=1)
                        )
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title='Telecom Network Performance Dashboard',
            showlegend=True,
            template='plotly_white',
            height=800
        )
        
        # Save dashboard
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"dashboard_{timestamp}.html"
        
        pyo.plot(fig, filename=str(save_path), auto_open=False)
        self.logger.info(f"Dashboard saved to {save_path}")
        
        return str(save_path)
    
    @log_function_call
    def plot_site_comparison(
        self,
        data: pd.DataFrame,
        site_ids: List[str],
        kpi_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create site comparison visualization.
        
        Args:
            data: Input data
            site_ids: List of site IDs to compare
            kpi_name: KPI to compare
            save_path: Optional save path
            
        Returns:
            Path to saved plot
        """
        if 'Site_ID' not in data.columns:
            raise ValueError("Site_ID column not found in data")
        
        if kpi_name not in data.columns:
            raise ValueError(f"KPI '{kpi_name}' not found in data")
        
        # Filter data for specified sites
        site_data = data[data['Site_ID'].isin(site_ids)].copy()
        
        if site_data.empty:
            raise ValueError(f"No data found for sites: {site_ids}")
        
        # Create box plot comparison
        fig = go.Figure()
        
        for site_id in site_ids:
            site_kpi_data = site_data[site_data['Site_ID'] == site_id][kpi_name].dropna()
            
            if not site_kpi_data.empty:
                fig.add_trace(go.Box(
                    y=site_kpi_data,
                    name=f'Site {site_id}',
                    boxpoints='outliers'
                ))
        
        fig.update_layout(
            title=f'{kpi_name} Comparison Across Sites',
            yaxis_title=kpi_name,
            xaxis_title='Sites',
            template='plotly_white'
        )
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"site_comparison_{kpi_name}_{timestamp}.html"
        
        pyo.plot(fig, filename=str(save_path), auto_open=False)
        self.logger.info(f"Site comparison plot saved to {save_path}")
        
        return str(save_path)
    
    def generate_summary_report(
        self,
        data: pd.DataFrame,
        anomaly_results: List[AnomalyResult]
    ) -> Dict:
        """
        Generate summary statistics for reporting.
        
        Args:
            data: Input data
            anomaly_results: Anomaly detection results
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'data_overview': {
                'total_records': len(data),
                'date_range': None,
                'sites_count': data['Site_ID'].nunique() if 'Site_ID' in data.columns else 0,
                'kpis_analyzed': len([col for col in self.config.data.kpi_columns if col in data.columns])
            },
            'anomaly_summary': {
                'total_anomalies': sum(1 for r in anomaly_results if r.is_anomaly),
                'anomaly_rate': sum(1 for r in anomaly_results if r.is_anomaly) / len(anomaly_results) if anomaly_results else 0,
                'severity_distribution': {},
                'kpi_breakdown': {}
            }
        }
        
        # Date range
        if 'Date' in data.columns:
            summary['data_overview']['date_range'] = [
                str(data['Date'].min()),
                str(data['Date'].max())
            ]
        
        # Severity distribution
        severities = [r.severity for r in anomaly_results if r.is_anomaly]
        for severity in ['low', 'medium', 'high']:
            summary['anomaly_summary']['severity_distribution'][severity] = severities.count(severity)
        
        # KPI breakdown
        for kpi in self.config.data.kpi_columns:
            kpi_anomalies = [r for r in anomaly_results if r.kpi_name == kpi and r.is_anomaly]
            kpi_total = [r for r in anomaly_results if r.kpi_name == kpi]
            
            if kpi_total:
                summary['anomaly_summary']['kpi_breakdown'][kpi] = {
                    'anomalies': len(kpi_anomalies),
                    'total_samples': len(kpi_total),
                    'anomaly_rate': len(kpi_anomalies) / len(kpi_total)
                }
        
        return summary


def create_visualizer(config: TelecomConfig) -> TelecomVisualizer:
    """
    Factory function to create a configured visualizer.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured TelecomVisualizer instance
    """
    return TelecomVisualizer(config)
