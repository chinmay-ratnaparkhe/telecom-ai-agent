#!/usr/bin/env python3

"""
Unified Streamlit App: Visualizations + LLM Chat (Chain-of-Thought + Web + MCP integrated)

Notes:
- Visualizations match the improved clean_ui version (site/sector/KPI tabs and heatmap).
- Single chat on the right with structured reasoning steps. Web search and MCP tools are used automatically when available.
- No emojis.
- Sidebar shows model catalog (unsupervised models + hyperparameters) and MCP tools if connected.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional
import os
import re
import json
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
try:
    # Load environment variables from .env (searches up the tree)
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# Optional: agents for chain-of-thought and MCP
AGENTS_AVAILABLE = False
try:
    from telecom_ai_platform.agents.chain_of_thought import ChainOfThoughtAgent
    from telecom_ai_platform.agents.mcp_bridge import MCPBridge
    from telecom_ai_platform.core.config import TelecomConfig
    AGENTS_AVAILABLE = True
except Exception:
    AGENTS_AVAILABLE = False

# UI-level anomaly detector (pre-trained models)
DETECTOR_AVAILABLE = False
try:
    from telecom_ai_platform.core.config import TelecomConfig as _Cfg
    from telecom_ai_platform.models.anomaly_detector import KPIAnomalyDetector, KPISpecificDetector
    # General cross-KPI autoencoder (optional)
    from telecom_ai_platform.models.enhanced_autoencoder import GeneralAutoEncoderDetector
    DETECTOR_AVAILABLE = True
except Exception:
    DETECTOR_AVAILABLE = False
    GeneralAutoEncoderDetector = None  # type: ignore

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:  # static type checking only
        from telecom_ai_platform.models.enhanced_autoencoder import GeneralAutoEncoderDetector

# Domain-based default anomaly direction per KPI (high values bad vs low values bad vs both)
KPI_DEFAULT_DIRECTION = {
    'Packet_Loss': 'high',
    'Call_Drop_Rate': 'high',
    'CPU_Utilization': 'high',
    'RTT': 'high',
    'Active_Users': 'both',
    'DL_Throughput': 'low',  # unusually low throughput typically problematic
    'UL_Throughput': 'low',
    'RSRP': 'low',           # low signal power
    'SINR': 'low',           # low SINR harmful
    'Handover_Success_Rate': 'low',
}


# ---------- Page config ----------
st.set_page_config(
    page_title="Telecom AI Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Data helpers ----------
@st.cache_data
def load_telecom_data() -> pd.DataFrame:
    """Load KPI CSV from repository dataset. No synthetic fallback."""
    # Resolve repo root relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)
    candidates = [
        os.path.join(repo_root, "data", "AD_data_10KPI.csv"),
        os.path.join(repo_root, "AD_data_10KPI.csv"),
        os.path.join("..", "data", "AD_data_10KPI.csv"),
        "AD_data_10KPI.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])  # daily
            return df
    raise FileNotFoundError("AD_data_10KPI.csv not found in known locations. Place it under the data/ directory.")


def detect_anomalies(df: pd.DataFrame, kpi: str, threshold: float = 2.0) -> pd.DataFrame:
    """Simple z-score style anomaly detection on a filtered dataframe."""
    if df.empty or kpi not in df.columns:
        return pd.DataFrame()
    values = df[kpi].dropna()
    if len(values) < 3:
        return pd.DataFrame()
    mean_val = values.mean()
    std_val = values.std()
    if std_val == 0:
        return pd.DataFrame()
    mask = (df[kpi] - mean_val).abs() > threshold * std_val
    return df[mask & df[kpi].notna()]


def get_ui_detector() -> Optional["KPIAnomalyDetector"]:
    """Lazy-init and cache the KPI anomaly detector for chat/fast paths.

    Returns:
        KPIAnomalyDetector if available and models were loaded, else None.
    """
    if not DETECTOR_AVAILABLE:
        return None
    ui_detector = st.session_state.get('ui_detector')
    if ui_detector is None:
        try:
            _cfg = _Cfg()
            ui_detector = KPIAnomalyDetector(_cfg)
            # Load pre-trained per-KPI models (.pkl) if available
            ui_detector.load_all_models()
            st.session_state['ui_detector'] = ui_detector
        except Exception:
            ui_detector = None
            st.session_state['ui_detector'] = None
    return ui_detector


from typing import Any as _Any  # alias to avoid confusion

@st.cache_resource(show_spinner=False)
def load_general_autoencoder(model_path: Optional[str] = None) -> Optional[_Any]:
    """Load a pre-trained general autoencoder (if available) for cross-KPI reconstruction error analysis.

    Looks for 'general_autoencoder.pkl' under the configured models directory unless an explicit path is provided.
    Returns None if not found or on failure (silent)."""
    if not DETECTOR_AVAILABLE:
        return None
    try:
        cfg = _Cfg()
        if model_path is None:
            candidate = os.path.join(cfg.models_dir, 'general_autoencoder.pkl')
        else:
            candidate = model_path
        if not os.path.exists(candidate):
            return None
        # Ensure class is available; attempt lazy import if previously failed
        global GeneralAutoEncoderDetector
        if GeneralAutoEncoderDetector is None:
            try:
                from telecom_ai_platform.models.enhanced_autoencoder import GeneralAutoEncoderDetector as _GAE
                GeneralAutoEncoderDetector = _GAE  # type: ignore
            except Exception:
                return None
        gae = GeneralAutoEncoderDetector()
        gae.load_model(candidate)
        return gae
    except Exception:
        return None

# Utility: attempt to guarantee GeneralAutoEncoderDetector is imported, return bool success
def ensure_general_ae_class() -> bool:
    """Attempt to import GeneralAutoEncoderDetector with rich diagnostics.

    Stores error message and traceback in session_state keys:
        gae_import_error, gae_import_error_trace, gae_import_sys_path, gae_import_candidates
    """
    import importlib, traceback, inspect, sys, pkgutil
    global GeneralAutoEncoderDetector
    if GeneralAutoEncoderDetector is not None:
        # Record resolved file path for transparency
        try:
            import telecom_ai_platform.models.enhanced_autoencoder as _mod
            st.session_state['gae_resolved_file'] = getattr(_mod, '__file__', 'unknown')
        except Exception:
            pass
        return True
    try:
        # Enumerate candidate modules BEFORE import for debugging
        candidates = [m.module_finder.path for m in pkgutil.iter_importers() if hasattr(m, 'path')]  # type: ignore
        st.session_state['gae_import_candidates'] = candidates
    except Exception:
        st.session_state['gae_import_candidates'] = []
    try:
        import importlib.util, os
        importlib.invalidate_caches()
        spec = importlib.util.find_spec('telecom_ai_platform.models.enhanced_autoencoder')
        if spec and spec.origin:
            st.session_state['gae_candidate_path'] = spec.origin
            try:
                with open(spec.origin, 'rb') as f:
                    raw = f.read()
                st.session_state['gae_file_size'] = len(raw)
                # Null byte scan
                st.session_state['gae_null_bytes_found'] = b'\x00' in raw
            except Exception:
                pass
        mod = importlib.import_module('telecom_ai_platform.models.enhanced_autoencoder')
        GeneralAutoEncoderDetector = getattr(mod, 'GeneralAutoEncoderDetector')  # type: ignore
        st.session_state['gae_resolved_file'] = getattr(mod, '__file__', 'unknown')
        return True
    except Exception as e:
        st.session_state['gae_import_error'] = str(e)
        st.session_state['gae_import_error_trace'] = traceback.format_exc()
        st.session_state['gae_import_sys_path'] = list(sys.path)
        return False


def create_kpi_visualization(
    data: pd.DataFrame,
    site_id: str,
    sector_id: str,
    kpi: str,
    *,
    detector: Optional["KPIAnomalyDetector"] = None,
    last_days: Optional[int] = None,
    show_model_threshold: bool = False,
):
    filtered = data[(data['Site_ID'] == site_id) & (data['Sector_ID'] == sector_id)].copy()
    if filtered.empty:
        return None, pd.DataFrame()

    # Optional time window
    if last_days and 'Date' in filtered.columns:
        try:
            end_date = filtered['Date'].max()
            from datetime import timedelta
            start_date = end_date - timedelta(days=int(last_days))
            filtered = filtered[filtered['Date'] >= start_date]
        except Exception:
            pass

    # Decide second subplot title depending on threshold toggle
    second_title = "Score Distribution (Model Threshold)" if show_model_threshold and detector is not None else "Distribution & Anomalies"
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[f"{kpi} Trends for {site_id}, {sector_id}", second_title],
                        vertical_spacing=0.15)

    fig.add_trace(
        go.Scatter(x=filtered['Date'], y=filtered[kpi], mode='lines+markers', name=f'{kpi} Values',
                    line=dict(color='#667eea', width=2), marker=dict(size=5),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'),
        row=1, col=1
    )

    # Prefer model-based anomalies if detector is available and fitted
    anomalies_df = pd.DataFrame()
    full_results = []
    if detector is not None and getattr(detector, 'is_fitted', False) and kpi in filtered.columns:
        try:
            results = detector.detect_anomalies(filtered, kpi_name=kpi)
            full_results = results or []
            # Convert anomalies to DataFrame
            if results:
                anomalies_raw = pd.DataFrame([{
                    'Date': pd.to_datetime(r.timestamp) if 'Date' in filtered.columns else r.timestamp,
                    'Site_ID': r.site_id,
                    'Sector_ID': r.sector_id,
                    'KPI': r.kpi_name,
                    'Value': r.value,
                    'Severity': r.severity,
                    'Score': r.anomaly_score,
                    'Threshold': r.threshold,
                } for r in results if r.is_anomaly])
                # Apply optional direction filtering (stored in session_state)
                direction = st.session_state.get('anomaly_direction_filter', 'both')
                if direction != 'both' and not anomalies_raw.empty:
                    # Use percentile-based directional thresholds if provided in session
                    p_hi = float(st.session_state.get('dir_high_pct', 95))
                    p_lo = float(st.session_state.get('dir_low_pct', 5))
                    hi_thr = np.percentile(anomalies_raw['Value'], p_hi)
                    lo_thr = np.percentile(anomalies_raw['Value'], p_lo)
                    if direction == 'high':
                        anomalies_df = anomalies_raw[anomalies_raw['Value'] >= hi_thr]
                    elif direction == 'low':
                        anomalies_df = anomalies_raw[anomalies_raw['Value'] <= lo_thr]
                    else:
                        anomalies_df = anomalies_raw.copy()
                else:
                    anomalies_df = anomalies_raw
        except Exception:
            anomalies_df = detect_anomalies(filtered, kpi, threshold=2.0)
            if not anomalies_df.empty:
                anomalies_df = anomalies_df.rename(columns={kpi: 'Value'})
                anomalies_df['KPI'] = kpi
    else:
        anomalies_df = detect_anomalies(filtered, kpi, threshold=2.0)
        if not anomalies_df.empty:
            anomalies_df = anomalies_df.rename(columns={kpi: 'Value'})
            anomalies_df['KPI'] = kpi

    if not anomalies_df.empty:
        # Optional jitter to avoid overplotting when multiple anomalies share the same timestamp/value
        use_jitter = st.session_state.get('anomaly_jitter_enabled', True)
        if use_jitter:
            # Compute small jitter proportional to local index
            jit_scale = (anomalies_df['Value'].std() or 1.0) * 0.01
            anomalies_df = anomalies_df.reset_index(drop=True).assign(_jitter=lambda d: ((d.index % 5) - 2) * jit_scale)
            plot_values = anomalies_df['Value'] + anomalies_df['_jitter']
        else:
            plot_values = anomalies_df['Value']
        # Plot by severity (if available) so nearby points differ in color/symbol
        if 'Severity' in anomalies_df.columns:
            sev_palette = {
                'high': dict(color='#d62728', symbol='x'),
                'medium': dict(color='#ff7f0e', symbol='diamond-open'),
                'low': dict(color='#bcbd22', symbol='circle-open'),
                'normal': dict(color='#1f77b4', symbol='circle')
            }
            for sev, sub in anomalies_df.groupby('Severity'):
                pal = sev_palette.get(sev, dict(color='red', symbol='x'))
                idxs = sub.index
                fig.add_trace(
                    go.Scatter(
                        x=sub['Date'],
                        y=plot_values.loc[idxs],
                        mode='markers',
                        name=f'Anomaly {sev}',
                        marker=dict(color=pal['color'], size=11, symbol=pal['symbol'], line=dict(width=1,color='#222')),
                        hovertemplate='<b>Anomaly</b><br><b>Severity:</b> '+sev+'<br><b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
        else:
            fig.add_trace(
                go.Scatter(x=anomalies_df['Date'], y=plot_values, mode='markers', name='Anomalies',
                            marker=dict(color='red', size=10, symbol='x'),
                            hovertemplate='<b>Anomaly</b><br><b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'),
                row=1, col=1
            )
        # Annotate total anomaly count
        fig.add_annotation(text=f"Anomalies: {len(anomalies_df)}", xref='paper', x=0.01, yref='paper', y=0.98,
                           showarrow=False, font=dict(size=11, color='#444'))

    if not show_model_threshold:
        # Classic statistical bands
        mean_val = filtered[kpi].mean()
        std_val = filtered[kpi].std()
        fig.add_hline(y=mean_val, line_dash="dash", line_color="green",
                      annotation=dict(text="Mean", standoff=10), row=1, col=1)
        fig.add_hline(y=mean_val + 2*std_val, line_dash="dash", line_color="orange",
                      annotation=dict(text="Upper Threshold (2σ)", standoff=10), row=1, col=1)
        fig.add_hline(y=mean_val - 2*std_val, line_dash="dash", line_color="orange",
                      annotation=dict(text="Lower Threshold (2σ)", standoff=10), row=1, col=1)

        # Value distribution histogram
        fig.add_trace(
            go.Histogram(x=filtered[kpi], name='Distribution', nbinsx=20, marker_color='lightblue',
                         opacity=0.7, histnorm='probability density'),
            row=2, col=1
        )
        if len(filtered[kpi].dropna()) > 5:
            kde_x = np.linspace(filtered[kpi].min(), filtered[kpi].max(), 100)
            kde = stats.gaussian_kde(filtered[kpi].dropna())
            kde_y = kde(kde_x)
            fig.add_trace(
                go.Scatter(x=kde_x, y=kde_y, mode='lines', name='Density', line=dict(color='darkblue', width=2)),
                row=2, col=1
            )
        if not anomalies_df.empty:
            fig.add_trace(
                go.Scatter(x=anomalies_df['Value'], y=[0]*len(anomalies_df), mode='markers', name='Anomalies',
                           showlegend=False, marker=dict(color='red', size=10, symbol='x')),
                row=2, col=1
            )
    else:
        # Model-threshold view: show anomaly score distribution instead of raw value distribution
        if detector is not None and full_results:
            scores = [r.anomaly_score for r in full_results]
            threshold = full_results[0].threshold if full_results else None
            fig.add_trace(
                go.Histogram(x=scores, name='Scores', nbinsx=30, marker_color='lightblue',
                             opacity=0.75, histnorm='probability density'),
                row=2, col=1
            )
            if threshold is not None:
                fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                              annotation=dict(text=f"Score Threshold {threshold:.2f}", standoff=10),
                              row=2, col=1)
            # Approximate value threshold (optional visualization): pick value at first score >= threshold
            if threshold is not None:
                try:
                    vals = filtered[kpi].values
                    paired = sorted(zip(scores, vals), key=lambda x: x[0])
                    approx_val = None
                    for sc, val in paired:
                        if sc >= threshold:
                            approx_val = val
                            break
                    if approx_val is not None:
                        fig.add_hline(y=approx_val, line_dash="dot", line_color="red",
                                      annotation=dict(text="Approx Value @ Threshold", standoff=10), row=1, col=1)
                except Exception:
                    pass
        else:
            # Fallback to statistical distribution if model results missing
            fig.add_trace(
                go.Histogram(x=filtered[kpi], name='Distribution', nbinsx=20, marker_color='lightblue',
                             opacity=0.7, histnorm='probability density'),
                row=2, col=1
            )

    fig.update_layout(title=f'{kpi} Analysis for {site_id}, {sector_id}', height=700,
                      showlegend=True, template='plotly_white', hovermode='closest',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text=f"{kpi} Value", row=1, col=1)
    if show_model_threshold and detector is not None:
        fig.update_xaxes(title_text="Anomaly Score", row=2, col=1)
        fig.update_yaxes(title_text="Score Density", row=2, col=1)
    else:
        fig.update_xaxes(title_text=f"{kpi} Value", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1)
    return fig, anomalies_df


# ---------- Query helpers (canonicalization + fast analysis) ----------
def canonicalize_site(token: str, data: pd.DataFrame) -> Optional[str]:
    if not token:
        return None
    tok = str(token).strip().upper().replace("SITE", "").replace("_", "").replace("-", "")
    # Extract numeric part
    m = re.search(r"(\d+)", tok)
    candidates = sorted(data['Site_ID'].unique()) if 'Site_ID' in data.columns else []
    if m and candidates:
        num = int(m.group(1))
        # Try zero-padded SITE_###
        site_id = f"SITE_{num:03d}"
        if site_id in candidates:
            return site_id
    # Fallback: best substring match
    tok_l = token.lower()
    for s in candidates:
        if tok_l in s.lower():
            return s
    return candidates[0] if candidates else None


def canonicalize_sector(site_id: str, token: str, data: pd.DataFrame) -> Optional[str]:
    if not site_id or not token:
        return None
    site_df = data[data['Site_ID'] == site_id]
    sectors = site_df['Sector_ID'].unique().tolist() if not site_df.empty else []
    if not sectors:
        return None
    t = str(token).strip().upper()
    # Accept single letter like 'A' or patterns like 'SECTOR_A'
    suffix = t[-1] if len(t) == 1 else t.split('_')[-1]
    for sec in sectors:
        if sec.upper().endswith(f"SECTOR_{suffix}"):
            return sec
    # Fallback: substring match
    for sec in sectors:
        if t in sec.upper():
            return sec
    return sectors[0]


def _window_by_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty or 'Date' not in df.columns:
        return df
    try:
        end_date = pd.to_datetime(df['Date']).max()
        from datetime import timedelta
        start = end_date - timedelta(days=int(days))
        return df[pd.to_datetime(df['Date']) >= start]
    except Exception:
        return df


def fast_answer_if_applicable(query: str, data: pd.DataFrame) -> Optional[Dict]:
    ql = query.lower()
    # Parse explicit last N days (no implicit default: use full range unless user specifies)
    m_days = re.search(r"last\s+(\d+)\s*days?", ql)
    days = int(m_days.group(1)) if m_days else (7 if "last week" in ql or "last 7" in ql else None)

    # Case 0: Generic "Which site has highest anomalies ..." with no explicit KPI
    if "which site" in ql and "anomal" in ql and not any(k in ql for k in [
        "sinr", "rsrp", "throughput", "dl_throughput", "ul_throughput", "rtt", "packet", "cpu", "active_users", "handover", "call_drop"
    ]):
        win = _window_by_last_days(data, days) if days is not None else data
        if win.empty:
            return {"steps": ["No data in selected window"], "context": "", "answer": "No data available for the requested window."}
        # Prefer model-based detectors if available
        det = get_ui_detector()
        if det is not None and getattr(det, 'is_fitted', False):
            by_site = []
            for site in sorted(win['Site_ID'].unique()):
                site_df = win[win['Site_ID'] == site]
                try:
                    results = det.detect_anomalies(site_df)
                    count = sum(1 for r in results if r.is_anomaly)
                except Exception:
                    count = 0
                by_site.append((site, count))
            if by_site:
                by_site.sort(key=lambda x: x[1], reverse=True)
                top_site, top_count = by_site[0]
                steps = [
                    (f"Filtered dataset to last {days} days" if days is not None else "Used entire dataset"),
                    "Used KPI-specific trained detectors (IF/OCSVM/GMM/AE) across available KPIs",
                    f"Ranked sites by total model-detected anomalies; top is {top_site} with {top_count}",
                ]
                context = "Top 5 counts:\n" + "\n".join([f"{s}: {c}" for s, c in by_site[:5]])
                ans = f"{top_site} has the highest anomalies in {('the last ' + str(days) + ' days') if days is not None else 'the entire dataset period'} (count={top_count})."
                return {"steps": steps, "context": context, "answer": ans}
            else:
                return {"steps": ["Model returned no results in window"], "context": "", "answer": "No anomalies found in the requested window."}
        else:
            # Fallback to simple z-score if models are unavailable
            all_kpis = [c for c in win.columns if c not in ['Date', 'Site_ID', 'Sector_ID']]
            kpis = ['SINR'] if 'SINR' in all_kpis else [k for k in ['DL_Throughput', 'RSRP', 'RTT'] if k in all_kpis]
            if not kpis:
                kpis = all_kpis[:3]
            by_site = []
            for site in sorted(win['Site_ID'].unique()):
                site_df = win[win['Site_ID'] == site]
                total = 0
                for k in kpis:
                    an_df = detect_anomalies(site_df, k, threshold=2.0)
                    total += len(an_df)
                by_site.append((site, total))
            if by_site:
                by_site.sort(key=lambda x: x[1], reverse=True)
                top_site, top_count = by_site[0]
                steps = [
                    (f"Filtered dataset to last {days} days" if days is not None else "Used entire dataset"),
                    f"Used z-score fallback across KPIs: {', '.join(kpis)}",
                    f"Ranked sites by total anomaly count; top is {top_site} with {top_count}",
                ]
                context = "Top 5 counts:\n" + "\n".join([f"{s}: {c}" for s, c in by_site[:5]])
                ans = f"{top_site} has the highest anomalies in {('the last ' + str(days) + ' days') if days is not None else 'the entire dataset period'} (count={top_count})."
                return {"steps": steps, "context": context, "answer": ans}
            else:
                return {"steps": ["No KPIs available for anomaly computation"], "context": "", "answer": "No anomalies found in the requested window."}

    # Case 1: Which site has highest SINR anomalies ...
    if "which site" in ql and "sinr" in ql and "anomal" in ql:
        win = _window_by_last_days(data, days) if days is not None else data
        det = get_ui_detector()
        by_site = []
        if det is not None and getattr(det, 'is_fitted', False):
            for site in sorted(win['Site_ID'].unique()):
                site_df = win[win['Site_ID'] == site]
                try:
                    results = det.detect_anomalies(site_df, kpi_name='SINR')
                    count = sum(1 for r in results if r.kpi_name == 'SINR' and r.is_anomaly)
                except Exception:
                    count = 0
                by_site.append((site, count))
            if by_site:
                by_site.sort(key=lambda x: x[1], reverse=True)
                top_site, top_count = by_site[0]
                steps = [
                    (f"Filtered dataset to last {days} days" if days is not None else "Used entire dataset"),
                    "Used KPI-specific SINR detector to count anomalies per site",
                    f"Ranked sites by anomaly count; top is {top_site} with {top_count}",
                ]
                context = "Top 5 counts:\n" + "\n".join([f"{s}: {c}" for s, c in by_site[:5]])
                ans = f"{top_site} has the highest SINR anomalies in {('the last ' + str(days) + ' days') if days is not None else 'the entire dataset period'} (count={top_count})."
                return {"steps": steps, "context": context, "answer": ans}
            else:
                return {"steps": ["Model returned no SINR results in window"], "context": "", "answer": "No anomalies found in the requested window."}
        else:
            # Fallback to z-score if models unavailable
            by_site = []
            for site in sorted(win['Site_ID'].unique()):
                site_df = win[win['Site_ID'] == site]
                if 'SINR' in site_df.columns and not site_df.empty:
                    an_df = detect_anomalies(site_df, 'SINR', threshold=2.0)
                    count = len(an_df)
                    by_site.append((site, count))
            if by_site:
                by_site.sort(key=lambda x: x[1], reverse=True)
                top_site, top_count = by_site[0]
                steps = [
                    (f"Filtered dataset to last {days} days" if days is not None else "Used entire dataset"),
                    "Used z-score fallback for SINR per site",
                    f"Ranked sites by anomaly count; top is {top_site} with {top_count}",
                ]
                context = "Top 5 counts:\n" + "\n".join([f"{s}: {c}" for s, c in by_site[:5]])
                ans = f"{top_site} has the highest SINR anomalies in {('the last ' + str(days) + ' days') if days is not None else 'the entire dataset period'} (count={top_count})."
                return {"steps": steps, "context": context, "answer": ans}
            else:
                return {"steps": ["No SINR data found in the selected window"], "context": "", "answer": "No anomalies found in the requested window."}

    # Case 2: Analyze SINR anomalies for Site X Sector Y
    if "sinr" in ql and "anomal" in ql and "site" in ql:
        # Extract site token and sector token
        site_m = re.search(r"site\s*[:#_-]?\s*([\w-]+)", ql)
        sec_m = re.search(r"sector\s*[:#_-]?\s*([a-zA-Z])", ql)
        site_id = canonicalize_site(site_m.group(1), data) if site_m else None
        sector_id = canonicalize_sector(site_id, sec_m.group(1), data) if (site_id and sec_m) else None
        if site_id:
            df = data.copy()
            df = df[df['Site_ID'] == site_id]
            if sector_id:
                df = df[df['Sector_ID'] == sector_id]
            if days is not None:
                df = _window_by_last_days(df, days)
            if 'SINR' in df.columns and not df.empty:
                det = get_ui_detector()
                if det is not None and getattr(det, 'is_fitted', False):
                    try:
                        results = det.detect_anomalies(df, kpi_name='SINR')
                        # Filter anomalies and optional sector match
                        anomalies = [r for r in results if r.kpi_name == 'SINR' and r.is_anomaly and (not sector_id or (r.sector_id == sector_id))]
                    except Exception:
                        anomalies = []
                    cnt = len(anomalies)
                    recent_lines = []
                    for r in sorted(anomalies, key=lambda x: str(x.timestamp))[-3:]:
                        recent_lines.append(f"{r.timestamp}: {r.value:.2f} (score={r.anomaly_score:.2f}, sev={r.severity})")
                    steps = [
                        f"Selected {site_id}{' / ' + sector_id if sector_id else ''}",
                        (f"Applied last {days} days window" if days is not None else "Used entire dataset (no time window specified)"),
                        "Used KPI-specific SINR detector to identify anomalies",
                    ]
                    ctx = "" if not recent_lines else ("Recent anomalies:\n" + "\n".join(recent_lines))
                    ans = f"Detected {cnt} SINR anomalies for {site_id}{' / ' + sector_id if sector_id else ''} over {('the last ' + str(days) + ' days') if days is not None else 'the entire dataset period'}."
                    return {"steps": steps, "context": ctx, "answer": ans}
                else:
                    # Fallback z-score
                    an_df = detect_anomalies(df, 'SINR', threshold=2.0)
                    cnt = len(an_df)
                    recent = an_df.sort_values('Date').tail(3) if cnt else pd.DataFrame()
                    steps = [
                        f"Selected {site_id}{' / ' + sector_id if sector_id else ''}",
                        (f"Applied last {days} days window" if days is not None else "Used entire dataset (no time window specified)"),
                        "Used z-score fallback on SINR",
                    ]
                    ctx = "" if recent.empty else ("Recent anomalies:\n" + "\n".join([f"{r['Date']}: {r['SINR']:.2f}" for _, r in recent.iterrows()]))
                    ans = f"Detected {cnt} SINR anomalies for {site_id}{' / ' + sector_id if sector_id else ''} over {('the last ' + str(days) + ' days') if days is not None else 'the entire dataset period'}."
                    return {"steps": steps, "context": ctx, "answer": ans}

    # Case 3: Analyze DL or UL Throughput anomalies for Site X Sector Y
    if ("throughput" in ql or "dl_throughput" in ql or "ul_throughput" in ql or "uplink" in ql) and "anomal" in ql and "site" in ql:
        site_m = re.search(r"site\s*[:#_-]?\s*([\w-]+)", ql)
        sec_m = re.search(r"sector\s*[:#_-]?\s*([a-zA-Z])", ql)
        site_id = canonicalize_site(site_m.group(1), data) if site_m else None
        sector_id = canonicalize_sector(site_id, sec_m.group(1), data) if (site_id and sec_m) else None
        if site_id:
            is_ul = any(tok in ql for tok in ["ul_throughput", "ul throughput", "uplink"]) and not any(tok in ql for tok in ["dl_throughput", "downlink"])
            kpi_name = 'UL_Throughput' if is_ul else 'DL_Throughput'
            df = data.copy()
            df = df[df['Site_ID'] == site_id]
            if sector_id:
                df = df[df['Sector_ID'] == sector_id]
            applied_window = False
            if days is not None:
                df = _window_by_last_days(df, days)
                applied_window = True
            if kpi_name in df.columns and not df.empty:
                det = get_ui_detector()
                if det is not None and getattr(det, 'is_fitted', False):
                    try:
                        results = det.detect_anomalies(df, kpi_name=kpi_name)
                        anomalies = [r for r in results if r.kpi_name == kpi_name and r.is_anomaly and (not sector_id or (r.sector_id == sector_id))]
                    except Exception:
                        anomalies = []
                    cnt = len(anomalies)
                    recent_lines = []
                    for r in sorted(anomalies, key=lambda x: str(x.timestamp))[-3:]:
                        recent_lines.append(f"{r.timestamp}: {r.value:.2f} (score={r.anomaly_score:.2f}, sev={r.severity})")
                    steps = [f"Selected {site_id}{' / ' + sector_id if sector_id else ''}"]
                    if applied_window:
                        steps.append(f"Applied last {days} days window (relative to dataset max date)")
                    else:
                        steps.append("Used entire dataset (no time window specified)")
                    steps.append(f"Used KPI-specific {kpi_name} detector to identify anomalies")
                    ctx = "" if not recent_lines else ("Recent anomalies:\n" + "\n".join(recent_lines))
                    period_text = f"the last {days} days" if applied_window else "the entire dataset period"
                    ans = f"Detected {cnt} {kpi_name.replace('_',' ')} anomalies for {site_id}{' / ' + sector_id if sector_id else ''} over {period_text}."
                    return {"steps": steps, "context": ctx, "answer": ans}
                else:
                    # Fallback z-score
                    an_df = detect_anomalies(df, kpi_name, threshold=2.0)
                    cnt = len(an_df)
                    recent = an_df.sort_values('Date').tail(3) if cnt else pd.DataFrame()
                    steps = [f"Selected {site_id}{' / ' + sector_id if sector_id else ''}"]
                    if applied_window:
                        steps.append(f"Applied last {days} days window (relative to dataset max date)")
                    else:
                        steps.append("Used entire dataset (no time window specified)")
                    steps.append(f"Used z-score fallback on {kpi_name}")
                    ctx = "" if recent.empty else ("Recent anomalies:\n" + "\n".join([f"{r['Date']}: {r[kpi_name]:.2f}" for _, r in recent.iterrows()]))
                    period_text = f"the last {days} days" if applied_window else "the entire dataset period"
                    ans = f"Detected {cnt} {kpi_name.replace('_',' ')} anomalies for {site_id}{' / ' + sector_id if sector_id else ''} over {period_text}."
                    return {"steps": steps, "context": ctx, "answer": ans}
    return None


def create_multi_kpi_comparison(data: pd.DataFrame, site_id: str, sector_id: str, kpis: List[str]):
    if not kpis or len(kpis) < 2:
        return None
    filtered = data[(data['Site_ID'] == site_id) & (data['Sector_ID'] == sector_id)].copy()
    if filtered.empty:
        return None
    for k in kpis:
        if k in filtered.columns:
            min_val = filtered[k].min()
            max_val = filtered[k].max()
            filtered[f"{k}_normalized"] = 1.0 if max_val == min_val else (filtered[k]-min_val)/(max_val-min_val)
    fig = go.Figure()
    for k in kpis:
        fig.add_trace(go.Scatter(x=filtered['Date'], y=filtered[f"{k}_normalized"], mode='lines', name=k,
                                 hovertemplate=f'<b>{k}</b><br>Date: %{{x}}<br>Normalized: %{{y:.2f}}<br>Actual: %{{customdata:.2f}}<extra></extra>',
                                 customdata=filtered[k]))
    fig.update_layout(title=f'Multi-KPI Comparison for {site_id}, {sector_id}', height=500, showlegend=True,
                      template='plotly_white', hovermode='x unified',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Normalized Value (0-1)")
    return fig


def create_sector_comparison(data: pd.DataFrame, site_id: str, kpi: str):
    site_df = data[data['Site_ID'] == site_id].copy()
    if site_df.empty:
        return None
    sectors = site_df['Sector_ID'].unique()
    if len(sectors) <= 1:
        return None
    fig = go.Figure()
    for sec in sectors:
        sec_df = site_df[site_df['Sector_ID'] == sec]
        fig.add_trace(go.Scatter(x=sec_df['Date'], y=sec_df[kpi], mode='lines+markers', name=sec,
                                 marker=dict(size=6),
                                 hovertemplate=f'<b>{sec}</b><br>Date: %{{x}}<br>{kpi}: %{{y:.2f}}<extra></extra>'))
    fig.update_layout(title=f'{kpi} Comparison Across All Sectors for {site_id}', height=500, showlegend=True,
                      template='plotly_white', hovermode='x unified',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=f"{kpi} Value")
    return fig


def create_site_sector_kpi_heatmap(data: pd.DataFrame, kpi: str):
    if kpi not in data.columns:
        return None
    latest = data.sort_values('Date').groupby(['Site_ID', 'Sector_ID']).last().reset_index()
    pivot = latest.pivot(index='Site_ID', columns='Sector_ID', values=kpi)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(title=kpi),
        hovertemplate=f'<b>Site:</b> %{{y}}<br><b>Sector:</b> %{{x}}<br><b>{kpi}:</b> %{{z:.2f}}<extra></extra>'
    ))
    fig.update_layout(title=f'{kpi} Heatmap - All Sites and Sectors (Latest Values)', height=600, width=900,
                      xaxis_title='Sector ID', yaxis_title='Site ID', template='plotly_white')
    return fig


# ---------- Model catalog (visible + usable in chat context) ----------
DEFAULT_MODEL_CATALOG = [
    {"KPI": "RSRP", "Primary Model": "Isolation Forest", "Secondary Model": "Statistical (Z-score)", "Rationale": "Point anomalies in signal strength", "Key Parameters": "contamination=0.1, n_estimators=100"},
    {"KPI": "SINR", "Primary Model": "Local Outlier Factor", "Secondary Model": "Isolation Forest", "Rationale": "Density-based, interference detection", "Key Parameters": "n_neighbors=20, contamination=0.1"},
    {"KPI": "DL_Throughput", "Primary Model": "LSTM Autoencoder", "Secondary Model": "Seasonal Hybrid ESD", "Rationale": "Temporal patterns, traffic variations", "Key Parameters": "sequence_length=7, threshold=3"},
    {"KPI": "UL_Throughput", "Primary Model": "LSTM Autoencoder", "Secondary Model": "Statistical Process Control", "Rationale": "Temporal patterns, capacity limits", "Key Parameters": "sequence_length=7, reconstruction_error"},
    {"KPI": "Call_Drop_Rate", "Primary Model": "Ensemble (IF + GMM)", "Secondary Model": "One-Class SVM", "Rationale": "Multi-modal distribution", "Key Parameters": "n_components=3, contamination=0.05"},
    {"KPI": "RTT", "Primary Model": "Isolation Forest", "Secondary Model": "Moving Average + Control", "Rationale": "Latency spikes detection", "Key Parameters": "contamination=0.08, window=7"},
    {"KPI": "CPU_Utilization", "Primary Model": "Time Series Decomposition", "Secondary Model": "Prophet", "Rationale": "Strong temporal patterns", "Key Parameters": "seasonal_periods=[7, 30]"},
    {"KPI": "Active_Users", "Primary Model": "Seasonal Hybrid ESD", "Secondary Model": "DBSCAN", "Rationale": "Seasonal usage patterns", "Key Parameters": "alpha=0.05, max_anoms=0.1"},
    {"KPI": "Handover_Success", "Primary Model": "Gaussian Mixture Model", "Secondary Model": "Statistical Control Charts", "Rationale": "Success rate boundaries", "Key Parameters": "n_components=2, threshold=2σ"},
    {"KPI": "Packet_Loss", "Primary Model": "One-Class SVM", "Secondary Model": "Isolation Forest", "Rationale": "Boundary-based detection", "Key Parameters": "nu=0.05, gamma='scale'"},
]

def get_model_catalog() -> List[Dict]:
    return st.session_state.get('model_catalog', DEFAULT_MODEL_CATALOG)


# ---------- Simple web search (DuckDuckGo) ----------
def web_search_snippets(query: str) -> str:
    try:
        import requests
        params = {'q': f"{query} telecom network KPI", 'format': 'json', 'no_html': '1', 'skip_disambig': '1'}
        r = requests.get("https://api.duckduckgo.com/", params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            parts = []
            if data.get('AbstractText'):
                parts.append(data['AbstractText'])
            for t in data.get('RelatedTopics', [])[:2]:
                if isinstance(t, dict) and t.get('Text'):
                    parts.append(t['Text'])
            return "\n".join(parts[:3])
    except Exception:
        pass
    return ""


# ---------- NL routing to MCP tools ----------
def route_and_call_mcp(query: str, mcp_bridge: Optional[object], tools: List[str], data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    if not mcp_bridge or not tools:
        return None
    ql = query.lower()
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Parse common entities
    # KPI
    kpi_map = {
    'sinr': 'SINR', 'rsrp': 'RSRP', 'dl_throughput': 'DL_Throughput', 'ul_throughput': 'UL_Throughput', 'throughput': 'DL_Throughput',
        'rtt': 'RTT', 'packet_loss': 'Packet_Loss', 'active_users': 'Active_Users',
        'cpu': 'CPU_Utilization', 'cpu_utilization': 'CPU_Utilization',
        'handover': 'Handover_Success_Rate', 'handover_success_rate': 'Handover_Success_Rate',
        'call_drop': 'Call_Drop_Rate', 'call_drop_rate': 'Call_Drop_Rate'
    }
    kpi = next((k for k in kpi_map.keys() if k in ql), None)
    kpi_norm = kpi_map.get(kpi) if kpi else st.session_state.get('selected_kpi')
    # Site
    site_m = re.search(r"site\s*[:#_-]?\s*([\w-]+)", ql)
    site_id = canonicalize_site(site_m.group(1), data) if (site_m and data is not None) else st.session_state.get('selected_site')
    # Time window
    m_days = re.search(r"last\s+(\d+)\s*days?", ql)
    days = int(m_days.group(1)) if m_days else (7 if "last week" in ql or "last 7" in ql else None)
    date_range = 'last_week'
    if days:
        # Convert to server format as absolute range relative to dataset window is not available server-side; keep preset
        date_range = 'last_week' if days <= 7 else 'last_month'

    # Tool 1: detect_anomalies
    if ("anomal" in ql or "outlier" in ql) and 'detect_anomalies' in tools and kpi_norm:
        try:
            res = loop.run_until_complete(
                mcp_bridge.detect_anomalies_with_mcp(
                    kpi_name=kpi_norm,
                    site_id=site_id,
                    date_range=date_range
                )
            )
            return {"tool": "detect_anomalies", "args": {"kpi_name": kpi_norm, "site_id": site_id, "date_range": date_range}, "result": res}
        except Exception as e:
            return {"tool": "detect_anomalies", "error": str(e)}

    # Tool 2: analyze_kpi_trends
    if ("trend" in ql or "over time" in ql or "pattern" in ql) and 'analyze_kpi_trends' in tools and kpi_norm:
        try:
            res = loop.run_until_complete(
                mcp_bridge.analyze_kpi_with_mcp(
                    kpi_name=kpi_norm,
                    site_id=site_id,
                    date_range=date_range
                )
            )
            return {"tool": "analyze_kpi_trends", "args": {"kpi_name": kpi_norm, "site_id": site_id, "date_range": date_range}, "result": res}
        except Exception as e:
            return {"tool": "analyze_kpi_trends", "error": str(e)}

    return None


# ---------- LLM chat (simple) ----------
def simple_llm_chat(query: str, *, mcp: Optional[object], skip_llm: bool = False) -> Dict:
    """Return structured reasoning steps + answer. Uses CoT + Web + MCP automatically when available."""
    steps: List[str] = []
    context_bits: List[str] = []
    context_payload: Dict = {}

    # Include model catalog context
    model_catalog_text = json.dumps(get_model_catalog(), indent=2)
    context_bits.append("Model Catalog:\n" + model_catalog_text)
    context_payload["model_catalog"] = get_model_catalog()

    # Lightweight in-app data summarization for queries that mention a KPI/site/sector so answers include REAL values
    try:
        data = load_telecom_data()
        ql = query.lower()
        # Heuristic extractors
        kpi_alias_map = {
            'cpu': 'CPU_Utilization', 'cpu_utilization': 'CPU_Utilization',
            'active_users': 'Active_Users', 'users': 'Active_Users',
            'dl_throughput': 'DL_Throughput', 'downlink': 'DL_Throughput',
            'ul_throughput': 'UL_Throughput', 'uplink': 'UL_Throughput',
            'rtt': 'RTT', 'latency': 'RTT',
            'packet_loss': 'Packet_Loss', 'loss': 'Packet_Loss',
            'call_drop_rate': 'Call_Drop_Rate', 'call_drop': 'Call_Drop_Rate',
            'sinr': 'SINR', 'rsrp': 'RSRP', 'handover': 'Handover_Success_Rate'
        }
        kpi_in_q = None
        for alias, norm in kpi_alias_map.items():
            if re.search(rf"\b{alias}\b", ql):
                kpi_in_q = norm
                break
        # Site & sector tokens like 'site 1', 'sector a'
        site_match = re.search(r"site\s*(\d{1,3})", ql)
        site_id = None
        if site_match:
            site_num = int(site_match.group(1))
            site_id = f"SITE_{site_num:03d}"
        # Sector letter (A/B/C). Accept 'sector a'
        sector_match = re.search(r"sector\s*([a-z])", ql)
        sector_id = None
        if sector_match and site_id:
            sector_letter = sector_match.group(1).upper()
            # Build expected pattern present in dataset
            sector_id = f"{site_id}_SECTOR_{sector_letter}"
        # If user hasn't specified, fallback to current selections stored in session
        if not site_id:
            site_id = st.session_state.get('selected_site')
        if not sector_id and site_id:
            # Use selected sector if matches site
            sel_sec = st.session_state.get('selected_sector')
            if sel_sec and isinstance(sel_sec, str) and sel_sec.startswith(site_id):
                sector_id = sel_sec
        if kpi_in_q and site_id and sector_id and kpi_in_q in data.columns:
            df_slice = data[(data['Site_ID'] == site_id) & (data['Sector_ID'] == sector_id)][['Date', kpi_in_q]].dropna().copy()
            if not df_slice.empty:
                # Last 60 days (or all if shorter)
                df_slice.sort_values('Date', inplace=True)
                window_df = df_slice.tail(60)
                stats_summary = {
                    'site': site_id,
                    'sector': sector_id,
                    'kpi': kpi_in_q,
                    'records_window': int(len(window_df)),
                    'start_date': window_df['Date'].iloc[0].strftime('%Y-%m-%d'),
                    'end_date': window_df['Date'].iloc[-1].strftime('%Y-%m-%d'),
                    'min': float(window_df[kpi_in_q].min()),
                    'max': float(window_df[kpi_in_q].max()),
                    'mean': float(window_df[kpi_in_q].mean()),
                    'std': float(window_df[kpi_in_q].std()),
                    'latest_value': float(window_df[kpi_in_q].iloc[-1])
                }
                context_bits.append("Data Snapshot:\n" + json.dumps(stats_summary, indent=2))
                context_payload['data_snapshot'] = stats_summary
                # Add top anomalies if detector available
                det = get_ui_detector()
                if det is not None and getattr(det, 'is_fitted', False):
                    try:
                        res = det.detect_anomalies(window_df.rename(columns={kpi_in_q: kpi_in_q}), kpi_name=kpi_in_q)
                        an_vals = [r for r in res if r.is_anomaly]
                        if an_vals:
                            top_anoms = sorted(an_vals, key=lambda r: r.anomaly_score, reverse=True)[:5]
                            an_payload = [
                                {'value': a.value, 'score': a.anomaly_score, 'severity': a.severity} for a in top_anoms
                            ]
                            context_bits.append("Top Anomalies (model-based):\n" + json.dumps(an_payload, indent=2))
                            context_payload['top_anomalies'] = an_payload
                    except Exception:
                        pass
    except Exception:
        # Non-fatal; ignore summarization errors
        pass

    # Use web search first to enrich context (best-effort)
    try:
        steps.append("Performing targeted web search for telecom definitions and standards")
        web_text = web_search_snippets(query)
        if web_text:
            context_bits.append("Web Search:\n" + web_text)
            context_payload["web_search"] = web_text
        else:
            context_bits.append("Web Search: No additional context found")
    except Exception as e:
        context_bits.append(f"Web Search error: {e}")

    # Consult MCP for structured analysis if available
    if mcp is not None:
        steps.append("Consulting MCP tools for available capabilities and telemetry access")
        try:
            status = st.session_state.get('mcp_status', {})
            tools = status.get('available_tools', [])
            context_bits.append("MCP Tools:\n" + "\n".join(tools) if tools else "MCP Tools: None discovered")
            context_payload["mcp_tools"] = tools

            # Natural-language routing to MCP tools
            if tools:
                routed = route_and_call_mcp(query, mcp, tools, data=st.session_state.get('cot_agent').network_tools.current_data if st.session_state.get('cot_agent') else None)
                if routed:
                    steps.append(f"Used MCP tool: {routed['tool']}")
                    context_bits.append("MCP routed result:\n" + json.dumps(routed.get('result', {}), indent=2))
                    context_payload["mcp_routed_result"] = routed
        except Exception as e:
            context_bits.append(f"MCP Tools: Error during invocation ({e})")

    # Prefer the real LLM if available (unless explicitly skipped)
    cot = st.session_state.get('cot_agent') if AGENTS_AVAILABLE else None
    if cot is None and AGENTS_AVAILABLE:
        # Lazy init attempt
        try:
            cot = ChainOfThoughtAgent(TelecomConfig())
            st.session_state['cot_agent'] = cot
        except Exception as e:
            steps.append(f"LLM unavailable: {e}")
            cot = None

    if not skip_llm and cot is not None:
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            steps.append("Invoking ChainOfThoughtAgent to synthesize an answer")
            result = loop.run_until_complete(
                cot.analyze_with_chain_of_thought(
                    query=query,
                    context=context_payload,
                    use_web_search=True,
                    use_mcp=True,
                )
            )

            # Convert reasoning steps to full display (no truncation)
            rendered_steps: List[str] = []
            for i, s in enumerate(result.reasoning_chain, start=1):
                title = s.reasoning_type.value.replace('_', ' ').title()
                summary = (s.analysis or '').strip()
                rendered_steps.append(f"Step {i}: {title} — {summary}")

            return {
                "steps": rendered_steps,
                "context": "\n\n".join(context_bits),
                "answer": result.final_conclusion or ""
            }
        except Exception as e:
            steps.append(f"LLM error while generating answer: {e}")

    # Fallback or context-only: compose a concise answer from available context
    steps.append("Formulating answer from local context (no LLM)")
    answer_parts = []
    if "web_search" in context_payload:
        answer_parts.append("Included web context where relevant.")
    if "mcp_anomaly_result" in context_payload or "mcp_kpi_result" in context_payload:
        answer_parts.append("Included MCP tool results.")
    answer_parts.append("Analysis relies on the loaded dataset with per-site and per-sector KPIs.")
    return {
        "steps": steps,
        "context": "\n\n".join(context_bits),
        "answer": "\n".join(answer_parts)
    }


# ---------- Sidebar ----------
def render_sidebar(data: pd.DataFrame):
    with st.sidebar:
        st.header("Controls")

        # Site and sector
        sites = sorted(data['Site_ID'].unique()) if not data.empty else []
    site = st.selectbox("Site", sites, index=0 if sites else None, key="sidebar_site")
    sectors = sorted(data[data['Site_ID'] == site]['Sector_ID'].unique()) if site else []
    sector = st.selectbox("Sector", sectors, index=0 if sectors else None, key="sidebar_sector")

    # KPI selection
    kpis = [c for c in data.columns if c not in ['Date', 'Site_ID', 'Sector_ID']]
    kpi = st.selectbox("KPI", sorted(kpis) if kpis else [], key="sidebar_kpi")
    multi_kpi = st.multiselect("Compare Multiple KPIs (optional)", sorted(kpis) if kpis else [], default=[], key="sidebar_multi_kpi")

    # Anomaly direction filter (high vs low vs both) with domain default
    default_dir = KPI_DEFAULT_DIRECTION.get(kpi, 'both') if kpi else 'both'
    direction = st.radio("Anomaly Direction", ['both','high','low'], index=['both','high','low'].index(default_dir), key='anomaly_direction_filter', help="Filter displayed anomalies: high spikes, low dips, or both.")
    if direction != 'both':
        cdir1, cdir2 = st.columns(2)
        with cdir1:
            st.number_input("High %ile", 50.0, 100.0, value=float(st.session_state.get('dir_high_pct', 95.0)), step=0.5, key='dir_high_pct', help="Percentile threshold for high anomalies (values >= this).")
        with cdir2:
            st.number_input("Low %ile", 0.0, 50.0, value=float(st.session_state.get('dir_low_pct', 5.0)), step=0.5, key='dir_low_pct', help="Percentile threshold for low anomalies (values <= this).")

    # Time window for charts/anomalies
    st.markdown("---")
    st.subheader("Time Window")
    last_days = st.number_input("Days (applies to charts & anomaly overlay)", min_value=1, max_value=120, value=60, step=1)
    st.session_state['last_days'] = int(last_days)

    # MCP connection (auto-connect using config)
    st.markdown("---")
    st.subheader("MCP Server")
    status = st.session_state.get('mcp_status')
    # MCP URL configuration and discovery
    default_url = None
    try:
        if AGENTS_AVAILABLE:
            cfg = TelecomConfig()
            default_url = cfg.agent.mcp_server_url
    except Exception:
        default_url = st.session_state.get('mcp_url', 'http://localhost:8000')

    # Allow user to override MCP URL
    mcp_url = st.text_input("MCP URL", value=st.session_state.get('mcp_url', default_url or 'http://localhost:8000'))
    col_a, col_b = st.columns([1,1])
    with col_a:
        reconnect = st.button("Connect")
    with col_b:
        discover = st.button("Discover")

    # Simple discovery: probe common ports on localhost
    if discover:
        import requests
        discovered = None
        for p in [8000, 8001, 8002, 8010, 9000, 7000]:
            try:
                url = f"http://localhost:{p}/health"
                r = requests.get(url, timeout=0.5)
                if r.ok and 'status' in r.json():
                    discovered = f"http://localhost:{p}"
                    break
            except Exception:
                continue
        if discovered:
            st.session_state['mcp_url'] = discovered
            mcp_url = discovered
            st.success(f"Discovered MCP at {discovered}")
        else:
            st.warning("No MCP server discovered on common ports.")

    # Trigger connect if URL changed or Connect pressed
    if AGENTS_AVAILABLE and (reconnect or (mcp_url and mcp_url != st.session_state.get('mcp_url')) or not status):
        try:
            # Persist chosen URL
            st.session_state['mcp_url'] = mcp_url
            cfg = TelecomConfig()
            # Override config's mcp_server_url dynamically if available
            try:
                cfg.agent.mcp_server_url = mcp_url
            except Exception:
                pass
            bridge = MCPBridge(cfg)
            st.session_state['mcp_bridge'] = bridge
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            ok = loop.run_until_complete(bridge.connect_to_server(mcp_url, alias="default"))
            status = loop.run_until_complete(bridge.get_server_status()) if ok else {"status": "disconnected"}
            st.session_state['mcp_status'] = status
        except Exception as e:
            status = {"status": "disconnected", "error": str(e)}
            st.session_state['mcp_status'] = status
    # Display MCP status and tools
    if status:
        st.text(f"Status: {status.get('status')}")
        if status.get('error'):
            st.text(f"Error: {status['error']}")
        tools = status.get('available_tools', [])
        st.text(f"URL: {st.session_state.get('mcp_url','n/a')}")
        if tools:
            st.markdown("**Available MCP Tools**")
            for t in tools:
                st.text(f"- {t}")

        # LLM status
        st.markdown("---")
        st.subheader("LLM")
        try:
            key_present = bool(os.getenv("GEMINI_API_KEY"))
        except Exception:
            key_present = False
        st.text(f"Gemini key: {'set' if key_present else 'missing'}")
        st.text(f"CoT agent: {'ready' if st.session_state.get('cot_agent') else 'unavailable'}")

        # Model catalog visible
        st.markdown("---")
        st.subheader("Model Catalog")
        st.dataframe(pd.DataFrame(get_model_catalog()), use_container_width=True)

    # --- Per-KPI model parameters viewer (if detectors loaded) ---
    ui_det = get_ui_detector()
    if ui_det and getattr(ui_det, 'is_fitted', False):
        with st.expander("KPI Model Parameters", expanded=False):
            try:
                kpi_list = sorted(list(ui_det.detectors.keys()))
                sel_param_kpi = st.selectbox("Select KPI", kpi_list, key='sidebar_param_kpi') if kpi_list else None
                if sel_param_kpi:
                    det = ui_det.detectors.get(sel_param_kpi)
                    if det and getattr(det, 'is_fitted', False):
                        if hasattr(det, 'get_full_params'):
                            st.json(det.get_full_params())
                        else:
                            st.json({
                                'kpi_name': det.kpi_name,
                                'algorithm': det.algorithm,
                                'threshold': float(det.threshold) if det.threshold is not None else None,
                                'params': getattr(det, 'params', {}),
                                'feature_names': getattr(det, 'feature_names', []),
                            })
                    else:
                        st.caption("Detector not fitted yet.")
                if st.button("Export All KPI Params", key='export_all_kpi_params_sidebar'):
                    try:
                        all_params = ui_det.get_all_parameters() if hasattr(ui_det, 'get_all_parameters') else {}
                        cfg_tmp = _Cfg()
                        out_path = os.path.join(cfg_tmp.models_dir, 'kpi_models_parameters.json')
                        with open(out_path, 'w') as f:
                            json.dump(all_params, f, indent=2)
                        st.success(f"Saved to {out_path}")
                    except Exception as _exp_err:
                        st.warning(f"Export failed: {_exp_err}")
            except Exception as _param_err:
                st.warning(f"Param viewer error: {_param_err}")
    else:
        st.caption("(KPI models not loaded – place .pkl files in models dir or train in Fine Tuning tab.)")

    # Chat behavior (outside sidebar)
    st.markdown("---")
    st.subheader("Chat Settings")
    st.checkbox("Conversational tone", value=True, key="chat_tone_conversational")
    st.checkbox("Live stream reasoning", value=True, key="chat_stream_steps")

    # Save selections in session
    st.session_state['selected_site'] = site
    st.session_state['selected_sector'] = sector
    st.session_state['selected_kpi'] = kpi
    st.session_state['multi_kpi'] = multi_kpi


# ---------- Main ----------
def main():
    # Initialize agents if available (non-blocking)
    if AGENTS_AVAILABLE and 'cot_agent' not in st.session_state:
        try:
            st.session_state['cot_agent'] = ChainOfThoughtAgent(TelecomConfig())
        except Exception:
            st.session_state['cot_agent'] = None

    data = load_telecom_data()
    # Share loaded dataset with the CoT agent so its tools can analyze real data
    if AGENTS_AVAILABLE and st.session_state.get('cot_agent') is not None:
        try:
            st.session_state['cot_agent'].network_tools.current_data = data
        except Exception:
            pass
    render_sidebar(data)

    site = st.session_state.get('selected_site')
    sector = st.session_state.get('selected_sector')
    kpi = st.session_state.get('selected_kpi')
    multi_kpi = st.session_state.get('multi_kpi', [])

    col_left, col_right = st.columns([2, 1])

    with col_left:
        tabs = st.tabs([
            "Single KPI Analysis",
            "Multi-KPI Comparison",
            "Sector Comparison",
            "Site-Sector Heatmap",
            "Data Overview",
            "Docs",
            "KPI Fine Tuning",
            "Completeness Audit",
        ])

        with tabs[0]:
            if site and sector and kpi:
                # Create or reuse a pre-trained detector for overlay
                ui_detector = st.session_state.get('ui_detector')
                if DETECTOR_AVAILABLE and ui_detector is None:
                    try:
                        _cfg = _Cfg()
                        ui_detector = KPIAnomalyDetector(_cfg)
                        ui_detector.load_all_models()  # use existing .pkl models only
                        st.session_state['ui_detector'] = ui_detector
                    except Exception:
                        ui_detector = None
                        st.session_state['ui_detector'] = None

                show_model_thr = st.toggle("Show model threshold & score distribution (instead of ±2σ)", value=False, help="Switches second panel to anomaly score distribution and overlays model-derived threshold.")
                # Build KPI-specific figure first
                kpi_fig, anomalies_df = create_kpi_visualization(
                    data, site, sector, kpi,
                    detector=ui_detector,
                    last_days=st.session_state.get('last_days', 7),
                    show_model_threshold=show_model_thr,
                )
                # Container with two columns: left KPI plot, right General AE plot
                col_kpi_plot, col_gen_plot = st.columns(2)
                with col_kpi_plot:
                    if kpi_fig is not None:
                        st.plotly_chart(kpi_fig, use_container_width=True)
                    else:
                        st.warning("No data available for selection")
                # Anomaly table
                filtered = data[(data['Site_ID'] == site) & (data['Sector_ID'] == sector)]
                st.subheader("Anomaly Details")
                if 'anomalies_df' in locals() and anomalies_df is not None and not anomalies_df.empty:
                    desired_cols = ['Date', 'Site_ID', 'Sector_ID', 'KPI', 'Value', 'Severity', 'Score', 'Threshold']
                    # Coerce Date for display
                    try:
                        anomalies_df['Date'] = pd.to_datetime(anomalies_df['Date']).dt.strftime('%Y-%m-%d')
                    except Exception:
                        pass
                    show_cols = [c for c in desired_cols if c in anomalies_df.columns]
                    if not show_cols:
                        show_cols = anomalies_df.columns.tolist()
                    st.dataframe(anomalies_df.sort_values('Date', ascending=False)[show_cols], use_container_width=True)
                else:
                    # Fallback simple detection
                    fallback = detect_anomalies(filtered, kpi)
                    if not fallback.empty:
                        disp = fallback[['Date', kpi]].copy()
                        disp['Date'] = pd.to_datetime(disp['Date']).dt.strftime('%Y-%m-%d')
                        disp.columns = ['Date', f'{kpi}']
                        st.dataframe(disp.sort_values('Date', ascending=False), use_container_width=True)
                    else:
                        st.info("No anomalies detected in the selected window.")

                # --- NEW: General AutoEncoder overlay plot for this KPI (quick view) ---
                try:
                    gae_quick = load_general_autoencoder()
                    if gae_quick is not None:
                        st.markdown("#### General AutoEncoder Perspective (Quick Overlay)")
                        # Align features to model expectations
                        expected_feats = list(getattr(gae_quick, 'feature_names', []))
                        base_cols = [c for c in expected_feats if c in filtered.columns]
                        if not base_cols:
                            base_cols = [c for c in filtered.columns if c not in ['Date','Site_ID','Sector_ID']]
                        feat_block = filtered[base_cols].fillna(0.0)
                        # Dimension adaptation
                        if expected_feats and len(base_cols) != len(expected_feats):
                            arr_tmp = feat_block.values
                            cur_dim = arr_tmp.shape[1]; exp_dim = len(expected_feats)
                            if cur_dim == 1 and exp_dim > 1:
                                arr_tmp = np.repeat(arr_tmp, exp_dim, axis=1)
                            elif cur_dim > exp_dim:
                                arr_tmp = arr_tmp[:, :exp_dim]
                            elif cur_dim < exp_dim:
                                arr_tmp = np.hstack([arr_tmp, np.zeros((arr_tmp.shape[0], exp_dim - cur_dim))])
                            feat_block = pd.DataFrame(arr_tmp, columns=expected_feats[:arr_tmp.shape[1]])
                        arr_full = feat_block.values
                        gae_out = gae_quick.detect_anomalies(arr_full)
                        rec_err = gae_out['errors']
                        rec_anom = gae_out['is_anomaly']
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        fig_gae_overlay = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(f"{kpi} Value", "General AE Reconstruction Error"))
                        # Row 1: KPI series + anomaly markers (where GAEs says anomaly)
                        if kpi in filtered.columns:
                            fig_gae_overlay.add_trace(go.Scatter(x=filtered['Date'], y=filtered[kpi], mode='lines', name=kpi), row=1, col=1)
                            if rec_anom.any():
                                fig_gae_overlay.add_trace(go.Scatter(x=filtered['Date'][rec_anom], y=filtered[kpi][rec_anom], mode='markers', name='GAE Anomaly', marker=dict(color='red', size=7, symbol='x')), row=1, col=1)
                        # Row 2: Reconstruction error + threshold
                        fig_gae_overlay.add_trace(go.Scatter(x=filtered['Date'], y=rec_err, mode='lines', name='Reconstruction Error'), row=2, col=1)
                        if gae_quick.threshold is not None:
                            fig_gae_overlay.add_hline(y=gae_quick.threshold, line_color='red', line_dash='dash', annotation=dict(text='Threshold'), row=2, col=1)
                        fig_gae_overlay.update_layout(height=500, margin=dict(l=10,r=10,t=60,b=10))
                        st.plotly_chart(fig_gae_overlay, use_container_width=True)
                    else:
                        st.caption("(General AE model not yet trained for overlay plot.)")
                except Exception as gae_ov_err:
                    st.warning(f"General AE overlay failed: {gae_ov_err}")

                # General AutoEncoder side-by-side visualization
                st.markdown("---")
                st.subheader("General AutoEncoder (Cross-KPI) Reconstruction")
                all_numeric_kpis = [c for c in data.columns if c not in ['Date','Site_ID','Sector_ID'] and pd.api.types.is_numeric_dtype(data[c])]
                default_kpis = st.session_state.get('gae_selected_kpis') or all_numeric_kpis
                with st.expander("General AE Configuration & Training", expanded=False):
                    # --- Load existing model once for defaults / display ---
                    existing_model = load_general_autoencoder()
                    # Initialize session defaults from loaded model (only first time)
                    if existing_model is not None and 'gae_param_defaults_loaded' not in st.session_state:
                        # Threshold percentile
                        if getattr(existing_model, 'threshold_percentile', None) is not None:
                            st.session_state.setdefault('gae_thresh', int(existing_model.threshold_percentile))
                        # Epochs default to 100 (notebook) if no history
                        hist = getattr(existing_model, 'training_history', []) or []
                        if hist:
                            last_ep = hist[-1].get('epoch') or len(hist)
                            st.session_state.setdefault('gae_epochs', int(min(500, max(20, last_ep))))
                        else:
                            st.session_state.setdefault('gae_epochs', 100)
                        # Selected KPIs -> adopt feature_names intersection with available columns
                        feat_names = getattr(existing_model, 'feature_names', []) or []
                        if feat_names:
                            st.session_state.setdefault('gae_selected_kpis', [f for f in feat_names if f in all_numeric_kpis][:50])
                        # Advanced config prefill from last_training_config / attributes
                        conf = getattr(existing_model, 'last_training_config', {}) or {}
                        st.session_state.setdefault('gae_hidden_dims', ','.join(map(str, conf.get('hidden_dims', getattr(existing_model,'hidden_dims',[64,32,16])))))
                        st.session_state.setdefault('gae_dropout', float(conf.get('dropout_rate', getattr(existing_model,'dropout_rate',0.2))))
                        st.session_state.setdefault('gae_batch', int(conf.get('batch_size', getattr(existing_model,'batch_size',128))))
                        st.session_state.setdefault('gae_lr', float(conf.get('learning_rate', getattr(existing_model,'learning_rate',1e-3))))
                        st.session_state.setdefault('gae_pat', int(conf.get('early_stopping_patience', getattr(existing_model,'early_stopping_patience',30))))
                        st.session_state.setdefault('gae_test_split', float(conf.get('test_split', getattr(existing_model,'test_split',0.0))))
                        st.session_state.setdefault('gae_thr_mode', conf.get('threshold_mode', getattr(existing_model,'threshold_mode','percentile')))
                        st.session_state.setdefault('gae_sigma_mult', float(conf.get('sigma_multiplier', getattr(existing_model,'sigma_multiplier',3.0))))
                        st.session_state.setdefault('gae_target_frac', float(conf.get('target_anomaly_fraction', getattr(existing_model,'target_anomaly_fraction',0.05))))
                        st.session_state.setdefault('gae_prob_cutoff', 0.05)
                        st.session_state.setdefault('gae_z_cutoff', 3.0)
                        st.session_state['gae_param_defaults_loaded'] = True
                    # --- Show current model parameters if available ---
                    if existing_model is not None:
                        try:
                            model_params = {
                                'hidden_dims': getattr(existing_model, 'hidden_dims', None),
                                'dropout_rate': getattr(existing_model, 'dropout_rate', None),
                                'learning_rate': getattr(existing_model, 'learning_rate', None),
                                'batch_size': getattr(existing_model, 'batch_size', None),
                                'threshold_mode': getattr(existing_model, 'threshold_mode', None),
                                'threshold_percentile': getattr(existing_model, 'threshold_percentile', None),
                                'threshold': getattr(existing_model, 'threshold', None),
                                'n_trained_epochs': (existing_model.training_history[-1]['epoch'] if getattr(existing_model, 'training_history', None) and existing_model.training_history[-1].get('epoch') else len(getattr(existing_model, 'training_history', []))),
                                'feature_count': len(getattr(existing_model, 'feature_names', []) or []),
                                'test_split': getattr(existing_model, 'test_split', None),
                            }
                            st.markdown("**Current Trained Model Parameters**")
                            st.json(model_params)
                        except Exception as _mp_err:
                            st.caption(f"Model parameter display failed: {_mp_err}")
                    sel_kpis = st.multiselect("Select KPIs to include in general model (features)", all_numeric_kpis, default=default_kpis)
                    st.session_state['gae_selected_kpis'] = sel_kpis
                    col_tr1, col_tr2, col_tr3 = st.columns([1,1,2])
                    with col_tr1:
                        # Use session default (possibly loaded from model)
                        train_epochs_default = st.session_state.get('gae_epochs', 100)
                        train_epochs = st.slider("Max Epochs", 20, 500, int(train_epochs_default), 10, key='gae_epochs')
                    with col_tr2:
                        thresh_pct_default = int(st.session_state.get('gae_thresh', 95))
                        thresh_pct = st.slider("Threshold %", 80, 99, thresh_pct_default, 1, key='gae_thresh')
                    with col_tr3:
                        st.caption("Adjust reconstruction threshold percentile (higher -> fewer anomalies).")
                    # Advanced settings
                    adv_open = st.checkbox("Show Advanced Settings", value=False, key='gae_adv_toggle')
                    if adv_open:
                        c_adv1, c_adv2, c_adv3 = st.columns(3)
                        with c_adv1:
                            hidden_dims_input = st.text_input("Hidden Dims (comma)", value=st.session_state.get('gae_hidden_dims','128,64,32,16'), key='gae_hidden_dims')
                            dropout_rate = st.number_input("Dropout", 0.0, 0.9, float(st.session_state.get('gae_dropout',0.25)), 0.05, key='gae_dropout')
                            batch_size = st.number_input("Batch Size", 16, 1024, int(st.session_state.get('gae_batch',128)), 16, key='gae_batch')
                        with c_adv2:
                            learning_rate = st.number_input("Learning Rate", 1e-5, 1e-1, float(st.session_state.get('gae_lr',1e-3)), format='%e', key='gae_lr')
                            early_stop_pat = st.number_input("EarlyStop Patience", 5, 100, int(st.session_state.get('gae_pat',30)), 1, key='gae_pat')
                            test_split = st.number_input("Test Split", 0.0, 0.4, float(st.session_state.get('gae_test_split',0.0)), 0.05, key='gae_test_split')
                        with c_adv3:
                            threshold_mode = st.selectbox("Threshold Mode", ['percentile','val_sigma','target_anom_frac'], index=['percentile','val_sigma','target_anom_frac'].index(st.session_state.get('gae_thr_mode','percentile')), key='gae_thr_mode')
                            sigma_mult = st.number_input("Sigma Multiplier", 1.0, 10.0, float(st.session_state.get('gae_sigma_mult',3.0)), 0.5, key='gae_sigma_mult')
                            target_frac = st.number_input("Target Anom Fraction", 0.001, 0.5, float(st.session_state.get('gae_target_frac',0.05)), 0.005, key='gae_target_frac')
                        c_adv4, c_adv5 = st.columns(2)
                        with c_adv4:
                            st.number_input("Prob Cutoff (tail)", 0.0001, 0.5, float(st.session_state.get('gae_prob_cutoff',0.05)), 0.005, key='gae_prob_cutoff')
                        with c_adv5:
                            st.number_input("Z-Score Cutoff", 0.5, 10.0, float(st.session_state.get('gae_z_cutoff',3.0)), 0.1, key='gae_z_cutoff')
                    use_eng = st.checkbox("Use feature engineering (rolling / lags / ratios / temporal / site stats)", value=True, key='gae_use_eng')
                    cols_train = st.columns([1,1,2])
                    with cols_train[0]:
                        start_train = st.button("Train / Retrain General AutoEncoder", key='gae_train_btn')
                    with cols_train[1]:
                        quick_apply = st.button("Apply & Retrain", key='gae_quick_apply', help="Retrain with current settings and immediately refresh plots")
                    train_status_ph = st.empty()
                    progress_bar = st.progress(0)
                    loss_chart_ph = st.empty()
                    feature_importance_ph = st.empty()
                    # Training routine (site/sector agnostic: uses ALL rows for selected KPIs)
                    if start_train or quick_apply:
                        if not sel_kpis:
                            st.warning("Select at least one KPI to train the general model.")
                        else:
                            try:
                                if not ensure_general_ae_class():
                                    err_msg = st.session_state.get('gae_import_error', 'Unknown import error')
                                    st.error(f"General autoencoder class not available: {err_msg}")
                                    # Inline debug (avoid nested expander)
                                    st.markdown("**General AE Import Debug Details**")
                                    st.write("Resolved File:", st.session_state.get('gae_resolved_file'))
                                    st.write("Sys.path (first 10):", st.session_state.get('gae_import_sys_path', [])[:10])
                                    st.write("Candidates:", st.session_state.get('gae_import_candidates'))
                                    st.code(st.session_state.get('gae_import_error_trace', 'n/a'))
                                    raise RuntimeError("GeneralAutoEncoderDetector class unavailable")
                                # Parse advanced
                                hidden_dims = [int(x.strip()) for x in st.session_state.get('gae_hidden_dims','64,32,16').split(',') if x.strip().isdigit()] if adv_open else [64,32,16]
                                gae_local = GeneralAutoEncoderDetector(
                                    hidden_dims=hidden_dims,
                                    dropout_rate=st.session_state.get('gae_dropout',0.2) if adv_open else 0.2,
                                    learning_rate=st.session_state.get('gae_lr',1e-3) if adv_open else 1e-3,
                                    batch_size=st.session_state.get('gae_batch',128) if adv_open else 128,
                                    max_epochs=train_epochs,
                                    early_stopping_patience=st.session_state.get('gae_pat',30) if adv_open else 30,
                                    validation_split=0.2,
                                    test_split=st.session_state.get('gae_test_split',0.0) if adv_open else 0.0,
                                    threshold_percentile=thresh_pct,
                                    threshold_mode=st.session_state.get('gae_thr_mode','percentile') if adv_open else 'percentile',
                                    sigma_multiplier=st.session_state.get('gae_sigma_mult',3.0) if adv_open else 3.0,
                                    target_anomaly_fraction=st.session_state.get('gae_target_frac',0.05) if adv_open else 0.05,
                                )  # type: ignore
                                # Fit from dataframe using optional feature engineering
                                from telecom_ai_platform.models.enhanced_autoencoder import engineer_features, build_feature_subset
                                if use_eng:
                                    eng_df = engineer_features(data, sel_kpis)
                                    feat_df_full, feat_names = build_feature_subset(eng_df, sel_kpis)
                                else:
                                    feat_df_full = data[sel_kpis].fillna(0.0)
                                    feat_names = sel_kpis
                                total_epochs = train_epochs
                                train_loss_hist: List[float] = []
                                val_loss_hist: List[float] = []
                                def _cb(ep, tot, tr, vl):
                                    train_loss_hist.append(tr)
                                    val_loss_hist.append(vl)
                                    pct = int(ep / tot * 100)
                                    progress_bar.progress(min(pct,100))
                                    train_status_ph.info(f"Epoch {ep}/{tot} train_loss={tr:.5f} val_loss={vl:.5f}")
                                    # Live loss curve (Plotly)
                                    try:
                                        import plotly.graph_objects as go
                                        fig_loss = go.Figure()
                                        fig_loss.add_trace(go.Scatter(y=train_loss_hist, x=list(range(1,len(train_loss_hist)+1)), mode='lines', name='Train Loss'))
                                        fig_loss.add_trace(go.Scatter(y=val_loss_hist, x=list(range(1,len(val_loss_hist)+1)), mode='lines', name='Val Loss'))
                                        fig_loss.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10), title='Training Progress (Loss)')
                                        loss_chart_ph.plotly_chart(fig_loss, use_container_width=True)
                                    except Exception:
                                        pass
                                st.session_state['gae_training_active'] = True
                                fit_summary = gae_local.fit(feat_df_full.values, feature_names=feat_names, progress_callback=_cb)
                                st.session_state['gae_training_active'] = False
                                # Save model
                                try:
                                    cfg_tmp = _Cfg()
                                    save_path = os.path.join(cfg_tmp.models_dir, 'general_autoencoder.pkl')
                                    gae_local.save_model(save_path)
                                    st.success("General autoencoder trained & saved." if start_train else "General autoencoder quick retrain complete.")
                                    st.code(save_path)
                                    st.json(fit_summary)
                                    # Clear cached loader so new model loads
                                    load_general_autoencoder.clear()
                                    # Inline feature importance (top 15)
                                    try:
                                        import plotly.express as px
                                        fi_df = gae_local.get_feature_importance(feat_df_full.values)
                                        top_fi = fi_df.head(15)
                                        fi_fig = px.bar(top_fi, x='importance', y='feature', orientation='h', title='Top 15 Feature Importance (Reconstruction Error)', height=450)
                                        fi_fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
                                        feature_importance_ph.plotly_chart(fi_fig, use_container_width=True)
                                    except Exception as _fe:
                                        st.warning(f"Feature importance unavailable: {_fe}")
                                except Exception as se:
                                    st.warning(f"Model trained but save failed: {se}")
                            except Exception as te:
                                st.error(f"Training failed: {te}")
                gae = load_general_autoencoder()
                if gae is None:
                    st.caption("General autoencoder not available (train it above).")
                else:
                    try:
                        # Build feature matrix for current site/sector over selected KPIs (or all available if mismatch)
                        use_kpis = st.session_state.get('gae_selected_kpis') or [c for c in filtered.columns if c not in ['Date','Site_ID','Sector_ID']]
                        use_kpis = [k for k in use_kpis if k in filtered.columns]
                        # Attempt to reconstruct engineered feature set if model trained with engineering
                        raw_feat_df = filtered[use_kpis].copy()
                        feat_df = raw_feat_df.fillna(0.0)
                        expected_features = list(getattr(gae, 'feature_names', []))
                        used_engineering = False
                        if expected_features and len(expected_features) > len(use_kpis):
                            try:
                                from telecom_ai_platform.models.enhanced_autoencoder import engineer_features, build_feature_subset
                                eng_df = engineer_features(filtered, use_kpis)
                                eng_feat_df, eng_names = build_feature_subset(eng_df, use_kpis)
                                # Reorder to expected_features intersection
                                cols = [c for c in expected_features if c in eng_feat_df.columns]
                                if len(cols) >= 1:
                                    feat_df = eng_feat_df[cols].fillna(0.0)
                                    used_engineering = True
                            except Exception as fe:
                                st.warning(f"Feature engineering reconstruction failed, falling back to raw KPIs: {fe}")
                        # Dimension adaptation fallback if still mismatched
                        if expected_features:
                            exp_dim = len(expected_features)
                            cur_dim = feat_df.shape[1]
                            if cur_dim != exp_dim:
                                arr_tmp = feat_df.values
                                if cur_dim == 1 and exp_dim > 1:
                                    arr_tmp = np.repeat(arr_tmp, exp_dim, axis=1)
                                elif cur_dim > exp_dim:
                                    arr_tmp = arr_tmp[:, :exp_dim]
                                else:  # pad zeros
                                    pad = exp_dim - cur_dim
                                    arr_tmp = np.hstack([arr_tmp, np.zeros((arr_tmp.shape[0], pad))])
                                feat_df = pd.DataFrame(arr_tmp, columns=expected_features[:arr_tmp.shape[1]])
                        if used_engineering:
                            st.caption(f"Evaluation used engineered features ({feat_df.shape[1]} features)")
                        else:
                            if expected_features and feat_df.shape[1] == len(expected_features):
                                st.caption("Evaluation using raw features matching trained set")
                            elif expected_features:
                                st.caption(f"Feature dimension adapted: model expects {len(expected_features)}, provided {feat_df.shape[1]}")
                        if feat_df.empty:
                            st.info("No KPI data available for reconstruction with selected features.")
                        else:
                            arr = feat_df.values
                            out = gae.detect_anomalies(arr)
                            errs = out['errors']; base_mask = out['is_anomaly']
                            zscores = out.get('z_scores'); psi = out.get('psi'); probs = out.get('probabilities')
                            # Anomaly mode selection
                            anom_mode = st.selectbox("Anomaly Mode", ["reconstruction_error","probability","z_score"], key='gae_anom_mode')
                            if anom_mode == 'probability' and probs is not None:
                                prob_cut = st.session_state.get('gae_prob_cutoff',0.05)
                                mask = probs < prob_cut
                            elif anom_mode == 'z_score' and zscores is not None:
                                z_cut = st.session_state.get('gae_z_cutoff',3.0)
                                mask = zscores > z_cut
                            else:
                                mask = base_mask
                            # Drift alert classification
                            drift_msg = None; drift_level = None
                            if psi == psi:  # not NaN
                                if psi < 0.1:
                                    drift_level = 'low'
                                elif psi < 0.25:
                                    drift_level = 'moderate'
                                else:
                                    drift_level = 'high'
                                drift_msg = f"PSI={psi:.3f} ({drift_level})"
                                # Maintain history
                                hist_list = st.session_state.get('gae_psi_history', [])
                                hist_list.append({'ts': pd.Timestamp.utcnow().isoformat(), 'psi': float(psi)})
                                st.session_state['gae_psi_history'] = hist_list[-200:]
                            cga, cgb, cgc, cgd, cge = st.columns(5)
                            with cga: st.metric("Samples", len(errs))
                            with cgb: st.metric("Mean Err", f"{errs.mean():.4f}")
                            with cgc: st.metric("Anomalies", int(mask.sum()))
                            with cgd: st.metric("Mode", anom_mode)
                            with cge: st.metric("PSI", drift_msg or "N/A")
                            if drift_level == 'moderate':
                                st.warning("Moderate distribution shift detected vs training baseline.")
                            elif drift_level == 'high':
                                st.error("Significant drift detected – consider retraining.")
                            # Drift history chart
                            # Ensure plotly.graph_objects imported BEFORE first use (psi history uses go)
                            import plotly.graph_objects as go  # moved up to avoid UnboundLocalError
                            if st.session_state.get('gae_psi_history'):
                                psi_df = pd.DataFrame(st.session_state['gae_psi_history'])
                                psi_df['ts'] = pd.to_datetime(psi_df['ts'])
                                psi_fig = go.Figure(); psi_fig.add_trace(go.Scatter(x=psi_df['ts'], y=psi_df['psi'], mode='lines+markers', name='PSI'))
                                psi_fig.update_layout(height=160, margin=dict(l=10,r=10,t=30,b=10), title='PSI History')
                                st.plotly_chart(psi_fig, use_container_width=True)
                            rec_fig = go.Figure()
                            rec_fig.add_trace(go.Scatter(x=filtered['Date'], y=errs, mode='lines', name='Reconstruction Error', hovertemplate='Date=%{x}<br>Error=%{y:.4f}<extra></extra>'))
                            # Provide context: show selected KPI raw values scaled (optional) if available
                            if kpi in filtered.columns:
                                try:
                                    vals = filtered[kpi].astype(float)
                                    # Normalize KPI values to error scale for dual-display (min-max)
                                    vmin, vmax = vals.min(), vals.max()
                                    if vmax > vmin:
                                        norm_vals = (vals - vmin) / (vmax - vmin)
                                        # Scale to 90% of max error for visibility
                                        scale = (np.max(errs) * 0.9) if len(errs) else 1.0
                                        rec_fig.add_trace(go.Scatter(x=filtered['Date'], y=norm_vals * scale, mode='lines', name=f'{kpi} (normalized)', line=dict(dash='dot', width=1), opacity=0.6, hovertemplate='Date=%{x}<br>Norm {kpi}=%{y:.4f}<extra></extra>'))
                                except Exception:
                                    pass
                            if gae.threshold is not None:
                                rec_fig.add_hline(y=gae.threshold, line_color='red', line_dash='dash', annotation=dict(text='Threshold'))
                            if mask.any():
                                rec_fig.add_trace(go.Scatter(x=filtered['Date'][mask], y=errs[mask], mode='markers', name='Anomaly', marker=dict(color='red', size=8, symbol='x')))
                            rec_fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10), title=f"General AE Reconstruction Error (Mode: {anom_mode})")
                            st.plotly_chart(rec_fig, use_container_width=True)
                            dist_fig = go.Figure()
                            dist_fig.add_trace(go.Histogram(x=errs, nbinsx=40, name='Errors', marker_color='#88c'))
                            if gae.threshold is not None:
                                dist_fig.add_vline(x=gae.threshold, line_color='red', line_dash='dash')
                            dist_fig.update_layout(height=250, margin=dict(l=10,r=10,t=40,b=10), title="Reconstruction Error Distribution")
                            st.plotly_chart(dist_fig, use_container_width=True)
                            with st.expander("Per-sample reconstruction (head)", expanded=False):
                                head_n = min(100, len(errs))
                                tbl = pd.DataFrame({
                                    'Date': pd.to_datetime(filtered['Date']).dt.strftime('%Y-%m-%d %H:%M:%S'),
                                    'Reconstruction_Error': errs,
                                    'GenAE_Anomaly': mask
                                }).head(head_n)
                                st.dataframe(tbl, use_container_width=True)
                            # Feature importance
                            with st.expander("Feature Importance & Per-Feature Error", expanded=False):
                                try:
                                    imp_df = gae.get_feature_importance(arr)
                                    import plotly.express as px
                                    top_n = st.slider("Top N", 5, min(50, len(imp_df)), min(20, len(imp_df)), 1, key='gae_feat_topn') if len(imp_df) > 5 else len(imp_df)
                                    show_df = imp_df.head(top_n)
                                    bar_fig = px.bar(show_df[::-1], x='importance', y='feature', orientation='h')
                                    bar_fig.update_layout(height=400, margin=dict(l=10,r=10,t=40,b=10))
                                    st.plotly_chart(bar_fig, use_container_width=True)
                                    std_imp = imp_df.copy()
                                    std_imp['std_z'] = (std_imp['importance'] - std_imp['importance'].mean()) / (std_imp['importance'].std()+1e-12)
                                    st.dataframe(show_df.merge(std_imp[['feature','std_z']], on='feature'), use_container_width=True)
                                except Exception as fe:
                                    st.warning(f"Feature importance failed: {fe}")
                    except Exception as e:
                        st.warning(f"General autoencoder evaluation failed: {e}")
            else:
                st.info("Select Site, Sector, and KPI in the sidebar.")

        with tabs[1]:
            if site and sector and multi_kpi and len(multi_kpi) >= 2:
                fig = create_multi_kpi_comparison(data, site, sector, multi_kpi)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("KPIs are normalized to 0-1 for comparison.")
            else:
                st.info("Select Site, Sector, and at least two KPIs in the sidebar.")

        with tabs[2]:
            if site and kpi:
                fig = create_sector_comparison(data, site, kpi)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                # Sector stats
                site_df = data[data['Site_ID'] == site]
                sectors = site_df['Sector_ID'].unique()
                stats_rows = []
                det_for_stats = get_ui_detector()
                for sec in sectors:
                    sec_df = site_df[site_df['Sector_ID'] == sec]
                    if kpi in sec_df.columns and not sec_df.empty:
                        # Prefer model-based anomaly counts when detectors are available
                        if det_for_stats is not None and getattr(det_for_stats, 'is_fitted', False):
                            try:
                                res = det_for_stats.detect_anomalies(sec_df, kpi_name=kpi)
                                an_count = sum(1 for r in res if r.kpi_name == kpi and r.is_anomaly)
                            except Exception:
                                an_count = len(detect_anomalies(sec_df, kpi))
                        else:
                            an_count = len(detect_anomalies(sec_df, kpi))
                        stats_rows.append({
                            'Sector': sec,
                            'Mean': round(sec_df[kpi].mean(), 2),
                            'Min': round(sec_df[kpi].min(), 2),
                            'Max': round(sec_df[kpi].max(), 2),
                            'Std Dev': round(sec_df[kpi].std(), 2),
                            'Anomalies': an_count
                        })
                if stats_rows:
                    st.subheader("Sector Statistics")
                    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
            else:
                st.info("Select Site and KPI in the sidebar.")

        with tabs[3]:
            if kpi:
                fig = create_site_sector_kpi_heatmap(data, kpi)
                if fig:
                    st.plotly_chart(fig, use_container_width=False)
            else:
                st.info("Select a KPI in the sidebar.")

        with tabs[4]:
            if site and sector and kpi:
                filtered = data[(data['Site_ID'] == site) & (data['Sector_ID'] == sector)]
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Records", len(filtered))
                with c2:
                    st.metric(f"Avg {kpi}", f"{filtered[kpi].mean():.2f}")
                with c3:
                    # Prefer model-based anomaly count
                    det_for_overview = get_ui_detector()
                    if det_for_overview is not None and getattr(det_for_overview, 'is_fitted', False):
                        try:
                            res = det_for_overview.detect_anomalies(filtered, kpi_name=kpi)
                            an_cnt = sum(1 for r in res if r.kpi_name == kpi and r.is_anomaly)
                        except Exception:
                            an_cnt = len(detect_anomalies(filtered, kpi))
                    else:
                        an_cnt = len(detect_anomalies(filtered, kpi))
                    st.metric("Anomalies", an_cnt)
                with c4:
                    st.metric("Std Dev", f"{filtered[kpi].std():.2f}")
                with st.expander("View Raw Data"):
                    disp = filtered[['Date', 'Site_ID', 'Sector_ID', kpi]].copy()
                    disp['Date'] = disp['Date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(disp.sort_values('Date'), use_container_width=True)
            else:
                st.info("Select Site, Sector, and KPI in the sidebar.")

        with tabs[5]:
            st.subheader("Chain-of-Thought and MCP Overview")
            st.markdown("""
            Chain-of-Thought
            - The assistant breaks a query into steps: extract entities (site, sector, KPI), decide tools (web/MCP), compute statistics from the dataset, and synthesize an answer.
            - Reasoning steps are displayed in the chat panel for transparency.

            MCP Architecture (Block Diagram)
            """)
            st.code(
                """
                +-----------------------------+           +-----------------------+
                |       Streamlit UI          |           |   Web Search (DDG)   |
                |  - Filters (Site/Sector)    |<--------->|  Supporting Context   |
                |  - Visualizations           |           +-----------------------+
                |  - LLM Chat (CoT)           |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |   Chain-of-Thought Agent    |
                | - Plans reasoning steps     |
                | - Chooses MCP tools/Web     |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |         MCP Bridge          |
                | - Connects to MCP server    |
                | - Discovers tools/resources |
                | - Executes tool calls       |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |         MCP Server          |
                | - KPI analyzer, anomalies   |
                | - Trend predictor, reports  |
                | - Model parameters served   |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |        Data Sources         |
                | - AD_data_10KPI.csv         |
                | - Trained models, configs   |
                +-----------------------------+
                """,
                language="text",
            )

        # --- KPI Fine Tuning ---
    with tabs[6]:
            st.subheader("KPI Fine Tuning")
            with st.expander("Model Catalog", expanded=False):
                st.dataframe(pd.DataFrame(get_model_catalog()), use_container_width=True)
            kpi_list = [c for c in data.columns if c not in ['Date', 'Site_ID', 'Sector_ID']]
            ft_kpi = st.selectbox("Select KPI to tune", sorted(kpi_list) if kpi_list else [], key="ft_kpi")
            if not ft_kpi:
                st.info("Select a KPI to begin tuning.")
            else:
                with st.expander("Data Availability (per Site/Sector for selected KPI)", expanded=False):
                    try:
                        avail_df = data[['Site_ID','Sector_ID', ft_kpi]].copy()
                        summary = avail_df.groupby(['Site_ID','Sector_ID']).agg(
                            total_rows=('Site_ID','size'),
                            non_null=(ft_kpi, lambda s: s.notna().sum())
                        ).reset_index()
                        summary['non_null_pct'] = (summary['non_null'] / summary['total_rows'] * 100).round(2)
                        st.dataframe(summary.sort_values(['Site_ID','Sector_ID']), use_container_width=True)
                    except Exception as e:
                        st.write(f"Availability check failed: {e}")
                # Filters for training/preview
                c1, c2, c3 = st.columns(3)
                with c1:
                    sites_all = sorted(data['Site_ID'].unique())
                    ft_site = st.selectbox("Site", sites_all, index=0, key="tune_site")
                with c2:
                    secs_for_site = sorted(data[data['Site_ID'] == ft_site]['Sector_ID'].unique()) if sites_all else []
                    ft_sector = st.selectbox("Sector", secs_for_site, index=0, key="tune_sector")
                with c3:
                    ft_days = st.number_input("Training/Preview window (days)", min_value=7, max_value=180, value=60, step=1, key="tune_days")

                # Algorithm choices (include advanced ones actually implemented)
                internal_default = KPISpecificDetector.KPI_ALGORITHM_MAP.get(ft_kpi, 'isolation_forest')
                algo_options = [
                    'isolation_forest', 'local_outlier_factor', 'one_class_svm', 'gaussian_mixture',
                    'autoencoder', 'ensemble_if_gmm', 'time_series_decomposition', 'seasonal_hybrid_esd'
                ]
                # Preselect mapped algorithm
                selected_algo = st.selectbox("Algorithm", options=algo_options, index=algo_options.index(internal_default) if internal_default in algo_options else 0, key="tune_algo")
                st.caption(f"Default mapping for {ft_kpi}: {internal_default}")

                # Parameter widgets (debounced training) - collect into params dict
                st.markdown("---")
                st.markdown("### Hyperparameters")
                params: Dict[str, any] = {}
                if selected_algo == 'isolation_forest':
                    params['contamination'] = st.slider("contamination", 0.01, 0.5, 0.05, 0.01)
                    params['n_estimators'] = st.slider("n_estimators", 50, 500, 100, 10)
                    params['random_state'] = 42
                elif selected_algo == 'local_outlier_factor':
                    params['n_neighbors'] = st.slider("n_neighbors", 5, 100, 20, 1)
                    params['contamination'] = st.slider("contamination", 0.01, 0.5, 0.1, 0.01)
                elif selected_algo == 'one_class_svm':
                    params['nu'] = st.slider("nu", 0.01, 0.5, 0.05, 0.01)
                    params['kernel'] = st.selectbox("kernel", ['rbf', 'linear', 'poly', 'sigmoid'])
                    params['gamma'] = st.selectbox("gamma", ['scale', 'auto'])
                elif selected_algo == 'gaussian_mixture':
                    params['n_components'] = st.slider("n_components", 1, 10, 2, 1)
                    params['random_state'] = 42
                elif selected_algo == 'autoencoder':
                    params['encoding_dim'] = st.slider("encoding_dim", 4, 256, 32, 4)
                    params['epochs'] = st.slider("epochs", 10, 500, 50, 10)
                    params['learning_rate'] = st.number_input("learning_rate", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-4, format='%e')
                    params['batch_size'] = st.slider("batch_size", 16, 512, 64, 16)
                    params['sequence_length'] = st.slider("sequence_length (LSTM if >1 for Throughput KPIs)", 1, 30, 7, 1)
                elif selected_algo == 'time_series_decomposition':
                    params['period'] = st.slider("Seasonal Period (days)", 3, 30, 7, 1)
                    params['use_prophet'] = st.checkbox("Try Prophet residual enhancement", value=True)
                elif selected_algo == 'seasonal_hybrid_esd':
                    params['seasonal_window'] = st.slider("Seasonal Window (approx period)", 3, 30, 7, 1)
                    params['zscore_sigma'] = st.slider("Z-score Sigma for candidate anomalies", 1.0, 6.0, 3.0, 0.5)
                    params['max_anom_frac'] = st.slider("Max anomaly fraction", 0.01, 0.5, 0.1, 0.01)
                # --- Prepare training slice ---
                cfg = _Cfg()
                train_df = data[(data['Site_ID'] == ft_site) & (data['Sector_ID'] == ft_sector)].copy()
                if 'Date' in train_df.columns and len(train_df):
                    try:
                        train_df['Date'] = pd.to_datetime(train_df['Date'])
                        cutoff = train_df['Date'].max() - pd.Timedelta(days=int(ft_days))
                        train_df = train_df[train_df['Date'] >= cutoff]
                    except Exception:
                        pass
                X = train_df[[ft_kpi]].fillna(0.0).values if ft_kpi in train_df.columns else np.empty((0,1))

                tuned_key = f"tuned_model__{ft_kpi}"
                last_sig_key = f"tuned_signature__{ft_kpi}"
                current_signature = f"{ft_site}|{ft_sector}|{ft_kpi}|{ft_days}|{selected_algo}|{params}"
                train_clicked = st.button("Train / Preview Model", key="train_preview_btn")

                def _train():
                    if X.shape[0] == 0:
                        st.warning("No data available for selected filters.")
                        return None
                    det = KPISpecificDetector(ft_kpi, cfg)
                    det.algorithm = selected_algo
                    det.params = params.copy()
                    # Fit according to algorithm
                    if selected_algo == 'isolation_forest':
                        from sklearn.ensemble import IsolationForest
                        Xs = det.scaler.fit_transform(X)
                        det.model = IsolationForest(contamination=float(params['contamination']), n_estimators=int(params['n_estimators']), random_state=42).fit(Xs)
                        det.is_fitted = True
                        det._calculate_threshold(Xs)
                    elif selected_algo == 'local_outlier_factor':
                        from sklearn.neighbors import LocalOutlierFactor
                        Xs = det.scaler.fit_transform(X)
                        det.model = LocalOutlierFactor(n_neighbors=int(params['n_neighbors']), contamination=float(params['contamination']), novelty=True).fit(Xs)
                        det.is_fitted = True
                        det._calculate_threshold(Xs)
                    elif selected_algo == 'one_class_svm':
                        from sklearn.svm import OneClassSVM
                        Xs = det.scaler.fit_transform(X)
                        det.model = OneClassSVM(nu=float(params['nu']), kernel=params['kernel'], gamma=params['gamma']).fit(Xs)
                        det.is_fitted = True
                        det._calculate_threshold(Xs)
                    elif selected_algo == 'gaussian_mixture':
                        from sklearn.mixture import GaussianMixture
                        Xs = det.scaler.fit_transform(X)
                        det.model = GaussianMixture(n_components=int(params['n_components']), random_state=42).fit(Xs)
                        det.is_fitted = True
                        det._calculate_threshold(Xs)
                    elif selected_algo == 'autoencoder':
                        # Inject chosen hyperparameters into config before fit
                        try:
                            det.config.model.autoencoder_params['encoding_dim'] = int(params['encoding_dim'])
                            det.config.model.autoencoder_params['epochs'] = int(params['epochs'])
                            det.config.model.autoencoder_params['learning_rate'] = float(params['learning_rate'])
                            det.config.model.autoencoder_params['batch_size'] = int(params['batch_size'])
                            det.config.model.sequence_length = int(params['sequence_length'])
                        except Exception:
                            pass
                        det.fit(X)
                    elif selected_algo == 'ensemble_if_gmm':
                        from sklearn.ensemble import IsolationForest
                        from sklearn.mixture import GaussianMixture
                        Xs = det.scaler.fit_transform(X)
                        det.model = {
                            'if': IsolationForest(contamination=float(params.get('contamination', 0.05)), n_estimators=int(params.get('if_n_estimators', 100)), random_state=42).fit(Xs),
                            'gmm': GaussianMixture(n_components=2, random_state=42).fit(Xs),
                            'weights': (float(params.get('weight_if', 0.5)), float(params.get('weight_gmm', 0.5)))
                        }
                        det.is_fitted = True
                        scores = det._get_anomaly_scores(Xs)
                        det.threshold = np.percentile(scores, (1 - cfg.model.contamination_rate) * 100)
                    elif selected_algo == 'time_series_decomposition':
                        det.fit(X)
                    elif selected_algo == 'seasonal_hybrid_esd':
                        det.fit(X)
                    else:
                        st.error('Unsupported algorithm.')
                        return None
                    return det

                if train_clicked or (st.session_state.get(tuned_key) is not None and st.session_state.get(last_sig_key) != current_signature):
                    # Train if user requested OR params/site/sector/days changed since last preview
                    with st.spinner("Training model..."):
                        tuned = _train()
                        st.session_state[tuned_key] = tuned
                        st.session_state[last_sig_key] = current_signature
                        # Persist last-used params per KPI (sidecar JSON)
                        try:
                            cfgp = _Cfg(); sidecar = os.path.join(cfgp.models_dir, f"{ft_kpi}_last_params.json")
                            with open(sidecar, 'w') as f:
                                json.dump({'kpi': ft_kpi, 'algorithm': selected_algo, 'params': params}, f, indent=2)
                        except Exception:
                            pass
                tuned = st.session_state.get(tuned_key)
                if tuned is not None:
                    # Preview anomalies on training slice
                    class _PreviewAdapter:
                        is_fitted = True
                        def detect_anomalies(self, df, kpi_name=None, **_):
                            # Build results with real timestamps so plotting aligns
                            series_df = df.copy()
                            arr = series_df[[ft_kpi]].fillna(method='ffill').fillna(0.0).values
                            try:
                                results = tuned.predict(arr)
                            except Exception:
                                arr = series_df[[ft_kpi]].fillna(0.0).values
                                results = tuned.predict(arr)
                            # Attach metadata (Date, Site, Sector) to each result
                            dates = pd.to_datetime(series_df['Date']).tolist() if 'Date' in series_df.columns else [None]*len(results)
                            for i, r in enumerate(results):
                                if i < len(dates) and dates[i] is not None:
                                    r.timestamp = dates[i]
                                r.site_id = ft_site
                                r.sector_id = ft_sector
                            return results
                    adapter = _PreviewAdapter()
                    show_thr_preview = st.checkbox("Show model threshold view", value=True, key="ft_show_thr")
                    fig, anomalies_df = create_kpi_visualization(
                        data=train_df,
                        site_id=ft_site,
                        sector_id=ft_sector,
                        kpi=ft_kpi,
                        detector=adapter,
                        last_days=int(ft_days),
                        show_model_threshold=show_thr_preview,
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    if anomalies_df is not None and not anomalies_df.empty:
                        st.dataframe(anomalies_df.sort_values('Date', ascending=False), use_container_width=True)
                    else:
                        st.info("No anomalies detected with current settings on the selected window.")

                    # Diagnostics panel
                    with st.expander("Diagnostics", expanded=True):
                        try:
                            X_diag = train_df[[ft_kpi]].fillna(0.0).values
                            diag = tuned.get_diagnostics(X_diag)
                            if 'error' in diag:
                                st.warning(f"Diagnostics unavailable: {diag['error']}")
                            else:
                                cda, cdb, cdc, cdd = st.columns(4)
                                with cda: st.metric("Algorithm", diag.get('algorithm'))
                                with cdb: st.metric("Threshold", f"{diag.get('threshold'):.3f}" if diag.get('threshold') else "-")
                                with cdc: st.metric("Flagged", f"{diag.get('num_flagged')}/{diag.get('total')}")
                                with cdd: st.metric("Score Range", f"{diag.get('score_min'):.2f}-{diag.get('score_max'):.2f}")
                                # Mini histogram of scores
                                try:
                                    import plotly.express as px
                                    scores = []
                                    # reuse tuned._get_anomaly_scores if accessible
                                    if hasattr(tuned, '_get_anomaly_scores'):
                                        # need scaled input for most algos except time series ones
                                        if tuned.algorithm in ["time_series_decomposition", "seasonal_hybrid_esd"]:
                                            Xp = X_diag.astype(float)
                                        else:
                                            # dimension adapt like predict
                                            Xp = X_diag
                                            expected_dim = int(getattr(tuned.scaler, 'mean_', np.array([0])).shape[0])
                                            if Xp.shape[1] != expected_dim and expected_dim > 0:
                                                if Xp.shape[1] == 1:
                                                    Xp = np.repeat(Xp, expected_dim, axis=1)
                                                elif Xp.shape[1] > expected_dim:
                                                    Xp = Xp[:, :expected_dim]
                                                else:
                                                    pad = expected_dim - Xp.shape[1]
                                                    Xp = np.hstack([Xp, np.zeros((Xp.shape[0], pad))])
                                            try:
                                                Xp = tuned.scaler.transform(Xp)
                                            except Exception:
                                                pass
                                        scores = tuned._get_anomaly_scores(Xp)
                                    if len(scores):
                                        import plotly.graph_objects as go
                                        hist_fig = go.Figure()
                                        hist_fig.add_trace(go.Histogram(x=scores, nbinsx=30, name='Scores', marker_color='#6699cc'))
                                        if diag.get('threshold') is not None:
                                            hist_fig.add_vline(x=diag['threshold'], line_color='red', line_dash='dash')
                                        hist_fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10))
                                        st.plotly_chart(hist_fig, use_container_width=True)
                                except Exception:
                                    pass
                        except Exception as e:
                            st.warning(f"Diagnostics error: {e}")

                col_sv1, col_sv2 = st.columns([1,3])
                with col_sv1:
                    if st.button("Save tuned model", key="tune_save"):
                        tuned = st.session_state.get(tuned_key)
                        if not tuned:
                            st.warning("Train a model first.")
                        else:
                            try:
                                cfg = _Cfg()
                                models_dir = cfg.models_dir
                                path = os.path.join(models_dir, f"{ft_kpi}_detector.pkl")
                                # If LOF without novelty (older preview before patch), retrain quickly with same params enabling novelty for consistent scoring
                                if tuned.algorithm == 'local_outlier_factor' and not hasattr(tuned.model, 'score_samples'):
                                    try:
                                        from sklearn.neighbors import LocalOutlierFactor
                                        Xref = data[(data['Site_ID'] == ft_site) & (data['Sector_ID'] == ft_sector)][[ft_kpi]].dropna().values
                                        Xref = tuned.scaler.fit_transform(Xref) if Xref.shape[0] > 0 else Xref
                                        tuned.model = LocalOutlierFactor(n_neighbors=int(tuned.params.get('n_neighbors', 20)), contamination=float(tuned.params.get('contamination', 0.1)), novelty=True)
                                        if Xref.shape[0] > 0:
                                            tuned.model.fit(Xref)
                                            # Recompute threshold with new scoring function
                                            tuned._calculate_threshold(Xref)
                                    except Exception:
                                        pass
                                tuned.save_model(path)
                                ui_det = get_ui_detector()
                                if ui_det is None:
                                    ui_det = KPIAnomalyDetector(cfg)
                                    st.session_state['ui_detector'] = ui_det
                                ui_det.detectors[ft_kpi] = tuned
                                ui_det.is_fitted = True
                                # Mark fresh model timestamp to force main view refresh
                                st.session_state[f"model_refresh__{ft_kpi}"] = datetime.utcnow().isoformat()
                                # Update catalog displayed params
                                cat = get_model_catalog().copy()
                                kv = [f"{k}={v}" for k,v in params.items()]
                                for row in cat:
                                    if row.get('KPI') == ft_kpi:
                                        row['Key Parameters'] = ", ".join(kv)
                                        break
                                st.session_state['model_catalog'] = cat
                               
                                st.success(f"Saved tuned {ft_kpi} model to {path}")
                            except Exception as e:
                                st.error(f"Save failed: {e}")
                with col_sv2:
                    st.caption("Training is explicit now. Adjust parameters then click 'Train / Preview Model'. Only 'Save tuned model' updates production.")
            with tabs[7]:
                st.subheader("Site/Sector Completeness Audit")
                # Build completeness table: count records per Site/Sector, last date, missing KPIs (% null)
                try:
                    kpi_cols = [c for c in data.columns if c not in ['Date','Site_ID','Sector_ID']]
                    grouped = data.groupby(['Site_ID','Sector_ID'])
                    rows = []
                    for (s, sec), g in grouped:
                        recs = len(g)
                        last_date = pd.to_datetime(g['Date']).max() if 'Date' in g else None
                        kpi_avail = {k: 100.0 * (1 - g[k].isna().mean()) if k in g else 0.0 for k in kpi_cols}
                        completeness = round(np.mean(list(kpi_avail.values())), 2) if kpi_avail else 0.0
                        rows.append({
                            'Site_ID': s,
                            'Sector_ID': sec,
                            'Records': recs,
                            'Last_Date': last_date.strftime('%Y-%m-%d') if last_date else '-',
                            'Avg KPI % Present': completeness,
                        })
                    audit_df = pd.DataFrame(rows).sort_values(['Site_ID','Sector_ID'])
                    st.dataframe(audit_df, use_container_width=True)
                    st.caption("Avg KPI % Present = mean of non-null percentage across all KPI columns for that Site/Sector.")
                except Exception as e:
                    st.error(f"Audit failed: {e}")

    with col_right:
        st.subheader("LLM Assistant")
        user_query = st.text_area("Ask about network performance, models, or anomalies:", height=120,
                                  placeholder="Example: Analyze SINR anomalies for SITE_001 sector A and explain model choices")
        # Show LLM readiness hint
        from os import getenv
        if not getenv("GEMINI_API_KEY"):
            st.warning("GEMINI_API_KEY not found in environment. .env should contain it; we attempt to load it via python-dotenv. If you still see this, restart the app or set the variable in your shell.")
            if st.button("Retry LLM initialization"):
                # Minimal lazy init attempt
                try:
                    from telecom_ai_platform.agents.chain_of_thought import ChainOfThoughtAgent
                    from telecom_ai_platform.core.config import TelecomConfig
                    st.session_state['cot_agent'] = ChainOfThoughtAgent(TelecomConfig())
                    st.success("LLM initialized.")
                except Exception as e:
                    st.error(f"LLM init failed: {e}")
    if st.button("Send") and user_query.strip():
            # Lightly parse query to sync sidebar (site, KPI, time window) for visualization
            try:
                ql = user_query.lower()
                # Site pattern: "site 1" or "site: 1" or "site_1"
                site_match = re.search(r"site\s*[:#_-]?\s*([\w-]+)", ql)
                if site_match:
                    canon_site = canonicalize_site(site_match.group(1), data)
                    if canon_site:
                        st.session_state['selected_site'] = canon_site
                # KPI: pick known KPI names present in data
                known_kpis = [c for c in data.columns if c not in ['Date', 'Site_ID', 'Sector_ID']]
                for candidate in known_kpis:
                    if candidate.lower() in ql:
                        st.session_state['selected_kpi'] = candidate
                        break
                # Time window: last week -> 7 days
                if re.search(r"last\s+week", ql):
                    st.session_state['last_days'] = 7
                # Sector: default to first sector for selected site if sector not specified
                if st.session_state.get('selected_site') and not st.session_state.get('selected_sector'):
                    site_df = data[data['Site_ID'] == st.session_state['selected_site']]
                    sectors_avail = site_df['Sector_ID'].unique().tolist() if not site_df.empty else []
                    if sectors_avail:
                        st.session_state['selected_sector'] = sectors_avail[0]
            except Exception:
                pass
            # Fast-path: answer directly from dataset for common anomaly queries
            fast = fast_answer_if_applicable(user_query.strip(), data)
            if fast:
                st.markdown("**Reasoning Steps**")
                for i, s in enumerate(fast.get('steps', []), start=1):
                    st.markdown(f"{i}. {s}")
                if fast.get('context'):
                    with st.expander("Context Used"):
                        st.code(fast['context'])
                st.markdown("**Answer**")
                st.write(fast.get('answer', ''))
                return

            mcp_bridge = st.session_state.get('mcp_bridge')
            if st.session_state.get('chat_stream_steps') and AGENTS_AVAILABLE and st.session_state.get('cot_agent'):
                # Streaming mode: progressively render steps and final answer
                steps_ph = st.empty()
                context_ph = st.expander("Context Used")
                answer_ph_hdr = st.empty()
                answer_body = st.empty()

                # Compose conversational header
                if st.session_state.get('chat_tone_conversational'):
                    answer_body.markdown("Working on it — I’ll think it through step by step and share as I go…")

                rendered: List[str] = []

                def on_step(step):
                    try:
                        title = step.reasoning_type.value.replace('_', ' ').title()
                        summary = (step.analysis or '').strip()
                        rendered.append(f"Step {step.step_number}: {title} — {summary}")
                        with steps_ph.container():
                            st.markdown("**Reasoning Steps (live)**")
                            for i, s in enumerate(rendered, start=1):
                                st.markdown(f"{i}. {s}")
                    except Exception:
                        pass

                # Run analysis with callback and stream updates
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                cot = st.session_state.get('cot_agent')

                # Build context via simple_llm_chat pre-stage but without LLM phase
                pre = simple_llm_chat(user_query.strip(), mcp=mcp_bridge, skip_llm=True)
                if pre.get('context'):
                    with context_ph:
                        st.code(pre['context'])

                try:
                    result = loop.run_until_complete(
                        cot.analyze_with_chain_of_thought(
                            query=user_query.strip(),
                            context=pre,  # pass collected context
                            use_web_search=True,
                            use_mcp=True,
                            progress_cb=on_step,
                        )
                    )
                except Exception as e:
                    answer_body.error(f"LLM error: {e}")
                    return

                # Final answer streaming (token-by-token) using a background producer and a Queue
                import threading
                from queue import Queue, Empty
                import time as _time

                token_queue: Queue = Queue()
                _SENTINEL = object()

                def producer():
                    # Run async streaming in this background thread with its own event loop
                    import asyncio as _asyncio
                    def _push_token(t: str):
                        try:
                            token_queue.put(t)
                        except Exception:
                            pass
                    try:
                        _loop = _asyncio.new_event_loop()
                        _asyncio.set_event_loop(_loop)
                        final_text_local = _loop.run_until_complete(
                            cot.stream_final_conclusion(
                                query=user_query.strip(),
                                reasoning_chain=result.reasoning_chain,
                                token_cb=_push_token,
                            )
                        )
                        # Do not touch Streamlit APIs in background thread
                    except Exception:
                        # On error, push the non-streaming answer
                        try:
                            token_queue.put(result.final_conclusion or "")
                        except Exception:
                            pass
                    finally:
                        token_queue.put(_SENTINEL)

                # Start producer thread
                threading.Thread(target=producer, daemon=True).start()

                # Consume tokens on main thread and render incrementally
                answer_ph_hdr.markdown("**Answer**")
                buffer = ""
                while True:
                    try:
                        item = token_queue.get(timeout=0.2)
                    except Empty:
                        _time.sleep(0.05)
                        continue
                    if item is _SENTINEL:
                        break
                    buffer += str(item)
                    answer_body.markdown(buffer)

                # Finalize output
                final_text = buffer
                answer_body.markdown(final_text)
            else:
                # Non-streaming path (existing)
                result = simple_llm_chat(user_query.strip(), mcp=mcp_bridge)
                if result.get('steps'):
                    st.markdown("**Reasoning Steps**")
                    for i, s in enumerate(result['steps'], start=1):
                        st.markdown(f"{i}. {s}")
                if result.get('context'):
                    with st.expander("Context Used"):
                        st.code(result['context'])
                st.markdown("**Answer**")
                ans_text = result.get('answer', '')
                st.write(ans_text)


if __name__ == "__main__":
    main()
