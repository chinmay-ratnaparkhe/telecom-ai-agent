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
    DETECTOR_AVAILABLE = True
except Exception:
    DETECTOR_AVAILABLE = False


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
                anomalies_df = pd.DataFrame([{
                    'Date': pd.to_datetime(r.timestamp) if 'Date' in filtered.columns else r.timestamp,
                    'Site_ID': r.site_id,
                    'Sector_ID': r.sector_id,
                    'KPI': r.kpi_name,
                    'Value': r.value,
                    'Severity': r.severity,
                    'Score': r.anomaly_score,
                    'Threshold': r.threshold,
                } for r in results if r.is_anomaly])
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
        # Main anomalies series (time plot)
        fig.add_trace(
            go.Scatter(x=anomalies_df['Date'], y=anomalies_df['Value'], mode='markers', name='Anomalies',
                        marker=dict(color='red', size=10, symbol='x'),
                        hovertemplate='<b>Anomaly</b><br><b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'),
            row=1, col=1
        )

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
    # Parse last N days; default 7 if any time-window intent
    m_days = re.search(r"last\s+(\d+)\s*days?", ql)
    days = int(m_days.group(1)) if m_days else (7 if "last week" in ql or "last 7" in ql else None)

    # Case 0: Generic "Which site has highest anomalies ..." with no explicit KPI
    if "which site" in ql and "anomal" in ql and not any(k in ql for k in [
        "sinr", "rsrp", "throughput", "dl_throughput", "ul_throughput", "rtt", "packet", "cpu", "active_users", "handover", "call_drop"
    ]):
        win = _window_by_last_days(data, days or 7)
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
                    f"Filtered dataset to last {days or 7} days (relative to dataset max date)",
                    "Used KPI-specific trained detectors (IF/OCSVM/GMM/AE) across available KPIs",
                    f"Ranked sites by total model-detected anomalies; top is {top_site} with {top_count}",
                ]
                context = "Top 5 counts:\n" + "\n".join([f"{s}: {c}" for s, c in by_site[:5]])
                ans = f"{top_site} has the highest anomalies in the last {days or 7} days (count={top_count})."
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
                    f"Filtered dataset to last {days or 7} days (relative to dataset max date)",
                    f"Used z-score fallback across KPIs: {', '.join(kpis)}",
                    f"Ranked sites by total anomaly count; top is {top_site} with {top_count}",
                ]
                context = "Top 5 counts:\n" + "\n".join([f"{s}: {c}" for s, c in by_site[:5]])
                ans = f"{top_site} has the highest anomalies in the last {days or 7} days (count={top_count})."
                return {"steps": steps, "context": context, "answer": ans}
            else:
                return {"steps": ["No KPIs available for anomaly computation"], "context": "", "answer": "No anomalies found in the requested window."}

    # Case 1: Which site has highest SINR anomalies ...
    if "which site" in ql and "sinr" in ql and "anomal" in ql:
        win = _window_by_last_days(data, days or 7)
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
                    f"Filtered dataset to last {days or 7} days (relative to dataset max date)",
                    "Used KPI-specific SINR detector to count anomalies per site",
                    f"Ranked sites by anomaly count; top is {top_site} with {top_count}",
                ]
                context = "Top 5 counts:\n" + "\n".join([f"{s}: {c}" for s, c in by_site[:5]])
                ans = f"{top_site} has the highest SINR anomalies in the last {days or 7} days (count={top_count})."
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
                    f"Filtered dataset to last {days or 7} days (relative to dataset max date)",
                    "Used z-score fallback for SINR per site",
                    f"Ranked sites by anomaly count; top is {top_site} with {top_count}",
                ]
                context = "Top 5 counts:\n" + "\n".join([f"{s}: {c}" for s, c in by_site[:5]])
                ans = f"{top_site} has the highest SINR anomalies in the last {days or 7} days (count={top_count})."
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
            df = _window_by_last_days(df, days or 7)
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
                        f"Applied last {days or 7} days window (relative to dataset max date)",
                        "Used KPI-specific SINR detector to identify anomalies",
                    ]
                    ctx = "" if not recent_lines else ("Recent anomalies:\n" + "\n".join(recent_lines))
                    ans = f"Detected {cnt} SINR anomalies for {site_id}{' / ' + sector_id if sector_id else ''} in the last {days or 7} days."
                    return {"steps": steps, "context": ctx, "answer": ans}
                else:
                    # Fallback z-score
                    an_df = detect_anomalies(df, 'SINR', threshold=2.0)
                    cnt = len(an_df)
                    recent = an_df.sort_values('Date').tail(3) if cnt else pd.DataFrame()
                    steps = [
                        f"Selected {site_id}{' / ' + sector_id if sector_id else ''}",
                        f"Applied last {days or 7} days window (relative to dataset max date)",
                        "Used z-score fallback on SINR",
                    ]
                    ctx = "" if recent.empty else ("Recent anomalies:\n" + "\n".join([f"{r['Date']}: {r['SINR']:.2f}" for _, r in recent.iterrows()]))
                    ans = f"Detected {cnt} SINR anomalies for {site_id}{' / ' + sector_id if sector_id else ''} in the last {days or 7} days."
                    return {"steps": steps, "context": ctx, "answer": ans}

    # Case 3: Analyze DL Throughput anomalies for Site X Sector Y
    if ("throughput" in ql or "dl_throughput" in ql) and "anomal" in ql and "site" in ql:
        site_m = re.search(r"site\s*[:#_-]?\s*([\w-]+)", ql)
        sec_m = re.search(r"sector\s*[:#_-]?\s*([a-zA-Z])", ql)
        site_id = canonicalize_site(site_m.group(1), data) if site_m else None
        sector_id = canonicalize_sector(site_id, sec_m.group(1), data) if (site_id and sec_m) else None
        if site_id:
            df = data.copy()
            df = df[df['Site_ID'] == site_id]
            if sector_id:
                df = df[df['Sector_ID'] == sector_id]
            df = _window_by_last_days(df, days or 15)
            if 'DL_Throughput' in df.columns and not df.empty:
                det = get_ui_detector()
                if det is not None and getattr(det, 'is_fitted', False):
                    try:
                        results = det.detect_anomalies(df, kpi_name='DL_Throughput')
                        anomalies = [r for r in results if r.kpi_name == 'DL_Throughput' and r.is_anomaly and (not sector_id or (r.sector_id == sector_id))]
                    except Exception:
                        anomalies = []
                    cnt = len(anomalies)
                    recent_lines = []
                    for r in sorted(anomalies, key=lambda x: str(x.timestamp))[-3:]:
                        recent_lines.append(f"{r.timestamp}: {r.value:.2f} (score={r.anomaly_score:.2f}, sev={r.severity})")
                    steps = [
                        f"Selected {site_id}{' / ' + sector_id if sector_id else ''}",
                        f"Applied last {days or 15} days window (relative to dataset max date)",
                        "Used KPI-specific DL_Throughput detector to identify anomalies",
                    ]
                    ctx = "" if not recent_lines else ("Recent anomalies:\n" + "\n".join(recent_lines))
                    ans = f"Detected {cnt} DL Throughput anomalies for {site_id}{' / ' + sector_id if sector_id else ''} in the last {days or 15} days."
                    return {"steps": steps, "context": ctx, "answer": ans}
                else:
                    # Fallback z-score
                    an_df = detect_anomalies(df, 'DL_Throughput', threshold=2.0)
                    cnt = len(an_df)
                    recent = an_df.sort_values('Date').tail(3) if cnt else pd.DataFrame()
                    steps = [
                        f"Selected {site_id}{' / ' + sector_id if sector_id else ''}",
                        f"Applied last {days or 15} days window (relative to dataset max date)",
                        "Used z-score fallback on DL_Throughput",
                    ]
                    ctx = "" if recent.empty else ("Recent anomalies:\n" + "\n".join([f"{r['Date']}: {r['DL_Throughput']:.2f}" for _, r in recent.iterrows()]))
                    ans = f"Detected {cnt} DL Throughput anomalies for {site_id}{' / ' + sector_id if sector_id else ''} in the last {days or 15} days."
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
                fig, anomalies_df = create_kpi_visualization(
                    data,
                    site,
                    sector,
                    kpi,
                    detector=ui_detector,
                    last_days=st.session_state.get('last_days', 7),
                    show_model_threshold=show_model_thr,
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
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
                    params['encoding_dim'] = st.slider("encoding_dim", 4, 128, 32, 4)
                    params['epochs'] = st.slider("epochs", 10, 300, 50, 10)
                    params['learning_rate'] = st.select_slider("learning_rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
                    params['sequence_length'] = st.slider("sequence_length", 1, 30, 7, 1, help=">1 triggers LSTM AE for Throughput KPIs")
                elif selected_algo == 'ensemble_if_gmm':
                    params['contamination'] = st.slider("contamination (IF)", 0.01, 0.3, 0.05, 0.01)
                    params['if_n_estimators'] = st.slider("IF n_estimators", 50, 400, 100, 10)
                    params['weight_if'] = st.slider("Weight IF", 0.0, 1.0, 0.5, 0.05)
                    params['weight_gmm'] = 1.0 - params['weight_if']
                elif selected_algo == 'time_series_decomposition':
                    params['use_prophet'] = st.checkbox("Use Prophet if available", value=True)
                    params['season_period'] = st.slider("Seasonal period (days)", 3, 60, 7, 1)
                elif selected_algo == 'seasonal_hybrid_esd':
                    params['max_anoms_ratio'] = st.slider("Max anomalies ratio", 0.01, 0.3, 0.1, 0.01)
                    params['alpha'] = st.slider("Alpha (significance)", 0.001, 0.2, 0.05, 0.001)
                    params['period'] = st.slider("Seasonal period", 3, 60, 7, 1)

                # Data slice
                def _filter_for_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
                    if 'Date' not in df.columns:
                        return df
                    df = df.copy()
                    end = pd.to_datetime(df['Date']).max()
                    from datetime import timedelta
                    start = end - timedelta(days=int(days))
                    return df[pd.to_datetime(df['Date']) >= start]

                train_df = data[(data['Site_ID'] == ft_site) & (data['Sector_ID'] == ft_sector)].copy()
                train_df = _filter_for_window(train_df, ft_days)

                # Debounce: explicit Train button; also track hash of (algo, params, site, sector, days)
                current_signature = (selected_algo, tuple(sorted(params.items())), ft_site, ft_sector, int(ft_days))
                last_sig_key = f"last_sig__{ft_kpi}"
                train_clicked = st.button("Train / Preview Model", key="tune_train")
                tuned_key = f"tuned__{ft_kpi}"

                def _train():
                    cfg = _Cfg()
                    # Apply sequence_length if relevant
                    if selected_algo == 'autoencoder':
                        cfg.model.autoencoder_params['encoding_dim'] = int(params.get('encoding_dim', 32))
                        cfg.model.autoencoder_params['epochs'] = int(params.get('epochs', 50))
                        cfg.model.autoencoder_params['learning_rate'] = float(params.get('learning_rate', 1e-3))
                        cfg.model.sequence_length = int(params.get('sequence_length', 7))
                    if selected_algo == 'isolation_forest':
                        cfg.model.isolation_forest_params['contamination'] = float(params['contamination'])
                        cfg.model.isolation_forest_params['n_estimators'] = int(params['n_estimators'])
                    det = KPISpecificDetector(ft_kpi, cfg)
                    det.algorithm = selected_algo  # override mapping for tuning
                    X = train_df[[ft_kpi]].dropna().values
                    if X.shape[0] < 10:
                        st.warning("Not enough samples in window.")
                        return None
                    # Custom training paths
                    if selected_algo == 'isolation_forest':
                        from sklearn.ensemble import IsolationForest
                        det.model = IsolationForest(**cfg.model.isolation_forest_params)
                        Xs = det.scaler.fit_transform(X)
                        det.model.fit(Xs)
                        det.is_fitted = True
                        det._calculate_threshold(Xs)
                    elif selected_algo == 'local_outlier_factor':
                        from sklearn.neighbors import LocalOutlierFactor
                        # Use novelty=True so we can score new samples consistently outside training window
                        det.model = LocalOutlierFactor(n_neighbors=int(params['n_neighbors']), contamination=float(params['contamination']), novelty=True)
                        Xs = det.scaler.fit_transform(X)
                        det.model.fit(Xs)
                        det.is_fitted = True
                        det._calculate_threshold(Xs)
                    elif selected_algo == 'one_class_svm':
                        from sklearn.svm import OneClassSVM
                        det.model = OneClassSVM(nu=float(params['nu']), kernel=params['kernel'], gamma=params['gamma'])
                        Xs = det.scaler.fit_transform(X)
                        det.model.fit(Xs)
                        det.is_fitted = True
                        det._calculate_threshold(Xs)
                    elif selected_algo == 'gaussian_mixture':
                        from sklearn.mixture import GaussianMixture
                        det.model = GaussianMixture(n_components=int(params['n_components']), random_state=42)
                        Xs = det.scaler.fit_transform(X)
                        det.model.fit(Xs)
                        det.is_fitted = True
                        det._calculate_threshold(Xs)
                    elif selected_algo == 'autoencoder':
                        det.fit(X)  # uses AE/LSTM internally based on sequence_length
                    elif selected_algo == 'ensemble_if_gmm':
                        from sklearn.ensemble import IsolationForest
                        from sklearn.mixture import GaussianMixture
                        Xs = det.scaler.fit_transform(X)
                        det.model = {
                            'if': IsolationForest(contamination=float(params['contamination']), n_estimators=int(params['if_n_estimators']), random_state=42).fit(Xs),
                            'gmm': GaussianMixture(n_components=2, random_state=42).fit(Xs),
                            'weights': (float(params['weight_if']), float(params['weight_gmm']))
                        }
                        det.is_fitted = True
                        scores = det._get_anomaly_scores(Xs)
                        det.threshold = np.percentile(scores, (1 - cfg.model.contamination_rate) * 100)
                    elif selected_algo == 'time_series_decomposition':
                        det.fit(X)  # internal handles decomposition
                    elif selected_algo == 'seasonal_hybrid_esd':
                        det.fit(X)
                    else:
                        st.error("Unsupported algorithm.")
                        return None
                    det.params = params
                    return det

                if train_clicked or (st.session_state.get(tuned_key) is not None and st.session_state.get(last_sig_key) != current_signature):
                    # Train if user requested OR params/site/sector/days changed since last preview
                    with st.spinner("Training model..."):
                        tuned = _train()
                        st.session_state[tuned_key] = tuned
                        st.session_state[last_sig_key] = current_signature
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
