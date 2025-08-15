import os
import pandas as pd
from datetime import timedelta

from telecom_ai_platform.core.config import TelecomConfig
from telecom_ai_platform.models.anomaly_detector import KPIAnomalyDetector


def main():
    # Load data
    # Try common paths relative to repo root
    candidates = [
        os.path.join('data', 'AD_data_10KPI.csv'),
        'AD_data_10KPI.csv'
    ]
    df = None
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    if df is None:
        raise FileNotFoundError("AD_data_10KPI.csv not found")

    df['Date'] = pd.to_datetime(df['Date'])

    # Window: last 7 days relative to dataset max date
    end = df['Date'].max()
    start = end - timedelta(days=7)
    win = df[df['Date'] >= start].copy()

    # Load detectors
    cfg = TelecomConfig()
    det = KPIAnomalyDetector(cfg)
    det.load_all_models()
    if not det.is_fitted:
        print("No trained detectors found. Exiting.")
        return

    # Rank sites by total anomalies across all KPIs
    by_site = []
    for site in sorted(win['Site_ID'].unique()):
        sdf = win[win['Site_ID'] == site]
        try:
            res = det.detect_anomalies(sdf)
            cnt = sum(1 for r in res if r.is_anomaly)
        except Exception as e:
            cnt = 0
        by_site.append((site, cnt))
    by_site.sort(key=lambda x: x[1], reverse=True)

    # Print top five
    print("Top sites by anomalies (last 7 days, model-based):")
    for s, c in by_site[:5]:
        print(f"{s}: {c}")

    if by_site:
        print(f"\nHighest anomalies site: {by_site[0][0]} (count={by_site[0][1]})")


if __name__ == "__main__":
    main()
