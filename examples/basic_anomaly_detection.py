# Example: Basic Anomaly Detection

from src.telecom_ai_platform.models.anomaly_detector import AnomalyDetector
from src.telecom_ai_platform.core.data_processor import DataProcessor
import pandas as pd

# Load and process data
processor = DataProcessor()
df = pd.read_csv('data/AD_data_10KPI.csv')
processed_data = processor.process_kpi_data(df)

# Initialize anomaly detector
detector = AnomalyDetector()

# Detect anomalies for specific KPI
anomalies = detector.detect_anomalies(
    data=processed_data,
    kpi_name='DL_Throughput',
    site_id='SITE_001',
    method='autoencoder'
)

print(f"Found {len(anomalies)} anomalies")
for anomaly in anomalies[:5]:  # Show first 5
    print(f"Time: {anomaly['timestamp']}, Value: {anomaly['value']}, Score: {anomaly['anomaly_score']}")
