# test_server_fixes.py
import asyncio
import httpx
import json

async def test_server_health():
    """Test server health and basic functionality"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test health check
        try:
            health = await client.get("http://localhost:8000/health")
            print(f"Health check: {health.status_code}")
            print(json.dumps(health.json(), indent=2))
        except Exception as e:
            print(f"Health check failed: {e}")
        
        # Test data availability
        try:
            data_test = await client.post(
                "http://localhost:8000/call_tool",
                json={"tool": "list_available_data", "parameters": {}}
            )
            print(f"Data availability: {data_test.status_code}")
        except Exception as e:
            print(f"Data test failed: {e}")
        
        # Test simple anomaly detection
        try:
            anomaly_test = await client.post(
                "http://localhost:8000/call_tool",
                json={
                    "tool": "detect_anomalies",
                    "parameters": {
                        "kpi_name": "DL_Throughput",
                        "site_id": "SITE_001",
                        "date_range": "last_week",
                        "method": "statistical"
                    }
                }
            )
            print(f"Anomaly detection: {anomaly_test.status_code}")
        except Exception as e:
            print(f"Anomaly test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_server_health())