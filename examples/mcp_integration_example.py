# Example: MCP Server Integration

import requests
import json

# MCP Server URL
MCP_URL = "http://localhost:8000"

def call_mcp_tool(tool_name, parameters):
    """Call an MCP tool via HTTP API"""
    response = requests.post(
        f"{MCP_URL}/call_tool",
        json={
            "tool": tool_name,
            "parameters": parameters
        }
    )
    return response.json()

# Example 1: Detect anomalies
anomaly_result = call_mcp_tool("detect_anomalies", {
    "site_id": "SITE_001",
    "kpi_name": "RSRP",
    "date_range": "last_week",
    "method": "statistical"
})

print("Anomaly Detection Result:")
print(json.dumps(anomaly_result, indent=2))

# Example 2: Get site summary
site_summary = call_mcp_tool("get_site_summary", {
    "site_id": "SITE_001",
    "date_range": "last_month"
})

print("\nSite Summary:")
print(json.dumps(site_summary, indent=2))

# Example 3: Compare multiple sites
comparison = call_mcp_tool("compare_sites", {
    "site_ids": ["SITE_001", "SITE_002", "SITE_003"],
    "kpi_name": "DL_Throughput",
    "date_range": "last_week"
})

print("\nSite Comparison:")
print(json.dumps(comparison, indent=2))

# Example 4: List available data
data_inventory = call_mcp_tool("list_available_data", {})

print("\nData Inventory:")
print(json.dumps(data_inventory, indent=2))
