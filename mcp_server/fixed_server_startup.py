#!/usr/bin/env python3
"""
Fixed startup script for the MCP server with proper environment setup
Resolves OpenMP, matplotlib, and threading issues
"""

import os
import sys

# CRITICAL: Set environment variables BEFORE importing any libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict
os.environ['MPLBACKEND'] = 'Agg'  # Force matplotlib to use non-GUI backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings if used

# Now we can safely run the server
if __name__ == "__main__":
    print("🔧 Setting up environment...")
    print(f"   KMP_DUPLICATE_LIB_OK = {os.environ.get('KMP_DUPLICATE_LIB_OK')}")
    print(f"   MPLBACKEND = {os.environ.get('MPLBACKEND')}")
    
    # Import and run the enhanced server
    from updated_mcp_server import app, load_data_and_models
    import uvicorn
    
    print("\n🚀 Starting Enhanced Telecom KPI MCP Server...")
    print("📊 Version 3.0 with Visualization and Governance")
    
    try:
        print("📁 Loading data and models...")
        load_data_and_models()
        print("✅ Server initialization completed!")
    except Exception as e:
        print(f"⚠️  Initialization warning: {e}")
        print("🔄 Starting server in limited mode...")
    
    print("\n🌐 Starting HTTP server on localhost:8000...")
    
    # Production-ready server configuration
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        timeout_keep_alive=60,
        timeout_graceful_shutdown=30,
        limit_concurrency=10,
        limit_max_requests=1000,
        workers=1,  # Single worker to avoid threading issues
        reload=False,
        server_header=False,
        date_header=False
    )
    
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        print("👋 Server stopped")