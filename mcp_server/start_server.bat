# For Windows: start_server.bat
@echo off
echo Setting up environment variables...
set KMP_DUPLICATE_LIB_OK=TRUE
set MPLBACKEND=Agg
set TF_CPP_MIN_LOG_LEVEL=2

echo Starting Enhanced MCP Server...
python fixed_server_startup.py
pause