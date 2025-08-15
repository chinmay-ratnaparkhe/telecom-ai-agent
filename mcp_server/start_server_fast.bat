@echo off
echo Fast starting MCP Server (no conda activation)...
set "ENV_PY=%USERPROFILE%\anaconda3\envs\telecom-kpi\python.exe"
if exist "%ENV_PY%" (
    "%ENV_PY%" "%~dp0fixed_server_startup.py"
) else (
    python "%~dp0fixed_server_startup.py"
)
pause
