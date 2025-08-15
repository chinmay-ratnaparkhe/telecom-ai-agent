@echo off
echo Starting Telecom AI Agent (MCP server + Streamlit UI)...

set "ENV_PY=%USERPROFILE%\anaconda3\envs\telecom-kpi\python.exe"

if exist "%ENV_PY%" (
	echo Using env Python: %ENV_PY%
	start "MCP Server" "%ENV_PY%" "%~dp0mcp_server\fixed_server_startup.py"
	"%ENV_PY%" "%~dp0streamlit_ui\run_app.py"
) else (
	echo Env Python not found, trying conda hook...
	if exist "%USERPROFILE%\anaconda3\shell\condabin\conda-hook.ps1" (
		powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%USERPROFILE%\anaconda3\shell\condabin\conda-hook.ps1' ; conda activate telecom-kpi ; Start-Process -WindowStyle Minimized -FilePath python -ArgumentList '%~dp0mcp_server\fixed_server_startup.py' ; python '%~dp0streamlit_ui\run_app.py'"
	) else (
		echo Conda hook not found; using default Python.
		start "MCP Server" powershell -NoProfile -ExecutionPolicy Bypass -Command "python '%~dp0mcp_server\fixed_server_startup.py'"
		powershell -NoProfile -ExecutionPolicy Bypass -Command "python '%~dp0streamlit_ui\run_app.py'"
	)
)

pause
