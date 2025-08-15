@echo off
echo Fast starting Streamlit UI (no conda activation)...
set "ENV_PY=%USERPROFILE%\anaconda3\envs\telecom-kpi\python.exe"
if exist "%ENV_PY%" (
    "%ENV_PY%" "%~dp0run_app.py"
) else (
    python "%~dp0run_app.py"
)
pause
