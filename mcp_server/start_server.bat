@echo off
REM Windows startup for Enhanced MCP Server
echo Setting up environment variables...
set KMP_DUPLICATE_LIB_OK=TRUE
set MPLBACKEND=Agg
set TF_CPP_MIN_LOG_LEVEL=2

REM Activate the conda environment if available
if exist "%USERPROFILE%\anaconda3\shell\condabin\conda-hook.ps1" (
	powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%USERPROFILE%\anaconda3\shell\condabin\conda-hook.ps1' ; conda activate telecom-kpi ; python %~dp0fixed_server_startup.py"
) else (
	echo Conda hook not found, running with default Python...
	python %~dp0fixed_server_startup.py
)
pause