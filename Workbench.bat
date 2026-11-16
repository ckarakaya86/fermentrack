@echo off
REM One-click launcher for the FermenTrack Model Workbench (build / update / check models).
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3.12) else (set PY=python)

%PY% -c "import streamlit, plotly, sklearn, openpyxl" >nul 2>nul
if not %errorlevel%==0 (
    echo Installing required packages...
    %PY% -m pip install -r requirements.txt
)

if not exist examples\ftir_calibration.xlsx %PY% -m fermentrack.workbench.examples

echo Starting FermenTrack Workbench... (close this window to stop it)
%PY% -m streamlit run workbench_app.py --server.port 8502 --browser.gatherUsageStats false
pause
