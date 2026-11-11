@echo off
REM One-click launcher for the FermenTrack dashboard.
REM Installs missing packages the first time, then opens the app in your browser.
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3.12) else (set PY=python)

%PY% -c "import streamlit, plotly, sklearn" >nul 2>nul
if not %errorlevel%==0 (
    echo Installing required packages...
    %PY% -m pip install -r requirements.txt
)

echo Starting FermenTrack... (close this window to stop it)
%PY% -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
