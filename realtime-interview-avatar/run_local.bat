@echo off
chcp 65001 >nul
REM ==============================================
REM Local Lipsync Server
REM ==============================================

cd /d "%~dp0"

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated
)

REM Set PYTHONPATH for MuseTalk
set "PYTHONPATH=%PYTHONPATH%;%~dp0..\MuseTalk"

echo.
echo ==============================================
echo Starting Lipsync Server (Local Mode)
echo ==============================================
echo.

python app.py

pause
