@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
REM ==============================================
REM Local Lipsync Server
REM ==============================================

cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Failed to change directory
    pause
    exit /b 1
)

REM Kill existing Python processes running app.py
echo [INFO] Stopping existing server...

REM Kill process on port 5000
echo [INFO] Checking port 5000...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :5000 ^| findstr LISTENING') do (
    set PORT_PID=%%a
    set PORT_PID=!PORT_PID: =!
    if defined PORT_PID (
        echo [INFO] Killing process on port 5000 PID: !PORT_PID!
        taskkill /F /PID !PORT_PID! >nul 2>&1
    )
)

REM Wait and check again
timeout /t 1 /nobreak >nul

REM Kill remaining processes on port 5000
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :5000') do (
    set PORT_PID=%%a
    set PORT_PID=!PORT_PID: =!
    if defined PORT_PID (
        echo [INFO] Killing remaining process on port 5000 PID: !PORT_PID!
        taskkill /F /PID !PORT_PID! >nul 2>&1
    )
)

timeout /t 1 /nobreak >nul

REM Check Python installation
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+ and add it to PATH.
    echo.
    pause
    exit /b 1
)
python --version
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo [ERROR] Failed to activate virtual environment!
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment activated
) else (
    echo [WARNING] Virtual environment not found
    echo [WARNING] Continuing without virtual environment...
)

REM Check app.py file
if not exist "app.py" (
    echo [ERROR] app.py file not found!
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

REM Set PYTHONPATH for MuseTalk
set "PYTHONPATH=%PYTHONPATH%;%~dp0..\MuseTalk"

echo.
echo ==============================================
echo Starting Lipsync Server (Local Mode)
echo ==============================================
echo.

REM Start server in current window
echo [INFO] Starting server...
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Run Python and handle errors
python app.py
set EXIT_CODE=!ERRORLEVEL!

REM Wait after server stops (error or normal)
echo.
if !EXIT_CODE! neq 0 (
    echo ==============================================
    echo [ERROR] Server failed to start or crashed
    echo Exit code: !EXIT_CODE!
    echo ==============================================
    echo.
    echo Possible causes:
    echo   - Python module import error
    echo   - Port 5000 already in use
    echo   - Missing dependencies
    echo   - Configuration error
    echo.
    echo Check the error messages above for details.
    echo.
) else (
    echo ==============================================
    echo [INFO] Server stopped normally
    echo ==============================================
    echo.
)

REM Keep window open
echo.
echo Press any key to close this window...
pause
