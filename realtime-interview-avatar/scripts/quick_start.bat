@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion

echo.
echo ================================================
echo Realtime Interview Avatar - Auto Install
echo ================================================
echo.
echo Estimated time: 5-10 minutes
echo.

cd /d "%~dp0\.."

REM 1. Check Python
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed.
    echo Please install Python 3.9+: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

REM 2. Create virtual environment
echo [2/7] Creating Python virtual environment...
if not exist venv (
    echo Creating venv...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [SKIP] Virtual environment already exists
)
echo.

REM 3. Activate virtual environment and continue
echo [3/7] Activating virtual environment...
if not exist venv\Scripts\activate.bat (
    echo [ERROR] Cannot find venv\Scripts\activate.bat
    pause
    exit /b 1
)

REM Run the rest in activated environment
call venv\Scripts\activate.bat && (
    echo [OK] Virtual environment activated
    echo.

    REM 4. Upgrade pip
    echo [4/7] Upgrading pip...
    python -m pip install --upgrade pip --quiet
    if errorlevel 1 (
        echo [WARNING] pip upgrade failed, continuing anyway...
    ) else (
        echo [OK] pip upgraded
    )
    echo.

    REM 5. Install packages
    echo [5/7] Installing packages... ^(this may take a while^)
    echo.

    echo    - Installing web framework...
    pip install fastapi uvicorn python-dotenv websockets
    if errorlevel 1 (
        echo [ERROR] Web framework installation failed
        pause
        exit /b 1
    )
    echo [OK] Web framework installed
    echo.

    echo    - Installing AI libraries...
    pip install openai edge-tts
    if errorlevel 1 (
        echo [ERROR] AI libraries installation failed
        pause
        exit /b 1
    )
    echo [OK] AI libraries installed
    echo.

    echo    - Installing image/audio processing...
    pip install opencv-python numpy pillow soundfile
    if errorlevel 1 (
        echo [ERROR] Image/audio libraries installation failed
        pause
        exit /b 1
    )
    echo [OK] Image/audio libraries installed
    echo.

    echo    - Installing utilities...
    pip install aiofiles python-multipart httpx
    if errorlevel 1 (
        echo [ERROR] Utilities installation failed
        pause
        exit /b 1
    )
    echo [OK] Utilities installed
    echo.

    echo [OK] All packages installed successfully
    echo.

    REM 6. Create .env file
    echo [6/7] Setting up environment variables...
    if not exist .env (
        if exist .env.example (
            copy .env.example .env >nul
            echo [OK] .env file created from .env.example
        ) else (
            echo [WARNING] .env.example not found, creating basic .env...
            ^(
                echo SERVER_HOST=0.0.0.0
                echo SERVER_PORT=8000
                echo DEBUG=true
                echo API_KEY=dev_api_key_!random!!random!
                echo.
                echo OPENAI_API_KEY=your-openai-api-key-here
                echo.
                echo TTS_PROVIDER=edge
                echo TTS_EDGE_VOICE=ko-KR-SunHiNeural
                echo STT_PROVIDER=whisper
                echo LLM_PROVIDER=openai
                echo LLM_MODEL=gpt-3.5-turbo
                echo LLM_TEMPERATURE=0.7
            ^) > .env
            echo [OK] .env file created
        )
    ) else (
        echo [SKIP] .env file already exists
    )
    echo.

    REM 7. Create directory structure
    echo [7/7] Setting up directory structure...
    if not exist assets\images mkdir assets\images
    if not exist assets\audio mkdir assets\audio
    if not exist logs mkdir logs
    if not exist cache mkdir cache
    echo [OK] Directory structure ready
    echo.

    echo ================================================
    echo [SUCCESS] Auto installation complete!
    echo ================================================
    echo.
    echo Installed:
    echo   - Python virtual environment ^(venv^)
    echo   - All required packages
    echo   - Environment variables ^(.env^)
    echo   - Directory structure
    echo.
    echo Starting server...
    echo.
    echo Server info:
    echo   - API docs: http://localhost:8000/docs
    echo   - ReDoc: http://localhost:8000/redoc
    echo   - Health check: http://localhost:8000/health
    echo.
    echo To stop: Ctrl+C
    echo.
    echo ================================================
    echo.

    REM Start server
    if exist src\server\main.py (
        python src\server\main.py
    ) else (
        echo [ERROR] Cannot find src\server\main.py
        echo Current directory: %CD%
        dir src 2^>nul
        pause
        exit /b 1
    )
)

pause
