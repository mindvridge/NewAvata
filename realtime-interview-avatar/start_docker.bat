@echo off
REM ============================================================================
REM Realtime Lipsync Avatar Server - Docker Start Script
REM MuseTalk V1.5 + ElevenLabs TTS + FP16 Optimization
REM ============================================================================

title Avatar Server (Docker)

echo.
echo ==============================================================
echo     Realtime Lipsync Avatar Server (Docker)
echo     MuseTalk V1.5 + ElevenLabs TTS + FP16
echo ==============================================================
echo.

REM Change to script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ============================================================================
REM Check Docker
REM ============================================================================
echo [1/5] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Docker is not running!
    echo         Please start Docker Desktop first.
    echo.
    pause
    exit /b 1
)
echo        Docker is running [OK]

REM ============================================================================
REM Check NVIDIA GPU
REM ============================================================================
echo [2/5] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo        [WARNING] nvidia-smi not found.
    echo                  GPU acceleration may not work.
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader') do (
        echo        GPU: %%i [OK]
    )
)

REM ============================================================================
REM Check .env file
REM ============================================================================
echo [3/5] Checking .env file...
if not exist ".env" (
    echo.
    echo [ERROR] .env file not found!
    echo.
    echo Please create .env file with API keys:
    echo   - OPENAI_API_KEY or LLM_API_KEY
    echo   - ELEVENLABS_API_KEY
    echo.
    if exist ".env.example" (
        echo Copying .env.example to .env...
        copy ".env.example" ".env" >nul
        echo .env file created. Please set API keys and run again.
    )
    pause
    exit /b 1
)
echo        .env file found [OK]

REM ============================================================================
REM Check model directories
REM ============================================================================
echo [4/5] Checking model directories...
if not exist "models\musetalkV15" (
    echo        [WARNING] MuseTalk V1.5 model not found!
    echo                  Please download models to models\musetalkV15
) else (
    echo        MuseTalk V1.5 model [OK]
)

if not exist "precomputed" (
    echo        Creating precomputed directory...
    mkdir precomputed 2>nul
)

REM ============================================================================
REM Docker build and run
REM ============================================================================
echo [5/5] Starting Docker container...
echo.

REM Stop existing container
docker-compose down 2>nul

REM Check if image exists
docker images -q realtime-avatar:latest >nul 2>&1
if "%ERRORLEVEL%"=="1" (
    echo --------------------------------------------------------------
    echo  Docker image not found. Building...
    echo  This may take 10-20 minutes on first run.
    echo --------------------------------------------------------------
    echo.
    docker-compose build
    if errorlevel 1 (
        echo.
        echo [ERROR] Docker build failed.
        pause
        exit /b 1
    )
)

REM Start container
echo.
echo Starting Docker container...
docker-compose up -d

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start Docker container.
    pause
    exit /b 1
)

REM ============================================================================
REM Done
REM ============================================================================
echo.
echo ==============================================================
echo  Server started successfully!
echo ==============================================================
echo.
echo  URL: http://localhost:5000
echo.
echo  Model loading takes 1-3 minutes.
echo.
echo  Commands:
echo    View logs: docker-compose logs -f
echo    Stop:      docker-compose down
echo    Restart:   docker-compose restart
echo ==============================================================
echo.

set /p SHOW_LOGS="Show logs? (Y/N): "
if /i "%SHOW_LOGS%"=="Y" (
    echo.
    echo Showing logs... (Ctrl+C to exit)
    echo --------------------------------------------------------------
    docker-compose logs -f
) else (
    echo.
    pause
)
