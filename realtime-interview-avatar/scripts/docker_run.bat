@echo off
REM ============================================================================
REM Docker Run Script for Realtime Interview Avatar (Windows)
REM ============================================================================

echo ==============================================
echo Realtime Interview Avatar - Docker Runner
echo ==============================================
echo.

REM Change to project root
cd /d "%~dp0\.."

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running
    echo Please start Docker Desktop first
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo Error: .env file not found
    echo.
    echo Please create a .env file with:
    echo   DEEPGRAM_API_KEY=your_key
    echo   ELEVENLABS_API_KEY=your_key
    echo   OPENAI_API_KEY=your_key
    echo.
    pause
    exit /b 1
)

REM Check command argument
if "%1"=="build" goto build
if "%1"=="up" goto up
if "%1"=="down" goto down
if "%1"=="logs" goto logs
if "%1"=="restart" goto restart
goto help

:build
echo Building Docker images...
docker-compose -f docker-compose.realtime.yml build
echo.
echo Build complete!
goto end

:up
echo Starting services...
docker-compose -f docker-compose.realtime.yml up -d
echo.
echo Services started!
echo.
echo Access the API at: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo View logs: docker_run.bat logs
goto end

:down
echo Stopping services...
docker-compose -f docker-compose.realtime.yml down
echo.
echo Services stopped!
goto end

:logs
echo Showing logs (Ctrl+C to exit)...
docker-compose -f docker-compose.realtime.yml logs -f avatar
goto end

:restart
echo Restarting services...
docker-compose -f docker-compose.realtime.yml restart avatar
echo.
echo Services restarted!
goto end

:help
echo.
echo Usage: docker_run.bat [command]
echo.
echo Commands:
echo   build    - Build Docker images
echo   up       - Start all services
echo   down     - Stop all services
echo   logs     - View logs
echo   restart  - Restart avatar service
echo.
goto end

:end
