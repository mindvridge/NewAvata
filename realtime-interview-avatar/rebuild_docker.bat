@echo off
REM ============================================================================
REM Realtime Lipsync Avatar Server - Docker Rebuild Script
REM ============================================================================

title Docker Rebuild

echo.
echo ==============================================================
echo     Docker Image Rebuild
echo     (Run after code changes)
echo ==============================================================
echo.

cd /d "%~dp0"

REM Check Docker
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo         Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [1/3] Stopping existing container...
docker-compose down 2>nul

echo [2/3] Rebuilding Docker image...
echo        (This may take 10-20 minutes)
echo.
docker-compose build --no-cache

if errorlevel 1 (
    echo.
    echo [ERROR] Docker build failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting container...
docker-compose up -d

echo.
echo ==============================================================
echo  Rebuild complete!
echo  URL: http://localhost:5000
echo ==============================================================
echo.

set /p SHOW_LOGS="Show logs? (Y/N): "
if /i "%SHOW_LOGS%"=="Y" (
    docker-compose logs -f
) else (
    pause
)
