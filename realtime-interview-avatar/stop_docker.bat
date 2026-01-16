@echo off
REM ============================================================================
REM Realtime Lipsync Avatar Server - Docker Stop Script
REM ============================================================================

title Stop Docker Server

echo.
echo ==============================================================
echo     Stopping Avatar Server
echo ==============================================================
echo.

cd /d "%~dp0"

echo Stopping Docker container...
docker-compose down

echo.
echo ==============================================================
echo  Server stopped.
echo ==============================================================
echo.

pause
