@echo off
REM ============================================================================
REM Realtime Lipsync Avatar Server - Docker Logs
REM ============================================================================

title Docker Logs

echo.
echo ==============================================================
echo     Docker Logs (Ctrl+C to exit)
echo ==============================================================
echo.

cd /d "%~dp0"

docker-compose logs -f
