@echo off
chcp 65001 > nul
title AI Interview Avatar Server (Flask 5000)

echo.
echo ============================================================
echo   AI Interview Avatar Server (Flask)
echo   URL: http://localhost:5000
echo ============================================================
echo.

cd /d c:\NewAvata\NewAvata\realtime-interview-avatar

echo   Starting Flask server on port 5000...
echo.

python app.py

pause
