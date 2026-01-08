@echo off
chcp 65001 > nul
title AI Interview Avatar Server

echo.
echo ============================================================
echo   AI Interview Avatar Server
echo ============================================================
echo.

cd /d "%~dp0"

REM Python 가상환경 활성화 (있는 경우)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo   Virtual environment activated

    REM 필수 패키지 설치 확인
    pip show loguru >nul 2>&1
    if errorlevel 1 (
        echo   Installing required packages...
        pip install loguru python-dotenv pydantic pydantic-settings fastapi uvicorn openai openai-whisper -q
    )
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo   Virtual environment activated

    REM 필수 패키지 설치 확인
    pip show loguru >nul 2>&1
    if errorlevel 1 (
        echo   Installing required packages...
        pip install loguru python-dotenv pydantic pydantic-settings fastapi uvicorn openai openai-whisper -q
    )
)

echo   Starting server...
echo.

python scripts/run_server.py

pause
