@echo off
echo ============================================
echo 서버 재시작 스크립트
echo ============================================

echo.
echo [1] 기존 Python 프로세스 종료 중...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [2] 포트 5000 사용 중인 프로세스 종료 중...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a 2>nul
)
timeout /t 2 /nobreak >nul

echo.
echo [3] Python 캐시 정리 중...
cd /d c:\NewAvata\NewAvata\realtime-interview-avatar
del /s /q __pycache__\*.pyc 2>nul
rmdir /s /q __pycache__ 2>nul

echo.
echo [4] 서버 시작 중...
echo 로그 파일: server_log_new.txt
python app.py > server_log_new.txt 2>&1

pause
