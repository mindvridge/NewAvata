@echo off
chcp 65001 >nul
echo ============================================
echo   서버 재시작 중...
echo ============================================

REM 기존 Python 프로세스 종료
taskkill /F /IM python.exe 2>nul

REM 잠시 대기
timeout /t 2 /nobreak >nul

REM 서버 시작
cd /d c:\NewAvata\NewAvata\realtime-interview-avatar
python app.py
