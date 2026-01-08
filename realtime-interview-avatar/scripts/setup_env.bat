@echo off
REM ============================================================================
REM 환경 변수 설정 스크립트 (Windows)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ================================
echo 실시간 면접 아바타 - 환경 설정
echo ================================
echo.

cd /d "%~dp0\.."

REM .env 파일 존재 확인
if exist .env (
    echo [경고] .env 파일이 이미 존재합니다.
    set /p "overwrite=덮어쓰시겠습니까? (y/N): "
    if /i not "!overwrite!"=="y" (
        echo 취소되었습니다.
        exit /b 0
    )
    
    REM 백업 생성
    for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set "backup_date=%%a%%b%%c"
    for /f "tokens=1-2 delims=: " %%a in ("%time%") do set "backup_time=%%a%%b"
    copy .env .env.backup.!backup_date!_!backup_time! >nul
    echo [완료] 기존 .env 파일을 백업했습니다.
)

REM .env.example 복사
if not exist .env.example (
    echo [오류] .env.example 파일을 찾을 수 없습니다.
    exit /b 1
)

copy .env.example .env >nul
echo [완료] .env 파일 생성 완료
echo.

echo ================================
echo 필수 API 키를 입력하세요
echo (Enter로 건너뛰기)
echo ================================
echo.

REM OpenAI API Key
echo 1. OpenAI API 키
echo    발급: https://platform.openai.com/api-keys
set /p "openai_key=   입력: "
if not "!openai_key!"=="" (
    powershell -Command "(Get-Content .env) -replace 'OPENAI_API_KEY=.*', 'OPENAI_API_KEY=!openai_key!' | Set-Content .env"
    echo    [완료] 설정 완료
)
echo.

REM Deepgram API Key
echo 2. Deepgram API 키 (STT)
echo    발급: https://console.deepgram.com/
set /p "deepgram_key=   입력: "
if not "!deepgram_key!"=="" (
    powershell -Command "(Get-Content .env) -replace 'DEEPGRAM_API_KEY=.*', 'DEEPGRAM_API_KEY=!deepgram_key!' | Set-Content .env"
    echo    [완료] 설정 완료
)
echo.

REM ElevenLabs API Key
echo 3. ElevenLabs API 키 (TTS, 선택사항)
echo    발급: https://elevenlabs.io/
echo    [참고] 무료 대안: EdgeTTS (자동 설정됨)
set /p "elevenlabs_key=   입력: "
if not "!elevenlabs_key!"=="" (
    powershell -Command "(Get-Content .env) -replace 'ELEVENLABS_API_KEY=.*', 'ELEVENLABS_API_KEY=!elevenlabs_key!' | Set-Content .env"
    powershell -Command "(Get-Content .env) -replace 'TTS_PROVIDER=.*', 'TTS_PROVIDER=elevenlabs' | Set-Content .env"
    echo    [완료] 설정 완료 (ElevenLabs 사용)
) else (
    echo    [완료] EdgeTTS 사용 (무료)
)
echo.

REM API 키 생성
echo 4. 서버 API 키 생성
set "api_key=generated_%random%%random%%random%"
powershell -Command "(Get-Content .env) -replace 'API_KEY=.*', 'API_KEY=%api_key%' | Set-Content .env"
echo    [완료] API 키 생성 완료
echo.

REM JWT Secret 생성
set "jwt_secret=jwt_%random%%random%%random%"
powershell -Command "(Get-Content .env) -replace 'JWT_SECRET=.*', 'JWT_SECRET=%jwt_secret%' | Set-Content .env"

echo.
echo ================================
echo [완료] 환경 설정 완료!
echo ================================
echo.

REM 검증 실행
where python >nul 2>nul
if %errorlevel% equ 0 (
    echo 검증 중...
    python src\utils\env_validator.py
) else (
    echo [경고] Python이 설치되지 않아 검증을 건너뜁니다.
)

echo.
echo 다음 단계:
echo   1. .env 파일 확인: type .env
echo   2. 패키지 설치: pip install -r requirements.txt
echo   3. 서버 시작: python -m src.server.main
echo   4. API 문서: http://localhost:8000/docs
echo.
echo [주의] .env 파일을 Git에 커밋하지 마세요!
echo.

pause
