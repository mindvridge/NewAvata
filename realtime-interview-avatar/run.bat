@echo off
REM =============================================================================
REM Docker 실행 스크립트 (Windows)
REM =============================================================================

echo ======================================
echo 실시간 립싱크 아바타 서버 시작
echo ======================================

REM 프로젝트 루트로 이동
cd /d "%~dp0\.."

REM 환경 변수 파일 확인
if not exist "realtime-interview-avatar\.env" (
    echo [에러] .env 파일이 없습니다!
    echo        copy realtime-interview-avatar\.env.example realtime-interview-avatar\.env
    echo        .env 파일에 OPENAI_API_KEY를 설정하세요.
    pause
    exit /b 1
)

REM Docker Compose 실행
docker-compose -f realtime-interview-avatar/docker-compose.yml up -d

echo.
echo 서버가 시작되었습니다.
echo 접속: http://localhost:5000
echo.
echo 로그 확인: docker-compose -f realtime-interview-avatar/docker-compose.yml logs -f

pause
