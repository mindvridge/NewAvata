@echo off
REM ============================================
REM Cloudflare Tunnel 실행 스크립트
REM ============================================

echo [Cloudflare Tunnel] Starting...

REM 터널 설정 파일 경로
set CONFIG_FILE=%~dp0cloudflared-config.yml

REM 터널 이름 (tunnel create 시 지정한 이름)
set TUNNEL_NAME=interview-avatar

REM 터널 실행
cloudflared tunnel --config "%CONFIG_FILE%" run %TUNNEL_NAME%

pause
