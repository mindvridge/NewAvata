#!/bin/bash
# =============================================================================
# Docker 실행 스크립트
# =============================================================================

set -e

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

echo "======================================"
echo "실시간 립싱크 아바타 서버 시작"
echo "======================================"

# 환경 변수 파일 확인
if [ ! -f "realtime-interview-avatar/.env" ]; then
    echo "[에러] .env 파일이 없습니다!"
    echo "       cp realtime-interview-avatar/.env.example realtime-interview-avatar/.env"
    echo "       .env 파일에 OPENAI_API_KEY를 설정하세요."
    exit 1
fi

# Docker Compose 실행
docker-compose -f realtime-interview-avatar/docker-compose.yml up -d

echo ""
echo "서버가 시작되었습니다."
echo "접속: http://localhost:5000"
echo ""
echo "로그 확인: docker-compose -f realtime-interview-avatar/docker-compose.yml logs -f"
