#!/bin/bash
# =============================================================================
# Docker 빌드 스크립트
# =============================================================================

set -e

echo "======================================"
echo "실시간 립싱크 아바타 Docker 빌드"
echo "======================================"

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

# 환경 변수 파일 확인
if [ ! -f "realtime-interview-avatar/.env" ]; then
    echo "[경고] .env 파일이 없습니다. .env.example을 복사합니다."
    cp realtime-interview-avatar/.env.example realtime-interview-avatar/.env
    echo "[주의] realtime-interview-avatar/.env 파일에 OPENAI_API_KEY를 설정하세요!"
fi

# Docker Compose 빌드
echo ""
echo "[1/2] Docker 이미지 빌드 중..."
docker-compose -f realtime-interview-avatar/docker-compose.yml build

echo ""
echo "[2/2] 빌드 완료!"
echo ""
echo "======================================"
echo "사용법:"
echo "  실행: docker-compose -f realtime-interview-avatar/docker-compose.yml up -d"
echo "  로그: docker-compose -f realtime-interview-avatar/docker-compose.yml logs -f"
echo "  중지: docker-compose -f realtime-interview-avatar/docker-compose.yml down"
echo "  접속: http://localhost:5000"
echo "======================================"
