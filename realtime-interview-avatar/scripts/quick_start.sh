#!/bin/bash

# ============================================================================
# 빠른 시작 스크립트
# ============================================================================

set -e

echo "🚀 실시간 면접 아바타 - 빠른 시작"
echo "=================================="
echo ""

cd "$(dirname "$0")/.."

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. 환경 변수 확인
echo -e "${BLUE}1. 환경 변수 확인...${NC}"
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env 파일이 없습니다.${NC}"
    echo "자동으로 생성하시겠습니까? (y/N): "
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./scripts/setup_env.sh
    else
        echo "종료합니다."
        exit 1
    fi
fi
echo -e "${GREEN}✅ .env 파일 확인 완료${NC}"
echo ""

# 2. Python 가상환경 확인
echo -e "${BLUE}2. Python 가상환경 확인...${NC}"
if [ ! -d "venv" ]; then
    echo "가상환경을 생성하시겠습니까? (y/N): "
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -m venv venv
        echo -e "${GREEN}✅ 가상환경 생성 완료${NC}"
    fi
fi

# 가상환경 활성화
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✅ 가상환경 활성화${NC}"
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
    echo -e "${GREEN}✅ 가상환경 활성화${NC}"
fi
echo ""

# 3. 의존성 설치 확인
echo -e "${BLUE}3. 의존성 확인...${NC}"
if ! python -c "import fastapi" 2>/dev/null; then
    echo "패키지를 설치하시겠습니까? (y/N): "
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -q fastapi uvicorn python-dotenv websockets
        echo -e "${GREEN}✅ 기본 패키지 설치 완료${NC}"
    fi
fi
echo -e "${GREEN}✅ 의존성 확인 완료${NC}"
echo ""

# 4. 환경 변수 검증
echo -e "${BLUE}4. 환경 변수 검증...${NC}"
if [ -f "src/utils/env_validator.py" ]; then
    python src/utils/env_validator.py || true
fi
echo ""

# 5. 서버 시작
echo -e "${BLUE}5. 서버 시작...${NC}"
echo ""
echo "=================================="
echo -e "${GREEN}✅ 모든 준비 완료!${NC}"
echo "=================================="
echo ""
echo "서버 정보:"
echo "  - API 문서: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo "  - 헬스 체크: http://localhost:8000/health"
echo ""
echo "종료: Ctrl+C"
echo ""

# 서버 실행
python -m src.server.main
