#!/bin/bash

# ============================================================================
# 환경 변수 설정 스크립트
# ============================================================================

set -e

echo "🚀 실시간 면접 아바타 - 환경 설정"
echo "=================================="
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 루트 디렉토리로 이동
cd "$(dirname "$0")/.."

# .env 파일 존재 확인
if [ -f .env ]; then
    echo -e "${YELLOW}⚠️  .env 파일이 이미 존재합니다.${NC}"
    read -p "덮어쓰시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    mv .env .env.backup.$(date +%Y%m%d_%H%M%S)
    echo -e "${GREEN}✅ 기존 .env 파일을 백업했습니다.${NC}"
fi

# .env.example 복사
if [ ! -f .env.example ]; then
    echo -e "${RED}❌ .env.example 파일을 찾을 수 없습니다.${NC}"
    exit 1
fi

cp .env.example .env
echo -e "${GREEN}✅ .env 파일 생성 완료${NC}"
echo ""

# API 키 입력
echo "📝 필수 API 키를 입력하세요 (Enter로 건너뛰기)"
echo "================================================"
echo ""

# OpenAI API Key
echo -e "${BLUE}1. OpenAI API 키${NC}"
echo "   발급: https://platform.openai.com/api-keys"
read -p "   입력: " openai_key
if [ ! -z "$openai_key" ]; then
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sed -i.bak "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$openai_key/" .env
    else
        # Windows Git Bash
        sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$openai_key/" .env
    fi
    echo -e "${GREEN}   ✓ 설정 완료${NC}"
fi
echo ""

# Deepgram API Key
echo -e "${BLUE}2. Deepgram API 키 (STT)${NC}"
echo "   발급: https://console.deepgram.com/"
echo "   또는 로컬 Whisper 사용 가능"
read -p "   입력: " deepgram_key
if [ ! -z "$deepgram_key" ]; then
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sed -i.bak "s/DEEPGRAM_API_KEY=.*/DEEPGRAM_API_KEY=$deepgram_key/" .env
    else
        sed -i "s/DEEPGRAM_API_KEY=.*/DEEPGRAM_API_KEY=$deepgram_key/" .env
    fi
    echo -e "${GREEN}   ✓ 설정 완료${NC}"
fi
echo ""

# ElevenLabs API Key (선택)
echo -e "${BLUE}3. ElevenLabs API 키 (TTS, 선택사항)${NC}"
echo "   발급: https://elevenlabs.io/"
echo -e "   ${YELLOW}무료 대안: EdgeTTS (자동 설정됨)${NC}"
read -p "   입력: " elevenlabs_key
if [ ! -z "$elevenlabs_key" ]; then
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sed -i.bak "s/ELEVENLABS_API_KEY=.*/ELEVENLABS_API_KEY=$elevenlabs_key/" .env
        sed -i.bak "s/TTS_PROVIDER=.*/TTS_PROVIDER=elevenlabs/" .env
    else
        sed -i "s/ELEVENLABS_API_KEY=.*/ELEVENLABS_API_KEY=$elevenlabs_key/" .env
        sed -i "s/TTS_PROVIDER=.*/TTS_PROVIDER=elevenlabs/" .env
    fi
    echo -e "${GREEN}   ✓ 설정 완료 (ElevenLabs 사용)${NC}"
else
    echo -e "${GREEN}   ✓ EdgeTTS 사용 (무료)${NC}"
fi
echo ""

# API 키 생성
echo -e "${BLUE}4. 서버 API 키 생성${NC}"
api_key=$(openssl rand -hex 32 2>/dev/null || echo "generated_$(date +%s)")
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i.bak "s/API_KEY=.*/API_KEY=$api_key/" .env
else
    sed -i "s/API_KEY=.*/API_KEY=$api_key/" .env
fi
echo -e "${GREEN}   ✓ API 키 생성 완료${NC}"
echo ""

# JWT Secret 생성
jwt_secret=$(openssl rand -hex 32 2>/dev/null || echo "jwt_$(date +%s)")
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i.bak "s/JWT_SECRET=.*/JWT_SECRET=$jwt_secret/" .env
else
    sed -i "s/JWT_SECRET=.*/JWT_SECRET=$jwt_secret/" .env
fi

# 백업 파일 삭제
rm -f .env.bak

echo ""
echo "=================================="
echo -e "${GREEN}✅ 환경 설정 완료!${NC}"
echo "=================================="
echo ""

# 검증 실행
if command -v python3 &> /dev/null; then
    echo "🔍 환경 변수 검증 중..."
    python3 src/utils/env_validator.py || true
else
    echo -e "${YELLOW}⚠️  Python3가 설치되지 않아 검증을 건너뜁니다.${NC}"
fi

echo ""
echo "📚 다음 단계:"
echo "  1. .env 파일을 확인하세요: cat .env"
echo "  2. 필요한 패키지 설치: pip install -r requirements.txt"
echo "  3. 서버 시작: python -m src.server.main"
echo "  4. API 문서: http://localhost:8000/docs"
echo ""
echo -e "${RED}⚠️  주의: .env 파일을 Git에 커밋하지 마세요!${NC}"
echo ""
