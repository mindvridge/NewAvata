#!/bin/bash

# ============================================================================
# Local Development Environment Startup Script
# ============================================================================
# 로컬 개발 환경 시작 스크립트
# Docker Compose 또는 직접 실행 지원
# ============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 배너 출력
print_banner() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         로컬 개발 환경 - Interview Avatar                 ║"
    echo "║                                                            ║"
    echo "║         GPU 기반 실시간 면접 아바타 시스템                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

# 시스템 요구사항 확인
check_requirements() {
    log_info "시스템 요구사항 확인 중..."

    # Python 버전 확인
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3이 설치되어 있지 않습니다."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_info "Python 버전: $PYTHON_VERSION"

    # NVIDIA GPU 확인
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU 감지됨"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        GPU_AVAILABLE=true
    else
        log_warning "NVIDIA GPU가 감지되지 않았습니다. CPU 모드로 실행됩니다 (성능 저하)."
        GPU_AVAILABLE=false
    fi

    log_success "시스템 요구사항 확인 완료"
}

# 환경 변수 확인
check_env() {
    log_info "환경 변수 확인 중..."

    if [ ! -f .env ]; then
        log_warning ".env 파일이 없습니다."

        if [ -f .env.example ]; then
            log_info ".env.example을 .env로 복사 중..."
            cp .env.example .env
            log_warning ".env 파일을 편집하여 API 키를 입력하세요."
        else
            log_error ".env.example 파일이 없습니다."
            exit 1
        fi
    fi

    # .env 파일 로드
    source .env

    # 필수 API 키 확인
    MISSING_KEYS=()

    if [ -z "$DEEPGRAM_API_KEY" ]; then
        MISSING_KEYS+=("DEEPGRAM_API_KEY")
    fi

    if [ -z "$ELEVENLABS_API_KEY" ]; then
        MISSING_KEYS+=("ELEVENLABS_API_KEY")
    fi

    if [ -z "$OPENAI_API_KEY" ]; then
        MISSING_KEYS+=("OPENAI_API_KEY")
    fi

    if [ -z "$DAILY_API_KEY" ]; then
        MISSING_KEYS+=("DAILY_API_KEY")
    fi

    if [ ${#MISSING_KEYS[@]} -gt 0 ]; then
        log_warning "다음 API 키가 설정되지 않았습니다: ${MISSING_KEYS[*]}"
        log_warning "일부 기능이 제한될 수 있습니다."
    fi

    log_success "환경 변수 확인 완료"
}

# 실행 모드 선택
select_mode() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                    🚀 실행 모드 선택                       "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1) Docker Compose  ⭐ 권장 (완전한 환경, Redis 포함)"
    echo "2) 직접 실행       🔧 개발 (빠른 재시작, 디버깅 용이)"
    echo "3) Docker만        🐳 컨테이너만 (Redis 제외)"
    echo ""
    read -p "실행 모드를 선택하세요 [1-3] (기본: 1): " mode_choice

    mode_choice=${mode_choice:-1}

    case $mode_choice in
        1)
            RUN_MODE="docker-compose"
            ;;
        2)
            RUN_MODE="direct"
            ;;
        3)
            RUN_MODE="docker"
            ;;
        *)
            log_error "잘못된 선택입니다."
            exit 1
            ;;
    esac

    log_success "선택된 실행 모드: $RUN_MODE"
}

# Docker Compose로 실행
run_docker_compose() {
    log_info "Docker Compose로 실행 중..."

    # Docker 확인
    if ! command -v docker &> /dev/null; then
        log_error "Docker가 설치되어 있지 않습니다."
        log_info "Docker 설치: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose가 설치되어 있지 않습니다."
        exit 1
    fi

    # 개발 모드 확인
    read -p "개발 모드로 실행하시겠습니까? (핫 리로딩) [Y/n]: " dev_mode
    dev_mode=${dev_mode:-Y}

    if [[ "$dev_mode" =~ ^[Yy]$ ]]; then
        log_info "개발 모드로 실행 중 (핫 리로딩 활성화)..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
    else
        log_info "프로덕션 모드로 실행 중..."
        docker-compose up --build
    fi
}

# Docker만으로 실행
run_docker() {
    log_info "Docker 컨테이너로 실행 중..."

    # Docker 확인
    if ! command -v docker &> /dev/null; then
        log_error "Docker가 설치되어 있지 않습니다."
        exit 1
    fi

    # 이미지 빌드
    log_info "Docker 이미지 빌드 중..."
    docker build -t interview-avatar:latest -f docker/Dockerfile .

    # 컨테이너 실행
    log_info "Docker 컨테이너 시작 중..."
    docker run -it --rm \
        --gpus all \
        -p 8000:8000 \
        --env-file .env \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/logs:/app/logs \
        --name interview-avatar \
        interview-avatar:latest
}

# 직접 실행 (Python)
run_direct() {
    log_info "Python으로 직접 실행 중..."

    # 가상 환경 확인
    if [ ! -d "venv" ]; then
        log_info "가상 환경 생성 중..."
        python3 -m venv venv
    fi

    log_info "가상 환경 활성화 중..."
    source venv/bin/activate

    # 의존성 설치
    log_info "의존성 확인 중..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt

    # MuseTalk 모델 다운로드 확인
    if [ ! -d "models/musetalk" ]; then
        log_info "MuseTalk 모델 다운로드 중..."
        python scripts/setup_musetalk.py
    fi

    # Redis 확인 (옵션)
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            log_success "Redis 연결 확인됨"
        else
            log_warning "Redis가 실행되지 않고 있습니다."
            log_info "Redis를 시작하려면: sudo service redis-server start"
        fi
    else
        log_warning "Redis가 설치되지 않았습니다. 세션 관리가 제한됩니다."
    fi

    # 디렉토리 생성
    mkdir -p data logs uploads temp

    # 개발 서버 시작
    log_info "개발 서버 시작 중..."
    echo ""
    log_success "서버가 시작되었습니다!"
    log_success "URL: http://localhost:8000"
    log_success "API 문서: http://localhost:8000/docs"
    echo ""

    # 핫 리로딩 옵션
    read -p "핫 리로딩을 활성화하시겠습니까? [Y/n]: " reload
    reload=${reload:-Y}

    if [[ "$reload" =~ ^[Yy]$ ]]; then
        uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
    else
        uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --log-level info
    fi
}

# 헬스체크
health_check() {
    log_info "헬스체크 수행 중..."

    sleep 5

    for i in {1..30}; do
        if curl -f http://localhost:8000/api/health &> /dev/null; then
            log_success "헬스체크 성공!"
            log_success "애플리케이션이 정상적으로 실행 중입니다."
            return 0
        fi

        log_info "헬스체크 대기 중... ($i/30)"
        sleep 2
    done

    log_warning "헬스체크 실패: 애플리케이션이 응답하지 않습니다."
    return 1
}

# 유용한 정보 출력
print_info() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                    📌 유용한 정보                          "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "URL:"
    echo "  • 메인 페이지:    http://localhost:8000"
    echo "  • API 문서:       http://localhost:8000/docs"
    echo "  • ReDoc:          http://localhost:8000/redoc"
    echo "  • 헬스체크:       http://localhost:8000/api/health"
    echo ""

    if [ "$RUN_MODE" == "docker-compose" ]; then
        echo "Docker Compose 명령어:"
        echo "  • 로그 확인:      docker-compose logs -f app"
        echo "  • 중지:           docker-compose down"
        echo "  • 재시작:         docker-compose restart app"
        echo ""
    fi

    echo "개발 팁:"
    echo "  • STT 테스트:     tests/test_stt.py"
    echo "  • TTS 테스트:     tests/test_tts.py"
    echo "  • 로그 위치:      logs/"
    echo "  • 업로드 파일:    uploads/"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Cleanup 함수
cleanup() {
    echo ""
    log_info "정리 중..."

    if [ "$RUN_MODE" == "docker-compose" ]; then
        docker-compose down
    elif [ "$RUN_MODE" == "docker" ]; then
        docker stop interview-avatar 2>/dev/null || true
    fi

    log_success "정리 완료"
}

# Trap for cleanup
trap cleanup EXIT INT TERM

# 메인 함수
main() {
    print_banner
    check_requirements
    check_env
    select_mode

    case $RUN_MODE in
        docker-compose)
            run_docker_compose
            ;;
        docker)
            run_docker
            ;;
        direct)
            run_direct
            ;;
    esac

    # 백그라운드에서 실행되는 경우를 제외하고 헬스체크
    if [ "$RUN_MODE" == "direct" ]; then
        # 직접 실행 모드는 포그라운드에서 실행되므로 헬스체크 생략
        :
    else
        health_check
        print_info
    fi
}

# 스크립트 실행
main "$@"
