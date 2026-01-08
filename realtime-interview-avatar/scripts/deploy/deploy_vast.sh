#!/bin/bash

# ============================================================================
# Vast.ai Deployment Script for Interview Avatar System
# ============================================================================
# Vast.ai: https://vast.ai/
#
# 비용 추정 (시간당):
#   - RTX 4090 (24GB):  $0.20 - $0.40 (최저가) ⭐ 가장 저렴
#   - RTX 4090 (24GB):  $0.30 - $0.50 (평균)
#   - RTX 3090 (24GB):  $0.15 - $0.35
#   - A100 (40GB):      $0.60 - $1.20
#   - A100 (80GB):      $0.80 - $1.50
#
# 권장: RTX 4090 ($0.25/hr 평균) - 면접 1회당 약 $0.04-0.06
# 특징: 가장 저렴하지만, 인스턴스 안정성이 다를 수 있음
# ============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
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
    echo "║         Vast.ai Deployment - Interview Avatar             ║"
    echo "║                                                            ║"
    echo "║      가장 저렴한 GPU 클라우드 배포 솔루션                  ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

# 비용 정보 출력
print_cost_estimate() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                    💰 비용 추정 (Vast.ai)                  "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "GPU 타입별 시간당 비용 (실시간 최저가 기준):"
    echo "  • RTX 4090 (24GB):  \$0.20 - \$0.40 /hr  ⭐ 가장 저렴"
    echo "  • RTX 3090 (24GB):  \$0.15 - \$0.35 /hr"
    echo "  • RTX 3080 (10GB):  \$0.10 - \$0.25 /hr"
    echo "  • A100 (40GB):      \$0.60 - \$1.20 /hr"
    echo "  • A100 (80GB):      \$0.80 - \$1.50 /hr"
    echo ""
    echo "예상 사용량 (RTX 4090 기준):"
    echo "  • 면접 1회 (15분):           \$0.04 - \$0.06"
    echo "  • 일일 10회 면접:             \$0.40 - \$0.60"
    echo "  • 월간 운영 (8시간/일):       \$48 - \$76.80"
    echo ""
    echo "💡 팁: 신뢰도(reliability) 95% 이상 인스턴스 선택 권장"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# 환경 변수 확인
check_env() {
    log_info "환경 변수 확인 중..."

    if [ -z "$VAST_API_KEY" ]; then
        log_error "VAST_API_KEY 환경 변수가 설정되지 않았습니다."
        log_info "Vast.ai API 키를 생성하세요: https://cloud.vast.ai/api/"
        exit 1
    fi

    # .env 파일에서 API 키 로드
    if [ -f .env ]; then
        log_info ".env 파일에서 API 키 로드 중..."
        source .env
    else
        log_warning ".env 파일이 없습니다."
    fi

    log_success "환경 변수 확인 완료"
}

# Vast.ai CLI 설치 확인
check_vast_cli() {
    log_info "Vast.ai CLI 확인 중..."

    if ! command -v vast &> /dev/null; then
        log_warning "Vast.ai CLI가 설치되지 않았습니다. 설치 중..."

        pip install vastai

        log_success "Vast.ai CLI 설치 완료"
    else
        log_success "Vast.ai CLI 확인 완료"
    fi

    # API 키 설정
    vast set api-key "$VAST_API_KEY"
}

# 최적의 인스턴스 검색
search_instances() {
    log_info "최적의 GPU 인스턴스 검색 중..."

    # GPU 타입 선택
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                    🎮 GPU 타입 선택                        "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1) RTX 4090 (24GB)  ⭐ 권장 (\$0.25/hr 평균)"
    echo "2) RTX 3090 (24GB)  💰 저렴 (\$0.20/hr 평균)"
    echo "3) A100 (40GB)      🚀 고성능 (\$0.80/hr 평균)"
    echo "4) A100 (80GB)      🚀 초고성능 (\$1.00/hr 평균)"
    echo "5) 자동 선택 (가격 최적화)"
    echo ""
    read -p "GPU 타입을 선택하세요 [1-5] (기본: 1): " gpu_choice

    gpu_choice=${gpu_choice:-1}

    case $gpu_choice in
        1)
            GPU_NAME="RTX_4090"
            MIN_VRAM=20
            ;;
        2)
            GPU_NAME="RTX_3090"
            MIN_VRAM=20
            ;;
        3)
            GPU_NAME="A100"
            MIN_VRAM=35
            ;;
        4)
            GPU_NAME="A100"
            MIN_VRAM=75
            ;;
        5)
            GPU_NAME=""
            MIN_VRAM=20
            ;;
        *)
            log_error "잘못된 선택입니다."
            exit 1
            ;;
    esac

    # 검색 조건
    MIN_CUDA=11.8
    MIN_RELIABILITY=0.95
    MAX_PRICE=2.0

    # 인스턴스 검색
    log_info "검색 조건: GPU=$GPU_NAME, VRAM>=${MIN_VRAM}GB, CUDA>=$MIN_CUDA, Reliability>=${MIN_RELIABILITY}"

    if [ -z "$GPU_NAME" ]; then
        OFFERS=$(vast search offers \
            "cuda_max_good >= $MIN_CUDA gpu_ram >= $MIN_VRAM reliability >= $MIN_RELIABILITY dph < $MAX_PRICE" \
            --order "dph+" \
            --limit 10)
    else
        OFFERS=$(vast search offers \
            "cuda_max_good >= $MIN_CUDA gpu_ram >= $MIN_VRAM gpu_name = $GPU_NAME reliability >= $MIN_RELIABILITY dph < $MAX_PRICE" \
            --order "dph+" \
            --limit 10)
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                  🔍 검색 결과 (상위 10개)                  "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "$OFFERS"
    echo ""

    # 인스턴스 ID 입력
    read -p "사용할 인스턴스 ID를 입력하세요 (또는 Enter로 최저가 선택): " INSTANCE_ID

    if [ -z "$INSTANCE_ID" ]; then
        INSTANCE_ID=$(echo "$OFFERS" | awk 'NR==2 {print $1}')
        log_info "최저가 인스턴스 자동 선택: $INSTANCE_ID"
    fi

    log_success "선택된 인스턴스: $INSTANCE_ID"
}

# Docker 이미지 준비
prepare_docker_image() {
    log_info "Docker 이미지 준비 중..."

    if [ -z "$DOCKER_USERNAME" ]; then
        log_error "DOCKER_USERNAME 환경 변수가 설정되지 않았습니다."
        exit 1
    fi

    IMAGE_NAME="$DOCKER_USERNAME/interview-avatar:latest"

    # 이미지 빌드 확인
    read -p "Docker 이미지를 새로 빌드하시겠습니까? [y/N]: " build_image
    if [[ "$build_image" =~ ^[Yy]$ ]]; then
        log_info "Docker 이미지 빌드 중..."
        docker build -t "$IMAGE_NAME" -f docker/Dockerfile .

        log_info "Docker 이미지 푸시 중..."
        docker push "$IMAGE_NAME"
    fi

    log_success "이미지 준비 완료: $IMAGE_NAME"
}

# 인스턴스 생성 및 배포
deploy_instance() {
    log_info "Vast.ai 인스턴스 생성 중..."

    # Docker 실행 명령어
    DOCKER_CMD="docker run --gpus all -p 8000:8000 \
        -e DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY \
        -e ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY \
        -e OPENAI_API_KEY=$OPENAI_API_KEY \
        -e DAILY_API_KEY=$DAILY_API_KEY \
        -e APP_ENV=production \
        -e LOG_LEVEL=INFO \
        -e NVIDIA_VISIBLE_DEVICES=all \
        $IMAGE_NAME"

    # 인스턴스 생성
    INSTANCE_INFO=$(vast create instance $INSTANCE_ID \
        --image "$IMAGE_NAME" \
        --disk 50 \
        --env "DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY" \
        --env "ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY" \
        --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
        --env "DAILY_API_KEY=$DAILY_API_KEY" \
        --env "APP_ENV=production" \
        --env "NVIDIA_VISIBLE_DEVICES=all" \
        --label "interview-avatar" \
        --onstart-cmd "apt-get update && apt-get install -y curl")

    # 인스턴스 ID 추출
    CREATED_INSTANCE_ID=$(echo "$INSTANCE_INFO" | grep -oP 'instance \K[0-9]+')

    log_success "인스턴스 생성 완료: $CREATED_INSTANCE_ID"

    # 인스턴스 정보 저장
    echo "$CREATED_INSTANCE_ID" > .vast_instance_id
}

# 인스턴스 시작 대기
wait_for_instance() {
    log_info "인스턴스 시작 대기 중..."

    CREATED_INSTANCE_ID=$(cat .vast_instance_id)

    # 최대 5분 대기
    for i in {1..30}; do
        STATUS=$(vast show instances --raw | jq -r ".[] | select(.id==$CREATED_INSTANCE_ID) | .actual_status")

        if [ "$STATUS" == "running" ]; then
            log_success "인스턴스 시작 완료!"
            break
        fi

        log_info "인스턴스 상태: $STATUS ($i/30)"
        sleep 10
    done

    if [ "$STATUS" != "running" ]; then
        log_error "인스턴스 시작 실패: 타임아웃"
        exit 1
    fi
}

# SSH 연결 및 컨테이너 시작
start_container() {
    log_info "SSH를 통해 컨테이너 시작 중..."

    CREATED_INSTANCE_ID=$(cat .vast_instance_id)

    # SSH 정보 가져오기
    SSH_HOST=$(vast show instances --raw | jq -r ".[] | select(.id==$CREATED_INSTANCE_ID) | .ssh_host")
    SSH_PORT=$(vast show instances --raw | jq -r ".[] | select(.id==$CREATED_INSTANCE_ID) | .ssh_port")

    log_info "SSH 연결: $SSH_HOST:$SSH_PORT"

    # SSH를 통해 Docker 컨테이너 시작
    ssh -p $SSH_PORT root@$SSH_HOST << EOF
docker run -d --gpus all -p 8000:8000 \
    -e DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY \
    -e ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e DAILY_API_KEY=$DAILY_API_KEY \
    -e APP_ENV=production \
    -e LOG_LEVEL=INFO \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --name interview-avatar \
    --restart unless-stopped \
    $IMAGE_NAME
EOF

    log_success "컨테이너 시작 완료"
}

# 헬스체크
health_check() {
    log_info "헬스체크 수행 중..."

    CREATED_INSTANCE_ID=$(cat .vast_instance_id)

    # 공개 IP 가져오기
    PUBLIC_IP=$(vast show instances --raw | jq -r ".[] | select(.id==$CREATED_INSTANCE_ID) | .public_ipaddr")

    APP_URL="http://$PUBLIC_IP:8000"

    # 최대 5분 대기
    for i in {1..30}; do
        if curl -f "$APP_URL/api/health" &> /dev/null; then
            log_success "헬스체크 성공!"
            log_success "애플리케이션 URL: $APP_URL"
            return 0
        fi

        log_info "헬스체크 대기 중... ($i/30)"
        sleep 10
    done

    log_error "헬스체크 실패: 타임아웃"
    return 1
}

# 배포 정보 저장
save_deployment_info() {
    log_info "배포 정보 저장 중..."

    CREATED_INSTANCE_ID=$(cat .vast_instance_id)
    PUBLIC_IP=$(vast show instances --raw | jq -r ".[] | select(.id==$CREATED_INSTANCE_ID) | .public_ipaddr")
    INSTANCE_DETAILS=$(vast show instances --raw | jq ".[] | select(.id==$CREATED_INSTANCE_ID)")

    DEPLOYMENT_FILE="deployment_info_vast_$(date +%Y%m%d_%H%M%S).json"

    cat > "$DEPLOYMENT_FILE" <<EOF
{
    "provider": "vast.ai",
    "instance_id": $CREATED_INSTANCE_ID,
    "public_ip": "$PUBLIC_IP",
    "app_url": "http://$PUBLIC_IP:8000",
    "image_name": "$IMAGE_NAME",
    "deployed_at": "$(date -Iseconds)",
    "instance_details": $INSTANCE_DETAILS
}
EOF

    log_success "배포 정보 저장 완료: $DEPLOYMENT_FILE"
}

# 메인 함수
main() {
    print_banner
    print_cost_estimate

    check_env
    check_vast_cli
    search_instances
    prepare_docker_image
    deploy_instance
    wait_for_instance
    start_container
    health_check
    save_deployment_info

    CREATED_INSTANCE_ID=$(cat .vast_instance_id)
    PUBLIC_IP=$(vast show instances --raw | jq -r ".[] | select(.id==$CREATED_INSTANCE_ID) | .public_ipaddr")

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_success "배포가 완료되었습니다! 🎉"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "애플리케이션 URL: http://$PUBLIC_IP:8000"
    echo "인스턴스 ID: $CREATED_INSTANCE_ID"
    echo ""
    echo "유용한 명령어:"
    echo "  • 인스턴스 상태:  vast show instances"
    echo "  • SSH 연결:       vast ssh $CREATED_INSTANCE_ID"
    echo "  • 인스턴스 중지:  vast stop instance $CREATED_INSTANCE_ID"
    echo "  • 인스턴스 삭제:  vast destroy instance $CREATED_INSTANCE_ID"
    echo ""
    echo "💡 사용 후 반드시 인스턴스를 중지/삭제하세요!"
    echo ""
}

# 스크립트 실행
main "$@"
