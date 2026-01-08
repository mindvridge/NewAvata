#!/bin/bash

# ============================================================================
# RunPod Deployment Script for Interview Avatar System
# ============================================================================
# RunPod: https://www.runpod.io/
#
# 비용 추정 (시간당):
#   - RTX 4090 (24GB):  $0.44 - $0.69 (On-Demand)
#   - RTX 4090 (24GB):  $0.34 - $0.54 (Spot)
#   - A100 (40GB):      $1.29 - $1.89 (On-Demand)
#   - A100 (40GB):      $0.99 - $1.49 (Spot)
#   - A100 (80GB):      $1.89 - $2.49 (On-Demand)
#
# 권장: RTX 4090 Spot ($0.34/hr) - 면접 1회당 약 $0.06-0.10
# ============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo "║         RunPod Deployment - Interview Avatar              ║"
    echo "║                                                            ║"
    echo "║  GPU-accelerated realtime interview avatar deployment     ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

# 비용 정보 출력
print_cost_estimate() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                    💰 비용 추정                            "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "GPU 타입별 시간당 비용:"
    echo "  • RTX 4090 (24GB) Spot:      \$0.34 - \$0.54 /hr  ⭐ 권장"
    echo "  • RTX 4090 (24GB) On-Demand: \$0.44 - \$0.69 /hr"
    echo "  • A100 (40GB) Spot:          \$0.99 - \$1.49 /hr"
    echo "  • A100 (40GB) On-Demand:     \$1.29 - \$1.89 /hr"
    echo "  • A100 (80GB) On-Demand:     \$1.89 - \$2.49 /hr"
    echo ""
    echo "예상 사용량:"
    echo "  • 면접 1회 (15분):           \$0.06 - \$0.10"
    echo "  • 일일 10회 면접:             \$0.60 - \$1.00"
    echo "  • 월간 운영 (8시간/일):       \$81.60 - \$129.60"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# 환경 변수 확인
check_env() {
    log_info "환경 변수 확인 중..."

    if [ -z "$RUNPOD_API_KEY" ]; then
        log_error "RUNPOD_API_KEY 환경 변수가 설정되지 않았습니다."
        log_info "RunPod API 키를 생성하세요: https://www.runpod.io/console/user/settings"
        exit 1
    fi

    # .env 파일에서 API 키 로드
    if [ -f .env ]; then
        log_info ".env 파일에서 API 키 로드 중..."
        source .env
    else
        log_warning ".env 파일이 없습니다. .env.example을 복사하여 설정하세요."
    fi

    log_success "환경 변수 확인 완료"
}

# RunPod CLI 설치 확인
check_runpod_cli() {
    log_info "RunPod CLI 확인 중..."

    if ! command -v runpodctl &> /dev/null; then
        log_warning "RunPod CLI가 설치되지 않았습니다. 설치 중..."

        # OS 감지
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            wget -qO- https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -O /usr/local/bin/runpodctl
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            wget -qO- https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-darwin-amd64 -O /usr/local/bin/runpodctl
        else
            log_error "지원하지 않는 OS입니다."
            exit 1
        fi

        chmod +x /usr/local/bin/runpodctl
        log_success "RunPod CLI 설치 완료"
    else
        log_success "RunPod CLI 확인 완료"
    fi

    # API 키 설정
    runpodctl config --apiKey "$RUNPOD_API_KEY"
}

# GPU 타입 선택
select_gpu() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                    🎮 GPU 타입 선택                        "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1) RTX 4090 (24GB) - Spot         ⭐ 권장 (\$0.34-0.54/hr)"
    echo "2) RTX 4090 (24GB) - On-Demand    (\$0.44-0.69/hr)"
    echo "3) A100 (40GB) - Spot             (\$0.99-1.49/hr)"
    echo "4) A100 (40GB) - On-Demand        (\$1.29-1.89/hr)"
    echo "5) A100 (80GB) - On-Demand        (\$1.89-2.49/hr)"
    echo ""
    read -p "GPU 타입을 선택하세요 [1-5] (기본: 1): " gpu_choice

    gpu_choice=${gpu_choice:-1}

    case $gpu_choice in
        1)
            GPU_TYPE="NVIDIA GeForce RTX 4090"
            GPU_COUNT=1
            DISK_SIZE=50
            INSTANCE_TYPE="spot"
            ;;
        2)
            GPU_TYPE="NVIDIA GeForce RTX 4090"
            GPU_COUNT=1
            DISK_SIZE=50
            INSTANCE_TYPE="on-demand"
            ;;
        3)
            GPU_TYPE="NVIDIA A100-SXM4-40GB"
            GPU_COUNT=1
            DISK_SIZE=50
            INSTANCE_TYPE="spot"
            ;;
        4)
            GPU_TYPE="NVIDIA A100-SXM4-40GB"
            GPU_COUNT=1
            DISK_SIZE=50
            INSTANCE_TYPE="on-demand"
            ;;
        5)
            GPU_TYPE="NVIDIA A100-SXM4-80GB"
            GPU_COUNT=1
            DISK_SIZE=50
            INSTANCE_TYPE="on-demand"
            ;;
        *)
            log_error "잘못된 선택입니다."
            exit 1
            ;;
    esac

    log_success "선택된 GPU: $GPU_TYPE ($INSTANCE_TYPE)"
}

# 배포 모드 선택
select_deployment_mode() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                  📦 배포 모드 선택                         "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1) Pod (지속적 실행)        - 24/7 운영, 고정 비용"
    echo "2) Serverless (요청 기반)  - 사용량 기반, 자동 스케일링"
    echo ""
    read -p "배포 모드를 선택하세요 [1-2] (기본: 1): " mode_choice

    mode_choice=${mode_choice:-1}

    case $mode_choice in
        1)
            DEPLOYMENT_MODE="pod"
            ;;
        2)
            DEPLOYMENT_MODE="serverless"
            ;;
        *)
            log_error "잘못된 선택입니다."
            exit 1
            ;;
    esac

    log_success "선택된 배포 모드: $DEPLOYMENT_MODE"
}

# Docker 이미지 빌드 및 푸시
build_and_push_image() {
    log_info "Docker 이미지 빌드 중..."

    # Docker Hub 로그인 확인
    if [ -z "$DOCKER_USERNAME" ]; then
        log_error "DOCKER_USERNAME 환경 변수가 설정되지 않았습니다."
        exit 1
    fi

    IMAGE_NAME="$DOCKER_USERNAME/interview-avatar:latest"

    # 이미지 빌드
    docker build -t "$IMAGE_NAME" -f docker/Dockerfile .

    log_info "Docker 이미지 푸시 중..."
    docker push "$IMAGE_NAME"

    log_success "이미지 빌드 및 푸시 완료: $IMAGE_NAME"
}

# Pod 배포
deploy_pod() {
    log_info "RunPod Pod 배포 중..."

    POD_NAME="interview-avatar-$(date +%s)"

    # Pod 생성
    runpodctl create pod \
        --name "$POD_NAME" \
        --gpuType "$GPU_TYPE" \
        --gpuCount $GPU_COUNT \
        --containerDiskSize $DISK_SIZE \
        --imageName "$IMAGE_NAME" \
        --volumeSize 100 \
        --ports "8000/http" \
        --env "DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY" \
        --env "ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY" \
        --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
        --env "DAILY_API_KEY=$DAILY_API_KEY" \
        --env "APP_ENV=production" \
        --env "LOG_LEVEL=INFO" \
        --env "NVIDIA_VISIBLE_DEVICES=all" \
        --env "CUDA_VISIBLE_DEVICES=0"

    log_success "Pod 배포 완료: $POD_NAME"

    # Pod 정보 조회
    sleep 5
    log_info "Pod 정보 조회 중..."
    runpodctl get pod "$POD_NAME"
}

# Serverless 배포
deploy_serverless() {
    log_info "RunPod Serverless 배포 중..."

    ENDPOINT_NAME="interview-avatar-$(date +%s)"

    # Serverless 엔드포인트 생성 (RunPod API 사용)
    curl -X POST "https://api.runpod.io/graphql" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d '{
            "query": "mutation { saveEndpoint(input: { name: \"'$ENDPOINT_NAME'\", gpuIds: \"'$GPU_TYPE'\", workersMin: 0, workersMax: 3, idleTimeout: 5, imageName: \"'$IMAGE_NAME'\", dockerArgs: \"-p 8000:8000\", env: [{ key: \"DEEPGRAM_API_KEY\", value: \"'$DEEPGRAM_API_KEY'\" }, { key: \"ELEVENLABS_API_KEY\", value: \"'$ELEVENLABS_API_KEY'\" }, { key: \"OPENAI_API_KEY\", value: \"'$OPENAI_API_KEY'\" }, { key: \"DAILY_API_KEY\", value: \"'$DAILY_API_KEY'\" }] }) { id name } }"
        }'

    log_success "Serverless 엔드포인트 배포 완료: $ENDPOINT_NAME"
}

# 헬스체크
health_check() {
    log_info "헬스체크 수행 중..."

    if [ "$DEPLOYMENT_MODE" == "pod" ]; then
        # Pod URL 가져오기
        POD_URL=$(runpodctl get pod "$POD_NAME" --json | jq -r '.httpEndpoint')

        # 최대 5분 대기
        for i in {1..30}; do
            if curl -f "$POD_URL/api/health" &> /dev/null; then
                log_success "헬스체크 성공!"
                log_success "애플리케이션 URL: $POD_URL"
                return 0
            fi

            log_info "헬스체크 대기 중... ($i/30)"
            sleep 10
        done

        log_error "헬스체크 실패: 타임아웃"
        return 1
    fi
}

# 배포 정보 저장
save_deployment_info() {
    log_info "배포 정보 저장 중..."

    DEPLOYMENT_FILE="deployment_info_runpod_$(date +%Y%m%d_%H%M%S).json"

    cat > "$DEPLOYMENT_FILE" <<EOF
{
    "provider": "runpod",
    "deployment_mode": "$DEPLOYMENT_MODE",
    "pod_name": "$POD_NAME",
    "gpu_type": "$GPU_TYPE",
    "instance_type": "$INSTANCE_TYPE",
    "image_name": "$IMAGE_NAME",
    "deployed_at": "$(date -Iseconds)",
    "pod_url": "$POD_URL"
}
EOF

    log_success "배포 정보 저장 완료: $DEPLOYMENT_FILE"
}

# 메인 함수
main() {
    print_banner
    print_cost_estimate

    check_env
    check_runpod_cli
    select_gpu
    select_deployment_mode

    # Docker 이미지 빌드 확인
    read -p "Docker 이미지를 새로 빌드하시겠습니까? [y/N]: " build_image
    if [[ "$build_image" =~ ^[Yy]$ ]]; then
        build_and_push_image
    fi

    # 배포 실행
    if [ "$DEPLOYMENT_MODE" == "pod" ]; then
        deploy_pod
        health_check
    else
        deploy_serverless
    fi

    save_deployment_info

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_success "배포가 완료되었습니다! 🎉"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if [ "$DEPLOYMENT_MODE" == "pod" ]; then
        echo "Pod URL: $POD_URL"
        echo ""
        echo "유용한 명령어:"
        echo "  • Pod 상태 확인:  runpodctl get pod $POD_NAME"
        echo "  • Pod 로그 확인:  runpodctl logs $POD_NAME"
        echo "  • Pod 중지:       runpodctl stop pod $POD_NAME"
        echo "  • Pod 삭제:       runpodctl remove pod $POD_NAME"
    fi

    echo ""
}

# 스크립트 실행
main "$@"
