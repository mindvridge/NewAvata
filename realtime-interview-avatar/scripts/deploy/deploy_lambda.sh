#!/bin/bash

# ============================================================================
# Lambda Labs Deployment Script for Interview Avatar System
# ============================================================================
# Lambda Labs: https://lambdalabs.com/
#
# 비용 추정 (시간당):
#   - RTX 4090 (24GB):  N/A (Lambda Labs는 RTX 4090 미제공)
#   - A100 (40GB):      $1.10 /hr (On-Demand)
#   - A100 (80GB):      $1.50 /hr (On-Demand)
#   - H100 (80GB):      $2.49 /hr (On-Demand)
#   - V100 (16GB):      $0.50 /hr (On-Demand)
#
# 권장: A100 (40GB) $1.10/hr - 면접 1회당 약 $0.18
# 특징: 안정적이고 빠른 네트워크, 관리하기 쉬움
# ============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    echo "║      Lambda Labs Deployment - Interview Avatar            ║"
    echo "║                                                            ║"
    echo "║        안정적이고 관리하기 쉬운 GPU 클라우드               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

# 비용 정보 출력
print_cost_estimate() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                💰 비용 추정 (Lambda Labs)                  "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "GPU 타입별 시간당 비용:"
    echo "  • V100 (16GB):      \$0.50 /hr  💰 저렴 (성능 제한)"
    echo "  • A100 (40GB):      \$1.10 /hr  ⭐ 권장"
    echo "  • A100 (80GB):      \$1.50 /hr  🚀 고용량"
    echo "  • H100 (80GB):      \$2.49 /hr  ⚡ 최고 성능"
    echo ""
    echo "예상 사용량 (A100 40GB 기준):"
    echo "  • 면접 1회 (15분):           \$0.28"
    echo "  • 일일 10회 면접:             \$2.80"
    echo "  • 월간 운영 (8시간/일):       \$211.20"
    echo ""
    echo "💡 특징:"
    echo "  - 안정적인 인프라 (99.9% 가동률)"
    echo "  - 빠른 네트워크 (10-100 Gbps)"
    echo "  - 관리하기 쉬운 대시보드"
    echo "  - 1시간 최소 과금"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# 환경 변수 확인
check_env() {
    log_info "환경 변수 확인 중..."

    if [ -z "$LAMBDA_API_KEY" ]; then
        log_error "LAMBDA_API_KEY 환경 변수가 설정되지 않았습니다."
        log_info "Lambda Labs API 키를 생성하세요: https://cloud.lambdalabs.com/api-keys"
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

# GPU 타입 선택
select_gpu() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                    🎮 GPU 타입 선택                        "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1) A100 (40GB)  ⭐ 권장 (\$1.10/hr)"
    echo "2) A100 (80GB)  🚀 고용량 (\$1.50/hr)"
    echo "3) V100 (16GB)  💰 저렴 (\$0.50/hr, 성능 제한)"
    echo "4) H100 (80GB)  ⚡ 최고 성능 (\$2.49/hr)"
    echo ""
    read -p "GPU 타입을 선택하세요 [1-4] (기본: 1): " gpu_choice

    gpu_choice=${gpu_choice:-1}

    case $gpu_choice in
        1)
            INSTANCE_TYPE="gpu_1x_a100_sxm4"
            GPU_NAME="A100 40GB"
            ;;
        2)
            INSTANCE_TYPE="gpu_1x_a100_80gb_sxm4"
            GPU_NAME="A100 80GB"
            ;;
        3)
            INSTANCE_TYPE="gpu_1x_v100"
            GPU_NAME="V100 16GB"
            ;;
        4)
            INSTANCE_TYPE="gpu_1x_h100_pcie"
            GPU_NAME="H100 80GB"
            ;;
        *)
            log_error "잘못된 선택입니다."
            exit 1
            ;;
    esac

    log_success "선택된 GPU: $GPU_NAME"
}

# 지역 선택
select_region() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                    🌍 지역 선택                            "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1) us-west-1 (California)   ⭐ 권장 (한국과 가까움)"
    echo "2) us-west-2 (Oregon)"
    echo "3) us-east-1 (Virginia)"
    echo "4) us-south-1 (Texas)"
    echo "5) europe-central-1 (Germany)"
    echo ""
    read -p "지역을 선택하세요 [1-5] (기본: 1): " region_choice

    region_choice=${region_choice:-1}

    case $region_choice in
        1)
            REGION="us-west-1"
            ;;
        2)
            REGION="us-west-2"
            ;;
        3)
            REGION="us-east-1"
            ;;
        4)
            REGION="us-south-1"
            ;;
        5)
            REGION="europe-central-1"
            ;;
        *)
            log_error "잘못된 선택입니다."
            exit 1
            ;;
    esac

    log_success "선택된 지역: $REGION"
}

# SSH 키 생성 또는 로드
setup_ssh_key() {
    log_info "SSH 키 확인 중..."

    SSH_KEY_PATH="$HOME/.ssh/lambda_interview_avatar"

    if [ ! -f "$SSH_KEY_PATH" ]; then
        log_info "SSH 키 생성 중..."
        ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "interview-avatar"
        log_success "SSH 키 생성 완료"
    else
        log_success "SSH 키 확인 완료"
    fi

    SSH_PUBLIC_KEY=$(cat "${SSH_KEY_PATH}.pub")
}

# 인스턴스 생성
create_instance() {
    log_info "Lambda Labs 인스턴스 생성 중..."

    INSTANCE_NAME="interview-avatar-$(date +%s)"

    # Lambda Labs API를 통해 인스턴스 생성
    RESPONSE=$(curl -s -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/launch" \
        -H "Authorization: Bearer $LAMBDA_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "region_name": "'$REGION'",
            "instance_type_name": "'$INSTANCE_TYPE'",
            "ssh_key_names": ["interview-avatar"],
            "name": "'$INSTANCE_NAME'",
            "quantity": 1
        }')

    # 응답 파싱
    INSTANCE_IDS=$(echo "$RESPONSE" | jq -r '.data.instance_ids[0]')

    if [ "$INSTANCE_IDS" == "null" ] || [ -z "$INSTANCE_IDS" ]; then
        log_error "인스턴스 생성 실패"
        echo "$RESPONSE" | jq .
        exit 1
    fi

    log_success "인스턴스 생성 완료: $INSTANCE_IDS"

    # 인스턴스 ID 저장
    echo "$INSTANCE_IDS" > .lambda_instance_id
}

# 인스턴스 시작 대기
wait_for_instance() {
    log_info "인스턴스 시작 대기 중..."

    INSTANCE_ID=$(cat .lambda_instance_id)

    # 최대 10분 대기
    for i in {1..60}; do
        RESPONSE=$(curl -s -X GET "https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID" \
            -H "Authorization: Bearer $LAMBDA_API_KEY")

        STATUS=$(echo "$RESPONSE" | jq -r '.data.status')

        if [ "$STATUS" == "active" ]; then
            log_success "인스턴스 시작 완료!"

            # IP 주소 가져오기
            INSTANCE_IP=$(echo "$RESPONSE" | jq -r '.data.ip')
            echo "$INSTANCE_IP" > .lambda_instance_ip

            break
        fi

        log_info "인스턴스 상태: $STATUS ($i/60)"
        sleep 10
    done

    if [ "$STATUS" != "active" ]; then
        log_error "인스턴스 시작 실패: 타임아웃"
        exit 1
    fi
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

# SSH를 통해 컨테이너 배포
deploy_container() {
    log_info "SSH를 통해 컨테이너 배포 중..."

    INSTANCE_IP=$(cat .lambda_instance_ip)
    SSH_KEY_PATH="$HOME/.ssh/lambda_interview_avatar"

    # SSH 연결 대기
    log_info "SSH 연결 대기 중..."
    for i in {1..30}; do
        if ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "echo connected" &> /dev/null; then
            log_success "SSH 연결 성공"
            break
        fi
        log_info "SSH 연결 대기 중... ($i/30)"
        sleep 10
    done

    # Docker 설치 및 컨테이너 실행
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << EOF
# Docker 설치 (이미 설치되어 있을 수 있음)
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
fi

# NVIDIA Container Toolkit 설치
if ! command -v nvidia-container-toolkit &> /dev/null; then
    distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
fi

# 컨테이너 실행
sudo docker run -d --gpus all -p 8000:8000 \
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

    log_success "컨테이너 배포 완료"
}

# 헬스체크
health_check() {
    log_info "헬스체크 수행 중..."

    INSTANCE_IP=$(cat .lambda_instance_ip)
    APP_URL="http://$INSTANCE_IP:8000"

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

    INSTANCE_ID=$(cat .lambda_instance_id)
    INSTANCE_IP=$(cat .lambda_instance_ip)

    DEPLOYMENT_FILE="deployment_info_lambda_$(date +%Y%m%d_%H%M%S).json"

    cat > "$DEPLOYMENT_FILE" <<EOF
{
    "provider": "lambda_labs",
    "instance_id": "$INSTANCE_ID",
    "instance_ip": "$INSTANCE_IP",
    "app_url": "http://$INSTANCE_IP:8000",
    "instance_type": "$INSTANCE_TYPE",
    "gpu_name": "$GPU_NAME",
    "region": "$REGION",
    "image_name": "$IMAGE_NAME",
    "deployed_at": "$(date -Iseconds)"
}
EOF

    log_success "배포 정보 저장 완료: $DEPLOYMENT_FILE"
}

# 메인 함수
main() {
    print_banner
    print_cost_estimate

    check_env
    select_gpu
    select_region
    setup_ssh_key
    prepare_docker_image
    create_instance
    wait_for_instance
    deploy_container
    health_check
    save_deployment_info

    INSTANCE_ID=$(cat .lambda_instance_id)
    INSTANCE_IP=$(cat .lambda_instance_ip)

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_success "배포가 완료되었습니다! 🎉"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "애플리케이션 URL: http://$INSTANCE_IP:8000"
    echo "인스턴스 ID: $INSTANCE_ID"
    echo "인스턴스 IP: $INSTANCE_IP"
    echo ""
    echo "유용한 명령어:"
    echo "  • SSH 연결:       ssh -i ~/.ssh/lambda_interview_avatar ubuntu@$INSTANCE_IP"
    echo "  • 컨테이너 로그:  ssh -i ~/.ssh/lambda_interview_avatar ubuntu@$INSTANCE_IP 'sudo docker logs interview-avatar'"
    echo "  • 인스턴스 종료:  Lambda Labs 대시보드에서 수동 종료"
    echo ""
    echo "💡 사용 후 반드시 인스턴스를 종료하세요!"
    echo "   https://cloud.lambdalabs.com/instances"
    echo ""
}

# 스크립트 실행
main "$@"
