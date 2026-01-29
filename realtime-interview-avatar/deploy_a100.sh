#!/bin/bash
# ==============================================
# A100 GPU 립싱크 서버 전체 배포 스크립트
# ==============================================
#
# 이 스크립트는 다음을 순서대로 수행합니다:
#   1. Python 가상환경 설정 및 의존성 설치
#   2. MuseTalk 모델 다운로드 (HuggingFace)
#   3. 아바타 영상 사전계산 (precomputed/*.pkl)
#   4. TensorRT 엔진 정리 (sm_120 삭제)
#   5. 서버 실행 테스트
#
# 사용법:
#   chmod +x deploy_a100.sh
#   ./deploy_a100.sh
#
# 예상 소요 시간: 약 30분 (다운로드 속도에 따라 다름)
# ==============================================

set -e  # 오류 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}=============================================="
    echo -e "  $1"
    echo -e "==============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_info() {
    echo -e "[INFO] $1"
}

# ==============================================
# 0. 시스템 확인
# ==============================================
print_header "시스템 확인"

# Python 확인
if command -v python3.10 &> /dev/null; then
    PYTHON=python3.10
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    print_error "Python 3.10+ 가 필요합니다"
    exit 1
fi
print_info "Python: $($PYTHON --version)"

# GPU 확인
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_info "GPU: $GPU_NAME"
else
    print_warning "nvidia-smi를 찾을 수 없습니다"
fi

# ffmpeg 확인
if ! command -v ffmpeg &> /dev/null; then
    print_warning "ffmpeg가 설치되지 않았습니다. 설치: sudo apt install ffmpeg"
fi

# ==============================================
# 1. 가상환경 및 의존성 설치
# ==============================================
print_header "1단계: 가상환경 및 의존성 설치"

if [ ! -d "venv" ]; then
    print_info "가상환경 생성 중..."
    $PYTHON -m venv venv
fi

source venv/bin/activate
print_success "가상환경 활성화: $(python --version)"

# pip 업그레이드
pip install --upgrade pip -q

# PyTorch 설치 (CUDA 12.4)
print_info "PyTorch 설치 중 (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# 나머지 의존성
print_info "의존성 설치 중..."
pip install -r requirements_a100.txt -q

print_success "의존성 설치 완료"

# ==============================================
# 2. MuseTalk 소스 설정
# ==============================================
print_header "2단계: MuseTalk 소스 설정"

MUSETALK_DIR="$(dirname "$SCRIPT_DIR")/MuseTalk"

if [ -d "$MUSETALK_DIR" ]; then
    print_success "MuseTalk 경로: $MUSETALK_DIR"
else
    print_info "MuseTalk 클론 중..."
    git clone https://github.com/TMElyralab/MuseTalk.git "$MUSETALK_DIR"
    print_success "MuseTalk 클론 완료"
fi

export PYTHONPATH="${PYTHONPATH}:${MUSETALK_DIR}"

# ==============================================
# 3. 모델 다운로드
# ==============================================
print_header "3단계: 모델 다운로드 (HuggingFace)"

# huggingface_hub 설치 확인
pip install huggingface_hub -q

print_info "MuseTalk 모델 다운로드 중..."
python scripts/download_musetalk_models.py --models-dir ./models

print_success "모델 다운로드 완료"

# ==============================================
# 4. TensorRT 엔진 정리 (sm_120 삭제)
# ==============================================
print_header "4단계: TensorRT 엔진 정리"

ENGINE_COUNT=0
for engine_file in models/musetalkV15/*.engine 2>/dev/null; do
    if [ -f "$engine_file" ]; then
        echo "  삭제: $(basename "$engine_file")"
        rm -f "$engine_file"
        ENGINE_COUNT=$((ENGINE_COUNT + 1))
    fi
done

if [ $ENGINE_COUNT -eq 0 ]; then
    print_info "삭제할 TensorRT 엔진 없음"
else
    print_success "$ENGINE_COUNT개 TensorRT 엔진 삭제 (A100에서 자동 재생성됨)"
fi

# ==============================================
# 5. 아바타 사전계산 (precomputed)
# ==============================================
print_header "5단계: 아바타 사전계산"

if [ -d "precomputed" ] && [ "$(ls -A precomputed 2>/dev/null)" ]; then
    print_info "precomputed 폴더 존재 - 건너뜁니다"
    print_info "재생성하려면: rm -rf precomputed && ./deploy_a100.sh"
else
    print_info "precomputed 폴더 생성 중..."
    mkdir -p precomputed/720p precomputed/480p precomputed/360p

    # 각 해상도별 사전계산
    AVATARS=("new_talk_short" "new_talk_long" "talk_short" "talk_long")
    RESOLUTIONS=("720p" "480p" "360p")

    for res in "${RESOLUTIONS[@]}"; do
        print_info "[$res] 사전계산 시작..."
        for avatar in "${AVATARS[@]}"; do
            video_path="assets/${res}/${avatar}.mp4"
            if [ -f "$video_path" ]; then
                # 출력 파일명: {video_name}_precomputed.pkl 형식으로 자동 생성됨
                output_file="precomputed/${res}/${avatar}_precomputed.pkl"
                if [ ! -f "$output_file" ]; then
                    echo "  처리 중: $avatar ($res)"
                    python scripts/precompute_avatar.py \
                        --video_path "$video_path" \
                        --output_dir "precomputed/${res}" \
                        --face_index center \
                        --use_float16 2>/dev/null || {
                        print_warning "실패: $avatar ($res)"
                    }
                fi
            fi
        done
    done

    print_success "사전계산 완료"
fi

# ==============================================
# 6. .env 파일 확인
# ==============================================
print_header "6단계: 환경 변수 설정"

if [ ! -f ".env" ]; then
    if [ -f ".env.a100" ]; then
        cp .env.a100 .env
        print_success ".env.a100 → .env 복사 완료"
        print_warning ".env 파일의 API 키를 설정하세요!"
    else
        print_error ".env 파일이 없습니다. .env.a100을 참고하여 생성하세요."
    fi
else
    print_success ".env 파일 확인됨"
fi

# ==============================================
# 7. 설치 확인
# ==============================================
print_header "설치 확인"

echo "디렉토리 구조:"
echo "  assets/     : $(du -sh assets 2>/dev/null | cut -f1)"
echo "  models/     : $(du -sh models 2>/dev/null | cut -f1)"
echo "  precomputed/: $(du -sh precomputed 2>/dev/null | cut -f1)"

echo ""
echo "Python 패키지:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"

# ==============================================
# 완료
# ==============================================
print_header "배포 완료!"

echo -e "서버 실행:"
echo -e "  ${GREEN}bash run_server.sh${NC}"
echo ""
echo -e "주의사항:"
echo -e "  - 첫 실행 시 TensorRT 엔진 빌드로 5~10분 소요"
echo -e "  - .env 파일의 API 키 확인 필요"
echo -e "  - 포트: 5000 (app.py 하드코딩)"
echo ""
