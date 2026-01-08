#!/bin/bash

################################################################################
# MuseTalk 모델 및 의존성 자동 설치 스크립트
#
# 사용법:
#   bash scripts/setup_musetalk.sh
#
# 옵션:
#   --use-mirror    중국 Hugging Face 미러 사용 (다운로드 속도 향상)
#   --skip-clone    저장소 클론 건너뛰기 (이미 클론된 경우)
#   --models-only   모델만 다운로드 (저장소 클론 안함)
################################################################################

set -e  # 에러 발생 시 즉시 종료

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models/musetalk"
TEMP_DIR="${PROJECT_ROOT}/temp/musetalk_setup"

# Hugging Face 미러 설정
USE_MIRROR=false
SKIP_CLONE=false
MODELS_ONLY=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-mirror)
            USE_MIRROR=true
            shift
            ;;
        --skip-clone)
            SKIP_CLONE=true
            shift
            ;;
        --models-only)
            MODELS_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            exit 1
            ;;
    esac
done

################################################################################
# 유틸리티 함수
################################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1이(가) 설치되어 있지 않습니다."
        return 1
    fi
    return 0
}

################################################################################
# 환경 확인
################################################################################

print_header "환경 확인"

# Git 확인
if check_command git; then
    print_success "Git 설치됨 ($(git --version))"
else
    print_error "Git을 설치해주세요: https://git-scm.com/"
    exit 1
fi

# Python 확인
if check_command python3 || check_command python; then
    PYTHON_CMD=$(command -v python3 || command -v python)
    PYTHON_VERSION=$($PYTHON_CMD --version)
    print_success "Python 설치됨 ($PYTHON_VERSION)"
else
    print_error "Python을 설치해주세요: https://www.python.org/"
    exit 1
fi

# pip 확인
if $PYTHON_CMD -m pip --version &> /dev/null; then
    print_success "pip 설치됨"
else
    print_error "pip를 설치해주세요"
    exit 1
fi

# wget 또는 curl 확인
if check_command wget; then
    DOWNLOAD_CMD="wget -c"
    print_success "wget 설치됨"
elif check_command curl; then
    DOWNLOAD_CMD="curl -L -O -C -"
    print_success "curl 설치됨"
else
    print_warning "wget 또는 curl이 설치되어 있지 않습니다. Python으로 다운로드합니다."
    DOWNLOAD_CMD="python_download"
fi

################################################################################
# 디렉토리 생성
################################################################################

print_header "디렉토리 생성"

mkdir -p "${MODELS_DIR}"
mkdir -p "${TEMP_DIR}"

print_success "모델 디렉토리: ${MODELS_DIR}"
print_success "임시 디렉토리: ${TEMP_DIR}"

################################################################################
# MuseTalk 저장소 클론
################################################################################

if [ "$MODELS_ONLY" = false ]; then
    print_header "MuseTalk 저장소 클론"

    MUSETALK_REPO_DIR="${PROJECT_ROOT}/third_party/MuseTalk"

    if [ -d "${MUSETALK_REPO_DIR}" ] && [ "$SKIP_CLONE" = true ]; then
        print_info "저장소 클론 건너뛰기 (이미 존재함)"
    else
        mkdir -p "${PROJECT_ROOT}/third_party"

        if [ -d "${MUSETALK_REPO_DIR}" ]; then
            print_warning "기존 저장소 삭제 중..."
            rm -rf "${MUSETALK_REPO_DIR}"
        fi

        print_info "MuseTalk 저장소 클론 중..."
        git clone https://github.com/TMElyralab/MuseTalk.git "${MUSETALK_REPO_DIR}"

        if [ $? -eq 0 ]; then
            print_success "저장소 클론 완료"
        else
            print_error "저장소 클론 실패"
            exit 1
        fi
    fi
fi

################################################################################
# Hugging Face CLI 설치
################################################################################

print_header "Hugging Face CLI 설치"

if $PYTHON_CMD -c "import huggingface_hub" 2>/dev/null; then
    print_info "huggingface_hub 이미 설치됨"
else
    print_info "huggingface_hub 설치 중..."
    $PYTHON_CMD -m pip install -q huggingface_hub
    print_success "huggingface_hub 설치 완료"
fi

################################################################################
# 모델 다운로드 함수
################################################################################

download_from_hf() {
    local repo_id=$1
    local local_dir=$2
    local allow_patterns=$3

    print_info "다운로드 중: ${repo_id}"

    if [ "$USE_MIRROR" = true ]; then
        export HF_ENDPOINT=https://hf-mirror.com
        print_info "Hugging Face 미러 사용: ${HF_ENDPOINT}"
    fi

    if [ -n "$allow_patterns" ]; then
        $PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import sys

try:
    snapshot_download(
        repo_id='${repo_id}',
        local_dir='${local_dir}',
        allow_patterns='${allow_patterns}',
        resume_download=True
    )
    print('✓ 다운로드 완료')
except Exception as e:
    print(f'✗ 다운로드 실패: {e}', file=sys.stderr)
    sys.exit(1)
"
    else
        $PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import sys

try:
    snapshot_download(
        repo_id='${repo_id}',
        local_dir='${local_dir}',
        resume_download=True
    )
    print('✓ 다운로드 완료')
except Exception as e:
    print(f'✗ 다운로드 실패: {e}', file=sys.stderr)
    sys.exit(1)
"
    fi
}

################################################################################
# 1. VAE 모델 다운로드
################################################################################

print_header "1. VAE 모델 다운로드"

VAE_DIR="${MODELS_DIR}/sd-vae-ft-mse"

if [ -d "${VAE_DIR}" ] && [ -f "${VAE_DIR}/diffusion_pytorch_model.safetensors" ]; then
    print_info "VAE 모델이 이미 존재합니다. 건너뜁니다."
else
    download_from_hf \
        "stabilityai/sd-vae-ft-mse" \
        "${VAE_DIR}" \
        "*.safetensors"

    print_success "VAE 모델 다운로드 완료"
fi

################################################################################
# 2. Whisper 모델 다운로드
################################################################################

print_header "2. Whisper 모델 다운로드"

WHISPER_DIR="${MODELS_DIR}/whisper"

if [ -d "${WHISPER_DIR}" ] && [ -f "${WHISPER_DIR}/tiny.pt" ]; then
    print_info "Whisper 모델이 이미 존재합니다. 건너뜁니다."
else
    mkdir -p "${WHISPER_DIR}"

    print_info "Whisper tiny 모델 다운로드 중..."

    # OpenAI Whisper 모델 다운로드
    $PYTHON_CMD -c "
import whisper
import shutil

try:
    # tiny 모델 다운로드 (가장 빠름, 39MB)
    model = whisper.load_model('tiny')
    print('✓ Whisper tiny 모델 다운로드 완료')

    # 모델 파일 복사
    import os
    cache_dir = os.path.expanduser('~/.cache/whisper')
    tiny_path = os.path.join(cache_dir, 'tiny.pt')

    if os.path.exists(tiny_path):
        shutil.copy(tiny_path, '${WHISPER_DIR}/tiny.pt')
        print('✓ 모델 파일 복사 완료')
except Exception as e:
    print(f'⚠ Whisper 다운로드 중 경고: {e}')
    print('수동으로 다운로드해주세요: https://openaipublic.azureedge.net/main/whisper/models/tiny.pt')
"

    if [ -f "${WHISPER_DIR}/tiny.pt" ]; then
        print_success "Whisper 모델 설치 완료"
    else
        print_warning "Whisper 모델을 수동으로 다운로드해주세요"
    fi
fi

################################################################################
# 3. DWPose 모델 다운로드
################################################################################

print_header "3. DWPose 모델 다운로드"

DWPOSE_DIR="${MODELS_DIR}/dwpose"

if [ -d "${DWPOSE_DIR}" ]; then
    print_info "DWPose 모델이 이미 존재합니다. 건너뜁니다."
else
    download_from_hf \
        "yzd-v/DWPose" \
        "${DWPOSE_DIR}" \
        ""

    print_success "DWPose 모델 다운로드 완료"
fi

################################################################################
# 4. Face Parse BiSeNet 모델 다운로드
################################################################################

print_header "4. Face Parse BiSeNet 모델 다운로드"

FACE_PARSE_DIR="${MODELS_DIR}/face-parse-bisent"

if [ -d "${FACE_PARSE_DIR}" ]; then
    print_info "Face Parse 모델이 이미 존재합니다. 건너뜁니다."
else
    download_from_hf \
        "jonathandinu/face-parsing" \
        "${FACE_PARSE_DIR}" \
        "*.pth"

    print_success "Face Parse 모델 다운로드 완료"
fi

################################################################################
# 5. MuseTalk 체크포인트 다운로드
################################################################################

print_header "5. MuseTalk 체크포인트 다운로드"

MUSETALK_CHECKPOINT_DIR="${MODELS_DIR}/musetalk"

if [ -d "${MUSETALK_CHECKPOINT_DIR}" ]; then
    print_info "MuseTalk 체크포인트가 이미 존재합니다. 건너뜁니다."
else
    download_from_hf \
        "TMElyralab/MuseTalk" \
        "${MUSETALK_CHECKPOINT_DIR}" \
        ""

    print_success "MuseTalk 체크포인트 다운로드 완료"
fi

################################################################################
# 6. GFPGAN 모델 다운로드 (선택사항)
################################################################################

print_header "6. GFPGAN 모델 다운로드 (얼굴 향상)"

GFPGAN_DIR="${MODELS_DIR}/gfpgan"

read -p "GFPGAN 모델을 다운로드하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "${GFPGAN_DIR}/GFPGANv1.4.pth" ]; then
        print_info "GFPGAN 모델이 이미 존재합니다. 건너뜁니다."
    else
        mkdir -p "${GFPGAN_DIR}"

        print_info "GFPGAN v1.4 다운로드 중..."

        GFPGAN_URL="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

        if [ "$DOWNLOAD_CMD" = "python_download" ]; then
            $PYTHON_CMD -c "
import urllib.request
import sys

try:
    urllib.request.urlretrieve(
        '${GFPGAN_URL}',
        '${GFPGAN_DIR}/GFPGANv1.4.pth'
    )
    print('✓ GFPGAN 다운로드 완료')
except Exception as e:
    print(f'✗ GFPGAN 다운로드 실패: {e}', file=sys.stderr)
"
        else
            cd "${GFPGAN_DIR}"
            $DOWNLOAD_CMD "${GFPGAN_URL}"
            cd "${PROJECT_ROOT}"
        fi

        print_success "GFPGAN 모델 다운로드 완료"
    fi
else
    print_info "GFPGAN 다운로드 건너뜀"
fi

################################################################################
# 모델 경로 설정 파일 생성
################################################################################

print_header "모델 경로 설정 파일 생성"

CONFIG_FILE="${MODELS_DIR}/model_paths.yaml"

cat > "${CONFIG_FILE}" << EOF
# MuseTalk 모델 경로 설정
# 자동 생성됨: $(date)

vae:
  path: ${VAE_DIR}
  checkpoint: diffusion_pytorch_model.safetensors

whisper:
  path: ${WHISPER_DIR}
  checkpoint: tiny.pt

dwpose:
  path: ${DWPOSE_DIR}
  det_checkpoint: yolox_l.onnx
  pose_checkpoint: dw-ll_ucoco_384.onnx

face_parse:
  path: ${FACE_PARSE_DIR}
  checkpoint: 79999_iter.pth

musetalk:
  path: ${MUSETALK_CHECKPOINT_DIR}
  unet_checkpoint: musetalk_unet.pth
  audio_processor: audio_processor.pth

gfpgan:
  path: ${GFPGAN_DIR}
  checkpoint: GFPGANv1.4.pth
  enabled: $([ -f "${GFPGAN_DIR}/GFPGANv1.4.pth" ] && echo "true" || echo "false")
EOF

print_success "설정 파일 생성: ${CONFIG_FILE}"

################################################################################
# 모델 검증
################################################################################

print_header "모델 파일 검증"

VALIDATION_PASSED=true

# VAE 검증
if [ -f "${VAE_DIR}/diffusion_pytorch_model.safetensors" ]; then
    FILE_SIZE=$(du -h "${VAE_DIR}/diffusion_pytorch_model.safetensors" | cut -f1)
    print_success "VAE: ${FILE_SIZE}"
else
    print_error "VAE 모델 파일이 없습니다"
    VALIDATION_PASSED=false
fi

# Whisper 검증
if [ -f "${WHISPER_DIR}/tiny.pt" ]; then
    FILE_SIZE=$(du -h "${WHISPER_DIR}/tiny.pt" | cut -f1)
    print_success "Whisper: ${FILE_SIZE}"
else
    print_warning "Whisper 모델 파일이 없습니다"
fi

# DWPose 검증
if [ -d "${DWPOSE_DIR}" ]; then
    print_success "DWPose: 설치됨"
else
    print_error "DWPose 모델이 없습니다"
    VALIDATION_PASSED=false
fi

# Face Parse 검증
if [ -d "${FACE_PARSE_DIR}" ]; then
    print_success "Face Parse: 설치됨"
else
    print_error "Face Parse 모델이 없습니다"
    VALIDATION_PASSED=false
fi

# MuseTalk 검증
if [ -d "${MUSETALK_CHECKPOINT_DIR}" ]; then
    print_success "MuseTalk: 설치됨"
else
    print_error "MuseTalk 체크포인트가 없습니다"
    VALIDATION_PASSED=false
fi

################################################################################
# 테스트 실행
################################################################################

print_header "테스트 실행"

read -p "설치 테스트를 실행하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "테스트 스크립트 실행 중..."

    $PYTHON_CMD << EOF
import sys
import os

print("\n=== Python 환경 테스트 ===")

# 필수 패키지 확인
required_packages = [
    'torch',
    'torchvision',
    'diffusers',
    'transformers',
    'opencv-python',
    'mediapipe',
    'whisper'
]

missing_packages = []

for package in required_packages:
    package_import = package.replace('-', '_')
    try:
        __import__(package_import)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} (설치 필요)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠ 다음 패키지를 설치해주세요:")
    print(f"  pip install {' '.join(missing_packages)}")
    sys.exit(1)
else:
    print("\n✓ 모든 필수 패키지가 설치되어 있습니다")

# PyTorch CUDA 확인
import torch
print(f"\n=== PyTorch 정보 ===")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# MPS (Apple Silicon) 확인
if hasattr(torch.backends, 'mps'):
    print(f"MPS 사용 가능: {torch.backends.mps.is_available()}")

print("\n✓ 테스트 완료!")
EOF

    if [ $? -eq 0 ]; then
        print_success "테스트 통과"
    else
        print_warning "일부 테스트 실패 (requirements.txt 확인)"
    fi
else
    print_info "테스트 건너뜀"
fi

################################################################################
# 정리
################################################################################

print_header "정리"

# 임시 디렉토리 삭제
if [ -d "${TEMP_DIR}" ]; then
    print_info "임시 파일 삭제 중..."
    rm -rf "${TEMP_DIR}"
    print_success "임시 파일 삭제 완료"
fi

################################################################################
# 완료
################################################################################

print_header "설치 완료!"

echo -e "${GREEN}"
echo "MuseTalk 모델 설치가 완료되었습니다."
echo ""
echo "모델 위치: ${MODELS_DIR}"
echo "설정 파일: ${CONFIG_FILE}"
echo ""
echo "다음 명령어로 아바타를 사용할 수 있습니다:"
echo "  python -m src.avatar.example_usage"
echo -e "${NC}"

if [ "$VALIDATION_PASSED" = false ]; then
    print_warning "일부 모델이 누락되었습니다. 위의 에러를 확인해주세요."
    exit 1
fi

exit 0
