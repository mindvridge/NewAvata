#!/bin/bash
# ==============================================
# A100 GPU 립싱크 서버 초기 셋업
# ==============================================
# 사용법: bash setup_a100.sh
# 환경: Ubuntu 22.04+ / CUDA 12.4 / A100 80GB
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo " A100 립싱크 서버 셋업"
echo "=============================================="
echo ""

# ------------------------------------------
# 1. 시스템 확인
# ------------------------------------------
echo "[1/7] 시스템 확인..."

# Python 확인
if ! command -v python3.10 &> /dev/null; then
    echo "[ERROR] Python 3.10이 필요합니다."
    echo "  sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi
echo "  Python: $(python3.10 --version)"

# CUDA 확인
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    echo "[WARNING] nvidia-smi를 찾을 수 없습니다. GPU 드라이버를 확인하세요."
fi

# ffmpeg 확인
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARNING] ffmpeg가 설치되지 않았습니다."
    echo "  sudo apt install ffmpeg"
fi

echo ""

# ------------------------------------------
# 2. 가상환경 생성
# ------------------------------------------
echo "[2/7] Python 가상환경 생성..."

if [ -d "venv" ]; then
    echo "  기존 venv 발견 - 건너뜁니다."
    echo "  (새로 생성하려면: rm -rf venv 후 재실행)"
else
    python3.10 -m venv venv
    echo "  venv 생성 완료"
fi

source venv/bin/activate
echo "  venv 활성화: $(python --version)"
echo ""

# ------------------------------------------
# 3. PyTorch 설치 (CUDA 12.4)
# ------------------------------------------
echo "[3/7] PyTorch 설치 (CUDA 12.4)..."

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
python -c "import torch; print(f'  PyTorch {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# ------------------------------------------
# 4. 의존성 설치
# ------------------------------------------
echo "[4/7] 의존성 설치..."

pip install -r requirements_a100.txt
echo ""

# ------------------------------------------
# 5. MuseTalk 소스 설정
# ------------------------------------------
echo "[5/7] MuseTalk 소스 설정..."

MUSETALK_DIR="$(dirname "$SCRIPT_DIR")/MuseTalk"

if [ -d "$MUSETALK_DIR" ]; then
    echo "  MuseTalk 경로: $MUSETALK_DIR"
else
    echo "  MuseTalk 소스가 없습니다. 클론 중..."
    git clone https://github.com/TMElyralab/MuseTalk.git "$MUSETALK_DIR"
    echo "  클론 완료: $MUSETALK_DIR"
fi

echo ""

# ------------------------------------------
# 6. sm_120 TensorRT 엔진 파일 삭제
# ------------------------------------------
echo "[6/7] 비호환 TensorRT 엔진 파일 정리..."

ENGINE_COUNT=0
for engine_file in models/musetalkV15/*sm120*.engine; do
    if [ -f "$engine_file" ]; then
        echo "  삭제: $(basename "$engine_file")"
        rm -f "$engine_file"
        ENGINE_COUNT=$((ENGINE_COUNT + 1))
    fi
done

# sm_120 외 다른 .engine 파일도 확인
for engine_file in models/musetalkV15/*.engine; do
    if [ -f "$engine_file" ]; then
        echo "  삭제 (기존 엔진): $(basename "$engine_file")"
        rm -f "$engine_file"
        ENGINE_COUNT=$((ENGINE_COUNT + 1))
    fi
done

if [ $ENGINE_COUNT -eq 0 ]; then
    echo "  삭제할 엔진 파일 없음"
else
    echo "  $ENGINE_COUNT개 엔진 파일 삭제 완료"
fi

echo ""
echo "  [INFO] 첫 실행 시 ONNX Runtime이 A100(sm_80)용 TensorRT 엔진을"
echo "         자동으로 재생성합니다. (약 5~10분 소요)"
echo ""

# ------------------------------------------
# 7. .env 파일 확인
# ------------------------------------------
echo "[7/7] 환경 변수 설정..."

if [ ! -f ".env" ]; then
    if [ -f ".env.a100" ]; then
        cp .env.a100 .env
        echo "  .env.a100 → .env 복사 완료"
        echo "  [중요] .env 파일의 API 키를 설정하세요!"
    else
        echo "  [WARNING] .env 파일이 없습니다."
        echo "  .env.a100을 참고하여 .env 파일을 생성하세요."
    fi
else
    echo "  .env 파일 확인됨"
fi

echo ""
echo "=============================================="
echo " 셋업 완료!"
echo "=============================================="
echo ""
echo " 서버 실행:"
echo "   bash run_server.sh"
echo ""
echo " 주의사항:"
echo "   - 첫 실행 시 TensorRT 엔진 빌드로 시작이 느릴 수 있습니다"
echo "   - .env 파일의 API 키와 URL을 확인하세요"
echo "   - 모델 파일이 없으면: python scripts/download_musetalk_models.py"
echo ""
