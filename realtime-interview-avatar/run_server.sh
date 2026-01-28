#!/bin/bash
# ==============================================
# 립싱크 서버 실행 (A100 / Linux)
# ==============================================
# 사용법: bash run_server.sh
# ==============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "[INFO] venv 활성화됨"
else
    echo "[ERROR] venv가 없습니다. 먼저 setup_a100.sh를 실행하세요."
    exit 1
fi

# PYTHONPATH 설정 (MuseTalk)
export PYTHONPATH="${PYTHONPATH}:$(dirname "$SCRIPT_DIR")/MuseTalk"
echo "[INFO] PYTHONPATH: $PYTHONPATH"

# GPU 설정
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "=============================================="
echo " Lipsync Server 시작"
echo "=============================================="
echo ""

python app.py
