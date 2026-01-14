# =============================================================================
# 실시간 립싱크 아바타 서버 Docker 이미지
# MuseTalk V1.5 + CosyVoice 2.0 TTS
# =============================================================================

# CUDA 12.4.1 devel 이미지 (PyTorch 2.11 CUDA 12.9와 호환)
# 호스트 CUDA 13.0 드라이버 사용
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
# 호스트 CUDA 라이브러리 우선 사용하도록 LD_LIBRARY_PATH 설정
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# PyTorch의 CUDA 라이브러리 경로도 추가
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib"

# 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    sox \
    libsox-dev \
    libsox-fmt-all \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Python 기본 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 작업 디렉토리 생성
WORKDIR /app

# =============================================================================
# PyTorch 및 기본 의존성 설치 (CUDA 12.9 - RTX 50 시리즈 sm_120 완전 지원)
# PyTorch nightly를 사용하여 RTX 5060 Ti (sm_120 아키텍처) 지원
# =============================================================================
# torch 최신 nightly 설치 (CUDA 12.9 필요 for sm_120)
# torchaudio도 함께 설치하여 ABI 호환성 보장
RUN pip install --no-cache-dir --pre \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu129

# =============================================================================
# CosyVoice 의존성 (torch 의존성이 있는 패키지만 --no-deps 사용)
# =============================================================================
# PyYAML 먼저 설치
RUN pip install --no-cache-dir PyYAML==6.0.1

# ruamel.yaml은 PyYAML과 호환되는 구버전 사용 (0.17.x는 별도 namespace 사용)
RUN pip install --no-cache-dir "ruamel.yaml<0.18"

# torch를 재설치하지 않는 패키지들은 정상 설치
# HyperPyYAML은 --no-deps로 설치 (이미 PyYAML과 ruamel.yaml 설치 완료)
RUN pip install --no-cache-dir --no-deps HyperPyYAML==1.2.2

RUN pip install --no-cache-dir \
    inflect==7.3.1 \
    librosa==0.10.2 \
    modelscope==1.20.0 \
    onnx==1.16.0 \
    pydantic==2.7.0 \
    soundfile==0.12.1 \
    transformers==4.51.3 \
    wetext==0.0.4 \
    scipy \
    scikit-learn \
    tiktoken \
    omegaconf \
    hydra-core \
    einops \
    pyarrow==18.1.0 \
    pyworld==0.3.4 \
    x-transformers==2.11.24 \
    rich==13.7.1

# torch 의존성이 있는 패키지만 --no-deps로 설치
RUN pip install --no-cache-dir --no-deps \
    conformer==0.3.2 \
    diffusers==0.29.0 \
    openai-whisper==20231117

# matcha-tts 정상 설치 (torch는 이미 설치되어 있으므로 재설치되지 않음)
# 주의: torch 버전 충돌 발생 시 --no-deps로 전환 필요
RUN pip install --no-cache-dir matcha-tts

# torchcodec는 FFmpeg 의존성 문제로 제거
# torchaudio는 soundfile backend를 사용 (이미 설치됨)

# =============================================================================
# MuseTalk 의존성 (PyTorch nightly 버전 유지하도록 --no-deps 사용)
# =============================================================================
RUN pip install --no-cache-dir --no-deps \
    "numpy<2.0.0" \
    diffusers==0.30.2 \
    accelerate==0.28.0 \
    opencv-python==4.9.0.80 \
    einops==0.8.1 \
    ffmpeg-python \
    moviepy \
    imageio[ffmpeg] \
    huggingface_hub==0.30.2

# omegaconf는 CosyVoice에서 이미 정상 설치되므로 여기서는 제외

# face_alignment, facexlib, mediapipe도 PyTorch 다운그레이드 방지를 위해 --no-deps 사용
RUN pip install --no-cache-dir --no-deps \
    face_alignment \
    facexlib \
    mediapipe

# 위 패키지들의 실제 필요한 의존성만 설치
RUN pip install --no-cache-dir \
    scikit-image \
    scipy \
    matplotlib \
    opencv-python \
    absl-py \
    flatbuffers \
    sounddevice

# =============================================================================
# 웹 서버 의존성
# =============================================================================
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    flask-socketio==5.3.6 \
    flask-cors==4.0.0 \
    python-dotenv==1.0.0 \
    requests>=2.31.0 \
    gevent==23.9.1 \
    gevent-websocket==0.10.1 \
    openai>=1.0.0

# =============================================================================
# ONNX Runtime GPU (CUDA 12.4 호환)
# =============================================================================
RUN pip install --no-cache-dir \
    onnxruntime-gpu==1.20.0

# =============================================================================
# 프로젝트 파일 복사
# =============================================================================

# CosyVoice 소스 복사 (pretrained_models 제외)
COPY CosyVoice/cosyvoice /app/CosyVoice/cosyvoice
COPY CosyVoice/third_party /app/CosyVoice/third_party

# MuseTalk 소스 복사 (models 제외)
COPY MuseTalk/musetalk /app/MuseTalk/musetalk
COPY MuseTalk/configs /app/MuseTalk/configs

# 메인 애플리케이션 복사
COPY realtime-interview-avatar/app.py /app/realtime-interview-avatar/
COPY realtime-interview-avatar/templates /app/realtime-interview-avatar/templates
COPY realtime-interview-avatar/static /app/realtime-interview-avatar/static
COPY realtime-interview-avatar/assets /app/realtime-interview-avatar/assets

# =============================================================================
# 디렉토리 생성
# =============================================================================
RUN mkdir -p /app/realtime-interview-avatar/results/realtime \
    && mkdir -p /app/realtime-interview-avatar/models \
    && mkdir -p /app/realtime-interview-avatar/precomputed \
    && mkdir -p /app/CosyVoice/pretrained_models \
    && mkdir -p /app/MuseTalk/models

# 작업 디렉토리 설정
WORKDIR /app/realtime-interview-avatar

# 포트 노출
EXPOSE 5000

# 환경 변수 (런타임에 오버라이드 가능)
ENV OPENAI_API_KEY=""
ENV COSYVOICE_MODEL_PATH="/app/CosyVoice/pretrained_models/CosyVoice2-0.5B"
ENV MUSETALK_PATH="/app/MuseTalk"
ENV COSYVOICE_PATH="/app/CosyVoice"

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5000/api/v2/status || exit 1

# 시작 명령
CMD ["python", "app.py"]
