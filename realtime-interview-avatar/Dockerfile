# =============================================================================
# 실시간 립싱크 아바타 서버 Docker 이미지
# MuseTalk V1.5 + ElevenLabs TTS + TensorRT 최적화
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
RUN pip install --no-cache-dir --pre \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu129

# =============================================================================
# MuseTalk 의존성
# =============================================================================
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    scipy \
    scikit-learn \
    scikit-image \
    matplotlib \
    opencv-python==4.9.0.80 \
    einops==0.8.1 \
    omegaconf \
    hydra-core \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    transformers==4.51.3 \
    huggingface_hub==0.30.2

# torch 의존성이 있는 패키지만 --no-deps로 설치
RUN pip install --no-cache-dir --no-deps \
    diffusers==0.30.2 \
    accelerate==0.28.0 \
    openai-whisper==20231117

# face_alignment, facexlib, mediapipe
RUN pip install --no-cache-dir --no-deps \
    face_alignment \
    facexlib \
    mediapipe

RUN pip install --no-cache-dir \
    absl-py \
    flatbuffers \
    ffmpeg-python \
    moviepy \
    imageio[ffmpeg] \
    pydub

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
# TensorRT 10.x 설치 (NVIDIA 공식 저장소)
# =============================================================================
# NVIDIA 키링 설치
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update

# TensorRT 10.x 라이브러리 설치 (libnvinfer.so.10 포함)
RUN apt-get install -y --no-install-recommends \
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvinfer-vc-plugin10 \
    libnvinfer-headers-dev \
    libnvinfer-dev \
    libnvonnxparsers10 \
    libnvonnxparsers-dev \
    && rm -rf /var/lib/apt/lists/*

# TensorRT 라이브러리 경로 설정
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# TensorRT Python 의존성 (UNet 최적화용)
RUN pip install --no-cache-dir \
    onnx==1.16.0 \
    pycuda

# =============================================================================
# 프로젝트 파일 복사
# =============================================================================

# MuseTalk 소스 복사 (models 제외)
COPY MuseTalk/musetalk /app/MuseTalk/musetalk
COPY MuseTalk/configs /app/MuseTalk/configs

# 메인 애플리케이션 복사
COPY realtime-interview-avatar/app.py /app/realtime-interview-avatar/
COPY realtime-interview-avatar/convert_unet_onnx.py /app/realtime-interview-avatar/
COPY realtime-interview-avatar/templates /app/realtime-interview-avatar/templates
COPY realtime-interview-avatar/static /app/realtime-interview-avatar/static
COPY realtime-interview-avatar/assets /app/realtime-interview-avatar/assets

# =============================================================================
# 디렉토리 생성
# =============================================================================
RUN mkdir -p /app/realtime-interview-avatar/results/realtime \
    && mkdir -p /app/realtime-interview-avatar/models \
    && mkdir -p /app/realtime-interview-avatar/precomputed \
    && mkdir -p /app/MuseTalk/models

# 작업 디렉토리 설정
WORKDIR /app/realtime-interview-avatar

# 포트 노출
EXPOSE 5000

# 환경 변수 (런타임에 오버라이드 가능)
ENV OPENAI_API_KEY=""
ENV MUSETALK_PATH="/app/MuseTalk"

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5000/api/v2/status || exit 1

# 시작 명령
CMD ["python", "app.py"]
