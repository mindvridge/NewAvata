# ============================================================================
# Multi-stage Dockerfile for Realtime Interview Avatar System
# NVIDIA GPU 지원 (CUDA 11.8 + cuDNN 8)
# ============================================================================

# ----------------------------------------------------------------------------
# Stage 1: Builder - 의존성 설치 및 모델 다운로드
# ----------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 시스템 패키지 업데이트 및 기본 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.10
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # 빌드 도구
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # 미디어 처리
    ffmpeg \
    # OpenCV 의존성
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 오디오 처리
    libsndfile1 \
    libportaudio2 \
    # 기타 유틸리티
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

# 작업 디렉토리 설정
WORKDIR /build

# PyTorch 및 CUDA 관련 패키지 먼저 설치 (캐싱 최적화)
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# requirements.txt 복사 및 Python 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 애플리케이션 코드 복사
COPY . /build/app

# MuseTalk 모델 다운로드 스크립트 실행
WORKDIR /build/app
RUN python scripts/setup_musetalk.py --model-dir /build/models

# ----------------------------------------------------------------------------
# Stage 2: Runtime - 최소 런타임 이미지
# ----------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # NVIDIA/CUDA 설정
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    CUDA_VISIBLE_DEVICES=0 \
    # 애플리케이션 설정
    APP_HOME=/app \
    MODELS_DIR=/app/models \
    LOG_LEVEL=INFO \
    # Python 경로
    PYTHONPATH=/app/src

# 런타임 패키지만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.10
    python3.10 \
    python3.10-venv \
    python3-pip \
    # 미디어 처리
    ffmpeg \
    # OpenCV 의존성
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # 오디오 처리
    libsndfile1 \
    libportaudio2 \
    # 기타
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 비-root 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

# 작업 디렉토리 생성
WORKDIR ${APP_HOME}

# Builder 스테이지에서 Python 패키지 복사
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Builder 스테이지에서 애플리케이션 코드 복사
COPY --from=builder /build/app ${APP_HOME}

# Builder 스테이지에서 모델 파일 복사
COPY --from=builder /build/models ${MODELS_DIR}

# 디렉토리 생성 및 권한 설정
RUN mkdir -p \
    ${APP_HOME}/data \
    ${APP_HOME}/logs \
    ${APP_HOME}/uploads \
    ${APP_HOME}/temp \
    && chown -R appuser:appuser ${APP_HOME}

# 헬스체크 스크립트 생성
RUN echo '#!/bin/bash\ncurl -f http://localhost:8000/api/health || exit 1' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/healthcheck.sh"]

# 포트 노출
EXPOSE 8000

# 볼륨 설정
VOLUME ["${APP_HOME}/data", "${APP_HOME}/logs", "${APP_HOME}/uploads"]

# 사용자 전환
USER appuser

# 엔트리포인트 및 기본 명령어
ENTRYPOINT ["python"]
CMD ["-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

# ============================================================================
# 빌드 및 실행 명령어
# ============================================================================
#
# 빌드:
#   docker build -t interview-avatar:latest -f docker/Dockerfile .
#
# 실행 (GPU 사용):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/logs:/app/logs \
#     --env-file .env \
#     interview-avatar:latest
#
# 실행 (개발 모드):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/src:/app/src \
#     -v $(pwd)/data:/app/data \
#     --env-file .env \
#     interview-avatar:latest \
#     -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
#
# ============================================================================
