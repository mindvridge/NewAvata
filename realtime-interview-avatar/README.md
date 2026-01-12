# 🎭 Realtime Interview Avatar

실시간 AI 면접관 아바타 시스템 - 음성 인식, LLM 기반 대화, 음성 합성, 립싱크 아바타가 통합된 실시간 면접 시뮬레이션 플랫폼

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [기술 스택](#-기술-스택)
- [빠른 시작](#-빠른-시작)
- [상세 설정](#-상세-설정)
- [API 문서](#-api-문서)
- [배포 가이드](#-배포-가이드)
- [트러블슈팅](#-트러블슈팅)
- [비용 추정](#-비용-추정)
- [로드맵](#-로드맵)
- [기여 가이드](#-기여-가이드)
- [라이선스](#-라이선스)

---

## 🎯 프로젝트 소개

### 주요 기능

이 프로젝트는 실시간으로 작동하는 AI 면접관 아바타 시스템입니다. 사용자가 마이크로 질문하면 AI가 즉각 응답하고, 립싱크가 적용된 아바타가 자연스럽게 말합니다.

**핵심 기능:**

- 🎤 **실시간 음성 인식**: Deepgram Nova-3를 사용한 한국어 고정밀 STT (레이턴시 < 300ms)
- 🧠 **AI 면접관**: GPT-4o 기반 맥락 인식 대화 (면접 질문, 피드백, 후속 질문)
- 🔊 **고품질 음성 합성**: ElevenLabs/EdgeTTS/Naver 지원 (다국어)
- 👤 **립싱크 아바타**: MuseTalk 기반 실시간 얼굴 애니메이션
- ⚡ **낮은 레이턴시**: End-to-end 레이턴시 < 2초 (최적화 시 < 1초)
- 🌐 **웹 기반 UI**: WebRTC를 통한 브라우저 직접 접속
- 🚀 **GPU 최적화**: TensorRT/ONNX, 배치 처리, 캐싱으로 4배 성능 향상
- 🐳 **간편한 배포**: Docker/Docker Compose, 클라우드 스크립트 제공

### 데모 스크린샷

```
┌─────────────────────────────────────────────┐
│  🎭 AI 면접관                        [●] REC │
│─────────────────────────────────────────────│
│                                             │
│        ┌─────────────────────┐             │
│        │                     │             │
│        │    👤 아바타 화면    │             │
│        │   (립싱크 동작)      │             │
│        │                     │             │
│        └─────────────────────┘             │
│                                             │
│  💬 AI: "자기소개를 해주시겠어요?"          │
│  🎤 You: "안녕하세요, 저는..."            │
│                                             │
│  [🎤 말하기 시작]  [⏸️ 일시정지]  [🔄 재시작]│
└─────────────────────────────────────────────┘
```

### 아키텍처 개요

```
┌──────────────┐    WebRTC     ┌──────────────────────────────────┐
│   Browser    │◄─────────────►│       FastAPI Server             │
│  (WebSocket) │   Audio/Video  │  ┌────────────────────────────┐  │
└──────────────┘                │  │   Pipecat Pipeline        │  │
                                │  │                            │  │
                                │  │  ┌──────┐  ┌──────┐       │  │
                                │  │  │ STT  │→ │ LLM  │       │  │
                                │  │  └──────┘  └──────┘       │  │
                                │  │      ↓         ↓          │  │
                                │  │  ┌──────┐  ┌───────┐     │  │
                                │  │  │ TTS  │→ │Avatar │     │  │
                                │  │  └──────┘  └───────┘     │  │
                                │  └────────────────────────────┘  │
                                └──────────────────────────────────┘
                                         ↓           ↓
                                ┌─────────────┐ ┌──────────────┐
                                │  Deepgram   │ │  ElevenLabs  │
                                │  (STT API)  │ │  (TTS API)   │
                                └─────────────┘ └──────────────┘
                                         ↓
                                ┌─────────────────┐
                                │   OpenAI GPT-4  │
                                │   (LLM API)     │
                                └─────────────────┘
                                         ↓
                                ┌─────────────────┐
                                │  GPU (CUDA)     │
                                │  MuseTalk Model │
                                └─────────────────┘
```

---

## 🛠 기술 스택

### 프레임워크 & 라이브러리

| 카테고리 | 기술 | 버전 | 용도 |
|---------|------|------|------|
| **음성 인식** | Deepgram Nova-3 | Latest | 실시간 STT (한국어) |
| **언어 모델** | OpenAI GPT-4o | Latest | AI 면접관 로직 |
| **음성 합성** | ElevenLabs | v1 | 고품질 TTS |
| | EdgeTTS | Latest | 무료 대안 (MS) |
| | Naver Clova | Latest | 한국어 전용 |
| **아바타** | MuseTalk | Latest | 립싱크 생성 |
| **파이프라인** | Pipecat | 0.0.43 | 실시간 미디어 처리 |
| **WebRTC** | Daily.co | Latest | 웹 기반 통신 |
| | aiortc | Latest | Python WebRTC |
| **웹 서버** | FastAPI | 0.115+ | REST + WebSocket API |
| **프론트엔드** | Vanilla JS | ES6 | 경량 웹 UI |

### GPU 최적화

| 기술 | 용도 | 성능 향상 |
|------|------|----------|
| **TensorRT** | 모델 추론 가속 | 2-4배 ↑ |
| **ONNX Runtime** | 크로스 플랫폼 추론 | 1.5-2배 ↑ |
| **FP16 양자화** | 메모리 절감 | 메모리 50% ↓ |
| **INT8 양자화** | 극한 최적화 | 메모리 75% ↓ |
| **배치 처리** | GPU 활용률 증가 | 4배 ↑ |
| **LRU/TTL 캐싱** | TTS 중복 제거 | 500ms → 1ms |
| **비동기 파이프라인** | 병렬 처리 | 레이턴시 30% ↓ |

### 인프라

- **컨테이너**: Docker, Docker Compose
- **GPU**: NVIDIA CUDA 11.8, cuDNN 8
- **캐시**: Redis 7.0
- **프록시**: Nginx (웹소켓 프록시)
- **모니터링**: Prometheus, Grafana (선택사항)

### 시스템 아키텍처 다이어그램

```
┌───────────────────────────────────────────────────────────────────────┐
│                          Client Layer (브라우저)                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │  Microphone  │   │   Speaker    │   │   Display    │              │
│  └──────┬───────┘   └──────▲───────┘   └──────▲───────┘              │
│         │                  │                   │                       │
└─────────┼──────────────────┼───────────────────┼───────────────────────┘
          │ WebRTC           │ WebRTC            │ WebSocket
          ▼                  │                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Application Layer (FastAPI)                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                      Pipecat Pipeline                          │    │
│  │                                                                 │    │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐  │    │
│  │  │   STT    │──►│   LLM    │──►│   TTS    │──►│  Avatar   │  │    │
│  │  │ Service  │   │ Service  │   │ Service  │   │  Service  │  │    │
│  │  └────┬─────┘   └────┬─────┘   └────┬─────┘   └─────┬─────┘  │    │
│  │       │              │              │              │           │    │
│  └───────┼──────────────┼──────────────┼──────────────┼───────────┘    │
│          │              │              │              │                │
│  ┌───────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐ ┌─────▼──────┐        │
│  │ Optimization │ │   Cache    │ │  Batching │ │   Async    │        │
│  │   Module     │ │  Manager   │ │ Processor │ │  Pipeline  │        │
│  └──────────────┘ └────┬───────┘ └───────────┘ └────────────┘        │
│                        │                                               │
│                   ┌────▼────┐                                          │
│                   │  Redis  │                                          │
│                   └─────────┘                                          │
└─────────────────────────────────────────────────────────────────────────┘
                           │                  │
                    ┌──────▼──────┐    ┌──────▼──────┐
                    │  External   │    │    GPU      │
                    │  API Layer  │    │   Layer     │
                    │             │    │             │
                    │ ┌─────────┐ │    │ ┌─────────┐ │
                    │ │Deepgram │ │    │ │MuseTalk │ │
                    │ ├─────────┤ │    │ │ Model   │ │
                    │ │OpenAI   │ │    │ ├─────────┤ │
                    │ ├─────────┤ │    │ │TensorRT │ │
                    │ │ElevenLab│ │    │ │Engine   │ │
                    │ └─────────┘ │    │ └─────────┘ │
                    └─────────────┘    └─────────────┘
```

---

## 🚀 빠른 시작

### 필수 요구사항

#### 하드웨어

- **GPU**: NVIDIA GPU (최소 8GB VRAM, 권장 16GB+)
  - 지원 GPU: RTX 3060 이상, A4000, A5000, L4, A10, A100
- **CPU**: 4코어 이상
- **RAM**: 16GB 이상 (권장 32GB)
- **디스크**: 20GB 이상 여유 공간

#### 소프트웨어

- **OS**: Ubuntu 20.04/22.04 또는 Windows 10/11 (WSL2)
- **NVIDIA Driver**: 525.x 이상
- **CUDA**: 11.8 (Docker 사용 시 자동 설치)
- **Docker**: 20.10+ 및 Docker Compose v2
- **Python**: 3.10+ (로컬 개발 시)

### 설치

#### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/realtime-interview-avatar.git
cd realtime-interview-avatar
```

#### 2. 환경 변수 설정

`.env.example`을 복사하여 `.env` 파일 생성:

```bash
cp .env.example .env
```

`.env` 파일 편집:

```bash
# API Keys (필수)
OPENAI_API_KEY=sk-...                  # OpenAI API 키
DEEPGRAM_API_KEY=...                   # Deepgram API 키
ELEVENLABS_API_KEY=...                 # ElevenLabs API 키 (선택)

# Daily.co (WebRTC)
DAILY_API_KEY=...                      # Daily.co API 키
DAILY_ROOM_URL=https://your-domain.daily.co/room-name

# TTS Provider 선택
TTS_PROVIDER=elevenlabs                # elevenlabs | edge | naver

# 서버 설정
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=false

# GPU 설정
CUDA_VISIBLE_DEVICES=0                 # 사용할 GPU ID

# 캐시 설정
REDIS_URL=redis://localhost:6379
ENABLE_CACHE=true

# 최적화 설정
ENABLE_TENSORRT=true
ENABLE_BATCHING=true
BATCH_SIZE=4
```

#### 3. Docker로 실행 (권장)

```bash
# GPU 지원 Docker Compose로 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f app

# 서비스 상태 확인
docker-compose ps
```

서비스가 시작되면 브라우저에서 접속:
- **웹 UI**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

#### 4. 로컬 개발 환경 (선택)

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# MuseTalk 모델 다운로드
python scripts/download_models.py

# 개발 서버 실행
python -m src.server.main
```

### 첫 실행 테스트

1. 브라우저에서 http://localhost:8000 접속
2. "마이크 권한 허용" 클릭
3. "말하기 시작" 버튼 클릭
4. "안녕하세요"라고 말하기
5. AI 면접관의 응답과 아바타 립싱크 확인

---

## ⚙️ 상세 설정

### 컴포넌트별 설정 옵션

#### STT (음성 인식) 설정

[config/settings.py](config/settings.py):

```python
class STTConfig:
    provider: str = "deepgram"          # deepgram | whisper
    model: str = "nova-3"               # Deepgram 모델
    language: str = "ko"                # ko | en | ja
    smart_format: bool = True           # 자동 구두점, 대소문자
    vad_enabled: bool = True            # 음성 활동 감지
    interim_results: bool = True        # 중간 결과 전송
    encoding: str = "linear16"
    sample_rate: int = 16000
```

#### LLM (언어 모델) 설정

```python
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"               # gpt-4o | gpt-4-turbo
    temperature: float = 0.7            # 창의성 (0.0-1.0)
    max_tokens: int = 150               # 응답 최대 길이
    system_prompt: str = """
        당신은 전문 면접관입니다.
        지원자의 답변을 듣고 적절한 후속 질문을 하세요.
    """
    memory_turns: int = 10              # 대화 기억 턴 수
```

#### TTS (음성 합성) 설정

```python
class TTSConfig:
    provider: str = "elevenlabs"        # elevenlabs | edge | naver

    # ElevenLabs
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    model: str = "eleven_turbo_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75

    # EdgeTTS (무료 대안)
    edge_voice: str = "ko-KR-SunHiNeural"

    # Naver Clova
    naver_speaker: str = "nara"         # nara | jinho
```

#### Avatar (아바타) 설정

```python
class AvatarConfig:
    model_path: str = "models/musetalk"
    avatar_image: str = "assets/avatar.png"  # 기본 얼굴 이미지
    fps: int = 25                       # 프레임레이트
    resolution: tuple = (512, 512)      # 해상도
    use_tensorrt: bool = True           # TensorRT 가속
    batch_size: int = 8                 # 배치 크기
```

### 성능 튜닝 가이드

#### GPU 메모리 최적화

```python
# config/settings.py

# 메모리 부족 시 (8GB VRAM)
class OptimizationConfig:
    precision_mode: str = "fp16"        # fp32 | fp16 | int8
    use_tensorrt: bool = False          # TensorRT 비활성화
    batch_size: int = 1                 # 배치 크기 축소
    enable_cache: bool = True           # 캐시 활성화 (필수)

# 충분한 메모리 (16GB+ VRAM)
class OptimizationConfig:
    precision_mode: str = "fp16"
    use_tensorrt: bool = True           # TensorRT 활성화
    batch_size: int = 8                 # 배치 크기 증가
    enable_cache: bool = True
    max_cache_size_mb: int = 4096       # 캐시 크기 증가
```

#### 레이턴시 최적화

```bash
# .env 파일

# 저지연 설정 (< 1초 목표)
ENABLE_TENSORRT=true
ENABLE_BATCHING=true
BATCH_SIZE=4
MAX_WAIT_TIME_MS=30
ENABLE_CACHE=true
CACHE_PREWARM=true

# STT 최적화
STT_INTERIM_RESULTS=true
VAD_ENABLED=true

# TTS 최적화
TTS_PROVIDER=edge                      # ElevenLabs보다 빠름
TTS_STREAMING=true

# Avatar 최적화
AVATAR_FPS=25
AVATAR_RESOLUTION=512
```

#### 배치 처리 설정

```python
# src/optimization/batching.py

class BatchConfig:
    batch_size: int = 4                 # 기본 배치 크기
    max_wait_time_ms: int = 50          # 최대 대기 시간
    enable_dynamic_batching: bool = True  # 동적 배치 크기 조정
```

**동적 배치 크기 조정**:
- 레이턴시가 높으면 배치 크기 자동 감소
- 레이턴시가 낮으면 배치 크기 자동 증가
- 목표 레이턴시: 50ms

#### 캐싱 전략

```python
# src/optimization/caching.py

# TTS 캐시 (메모리 + 디스크)
class TTSAudioCache:
    max_memory_size_mb: int = 512       # 메모리 캐시 크기
    max_disk_size_mb: int = 2048        # 디스크 캐시 크기
    ttl_seconds: int = 3600             # 1시간 TTL

# 얼굴 특징 캐시
class FaceFeatureCache:
    max_size: int = 100                 # 최대 캐시 항목
    ttl_seconds: int = 1800             # 30분 TTL
```

**캐시 프리워밍**:
```python
# 자주 사용되는 질문 미리 캐싱
prewarm_questions = [
    "자기소개를 해주세요.",
    "지원 동기를 말씀해주세요.",
    "강점과 약점을 말씀해주세요.",
]

await cache_manager.prewarm(prewarm_questions)
```

---

## 📚 API 문서

### REST API 엔드포인트

#### 1. 헬스 체크

```bash
GET /health
```

**응답**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gpu_available": true,
  "gpu_memory_used_mb": 4096,
  "gpu_memory_total_mb": 16384
}
```

#### 2. 세션 생성

```bash
POST /api/sessions
Content-Type: application/json

{
  "user_id": "user123",
  "interview_type": "technical",
  "language": "ko"
}
```

**응답**:
```json
{
  "session_id": "sess_abc123",
  "daily_room_url": "https://your-domain.daily.co/sess_abc123",
  "expires_at": "2024-01-01T12:00:00Z"
}
```

**cURL 예시**:
```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "interview_type": "technical", "language": "ko"}'
```

#### 3. 세션 조회

```bash
GET /api/sessions/{session_id}
```

**응답**:
```json
{
  "session_id": "sess_abc123",
  "status": "active",
  "created_at": "2024-01-01T10:00:00Z",
  "duration_seconds": 1200,
  "message_count": 15
}
```

#### 4. 세션 종료

```bash
DELETE /api/sessions/{session_id}
```

**응답**:
```json
{
  "session_id": "sess_abc123",
  "status": "terminated",
  "summary": {
    "duration_seconds": 1800,
    "total_messages": 20,
    "ai_responses": 10
  }
}
```

### WebSocket API

#### 연결

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('WebSocket connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  handleMessage(data);
};
```

#### 메시지 타입

##### 1. STT 중간 결과

```json
{
  "type": "stt_interim",
  "transcript": "안녕하",
  "is_final": false
}
```

##### 2. STT 최종 결과

```json
{
  "type": "stt_final",
  "transcript": "안녕하세요",
  "is_final": true
}
```

##### 3. LLM 응답 (스트리밍)

```json
{
  "type": "llm_response",
  "text": "안녕하세요! 자기소개를",
  "done": false
}
```

##### 4. LLM 응답 완료

```json
{
  "type": "llm_response",
  "text": "안녕하세요! 자기소개를 해주시겠어요?",
  "done": true
}
```

##### 5. TTS 오디오

```json
{
  "type": "tts_audio",
  "audio_data": "base64_encoded_audio",
  "sample_rate": 24000,
  "format": "pcm16"
}
```

##### 6. Avatar 프레임

```json
{
  "type": "avatar_frame",
  "frame_data": "base64_encoded_image",
  "width": 512,
  "height": 512,
  "format": "jpeg"
}
```

##### 7. 에러

```json
{
  "type": "error",
  "code": "STT_ERROR",
  "message": "음성 인식 실패",
  "details": "..."
}
```

### Python SDK 예시

```python
import asyncio
from src.client import InterviewAvatarClient

async def main():
    # 클라이언트 생성
    client = InterviewAvatarClient(
        api_url="http://localhost:8000",
        api_key="your_api_key"
    )

    # 세션 시작
    session = await client.create_session(
        user_id="user123",
        interview_type="technical"
    )

    print(f"Session ID: {session.session_id}")
    print(f"Daily Room: {session.daily_room_url}")

    # WebSocket 연결
    await client.connect(session.session_id)

    # 메시지 핸들러 등록
    @client.on("stt_final")
    async def on_transcript(data):
        print(f"You: {data['transcript']}")

    @client.on("llm_response")
    async def on_response(data):
        if data['done']:
            print(f"AI: {data['text']}")

    # 대화 시작
    await client.start_conversation()

    # 대기
    await asyncio.sleep(300)  # 5분

    # 세션 종료
    await client.terminate_session(session.session_id)

asyncio.run(main())
```

---

## 🐳 배포 가이드

### Docker 배포

#### 단일 컨테이너 실행

```bash
# 이미지 빌드
docker build -t interview-avatar:latest -f docker/Dockerfile .

# 컨테이너 실행
docker run -d \
  --name interview-avatar \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  --env-file .env \
  interview-avatar:latest
```

#### Docker Compose (권장)

```bash
# 서비스 시작
docker-compose up -d

# 특정 서비스만 재시작
docker-compose restart app

# 로그 확인
docker-compose logs -f app

# 서비스 중지
docker-compose down

# 볼륨까지 삭제
docker-compose down -v
```

**docker-compose.yml** 구조:
- **app**: FastAPI 서버 (GPU 필요)
- **redis**: 캐시 서버
- **nginx**: 리버스 프록시 (선택사항)

#### 프로덕션 설정

`docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  app:
    image: interview-avatar:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: always
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

실행:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 클라우드 배포

#### RunPod 배포

**특징**: 저렴한 GPU 인스턴스, 시간당 과금, 즉시 시작

```bash
cd scripts/deploy
./deploy_runpod.sh
```

**수동 배포**:

1. RunPod 계정 생성 및 API 키 발급
2. `.env` 파일 설정:
```bash
RUNPOD_API_KEY=your_api_key
RUNPOD_GPU_TYPE=RTX_A5000
RUNPOD_REGION=US
```

3. 배포 스크립트 실행:
```bash
# 인스턴스 생성
runpod create pod \
  --name interview-avatar \
  --gpu-type "RTX A5000" \
  --image-name interview-avatar:latest \
  --ports 8000:8000 \
  --volume-mount /workspace/models:/app/models

# 인스턴스 IP 확인
runpod list pods
```

4. 접속 테스트:
```bash
curl http://<INSTANCE_IP>:8000/health
```

**비용**: ~$0.34/hour (RTX A5000 기준)

#### Vast.ai 배포 (가장 저렴)

**특징**: 개인 GPU 대여, 가장 저렴, 불안정할 수 있음

```bash
./deploy_vast.sh
```

**수동 배포**:

1. Vast.ai 계정 생성
2. CLI 설치:
```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

3. 인스턴스 검색 및 생성:
```bash
# RTX 3090 인스턴스 검색 (16GB VRAM)
vastai search offers 'gpu_ram >= 16 reliability > 0.95'

# 인스턴스 생성
vastai create instance <INSTANCE_ID> \
  --image interview-avatar:latest \
  --disk 50 \
  --env-file .env
```

**비용**: ~$0.20/hour (RTX 3090 기준)

#### Lambda Labs 배포 (가장 안정적)

**특징**: 고품질 GPU, 안정적, 월간 구독

```bash
./deploy_lambda.sh
```

**수동 배포**:

1. Lambda Labs 계정 생성
2. 인스턴스 생성:
```bash
lambda-cli instances create \
  --instance-type gpu_1x_a10 \
  --name interview-avatar \
  --ssh-key ~/.ssh/id_rsa.pub
```

3. SSH 접속 및 Docker 실행:
```bash
ssh ubuntu@<INSTANCE_IP>

# Docker 설치 확인
docker --version

# 프로젝트 배포
git clone <your-repo>
cd realtime-interview-avatar
docker-compose up -d
```

**비용**: ~$0.60/hour (A10 기준)

#### 클라우드 비교표

| 플랫폼 | GPU | 비용/시간 | 안정성 | 시작 속도 | 권장 용도 |
|--------|-----|----------|--------|----------|-----------|
| **RunPod** | RTX A5000 | $0.34 | ⭐⭐⭐⭐ | 즉시 | 프로덕션, 개발 |
| **Vast.ai** | RTX 3090 | $0.20 | ⭐⭐⭐ | 빠름 | 개발, 테스트 |
| **Lambda Labs** | A10 | $0.60 | ⭐⭐⭐⭐⭐ | 보통 | 프로덕션 |

### 로컬 배포 (스크립트)

```bash
# 전체 스택 시작
./scripts/start_local.sh

# 개발 모드 (핫 리로드)
./scripts/start_local.sh --dev

# 특정 포트 지정
./scripts/start_local.sh --port 9000

# GPU 지정
CUDA_VISIBLE_DEVICES=1 ./scripts/start_local.sh
```

---

## 🔧 트러블슈팅

### 자주 발생하는 문제

#### 1. GPU 메모리 부족 (CUDA Out of Memory)

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**해결 방법**:

```python
# config/settings.py 수정

# 배치 크기 축소
BATCH_SIZE = 1

# FP16 사용
PRECISION_MODE = "fp16"

# TensorRT 비활성화 (메모리 절약)
ENABLE_TENSORRT = False

# 해상도 축소
AVATAR_RESOLUTION = 256  # 512에서 256으로
```

또는 환경 변수로:
```bash
export BATCH_SIZE=1
export AVATAR_RESOLUTION=256
export ENABLE_TENSORRT=false
```

#### 2. WebRTC 연결 실패

**증상**:
```
WebSocket connection failed
DailyTransport: Unable to join room
```

**해결 방법**:

1. **Daily.co API 키 확인**:
```bash
# .env 파일
DAILY_API_KEY=your_valid_api_key
```

2. **방 URL 생성**:
```bash
curl -X POST https://api.daily.co/v1/rooms \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "interview-room", "privacy": "public"}'
```

3. **방화벽 확인**:
```bash
# 포트 8000 열기
sudo ufw allow 8000/tcp
sudo ufw allow 3478/udp  # STUN
sudo ufw allow 5349/tcp  # TURN
```

4. **CORS 설정** (프론트엔드가 다른 도메인일 경우):
```python
# src/server/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 3. STT API 에러 (Deepgram)

**증상**:
```
DeepgramError: Invalid API key
```

**해결 방법**:

1. API 키 확인:
```bash
curl -X GET https://api.deepgram.com/v1/projects \
  -H "Authorization: Token YOUR_API_KEY"
```

2. 잔액 확인:
- Deepgram 대시보드에서 크레딧 잔액 확인
- 무료 티어: $200 크레딧 (처음 가입 시)

3. 대체 STT 사용 (Whisper):
```python
# config/settings.py
STT_PROVIDER = "whisper"  # deepgram 대신
```

#### 4. TTS API 에러 (ElevenLabs)

**증상**:
```
ElevenLabsError: 401 Unauthorized
```

**해결 방법**:

1. 무료 대안 사용 (EdgeTTS):
```bash
# .env
TTS_PROVIDER=edge
```

2. Naver Clova 사용 (한국어 전용):
```bash
TTS_PROVIDER=naver
NAVER_CLIENT_ID=your_client_id
NAVER_CLIENT_SECRET=your_client_secret
```

#### 5. Docker 빌드 실패

**증상**:
```
ERROR: failed to solve: failed to compute cache key
```

**해결 방법**:

1. BuildKit 사용:
```bash
DOCKER_BUILDKIT=1 docker build -t interview-avatar -f docker/Dockerfile .
```

2. 캐시 없이 빌드:
```bash
docker build --no-cache -t interview-avatar -f docker/Dockerfile .
```

3. 디스크 공간 확인:
```bash
df -h
docker system prune -a  # 사용하지 않는 이미지 삭제
```

#### 6. MuseTalk 모델 로딩 실패

**증상**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/musetalk/...'
```

**해결 방법**:

1. 모델 다운로드:
```bash
python scripts/download_models.py
```

2. 수동 다운로드:
```bash
mkdir -p models/musetalk
cd models/musetalk

# Hugging Face에서 다운로드
wget https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk.pth
wget https://huggingface.co/TMElyralab/MuseTalk/resolve/main/dwpose.pth
```

3. 권한 확인:
```bash
chmod -R 755 models/
```

#### 7. 레이턴시가 너무 높음 (> 3초)

**해결 방법**:

1. **프로파일링 실행**:
```bash
python scripts/profile.py --duration 60
```

2. **병목 구간 확인**:
- STT: > 500ms → Deepgram 리전 확인
- LLM: > 1000ms → GPT-4 대신 GPT-3.5-turbo 사용
- TTS: > 800ms → EdgeTTS 사용 또는 캐싱 활성화
- Avatar: > 500ms → TensorRT 활성화, 해상도 축소

3. **최적화 활성화**:
```bash
# .env
ENABLE_TENSORRT=true
ENABLE_BATCHING=true
ENABLE_CACHE=true
CACHE_PREWARM=true
```

#### 8. Redis 연결 실패

**증상**:
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**해결 방법**:

1. Redis 서비스 확인:
```bash
docker-compose ps redis
docker-compose logs redis
```

2. Redis 재시작:
```bash
docker-compose restart redis
```

3. Redis 없이 실행 (캐시 비활성화):
```bash
# .env
ENABLE_CACHE=false
```

### 로그 확인

```bash
# Docker 로그
docker-compose logs -f app

# 특정 레벨만 필터링
docker-compose logs -f app | grep ERROR

# 파일로 저장
docker-compose logs app > logs.txt

# 파이썬 로그 (로컬 실행 시)
tail -f logs/app.log
```

### 디버그 모드

```bash
# .env
DEBUG=true
LOG_LEVEL=DEBUG

# 재시작
docker-compose restart app
```

---

## 💰 비용 추정

### API 사용 비용

#### Deepgram (STT)

| 플랜 | 비용 | 무료 크레딧 |
|------|------|-------------|
| Nova-3 | $0.0043/분 | $200 |
| Base | $0.0125/분 | - |

**월간 예상 (1시간/일 사용)**:
- 30시간 × 60분 × $0.0043 = **$7.74/월**

#### OpenAI (LLM)

| 모델 | 입력 (1M 토큰) | 출력 (1M 토큰) |
|------|---------------|---------------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4-turbo | $10.00 | $30.00 |
| GPT-3.5-turbo | $0.50 | $1.50 |

**월간 예상 (평균 100토큰 입력, 150토큰 출력, 1000회 대화)**:
- 입력: 100k 토큰 × $2.50/1M = $0.25
- 출력: 150k 토큰 × $10.00/1M = $1.50
- **합계: $1.75/월**

#### ElevenLabs (TTS)

| 플랜 | 비용 | 문자 수/월 |
|------|------|----------|
| Free | $0 | 10,000 |
| Starter | $5 | 30,000 |
| Creator | $22 | 100,000 |

**월간 예상 (평균 50자 응답, 1000회 대화)**:
- 50,000 문자 → **Creator 플랜: $22/월**

#### EdgeTTS (무료 대안)

- **비용: $0** (Microsoft 제공 무료 TTS)
- 제한: 없음
- 품질: ElevenLabs보다 약간 낮음

#### Daily.co (WebRTC)

| 플랜 | 비용 | 분/월 |
|------|------|-------|
| Free | $0 | 1,000 |
| Developer | $29 | 10,000 |
| Business | $99 | 50,000 |

**월간 예상 (30시간 사용)**:
- 1,800분 → **Developer 플랜: $29/월**

### 클라우드 인프라 비용

#### GPU 인스턴스 (24/7 운영)

| 플랫폼 | GPU | 시간당 | 월간 (730시간) |
|--------|-----|--------|---------------|
| RunPod | RTX A5000 | $0.34 | $248 |
| Vast.ai | RTX 3090 | $0.20 | $146 |
| Lambda Labs | A10 | $0.60 | $438 |
| AWS EC2 | g4dn.xlarge | $0.526 | $384 |
| GCP | T4 | $0.35 | $255 |

**권장 옵션**:
- **개발/테스트**: Vast.ai RTX 3090 ($146/월)
- **프로덕션**: RunPod RTX A5000 ($248/월)
- **엔터프라이즈**: Lambda Labs A10 ($438/월)

#### 주문형 사용 (On-Demand)

시간당만 과금 (인스턴스 중지 시 비용 없음):

| 사용 패턴 | 시간/일 | 일/월 | 월간 비용 (Vast.ai) |
|----------|--------|-------|-------------------|
| 가벼운 테스트 | 2 | 20 | $8 |
| 정기 개발 | 6 | 22 | $26 |
| 반일 운영 | 12 | 30 | $72 |
| 전일 운영 | 24 | 30 | $146 |

### 총 비용 예상

#### 시나리오 1: 개발/테스트 (최소 비용)

| 항목 | 비용/월 |
|------|---------|
| STT (Whisper 로컬) | $0 |
| LLM (GPT-3.5-turbo) | $0.50 |
| TTS (EdgeTTS 무료) | $0 |
| WebRTC (Free 플랜) | $0 |
| GPU (Vast.ai 2시간/일) | $8 |
| **합계** | **$8.50/월** |

#### 시나리오 2: 프로덕션 (최적화)

| 항목 | 비용/월 |
|------|---------|
| STT (Deepgram Nova-3) | $7.74 |
| LLM (GPT-4o) | $1.75 |
| TTS (EdgeTTS 무료) | $0 |
| WebRTC (Developer) | $29 |
| GPU (Vast.ai 24/7) | $146 |
| **합계** | **$184.49/월** |

#### 시나리오 3: 프로덕션 (고품질)

| 항목 | 비용/월 |
|------|---------|
| STT (Deepgram Nova-3) | $7.74 |
| LLM (GPT-4o) | $1.75 |
| TTS (ElevenLabs Creator) | $22 |
| WebRTC (Developer) | $29 |
| GPU (RunPod A5000 24/7) | $248 |
| **합계** | **$308.49/월** |

### 비용 절감 팁

1. **무료 TTS 사용**: EdgeTTS로 $22/월 절감
2. **캐싱 활성화**: TTS 중복 요청 80% 감소
3. **주문형 GPU**: 사용하지 않을 때 인스턴스 중지
4. **배치 처리**: GPU 활용률 증가로 처리량 4배 향상
5. **GPT-3.5 사용**: 간단한 대화는 GPT-3.5-turbo로 비용 80% 절감

---

## 🗺 로드맵

### v1.0 (현재)

- [x] 실시간 STT (Deepgram)
- [x] LLM 면접관 (GPT-4o)
- [x] TTS (ElevenLabs/Edge/Naver)
- [x] MuseTalk 아바타
- [x] WebRTC 통합
- [x] Docker 배포
- [x] 기본 최적화 (캐싱, 배치)

### v1.1 (1-2개월)

- [ ] **다중 언어 지원** (영어, 일본어, 중국어)
- [ ] **음성 감정 분석** (면접자의 감정 상태 파악)
- [ ] **실시간 자막** (STT 결과를 화면에 표시)
- [ ] **대화 요약** (면접 후 자동 피드백 생성)
- [ ] **Prometheus + Grafana** (모니터링 대시보드)

### v1.2 (3-4개월)

- [ ] **커스텀 아바타 업로드** (사용자 지정 얼굴 이미지)
- [ ] **다중 면접관 모드** (2명 이상의 AI 면접관)
- [ ] **이력서 분석** (PDF 업로드 후 맞춤형 질문)
- [ ] **실시간 화면 공유** (코딩 테스트 지원)
- [ ] **녹화 기능** (면접 영상 저장 및 재생)

### v2.0 (6개월+)

- [ ] **온프레미스 모델** (로컬 LLM, 로컬 TTS로 완전 오프라인)
  - Llama 3 70B (LLM)
  - XTTS v2 (TTS)
  - Faster Whisper (STT)
- [ ] **실시간 제스처 생성** (아바타 손동작, 표정 다양화)
- [ ] **VR 지원** (Meta Quest, PSVR2)
- [ ] **모바일 앱** (iOS/Android)
- [ ] **SaaS 플랫폼** (멀티 테넌트, 구독 결제)

### 커뮤니티 요청 기능

다음 기능을 구현할지 투표하세요! (GitHub Discussions):

- [ ] 면접 난이도 조절 (초급/중급/고급)
- [ ] 업종별 면접관 (IT/금융/마케팅 등)
- [ ] 그룹 면접 모드 (3-4명 동시 참여)
- [ ] AI 면접관 성격 설정 (친절/엄격/중립)
- [ ] 실시간 힌트 제공 (면접자가 막힐 때)

---

## 🤝 기여 가이드

### 기여 방법

이 프로젝트에 기여해주셔서 감사합니다! 다음 단계를 따라주세요:

#### 1. Fork 및 Clone

```bash
# Fork 버튼 클릭 (GitHub 웹)
git clone https://github.com/YOUR_USERNAME/realtime-interview-avatar.git
cd realtime-interview-avatar
```

#### 2. 브랜치 생성

```bash
git checkout -b feature/your-feature-name
# 또는
git checkout -b fix/your-bug-fix
```

브랜치 네이밍 규칙:
- `feature/` - 새로운 기능
- `fix/` - 버그 수정
- `docs/` - 문서 업데이트
- `refactor/` - 코드 리팩토링
- `test/` - 테스트 추가

#### 3. 개발 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 개발 의존성 설치
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Pre-commit 훅 설치
pre-commit install
```

#### 4. 코드 작성

**코딩 스타일**:
- PEP 8 준수 (Black 포맷터 사용)
- 타입 힌트 사용 (Python 3.10+)
- Docstring 작성 (Google 스타일)

예시:
```python
from typing import Optional

async def process_audio(
    audio_data: bytes,
    sample_rate: int = 16000,
    language: str = "ko",
) -> Optional[str]:
    """
    오디오 데이터를 처리하여 텍스트로 변환합니다.

    Args:
        audio_data: 원시 오디오 데이터 (PCM16)
        sample_rate: 샘플링 레이트 (Hz)
        language: 언어 코드 (ISO 639-1)

    Returns:
        변환된 텍스트, 실패 시 None

    Raises:
        ValueError: audio_data가 비어있을 경우
    """
    if not audio_data:
        raise ValueError("audio_data는 비어있을 수 없습니다.")

    # 처리 로직
    ...
```

#### 5. 테스트 작성

```bash
# 테스트 실행
pytest tests/

# 커버리지 확인
pytest --cov=src tests/

# 특정 테스트만 실행
pytest tests/test_stt.py::test_deepgram_transcription
```

새 기능에는 반드시 테스트를 포함해주세요:

```python
# tests/test_your_feature.py
import pytest
from src.your_module import your_function

@pytest.mark.asyncio
async def test_your_function():
    result = await your_function("test_input")
    assert result == "expected_output"
```

#### 6. 커밋 및 푸시

```bash
# 변경사항 스테이징
git add .

# 커밋 (Conventional Commits 사용)
git commit -m "feat: add new TTS provider support"

# 푸시
git push origin feature/your-feature-name
```

**커밋 메시지 규칙**:
- `feat:` - 새로운 기능
- `fix:` - 버그 수정
- `docs:` - 문서 변경
- `style:` - 코드 스타일 (포맷팅)
- `refactor:` - 리팩토링
- `test:` - 테스트 추가/수정
- `chore:` - 빌드/설정 변경

예시:
```
feat: add Whisper STT provider support

- Add WhisperSTTService class
- Update settings.py with whisper config
- Add tests for whisper transcription
```

#### 7. Pull Request 생성

1. GitHub에서 "Compare & pull request" 클릭
2. PR 템플릿 작성:

```markdown
## 변경 사항

- [ ] 새로운 TTS 프로바이더 추가
- [ ] 설정 파일 업데이트
- [ ] 테스트 추가

## 관련 이슈

Closes #123

## 테스트 방법

1. `.env`에 새 TTS API 키 추가
2. `python -m src.server.main` 실행
3. 브라우저에서 테스트

## 스크린샷 (해당 시)

[스크린샷 첨부]

## 체크리스트

- [x] 코드 스타일 확인 (Black, Flake8)
- [x] 테스트 추가 및 통과
- [x] 문서 업데이트
- [x] CHANGELOG.md 업데이트
```

#### 8. 코드 리뷰

- Maintainer가 코드를 리뷰합니다
- 변경 요청이 있을 수 있습니다
- 피드백에 따라 코드를 수정하고 푸시합니다

### 개발 가이드라인

#### 프로젝트 구조

```
src/
├── stt/              # 음성 인식 모듈
├── tts/              # 음성 합성 모듈
├── llm/              # 언어 모델 모듈
├── avatar/           # 아바타 렌더링 모듈
├── pipeline/         # 파이프라인 통합
├── server/           # FastAPI 서버
├── optimization/     # 최적화 모듈
└── utils/            # 유틸리티
```

#### 새 모듈 추가 시

1. `src/` 아래 새 디렉토리 생성
2. `__init__.py` 작성
3. 테스트 파일 추가 (`tests/test_your_module.py`)
4. `README.md` 업데이트

#### 버그 제보

[GitHub Issues](https://github.com/yourusername/realtime-interview-avatar/issues)에서:

- **Bug Report** 템플릿 사용
- 재현 가능한 최소 예제 제공
- 환경 정보 (OS, Python 버전, GPU 등)
- 에러 로그 첨부

#### 기능 제안

[GitHub Discussions](https://github.com/yourusername/realtime-interview-avatar/discussions)에서:

- 기능의 유즈 케이스 설명
- 예상되는 동작 기술
- 가능하면 프로토타입 코드 제공

---

## 📄 라이선스

MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📞 연락처 및 링크

- **GitHub**: https://github.com/yourusername/realtime-interview-avatar
- **Documentation**: https://docs.your-project.com
- **Discord**: https://discord.gg/your-invite
- **Email**: your-email@example.com

---

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들을 기반으로 합니다:

- [MuseTalk](https://github.com/TMElyralab/MuseTalk) - 립싱크 아바타 생성
- [Pipecat](https://github.com/pipecat-ai/pipecat) - 실시간 AI 파이프라인
- [FastAPI](https://github.com/tiangolo/fastapi) - 고성능 웹 프레임워크
- [Deepgram](https://deepgram.com) - 실시간 STT API
- [ElevenLabs](https://elevenlabs.io) - 고품질 TTS API

특별히 기여해주신 분들:
- [@contributor1](https://github.com/contributor1) - 초기 아키텍처 설계
- [@contributor2](https://github.com/contributor2) - TensorRT 최적화

---

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**

**🐛 버그를 발견하셨나요?** [Issue를 생성해주세요](https://github.com/yourusername/realtime-interview-avatar/issues/new)

**💡 기능 제안이 있으신가요?** [Discussion을 시작해주세요](https://github.com/yourusername/realtime-interview-avatar/discussions/new)

---

## 📝 변경 이력 (Changelog)

### 2026-01-12 - 멀티 클라이언트 WebSocket 분리 (중요 백업 포인트)

**커밋: `1317c10`** - 이 커밋은 중요한 백업 포인트입니다.

#### 주요 변경 사항

1. **멀티 클라이언트 WebSocket 분리**
   - 각 클라이언트마다 고유 SID(Session ID) 할당
   - `socketio.emit(to=sid)` 사용으로 해당 클라이언트에게만 메시지 전송
   - 여러 브라우저 창에서 접속해도 서로 간섭 없이 독립 동작

2. **동시 생성 방지**
   - `generation_lock` 추가로 한 번에 한 클라이언트만 립싱크 생성 가능
   - 다른 클라이언트 생성 중일 때 대기 메시지 표시

3. **립싱크 품질 향상**
   - 페이드 인/아웃 효과 (8프레임, ~0.3초) - 자연스러운 전환
   - Unsharp mask 샤프닝 (1.5x strength) - VAE 출력 선명도 향상
   - INTER_LANCZOS4 보간법 - 고품질 리사이즈
   - Gaussian blur 커널 크기 감소 (0.05 → 0.025) - 경계 선명도 향상

4. **UI 기능 추가**
   - 시스템 프롬프트 편집 기능 (테스트 페이지)
   - API: `GET/POST /api/prompt`

5. **기타 수정**
   - `start_server.bat` 인코딩 문제 수정
   - `landmarks[29]` 사용 (코 다리 하단) - 원본 MuseTalk과 동일

#### 관련 파일
- `realtime-interview-avatar/app.py`
- `realtime-interview-avatar/templates/index.html`
- `realtime-interview-avatar/start_server.bat`
- `MuseTalk/musetalk/utils/blending.py` (blur 커널 감소)

---

### 2026-01-12 - CosyVoice 한국어 TTS 수정 (중요 백업 포인트)

**커밋: `82198c4`** - CosyVoice 한국어 음성 합성 수정

#### 주요 변경 사항

1. **CosyVoice 프롬프트 설정 수정**
   - 프롬프트 오디오: `여성 50대 면접관` 음성 파일 (6.5초)
   - 프롬프트 텍스트: 오디오 내용과 정확히 일치하도록 수정
   - `"안녕하세요! 면접에 참여해 주셔서 감사합니다. 먼저, 본인에 대해 간단히 소개해 주시겠어요?"`

2. **한국어 숫자 읽기 문제 해결**
   - `text_frontend=False` 설정 추가
   - CosyVoice가 한국어를 영어로 인식하여 숫자를 영어로 변환하는 문제 해결
   - 예: "13년" → "thirteen년" 대신 "십삼년"으로 정상 발음

#### 관련 파일
- `realtime-interview-avatar/app.py` (CosyVoice 설정)
