# Docker로 Real-time MuseTalk 실행하기

## 개요

Docker를 사용하면 Windows에서도 mmpose가 필요한 Real-time MuseTalk을 실행할 수 있습니다.

```
┌─────────────────────────────────────────────────────────┐
│                 Host (Windows)                          │
│  - Docker Desktop                                       │
│  - 웹 브라우저 (http://localhost:8000)                  │
└─────────────────────┬───────────────────────────────────┘
                      │ GPU Passthrough
┌─────────────────────▼───────────────────────────────────┐
│            Docker Container (Linux)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  │
│  │   STT   │->│   LLM   │->│   TTS   │->│ MuseTalk  │  │
│  │Deepgram │  │ GPT-4o  │  │ElevenLabs│  │ Real-time │  │
│  └─────────┘  └─────────┘  └─────────┘  └───────────┘  │
│                                          mmpose OK!     │
└─────────────────────────────────────────────────────────┘
```

## 필수 조건

### 1. Docker Desktop 설치

Windows용 Docker Desktop을 설치하세요:
- 다운로드: https://www.docker.com/products/docker-desktop/
- WSL2 백엔드 활성화 (설치 시 자동 설정)

### 2. NVIDIA Container Toolkit

GPU를 사용하려면 NVIDIA Container Toolkit이 필요합니다:

```powershell
# WSL2에서 실행
wsl

# NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. API 키 준비

`.env` 파일에 다음 API 키들을 설정하세요:

```bash
# .env 파일 생성
cp .env.example .env
```

필수 API 키:
- `OPENAI_API_KEY`: GPT-4o (LLM)
- `DEEPGRAM_API_KEY`: Nova-3 (STT)
- `ELEVENLABS_API_KEY`: 음성 합성 (TTS)

## 빠른 시작

### Windows (PowerShell)

```powershell
# 1. 프로젝트 폴더로 이동
cd c:\NewAvata\NewAvata\realtime-interview-avatar

# 2. Docker 이미지 빌드
docker-compose -f docker-compose.realtime.yml build

# 3. 서비스 시작
docker-compose -f docker-compose.realtime.yml up -d

# 4. 로그 확인
docker-compose -f docker-compose.realtime.yml logs -f avatar

# 5. 중지
docker-compose -f docker-compose.realtime.yml down
```

### 배치 스크립트 사용

```cmd
# 빌드
scripts\docker_run.bat build

# 시작
scripts\docker_run.bat up

# 로그 보기
scripts\docker_run.bat logs

# 중지
scripts\docker_run.bat down
```

## API 사용

서비스가 시작되면:
- API 엔드포인트: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### WebSocket 연결

```javascript
// JavaScript 예시
const ws = new WebSocket('ws://localhost:8000/ws/session_id?api_key=your_api_key');

// 세션 시작
ws.send(JSON.stringify({
    type: 'start_session'
}));

// 텍스트 입력
ws.send(JSON.stringify({
    type: 'text_message',
    text: '안녕하세요'
}));

// 응답 수신
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
    // data.type: 'ai_response'
    // data.text: AI 응답
    // data.audio: 오디오 데이터 (base64)
};
```

## 성능 비교

| 환경 | 속도 | 비고 |
|------|------|------|
| Windows (face_alignment) | ~25 fps | mmpose 없음 |
| Docker (mmpose Real-time) | ~30 fps | 최적화됨 |
| Docker 오버헤드 | -1~3% | 무시 가능 |
| **순이익** | **+17~19%** | Docker가 더 빠름! |

## 문제 해결

### GPU가 인식되지 않음

```powershell
# Docker에서 GPU 확인
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 메모리 부족

```yaml
# docker-compose.realtime.yml에서 shm_size 증가
services:
  avatar:
    shm_size: '16gb'  # 8gb -> 16gb
```

### 빌드 실패

```powershell
# 캐시 없이 다시 빌드
docker-compose -f docker-compose.realtime.yml build --no-cache
```

### 모델 로딩 느림

모델 파일을 호스트에 다운로드하고 마운트하세요:

```yaml
volumes:
  - ../MuseTalk/models:/app/models:ro
```

## 파일 구조

```
realtime-interview-avatar/
├── docker/
│   ├── Dockerfile.realtime    # Real-time용 Dockerfile
│   └── Dockerfile             # 기본 Dockerfile
├── docker-compose.realtime.yml  # Real-time 구성
├── docker-compose.yml           # 기본 구성
├── scripts/
│   ├── docker_run.bat         # Windows 실행 스크립트
│   └── docker_build.sh        # Linux 빌드 스크립트
├── src/
│   ├── avatar/
│   │   └── musetalk_realtime.py  # Real-time Avatar
│   ├── pipeline/
│   │   └── realtime_avatar_pipeline.py  # 통합 파이프라인
│   ├── stt/
│   │   └── deepgram_service.py
│   ├── llm/
│   │   └── interviewer_agent.py
│   └── tts/
│       └── elevenlabs_service.py
└── .env  # API 키 (생성 필요)
```

## 주의사항

1. **첫 실행 시간**: Docker 이미지 빌드에 10-20분 소요
2. **모델 다운로드**: 첫 실행 시 모델 다운로드에 추가 시간 소요
3. **GPU 메모리**: RTX 4070 Ti (12GB) 권장
4. **디스크 공간**: Docker 이미지 약 15-20GB

## 다음 단계

1. Docker Desktop 설치
2. `.env` 파일 생성 및 API 키 설정
3. `docker-compose.realtime.yml` 실행
4. http://localhost:8000/docs 접속하여 API 테스트
