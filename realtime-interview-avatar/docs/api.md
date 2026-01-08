# API 문서

실시간 면접 아바타 시스템의 전체 API 문서입니다.

## 📋 목차

- [개요](#개요)
- [인증](#인증)
- [REST API 엔드포인트](#rest-api-엔드포인트)
- [WebSocket API](#websocket-api)
- [에러 코드](#에러-코드)
- [요청/응답 스키마](#요청응답-스키마)
- [예제 코드](#예제-코드)
- [보안 가이드](#보안-가이드)

---

## 개요

### Base URL

```
http://localhost:8000
```

프로덕션:
```
https://your-domain.com
```

### API 버전

현재 버전: `v1`

### Content-Type

모든 요청과 응답은 `application/json` 형식을 사용합니다.

### 자동 문서화

FastAPI는 자동으로 대화형 API 문서를 생성합니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI 스키마**: http://localhost:8000/openapi.json

---

## 인증

### API Key 인증

API 키는 HTTP 헤더를 통해 전달됩니다.

**헤더**:
```
X-API-Key: your_api_key_here
```

**예시**:
```bash
curl -X GET http://localhost:8000/api/sessions \
  -H "X-API-Key: your_api_key_here"
```

### API 키 발급

1. 관리자 계정으로 로그인
2. `/admin/api-keys` 엔드포인트에서 새 API 키 생성
3. 생성된 키를 안전하게 보관

⚠️ **보안 주의사항**:
- API 키는 절대 공개 저장소에 커밋하지 마세요
- `.env` 파일을 사용하여 환경 변수로 관리하세요
- 정기적으로 API 키를 교체하세요

### 인증 오류

잘못된 API 키 사용 시:

```json
{
  "detail": "Invalid API key",
  "status_code": 401
}
```

---

## REST API 엔드포인트

### 1. 헬스 체크

시스템 상태를 확인합니다.

**엔드포인트**: `GET /health`

**인증**: 불필요

**응답**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-01T12:00:00Z",
  "gpu_available": true,
  "gpu_memory_used_mb": 4096,
  "gpu_memory_total_mb": 16384,
  "services": {
    "stt": "operational",
    "llm": "operational",
    "tts": "operational",
    "avatar": "operational",
    "redis": "operational"
  }
}
```

**cURL 예시**:

```bash
curl -X GET http://localhost:8000/health
```

**Python 예시**:

```python
import requests

response = requests.get("http://localhost:8000/health")
data = response.json()

print(f"Status: {data['status']}")
print(f"GPU Available: {data['gpu_available']}")
```

---

### 2. 세션 생성

새로운 면접 세션을 생성합니다.

**엔드포인트**: `POST /api/sessions`

**인증**: 필수

**요청 본문**:

```json
{
  "user_id": "user123",
  "interview_type": "technical",
  "language": "ko",
  "difficulty": "medium",
  "duration_minutes": 30
}
```

**요청 스키마**:

| 필드 | 타입 | 필수 | 설명 | 기본값 |
|------|------|------|------|--------|
| user_id | string | O | 사용자 고유 ID | - |
| interview_type | string | X | 면접 유형 (technical/behavioral/general) | general |
| language | string | X | 언어 코드 (ko/en/ja) | ko |
| difficulty | string | X | 난이도 (easy/medium/hard) | medium |
| duration_minutes | integer | X | 세션 최대 시간 (분) | 30 |

**응답**:

```json
{
  "session_id": "sess_abc123def456",
  "user_id": "user123",
  "interview_type": "technical",
  "language": "ko",
  "difficulty": "medium",
  "status": "created",
  "created_at": "2024-01-01T12:00:00Z",
  "expires_at": "2024-01-01T12:30:00Z",
  "daily_room_url": "https://your-domain.daily.co/sess_abc123def456",
  "websocket_url": "ws://localhost:8000/ws/sess_abc123def456"
}
```

**cURL 예시**:

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "user_id": "user123",
    "interview_type": "technical",
    "language": "ko",
    "difficulty": "medium"
  }'
```

**Python 예시**:

```python
import requests

url = "http://localhost:8000/api/sessions"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your_api_key_here"
}
payload = {
    "user_id": "user123",
    "interview_type": "technical",
    "language": "ko",
    "difficulty": "medium"
}

response = requests.post(url, headers=headers, json=payload)
session = response.json()

print(f"Session ID: {session['session_id']}")
print(f"WebSocket URL: {session['websocket_url']}")
print(f"Daily Room: {session['daily_room_url']}")
```

**에러 응답**:

```json
{
  "detail": "Invalid interview_type. Must be one of: technical, behavioral, general",
  "status_code": 400
}
```

---

### 3. 세션 조회

기존 세션의 상태를 조회합니다.

**엔드포인트**: `GET /api/sessions/{session_id}`

**인증**: 필수

**경로 파라미터**:

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| session_id | string | 세션 고유 ID |

**응답**:

```json
{
  "session_id": "sess_abc123def456",
  "user_id": "user123",
  "status": "active",
  "interview_type": "technical",
  "language": "ko",
  "difficulty": "medium",
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:01:30Z",
  "expires_at": "2024-01-01T12:30:00Z",
  "duration_seconds": 450,
  "message_count": 15,
  "metadata": {
    "questions_asked": 5,
    "responses_received": 5,
    "average_response_time_ms": 1200
  }
}
```

**상태 값**:

- `created`: 세션 생성됨 (아직 시작 안됨)
- `active`: 면접 진행 중
- `paused`: 일시 정지
- `completed`: 정상 종료
- `terminated`: 강제 종료
- `expired`: 만료됨

**cURL 예시**:

```bash
curl -X GET http://localhost:8000/api/sessions/sess_abc123def456 \
  -H "X-API-Key: your_api_key_here"
```

**Python 예시**:

```python
import requests

session_id = "sess_abc123def456"
url = f"http://localhost:8000/api/sessions/{session_id}"
headers = {"X-API-Key": "your_api_key_here"}

response = requests.get(url, headers=headers)
session = response.json()

print(f"Status: {session['status']}")
print(f"Duration: {session['duration_seconds']}s")
print(f"Messages: {session['message_count']}")
```

---

### 4. 세션 목록 조회

사용자의 모든 세션을 조회합니다.

**엔드포인트**: `GET /api/sessions`

**인증**: 필수

**쿼리 파라미터**:

| 파라미터 | 타입 | 필수 | 설명 | 기본값 |
|---------|------|------|------|--------|
| user_id | string | O | 사용자 ID | - |
| status | string | X | 상태 필터 | all |
| limit | integer | X | 최대 개수 | 20 |
| offset | integer | X | 페이지 오프셋 | 0 |

**응답**:

```json
{
  "total": 45,
  "limit": 20,
  "offset": 0,
  "sessions": [
    {
      "session_id": "sess_abc123",
      "status": "completed",
      "created_at": "2024-01-01T12:00:00Z",
      "duration_seconds": 1800
    },
    {
      "session_id": "sess_def456",
      "status": "active",
      "created_at": "2024-01-01T15:30:00Z",
      "duration_seconds": 300
    }
  ]
}
```

**cURL 예시**:

```bash
curl -X GET "http://localhost:8000/api/sessions?user_id=user123&limit=10" \
  -H "X-API-Key: your_api_key_here"
```

**Python 예시**:

```python
import requests

url = "http://localhost:8000/api/sessions"
headers = {"X-API-Key": "your_api_key_here"}
params = {
    "user_id": "user123",
    "status": "completed",
    "limit": 10
}

response = requests.get(url, headers=headers, params=params)
data = response.json()

print(f"Total sessions: {data['total']}")
for session in data['sessions']:
    print(f"- {session['session_id']}: {session['status']}")
```

---

### 5. 세션 일시정지

진행 중인 세션을 일시정지합니다.

**엔드포인트**: `POST /api/sessions/{session_id}/pause`

**인증**: 필수

**응답**:

```json
{
  "session_id": "sess_abc123def456",
  "status": "paused",
  "paused_at": "2024-01-01T12:15:00Z"
}
```

**cURL 예시**:

```bash
curl -X POST http://localhost:8000/api/sessions/sess_abc123def456/pause \
  -H "X-API-Key: your_api_key_here"
```

---

### 6. 세션 재개

일시정지된 세션을 재개합니다.

**엔드포인트**: `POST /api/sessions/{session_id}/resume`

**인증**: 필수

**응답**:

```json
{
  "session_id": "sess_abc123def456",
  "status": "active",
  "resumed_at": "2024-01-01T12:20:00Z"
}
```

**cURL 예시**:

```bash
curl -X POST http://localhost:8000/api/sessions/sess_abc123def456/resume \
  -H "X-API-Key: your_api_key_here"
```

---

### 7. 세션 종료

세션을 종료하고 요약 정보를 반환합니다.

**엔드포인트**: `DELETE /api/sessions/{session_id}`

**인증**: 필수

**응답**:

```json
{
  "session_id": "sess_abc123def456",
  "status": "terminated",
  "terminated_at": "2024-01-01T12:25:00Z",
  "summary": {
    "duration_seconds": 1500,
    "total_messages": 20,
    "ai_questions": 10,
    "user_responses": 10,
    "average_response_time_ms": 1200,
    "average_ai_latency_ms": 800,
    "transcript_url": "https://storage.example.com/transcripts/sess_abc123def456.txt",
    "video_url": "https://storage.example.com/videos/sess_abc123def456.mp4"
  },
  "feedback": {
    "overall_score": 7.5,
    "communication": 8.0,
    "technical_knowledge": 7.0,
    "problem_solving": 7.5,
    "comments": "좋은 의사소통 능력을 보여주셨습니다. 기술적인 부분에서 좀 더 구체적인 답변을 제공하면 더 좋을 것 같습니다."
  }
}
```

**cURL 예시**:

```bash
curl -X DELETE http://localhost:8000/api/sessions/sess_abc123def456 \
  -H "X-API-Key: your_api_key_here"
```

**Python 예시**:

```python
import requests

session_id = "sess_abc123def456"
url = f"http://localhost:8000/api/sessions/{session_id}"
headers = {"X-API-Key": "your_api_key_here"}

response = requests.delete(url, headers=headers)
summary = response.json()

print(f"Duration: {summary['summary']['duration_seconds']}s")
print(f"Score: {summary['feedback']['overall_score']}/10")
print(f"Transcript: {summary['summary']['transcript_url']}")
```

---

### 8. 설정 조회

현재 시스템 설정을 조회합니다.

**엔드포인트**: `GET /api/config`

**인증**: 필수

**응답**:

```json
{
  "stt": {
    "provider": "deepgram",
    "model": "nova-3",
    "language": "ko"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7
  },
  "tts": {
    "provider": "elevenlabs",
    "voice_id": "21m00Tcm4TlvDq8ikWAM"
  },
  "avatar": {
    "fps": 25,
    "resolution": [512, 512],
    "use_tensorrt": true
  },
  "optimization": {
    "enable_cache": true,
    "enable_batching": true,
    "batch_size": 4
  }
}
```

**cURL 예시**:

```bash
curl -X GET http://localhost:8000/api/config \
  -H "X-API-Key: your_api_key_here"
```

---

### 9. 통계 조회

시스템 사용 통계를 조회합니다.

**엔드포인트**: `GET /api/stats`

**인증**: 필수

**쿼리 파라미터**:

| 파라미터 | 타입 | 필수 | 설명 | 기본값 |
|---------|------|------|------|--------|
| period | string | X | 기간 (today/week/month) | today |
| user_id | string | X | 특정 사용자 필터 | - |

**응답**:

```json
{
  "period": "today",
  "total_sessions": 125,
  "active_sessions": 8,
  "completed_sessions": 115,
  "terminated_sessions": 2,
  "average_duration_seconds": 1350,
  "average_latency_ms": 850,
  "total_messages": 2500,
  "gpu_utilization_percent": 65,
  "cache_hit_rate": 0.82
}
```

**cURL 예시**:

```bash
curl -X GET "http://localhost:8000/api/stats?period=week" \
  -H "X-API-Key: your_api_key_here"
```

---

## WebSocket API

### 연결

**엔드포인트**: `ws://localhost:8000/ws/{session_id}`

**인증**: 쿼리 파라미터로 API 키 전달

```
ws://localhost:8000/ws/sess_abc123def456?api_key=your_api_key_here
```

### 연결 예시

**JavaScript**:

```javascript
const sessionId = "sess_abc123def456";
const apiKey = "your_api_key_here";
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}?api_key=${apiKey}`);

ws.onopen = () => {
  console.log("WebSocket connected");
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  handleMessage(data);
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};

ws.onclose = () => {
  console.log("WebSocket disconnected");
};
```

**Python**:

```python
import asyncio
import websockets
import json

async def connect():
    session_id = "sess_abc123def456"
    api_key = "your_api_key_here"
    uri = f"ws://localhost:8000/ws/{session_id}?api_key={api_key}"

    async with websockets.connect(uri) as websocket:
        print("Connected")

        # 메시지 수신
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['type']}")

            if data['type'] == 'llm_response' and data['done']:
                print(f"AI: {data['text']}")

asyncio.run(connect())
```

---

### 메시지 타입

#### 1. 연결 확인 (Server → Client)

```json
{
  "type": "connected",
  "session_id": "sess_abc123def456",
  "timestamp": "2024-01-01T12:00:00Z",
  "message": "Connected to interview session"
}
```

#### 2. 세션 시작 (Client → Server)

```json
{
  "type": "start_session",
  "session_id": "sess_abc123def456"
}
```

**응답**:

```json
{
  "type": "session_started",
  "session_id": "sess_abc123def456",
  "message": "안녕하세요! 면접을 시작하겠습니다. 먼저 자기소개를 해주시겠어요?"
}
```

#### 3. STT 중간 결과 (Server → Client)

```json
{
  "type": "stt_interim",
  "transcript": "안녕하",
  "confidence": 0.85,
  "is_final": false,
  "timestamp": "2024-01-01T12:01:00Z"
}
```

#### 4. STT 최종 결과 (Server → Client)

```json
{
  "type": "stt_final",
  "transcript": "안녕하세요, 저는 김철수입니다.",
  "confidence": 0.92,
  "is_final": true,
  "timestamp": "2024-01-01T12:01:05Z"
}
```

#### 5. LLM 응답 시작 (Server → Client)

```json
{
  "type": "llm_start",
  "timestamp": "2024-01-01T12:01:06Z"
}
```

#### 6. LLM 응답 스트리밍 (Server → Client)

```json
{
  "type": "llm_response",
  "text": "안녕하세요",
  "done": false,
  "timestamp": "2024-01-01T12:01:06.100Z"
}
```

```json
{
  "type": "llm_response",
  "text": "안녕하세요, 김철수님.",
  "done": false,
  "timestamp": "2024-01-01T12:01:06.200Z"
}
```

```json
{
  "type": "llm_response",
  "text": "안녕하세요, 김철수님. 만나서 반갑습니다. 어떤 직무에 지원하셨나요?",
  "done": true,
  "timestamp": "2024-01-01T12:01:08Z"
}
```

#### 7. TTS 오디오 (Server → Client)

```json
{
  "type": "tts_audio",
  "audio_data": "base64_encoded_audio_data...",
  "sample_rate": 24000,
  "format": "pcm16",
  "duration_ms": 3500,
  "timestamp": "2024-01-01T12:01:09Z"
}
```

#### 8. Avatar 프레임 (Server → Client)

```json
{
  "type": "avatar_frame",
  "frame_data": "base64_encoded_image...",
  "width": 512,
  "height": 512,
  "format": "jpeg",
  "frame_number": 125,
  "timestamp": "2024-01-01T12:01:09.040Z"
}
```

#### 9. 오디오 데이터 전송 (Client → Server)

```json
{
  "type": "audio_data",
  "audio": "base64_encoded_audio...",
  "sample_rate": 16000,
  "format": "pcm16"
}
```

#### 10. 에러 (Server → Client)

```json
{
  "type": "error",
  "code": "STT_ERROR",
  "message": "Speech recognition failed",
  "details": "Deepgram API timeout after 5s",
  "timestamp": "2024-01-01T12:01:10Z"
}
```

#### 11. Ping/Pong (양방향)

**Ping** (Client → Server):

```json
{
  "type": "ping",
  "timestamp": "2024-01-01T12:01:00Z"
}
```

**Pong** (Server → Client):

```json
{
  "type": "pong",
  "timestamp": "2024-01-01T12:01:00.050Z",
  "latency_ms": 50
}
```

---

## 에러 코드

### HTTP 상태 코드

| 코드 | 의미 | 설명 |
|------|------|------|
| 200 | OK | 요청 성공 |
| 201 | Created | 리소스 생성 성공 |
| 400 | Bad Request | 잘못된 요청 |
| 401 | Unauthorized | 인증 실패 |
| 403 | Forbidden | 권한 없음 |
| 404 | Not Found | 리소스를 찾을 수 없음 |
| 429 | Too Many Requests | 요청 제한 초과 |
| 500 | Internal Server Error | 서버 내부 오류 |
| 503 | Service Unavailable | 서비스 이용 불가 |

### 커스텀 에러 코드

#### 세션 관련

| 코드 | 설명 |
|------|------|
| SESSION_NOT_FOUND | 세션을 찾을 수 없음 |
| SESSION_EXPIRED | 세션이 만료됨 |
| SESSION_ALREADY_ACTIVE | 이미 활성화된 세션 |
| SESSION_LIMIT_REACHED | 동시 세션 개수 초과 |

#### STT 관련

| 코드 | 설명 |
|------|------|
| STT_ERROR | 음성 인식 실패 |
| STT_TIMEOUT | 음성 인식 타임아웃 |
| STT_INVALID_AUDIO | 유효하지 않은 오디오 형식 |
| STT_NO_SPEECH | 음성이 감지되지 않음 |

#### LLM 관련

| 코드 | 설명 |
|------|------|
| LLM_ERROR | LLM 처리 실패 |
| LLM_TIMEOUT | LLM 응답 타임아웃 |
| LLM_RATE_LIMIT | LLM API 요청 제한 초과 |
| LLM_INVALID_RESPONSE | LLM 응답이 유효하지 않음 |

#### TTS 관련

| 코드 | 설명 |
|------|------|
| TTS_ERROR | 음성 합성 실패 |
| TTS_TIMEOUT | 음성 합성 타임아웃 |
| TTS_INVALID_TEXT | 유효하지 않은 텍스트 |
| TTS_QUOTA_EXCEEDED | TTS 할당량 초과 |

#### Avatar 관련

| 코드 | 설명 |
|------|------|
| AVATAR_ERROR | 아바타 렌더링 실패 |
| AVATAR_GPU_OOM | GPU 메모리 부족 |
| AVATAR_MODEL_NOT_LOADED | 아바타 모델 로딩 실패 |

#### 시스템 관련

| 코드 | 설명 |
|------|------|
| GPU_NOT_AVAILABLE | GPU를 사용할 수 없음 |
| REDIS_CONNECTION_ERROR | Redis 연결 실패 |
| STORAGE_ERROR | 스토리지 접근 실패 |

### 에러 응답 형식

```json
{
  "error": {
    "code": "SESSION_NOT_FOUND",
    "message": "Session not found",
    "details": "Session ID 'sess_abc123' does not exist or has expired",
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_xyz789"
  }
}
```

---

## 요청/응답 스키마

### SessionCreate

```json
{
  "user_id": "string",
  "interview_type": "technical | behavioral | general",
  "language": "ko | en | ja",
  "difficulty": "easy | medium | hard",
  "duration_minutes": "integer (5-120)"
}
```

### SessionResponse

```json
{
  "session_id": "string",
  "user_id": "string",
  "interview_type": "string",
  "language": "string",
  "difficulty": "string",
  "status": "created | active | paused | completed | terminated | expired",
  "created_at": "datetime",
  "started_at": "datetime | null",
  "expires_at": "datetime",
  "duration_seconds": "integer",
  "message_count": "integer",
  "daily_room_url": "string",
  "websocket_url": "string",
  "metadata": "object"
}
```

### SessionSummary

```json
{
  "session_id": "string",
  "status": "string",
  "terminated_at": "datetime",
  "summary": {
    "duration_seconds": "integer",
    "total_messages": "integer",
    "ai_questions": "integer",
    "user_responses": "integer",
    "average_response_time_ms": "number",
    "average_ai_latency_ms": "number",
    "transcript_url": "string",
    "video_url": "string"
  },
  "feedback": {
    "overall_score": "number (0-10)",
    "communication": "number (0-10)",
    "technical_knowledge": "number (0-10)",
    "problem_solving": "number (0-10)",
    "comments": "string"
  }
}
```

### HealthResponse

```json
{
  "status": "healthy | degraded | unhealthy",
  "version": "string",
  "timestamp": "datetime",
  "gpu_available": "boolean",
  "gpu_memory_used_mb": "integer",
  "gpu_memory_total_mb": "integer",
  "services": {
    "stt": "operational | degraded | down",
    "llm": "operational | degraded | down",
    "tts": "operational | degraded | down",
    "avatar": "operational | degraded | down",
    "redis": "operational | degraded | down"
  }
}
```

---

## 예제 코드

### Python SDK

완전한 Python SDK 예제:

```python
import asyncio
import requests
import websockets
import json
import base64
from typing import Callable, Dict, Any

class InterviewAvatarClient:
    """실시간 면접 아바타 클라이언트"""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        self.ws = None
        self.handlers = {}

    def on(self, event_type: str, handler: Callable):
        """이벤트 핸들러 등록"""
        self.handlers[event_type] = handler

    async def create_session(
        self,
        user_id: str,
        interview_type: str = "general",
        language: str = "ko",
        difficulty: str = "medium"
    ) -> Dict[str, Any]:
        """세션 생성"""
        url = f"{self.api_url}/api/sessions"
        payload = {
            "user_id": user_id,
            "interview_type": interview_type,
            "language": language,
            "difficulty": difficulty
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """세션 조회"""
        url = f"{self.api_url}/api/sessions/{session_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def terminate_session(self, session_id: str) -> Dict[str, Any]:
        """세션 종료"""
        url = f"{self.api_url}/api/sessions/{session_id}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def connect(self, session_id: str):
        """WebSocket 연결"""
        ws_url = self.api_url.replace('http', 'ws')
        uri = f"{ws_url}/ws/{session_id}?api_key={self.api_key}"

        self.ws = await websockets.connect(uri)
        print(f"Connected to WebSocket: {session_id}")

        # 메시지 리스너 시작
        asyncio.create_task(self._listen())

    async def _listen(self):
        """WebSocket 메시지 수신"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                event_type = data.get('type')

                # 핸들러 호출
                if event_type in self.handlers:
                    await self.handlers[event_type](data)
        except Exception as e:
            print(f"WebSocket error: {e}")

    async def start_session(self):
        """면접 시작"""
        await self.ws.send(json.dumps({
            "type": "start_session"
        }))

    async def send_audio(self, audio_data: bytes, sample_rate: int = 16000):
        """오디오 전송"""
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        await self.ws.send(json.dumps({
            "type": "audio_data",
            "audio": audio_b64,
            "sample_rate": sample_rate,
            "format": "pcm16"
        }))

    async def close(self):
        """연결 종료"""
        if self.ws:
            await self.ws.close()


# 사용 예시
async def main():
    # 클라이언트 생성
    client = InterviewAvatarClient(
        api_url="http://localhost:8000",
        api_key="your_api_key_here"
    )

    # 이벤트 핸들러 등록
    @client.on("connected")
    async def on_connected(data):
        print(f"✓ Connected: {data['session_id']}")

    @client.on("stt_final")
    async def on_transcript(data):
        print(f"🎤 You: {data['transcript']}")

    @client.on("llm_response")
    async def on_response(data):
        if data['done']:
            print(f"🤖 AI: {data['text']}")

    @client.on("error")
    async def on_error(data):
        print(f"❌ Error: {data['message']}")

    try:
        # 세션 생성
        session = await client.create_session(
            user_id="user123",
            interview_type="technical",
            language="ko"
        )
        print(f"Session ID: {session['session_id']}")

        # WebSocket 연결
        await client.connect(session['session_id'])

        # 면접 시작
        await client.start_session()

        # 5분 대기
        await asyncio.sleep(300)

        # 세션 종료
        summary = await client.terminate_session(session['session_id'])
        print(f"\n=== 면접 요약 ===")
        print(f"Duration: {summary['summary']['duration_seconds']}s")
        print(f"Score: {summary['feedback']['overall_score']}/10")
        print(f"Comments: {summary['feedback']['comments']}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript SDK

```javascript
class InterviewAvatarClient {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
    this.ws = null;
    this.handlers = {};
  }

  on(eventType, handler) {
    this.handlers[eventType] = handler;
  }

  async createSession({ userId, interviewType = 'general', language = 'ko', difficulty = 'medium' }) {
    const response = await fetch(`${this.apiUrl}/api/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey
      },
      body: JSON.stringify({
        user_id: userId,
        interview_type: interviewType,
        language,
        difficulty
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    return response.json();
  }

  async getSession(sessionId) {
    const response = await fetch(`${this.apiUrl}/api/sessions/${sessionId}`, {
      headers: {
        'X-API-Key': this.apiKey
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    return response.json();
  }

  async terminateSession(sessionId) {
    const response = await fetch(`${this.apiUrl}/api/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: {
        'X-API-Key': this.apiKey
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    return response.json();
  }

  connect(sessionId) {
    return new Promise((resolve, reject) => {
      const wsUrl = this.apiUrl.replace('http', 'ws');
      this.ws = new WebSocket(`${wsUrl}/ws/${sessionId}?api_key=${this.apiKey}`);

      this.ws.onopen = () => {
        console.log(`Connected to WebSocket: ${sessionId}`);
        resolve();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const eventType = data.type;

        if (this.handlers[eventType]) {
          this.handlers[eventType](data);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
      };
    });
  }

  startSession() {
    this.ws.send(JSON.stringify({
      type: 'start_session'
    }));
  }

  sendAudio(audioData, sampleRate = 16000) {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1];
      this.ws.send(JSON.stringify({
        type: 'audio_data',
        audio: base64,
        sample_rate: sampleRate,
        format: 'pcm16'
      }));
    };
    reader.readAsDataURL(new Blob([audioData]));
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// 사용 예시
async function main() {
  const client = new InterviewAvatarClient(
    'http://localhost:8000',
    'your_api_key_here'
  );

  // 이벤트 핸들러 등록
  client.on('connected', (data) => {
    console.log('✓ Connected:', data.session_id);
  });

  client.on('stt_final', (data) => {
    console.log('🎤 You:', data.transcript);
  });

  client.on('llm_response', (data) => {
    if (data.done) {
      console.log('🤖 AI:', data.text);
    }
  });

  client.on('error', (data) => {
    console.error('❌ Error:', data.message);
  });

  try {
    // 세션 생성
    const session = await client.createSession({
      userId: 'user123',
      interviewType: 'technical',
      language: 'ko'
    });
    console.log('Session ID:', session.session_id);

    // WebSocket 연결
    await client.connect(session.session_id);

    // 면접 시작
    client.startSession();

    // 5분 후 종료
    setTimeout(async () => {
      const summary = await client.terminateSession(session.session_id);
      console.log('\n=== 면접 요약 ===');
      console.log('Duration:', summary.summary.duration_seconds, 's');
      console.log('Score:', summary.feedback.overall_score, '/10');
      console.log('Comments:', summary.feedback.comments);

      client.close();
    }, 300000);

  } catch (error) {
    console.error('Error:', error);
    client.close();
  }
}

main();
```

---

## 보안 가이드

### API 키 관리

⚠️ **중요: API 키를 절대 공개하지 마세요!**

**올바른 방법**:

```bash
# .env 파일에 저장
API_KEY=your_secret_api_key_here

# .gitignore에 추가
echo ".env" >> .gitignore
```

**잘못된 방법** ❌:

```python
# 코드에 하드코딩 (절대 금지!)
api_key = "sk-1234567890abcdef"
```

### 환경 변수 사용

**Python**:

```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL", "http://localhost:8000")
```

**JavaScript**:

```javascript
require('dotenv').config();

const API_KEY = process.env.API_KEY;
const API_URL = process.env.API_URL || 'http://localhost:8000';
```

### 요청 제한

- **비율 제한**: 1분당 60 요청
- **동시 세션**: 사용자당 최대 5개
- **세션 시간**: 최대 2시간

제한 초과 시:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "retry_after": 30
  }
}
```

### HTTPS 사용

프로덕션 환경에서는 반드시 HTTPS를 사용하세요:

```
https://your-domain.com
```

### CORS 설정

프론트엔드가 다른 도메인에서 호스팅될 경우 CORS 설정이 필요합니다.

```python
# src/server/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

---

## 비용 절감 팁

### 1. EdgeTTS로 시작 (무료)

처음에는 무료인 EdgeTTS를 사용하여 시스템을 테스트하세요:

```bash
# .env
TTS_PROVIDER=edge
```

동작 확인 후 ElevenLabs로 전환:

```bash
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_key
```

### 2. Daily.co 무료 티어

WebRTC 테스트는 Daily.co 무료 플랜으로 충분합니다:

- 월 1,000분 무료
- 최대 20명 참여 가능

### 3. 클라우드 자동 종료

GPU 인스턴스는 사용하지 않을 때 자동으로 종료되도록 설정하세요:

```bash
# 2시간 후 자동 종료
sudo shutdown -h +120
```

### 4. 캐싱 활성화

TTS 결과를 캐싱하여 API 호출 80% 절감:

```bash
# .env
ENABLE_CACHE=true
CACHE_PREWARM=true
```

### 5. 모니터링 설정

비용 초과를 방지하기 위해 알림을 설정하세요:

- OpenAI: Usage dashboard에서 한도 설정
- Deepgram: Credit alerts 활성화
- ElevenLabs: Character limit alerts

---

## 추가 리소스

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **GitHub 저장소**: https://github.com/yourusername/realtime-interview-avatar
- **이슈 트래커**: https://github.com/yourusername/realtime-interview-avatar/issues
- **Discord 커뮤니티**: https://discord.gg/your-invite

---

**마지막 업데이트**: 2024-01-01
**API 버전**: v1.0.0
