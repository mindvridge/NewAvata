# 실시간 립싱크 아바타 API 문서

## 개요

텍스트를 입력받아 TTS(Text-to-Speech)와 립싱크 비디오를 생성하는 API 서버입니다.

**Base URL**: `http://localhost:5000`

---

## 아키텍처

```
[클라이언트]                         [립싱크 서버]
    │                                    │
    ├── 1. POST /api/v2/lipsync/stream ─►│
    │   {"text": "음성으로 변환할 텍스트"}│
    │                                    │
    ◄── 2. SSE: TTS 진행 상태            │
    ◄── 3. SSE: 립싱크 진행 상태         │
    ◄── 4. SSE: 완료 (video_url)         │
    │                                    │
    ├── 5. GET /video/xxx.mp4 ──────────►│
    ◄── 6. 립싱크 비디오 스트리밍        │
```

---

## 목차

1. [서버 상태 확인](#1-서버-상태-확인)
2. [립싱크 비디오 생성 API](#2-립싱크-비디오-생성-api) ⭐ 메인 API
3. [기타 API](#3-기타-api)

---

## 1. 서버 상태 확인

### GET /api/v2/status

서버 상태 및 사용 가능한 리소스를 확인합니다.

**Request**
```bash
curl -X GET http://localhost:5000/api/v2/status
```

**Response**
```json
{
    "status": "ok",
    "models_loaded": true,
    "available_avatars": ["talk_long_720p", "talk_short_720p"],
    "tts_engines": ["elevenlabs"]
}
```

---

## 2. 립싱크 비디오 생성 API ⭐

텍스트를 입력받아 TTS + 립싱크 비디오를 생성합니다.

### POST /api/v2/lipsync/stream (스트리밍 - 권장)

진행 상태를 실시간 SSE(Server-Sent Events)로 스트리밍합니다.

**Request**
```bash
curl -X POST http://localhost:5000/api/v2/lipsync/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 면접에 오신 것을 환영합니다.",
    "avatar": "talk_long",
    "resolution": "720p",
    "tts_engine": "elevenlabs"
  }'
```

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| text | string | O | 음성으로 변환할 텍스트 |
| avatar | string | △ | 아바타 이름 (avatar 또는 avatar_path 중 하나 필수) |
| avatar_path | string | △ | 아바타 파일 직접 경로 (.pkl) |
| resolution | string | X | 해상도 ("720p", "480p", "360p", 기본값: "720p") |
| frame_skip | integer | X | 프레임 스킵 (1=없음, 2=절반추론, 기본값: 1) |
| tts_engine | string | X | TTS 엔진 (기본값: "elevenlabs") |
| tts_voice | string | X | 음성 종류 (기본값: "Custom") |

**Response (SSE 이벤트)**

```
data: {"type": "status", "stage": "tts", "message": "TTS 음성 생성 중..."}

data: {"type": "status", "stage": "tts_done", "message": "TTS 완료 (3.5초)", "audio_url": "/assets/audio/tts_output.wav", "duration": 3.5, "elapsed": 2.1}

data: {"type": "status", "stage": "lipsync", "message": "립싱크 비디오 생성 중..."}

data: {"type": "done", "text": "안녕하세요...", "video_url": "/video/output_with_audio.mp4", "audio_url": "/assets/audio/tts_output.wav", "audio_duration": 3.5, "elapsed": {"tts": 2.1, "lipsync": 15.3, "total": 17.4}}
```

**이벤트 타입**

| type | 설명 |
|------|------|
| status | 진행 상태 (stage: tts, tts_done, lipsync) |
| done | 처리 완료 (video_url, audio_url, 소요시간 포함) |
| error | 오류 발생 |

**JavaScript 사용 예시**
```javascript
const response = await fetch('/api/v2/lipsync/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        text: '안녕하세요, 면접에 오신 것을 환영합니다.',
        avatar: 'talk_long',
        resolution: '720p'
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
                case 'status':
                    console.log(`[${data.stage}] ${data.message}`);
                    break;
                case 'done':
                    // 립싱크 비디오 재생
                    videoElement.src = data.video_url;
                    videoElement.play();
                    break;
                case 'error':
                    console.error('오류:', data.message);
                    break;
            }
        }
    }
}
```

**Python 사용 예시**
```python
import requests
import json

response = requests.post(
    'http://localhost:5000/api/v2/lipsync/stream',
    json={
        'text': '안녕하세요, 면접에 오신 것을 환영합니다.',
        'avatar': 'talk_long',
        'resolution': '720p'
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = json.loads(line_str[6:])

            if data['type'] == 'status':
                print(f"[{data['stage']}] {data['message']}")
            elif data['type'] == 'done':
                print(f"\n[완료] 비디오: {data['video_url']}")
                print(f"       소요시간: {data['elapsed']['total']}초")
            elif data['type'] == 'error':
                print(f"\n[오류] {data['message']}")
```

---

### POST /api/v2/lipsync (동기)

동기 방식으로 립싱크 비디오를 생성합니다. 생성이 완료될 때까지 대기합니다.

**Request**
```bash
curl -X POST http://localhost:5000/api/v2/lipsync \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 면접에 오신 것을 환영합니다.",
    "avatar": "talk_long",
    "resolution": "720p"
  }'
```

**Response**
```json
{
    "success": true,
    "video_path": "results/realtime/output_with_audio.mp4",
    "video_url": "/video/output_with_audio.mp4",
    "elapsed": 17.4
}
```

---

## 3. 기타 API

### GET /api/avatars

사용 가능한 아바타 목록을 조회합니다.

**Response**
```json
{
    "avatars": [
        {"id": "talk_long_720p", "name": "talk_long_720p", "path": "precomputed/720p/talk_long_720p_precomputed.pkl"},
        {"id": "talk_short_720p", "name": "talk_short_720p", "path": "precomputed/720p/talk_short_720p_precomputed.pkl"}
    ]
}
```

### GET /api/queue_status

생성 대기열 상태를 조회합니다.

**Response**
```json
{
    "queue_size": 0,
    "is_processing": false,
    "gpu_memory": {
        "total": 24576,
        "used": 8192,
        "free": 16384
    }
}
```

### POST /api/cancel/<task_id>

진행 중인 작업을 취소합니다.

---

## 에러 응답

모든 API는 오류 시 다음 형식으로 응답합니다:

```json
{
    "success": false,
    "error": "오류 메시지"
}
```

| HTTP 코드 | 설명 |
|----------|------|
| 400 | 잘못된 요청 (필수 파라미터 누락) |
| 404 | 리소스를 찾을 수 없음 (아바타 파일 등) |
| 500 | 서버 내부 오류 |
| 503 | 서비스 이용 불가 (모델 미로드) |

---

## 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| ELEVENLABS_API_KEY | ElevenLabs API 키 | - |
| ELEVENLABS_VOICE_ID | ElevenLabs 음성 ID | - |

---

## 서버 실행

```bash
cd realtime-interview-avatar
python app.py
```

서버가 시작되면 `http://localhost:5000`에서 API를 사용할 수 있습니다.
