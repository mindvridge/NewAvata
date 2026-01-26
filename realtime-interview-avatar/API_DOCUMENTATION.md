# 립싱크 아바타 API 문서

텍스트를 입력받아 TTS(Text-to-Speech)와 립싱크 비디오를 생성하는 REST API입니다.

**Base URL**:
- 로컬: `http://localhost:5000`
- 외부: `LIPSYNC_BASE_URL` 환경변수 설정 시 전체 URL 반환

---

## 사용 모드

| 모드 | 용도 | persist | 파일 보관 |
|------|------|---------|-----------|
| **실시간 대화** | 면접 연습, 일회성 재생 | `false` (기본값) | 10분 후 자동 삭제 |
| **녹화** | 영상 저장, 다운로드 | `true` | 영구 보관 |

---

## 빠른 시작

```bash
# 1. 서버 상태 확인
curl http://localhost:5000/api/v2/status

# 2. 실시간 대화 모드 (최고품질, 임시 저장)
curl -X POST http://localhost:5000/api/v2/lipsync/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "안녕하세요", "quality": "high", "persist": false}'

# 3. 녹화 모드 (최고품질, 영구 저장)
curl -X POST http://localhost:5000/api/v2/lipsync/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "안녕하세요", "quality": "high", "persist": true}'

# 4. 빠른 미리보기 (저품질)
curl -X POST http://localhost:5000/api/v2/lipsync/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "안녕하세요", "quality": "low"}'

# 5. 생성된 비디오 다운로드
curl -O http://localhost:5000/video/{filename}.mp4
```

---

## API 목록

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| GET | `/api/v2/status` | 서버 상태 확인 |
| POST | `/api/v2/lipsync/stream` | 립싱크 생성 (SSE 스트리밍) |
| POST | `/api/v2/lipsync` | 립싱크 생성 (동기) |
| POST | `/api/video/save` | 비디오 영구 저장 |
| GET | `/api/tts_engines` | TTS 엔진 목록 |
| GET | `/api/avatars` | 아바타 목록 |
| GET | `/api/queue_status` | 대기열/GPU 상태 |
| GET | `/video/{filename}` | 비디오 파일 제공 |
| GET | `/assets/{path}` | 에셋 파일 제공 |

---

## 1. 서버 상태 확인

### GET /api/v2/status

```bash
curl http://localhost:5000/api/v2/status
```

**Response**
```json
{
    "status": "ok",
    "models_loaded": true,
    "available_avatars": ["new_talk_short_720p", "new_talk_short_480p", "new_talk_short_360p"],
    "tts_engines": ["cosyvoice", "elevenlabs"]
}
```

---

## 2. 립싱크 비디오 생성

### POST /api/v2/lipsync/stream (스트리밍 - 권장)

진행 상태를 실시간 SSE(Server-Sent Events)로 스트리밍합니다.

**Request**
```bash
# quality 프리셋 사용 (권장)
curl -X POST http://localhost:5000/api/v2/lipsync/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 면접에 오신 것을 환영합니다.",
    "avatar": "auto",
    "quality": "high",
    "persist": false
  }'

# 또는 개별 파라미터 지정
curl -X POST http://localhost:5000/api/v2/lipsync/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 면접에 오신 것을 환영합니다.",
    "avatar": "auto",
    "resolution": "720p",
    "frame_skip": 1,
    "persist": false
  }'
```

**Request Body**

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| text | string | ✅ | - | 음성으로 변환할 텍스트 |
| avatar | string | - | "auto" | 아바타 ("auto", "new_talk_short", "new_talk_long") |
| **quality** | string | - | "high" | **품질 프리셋 ("high", "medium", "low")** |
| resolution | string | - | - | 해상도 ("720p", "480p", "360p") - quality 우선 |
| frame_skip | integer | - | - | 프레임 스킵 (1, 2, 3) - quality 우선 |
| tts_engine | string | - | "cosyvoice" | TTS 엔진 |
| tts_voice | string | - | "Custom" | 음성 종류 |
| output_format | string | - | "mp4" | 출력 포맷 ("mp4", "webm") |
| **persist** | boolean | - | **false** | **영구 저장 여부 (녹화 모드)** |

> **quality 프리셋** (권장):
> | quality | frame_skip | resolution | 설명 |
> |---------|------------|------------|------|
> | `high` | 1 | 720p | 최고품질 (모든 프레임 추론) |
> | `medium` | 2 | 480p | 균형 (2프레임당 1추론) |
> | `low` | 3 | 360p | 최고속 (3프레임당 1추론) |
>
> `resolution` 또는 `frame_skip`을 직접 지정하면 해당 값이 우선 적용됩니다.

> **persist 파라미터**:
> - `false` (기본값): 실시간 대화용. 10분 후 자동 삭제
> - `true`: 녹화용. 영구 보관 (`results/saved/` 폴더)

> **아바타 자동 선택**: `avatar: "auto"` 사용시 오디오 길이에 따라 자동으로 new_talk_short(5초) 또는 new_talk_long(10초)이 선택됩니다.

**Response (SSE)**

```
data: {"type": "status", "stage": "tts", "message": "TTS 음성 생성 중..."}

data: {"type": "status", "stage": "tts_done", "message": "TTS 완료", "audio_url": "/assets/audio/tts_output.wav", "duration": 3.5}

data: {"type": "status", "stage": "lipsync", "message": "립싱크 비디오 생성 중..."}

data: {"type": "done", "video_url": "https://avatar.mindprep.co.kr/video/안녕하세요_a1b2c3d4.mp4", "audio_duration": 3.5, "persistent": false, "elapsed": {"tts": 1.2, "lipsync": 8.5, "total": 9.7}}
```

> **참고**: `LIPSYNC_BASE_URL` 미설정 시 `video_url`은 상대 경로 (`/video/...`)로 반환됩니다.

**이벤트 타입**

| type | 설명 |
|------|------|
| status | 진행 상태 (stage: tts, tts_done, lipsync) |
| done | 처리 완료 |
| error | 오류 발생 |

---

### POST /api/v2/lipsync (동기)

생성 완료까지 대기 후 결과 반환합니다.

**Request**
```bash
# 실시간 대화 모드 (임시)
curl -X POST http://localhost:5000/api/v2/lipsync \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요",
    "resolution": "720p",
    "persist": false
  }'

# 녹화 모드 (영구 저장)
curl -X POST http://localhost:5000/api/v2/lipsync \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요",
    "resolution": "720p",
    "persist": true
  }'
```

**Response**
```json
{
    "success": true,
    "video_path": "results/realtime/안녕하세요_a1b2c3d4.mp4",
    "video_url": "/video/안녕하세요_a1b2c3d4.mp4",
    "elapsed": 9.7,
    "persistent": false
}

// LIPSYNC_BASE_URL 설정 시
{
    "success": true,
    "video_path": "results/realtime/안녕하세요_a1b2c3d4.mp4",
    "video_url": "https://avatar.mindprep.co.kr/video/안녕하세요_a1b2c3d4.mp4",
    "elapsed": 9.7,
    "persistent": false
}
```

> **파일명 형식**: `{텍스트앞10글자}_{uuid8자리}.mp4` (예: `안녕하세요_a1b2c3d4.mp4`)

---

## 3. 비디오 영구 저장

### POST /api/video/save

이미 생성된 비디오를 영구 저장 폴더로 복사합니다.
(실시간 대화 모드에서 생성 후 나중에 저장하고 싶을 때 사용)

**Request**
```bash
curl -X POST http://localhost:5000/api/video/save \
  -H "Content-Type: application/json" \
  -d '{"video_url": "/video/output_xxx.mp4"}'
```

**Response**
```json
{
    "success": true,
    "saved_url": "/video/output_xxx.mp4",
    "message": "파일이 영구 저장되었습니다"
}
```

---

## 4. TTS 엔진 목록

### GET /api/tts_engines

```bash
curl http://localhost:5000/api/tts_engines
```

**Response**
```json
[
    {
        "id": "cosyvoice",
        "name": "CosyVoice (로컬)",
        "voices": ["Custom"],
        "available": true
    },
    {
        "id": "elevenlabs",
        "name": "ElevenLabs",
        "voices": ["Rachel", "Bella", "Antoni"],
        "available": true
    }
]
```

---

## 5. 아바타 목록

### GET /api/avatars

```bash
curl http://localhost:5000/api/avatars
```

**Response**
```json
{
    "avatars": [
        {
            "id": "new_talk_short_720p",
            "name": "new_talk_short_720p",
            "path": "precomputed/720p/new_talk_short_720p_precomputed.pkl"
        }
    ]
}
```

---

## 6. 대기열/GPU 상태

### GET /api/queue_status

```bash
curl http://localhost:5000/api/queue_status
```

**Response**
```json
{
    "queue_size": 0,
    "is_processing": false,
    "gpu": {
        "memory_total_mb": 24576,
        "memory_used_mb": 8192,
        "memory_free_mb": 16384,
        "utilization_percent": 45
    }
}
```

---

## 클라이언트 코드 예시

### JavaScript (브라우저)

```javascript
// 실시간 대화 모드 (임시)
async function generateLipsyncConversation(text) {
    const response = await fetch('/api/v2/lipsync/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text,
            resolution: '720p',
            persist: false  // 10분 후 자동 삭제
        })
    });
    return processSSE(response);
}

// 녹화 모드 (영구 저장)
async function generateLipsyncRecording(text) {
    const response = await fetch('/api/v2/lipsync/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text,
            resolution: '720p',
            persist: true  // 영구 보관
        })
    });
    return processSSE(response);
}

// SSE 응답 처리
async function processSSE(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));

                if (data.type === 'status') {
                    console.log(`[${data.stage}] ${data.message}`);
                } else if (data.type === 'done') {
                    console.log('완료:', data.video_url, '영구저장:', data.persistent);
                    return data;
                } else if (data.type === 'error') {
                    throw new Error(data.message);
                }
            }
        }
    }
}

// 사용 예시
const videoUrl = await generateLipsyncConversation('안녕하세요');
document.querySelector('video').src = videoUrl.video_url;
```

### Python

```python
import requests
import json

def generate_lipsync(
    text: str,
    persist: bool = False,
    base_url: str = "http://localhost:5000"
):
    """
    립싱크 비디오 생성 (스트리밍)

    Args:
        text: 음성으로 변환할 텍스트
        persist: True면 영구 저장 (녹화 모드), False면 임시 저장 (대화 모드)
        base_url: API 서버 주소
    """
    response = requests.post(
        f"{base_url}/api/v2/lipsync/stream",
        json={
            "text": text,
            "resolution": "720p",
            "persist": persist
        },
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data = json.loads(line_str[6:])

                if data['type'] == 'status':
                    print(f"[{data.get('stage', '')}] {data['message']}")
                elif data['type'] == 'done':
                    print(f"완료: {data['video_url']} (영구저장: {data.get('persistent', False)})")
                    return data
                elif data['type'] == 'error':
                    raise Exception(data['message'])

# 사용 예시
if __name__ == "__main__":
    # 실시간 대화 모드 (임시)
    result = generate_lipsync("안녕하세요", persist=False)
    print(f"비디오 URL: {result['video_url']}")

    # 녹화 모드 (영구 저장)
    result = generate_lipsync("면접 영상입니다.", persist=True)
    print(f"비디오 URL: {result['video_url']} (영구 저장됨)")
```

### cURL (Bash)

```bash
#!/bin/bash

# 실시간 대화 모드 (임시)
generate_conversation() {
    local text="$1"
    curl -s -X POST "http://localhost:5000/api/v2/lipsync" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\", \"resolution\": \"720p\", \"persist\": false}"
}

# 녹화 모드 (영구 저장)
generate_recording() {
    local text="$1"
    curl -s -X POST "http://localhost:5000/api/v2/lipsync" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\", \"resolution\": \"720p\", \"persist\": true}"
}

# 비디오 다운로드
download_video() {
    local url="$1"
    local output="$2"
    curl -o "$output" "http://localhost:5000$url"
}

# 사용 예시
result=$(generate_recording "안녕하세요")
video_url=$(echo $result | jq -r '.video_url')
download_video "$video_url" "output.mp4"
```

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
| 404 | 리소스를 찾을 수 없음 |
| 500 | 서버 내부 오류 |
| 503 | 서비스 이용 불가 (모델 미로드) |

---

## 파일 저장 위치

| 폴더 | 용도 | 자동 삭제 |
|------|------|-----------|
| `results/realtime/` | 임시 파일 (persist=false) | 10분 후 |
| `results/saved/` | 영구 저장 (persist=true) | 삭제 안 함 |

---

## 환경 변수

| 변수명 | 설명 | 필수 | 예시 |
|--------|------|------|------|
| **LIPSYNC_BASE_URL** | **외부 접근용 베이스 URL** | 외부 연동시 | `https://avatar.mindprep.co.kr` |
| ELEVENLABS_API_KEY | ElevenLabs API 키 | ElevenLabs 사용시 | - |
| ELEVENLABS_VOICE_ID | ElevenLabs 음성 ID | ElevenLabs 사용시 | - |
| COSYVOICE_API_URL | CosyVoice 서버 URL | CosyVoice 사용시 | `http://172.16.10.200:5000` |

### LIPSYNC_BASE_URL 설정

외부 서비스(Railway, Cloudflare Tunnel 등)에서 API를 호출할 때 전체 URL을 반환받으려면 설정이 필요합니다.

```bash
# .env 파일에 추가
LIPSYNC_BASE_URL=https://avatar.mindprep.co.kr
```

**설정 전 응답:**
```json
{ "video_url": "/video/output_a1b2c3d4.mp4" }
```

**설정 후 응답:**
```json
{ "video_url": "https://avatar.mindprep.co.kr/video/output_a1b2c3d4.mp4" }
```

---

## 트러블슈팅

### 비디오 다운로드 시 404 에러

**원인:**
1. 파일이 10분 TTL 후 자동 삭제됨
2. `LIPSYNC_BASE_URL` 미설정으로 잘못된 URL 사용
3. 외부 서비스에서 상대 경로를 올바르게 처리하지 못함

**해결:**
1. `LIPSYNC_BASE_URL` 환경변수 설정
2. `persist: true` 사용으로 영구 저장
3. API 응답 즉시 비디오 다운로드

**서버 로그 확인:**
```
[설정] LIPSYNC_BASE_URL = 'https://avatar.mindprep.co.kr'
[API 응답] video_url = https://avatar.mindprep.co.kr/video/안녕하세요_a1b2c3d4.mp4
[비디오 요청] filename = 안녕하세요_a1b2c3d4.mp4
[비디오 서빙] results/realtime/안녕하세요_a1b2c3d4.mp4 (임시)
```

### 비디오 파일 자동 삭제 설정

```python
# app.py 상단에서 설정 변경 가능
VIDEO_CLEANUP_ENABLED = True   # 자동 정리 활성화/비활성화
VIDEO_CLEANUP_TTL = 600        # TTL (초) - 기본 10분
VIDEO_CLEANUP_INTERVAL = 60    # 정리 주기 (초)
```

---

## 서버 실행

```bash
cd realtime-interview-avatar
python app.py
```

서버가 `http://localhost:5000`에서 시작됩니다.

**서버 시작 로그:**
```
[설정] LIPSYNC_BASE_URL = 'https://avatar.mindprep.co.kr'
[자동 정리] 백그라운드 스레드 시작 (TTL: 600초, 주기: 60초)
...
URL: http://localhost:5000
```
