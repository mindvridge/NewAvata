# 실시간 면접 아바타 시스템 API 문서

## 개요

이 문서는 실시간 면접 아바타 시스템의 REST API 사용 방법을 설명합니다.
Flask + WebSocket 기반으로 구현되어 있으며, TTS(Text-to-Speech)와 립싱크 비디오 생성을 지원합니다.

**Base URL**: `http://localhost:5000`

---

## 권장 아키텍처

### 클라이언트 LLM + 서버 립싱크 (권장)

클라이언트에서 LLM을 직접 호출하고, 생성된 텍스트를 서버로 보내 립싱크만 생성하는 방식입니다.

```
[클라이언트]                         [립싱크 서버]
    │                                    │
    ├── 1. LLM API 호출 ──────────►      │
    │   (OpenAI, Claude 등)              │
    │                                    │
    ◄── 2. LLM 응답 텍스트               │
    │                                    │
    ├── 3. POST /api/v2/lipsync/stream ─►│
    │   {"text": "LLM 응답 텍스트"}      │
    │                                    │
    ◄── 4. SSE: TTS 진행 상태            │
    ◄── 5. SSE: 립싱크 진행 상태         │
    ◄── 6. SSE: 완료 (video_url)         │
    │                                    │
    ├── 7. GET /video/xxx.mp4 ──────────►│
    ◄── 8. 립싱크 비디오 스트리밍        │
```

**장점:**
- LLM API 키를 클라이언트에서 관리 (보안)
- LLM 선택의 유연성 (OpenAI, Claude, 자체 LLM 등)
- 립싱크 서버는 GPU 리소스만 사용

---

## 목차

1. [서버 상태 확인](#1-서버-상태-확인)
2. [립싱크 비디오 생성 API](#2-립싱크-비디오-생성-api) ⭐ 권장
3. [TTS 생성 API](#3-tts-생성-api)
4. [채팅 API](#4-채팅-api) (서버 LLM 사용 시)
5. [채팅 + 립싱크 통합 API](#5-채팅--립싱크-통합-api) (서버 LLM 사용 시)
6. [기타 API](#6-기타-api)
7. [WebSocket 이벤트](#7-websocket-이벤트)

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
    "available_avatars": ["talk_long_720p", "talk_short_720p", "talk_long_480p", "talk_short_480p"],
    "tts_engines": ["elevenlabs"]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 서버 상태 ("ok") |
| models_loaded | boolean | 립싱크 모델 로드 여부 |
| available_avatars | array | 사용 가능한 아바타 목록 |
| tts_engines | array | 사용 가능한 TTS 엔진 목록 |

---

## 2. 채팅 API

### POST /api/chat

LLM을 통해 면접관 응답을 생성합니다.

**Stateless 모드 (권장)**: 클라이언트가 대화 기록(`history`)을 함께 전송하면 서버는 상태를 저장하지 않습니다.

**Request (Stateless 모드 - 권장)**
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "지원 동기를 말씀해주세요",
    "history": [
      {"role": "user", "content": "면접을 시작합니다."},
      {"role": "assistant", "content": "안녕하세요. 삼성전자 개발엔지니어링 직무 면접에 오신 것을 환영합니다."},
      {"role": "user", "content": "안녕하세요, 반갑습니다."},
      {"role": "assistant", "content": "네, 반갑습니다. 먼저 간단히 자기소개를 해주시겠어요?"}
    ],
    "org_type": "일반기업",
    "company_name": "삼성전자",
    "job_name": "개발엔지니어링"
  }'
```

**Request (Legacy 모드 - 하위 호환)**
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "자기소개를 해주세요",
    "org_type": "일반기업",
    "company_name": "삼성전자",
    "job_name": "개발엔지니어링"
  }'
```

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| message | string | O | 현재 사용자 메시지 |
| history | array | X | 이전 대화 기록 (Stateless 모드, 최대 50턴) |
| org_type | string | X | 분야 ("일반기업" 또는 "병원", 기본값: "일반기업") |
| company_name | string | X | 기업/병원명 |
| job_name | string | X | 직무명 (설정 시 질문셋 RAG 활성화) |

**history 배열 형식**
```json
[
  {"role": "user", "content": "사용자 메시지"},
  {"role": "assistant", "content": "면접관 응답"},
  ...
]
```

**org_type별 company_name 옵션**

| 분야 | 기업/병원명 |
|------|------------|
| 일반기업 | 삼성전자, SK하이닉스, 현대자동차, LG에너지솔루션, 삼성바이오로직스, 기아, LG전자, 포스코홀딩스, 네이버, 현대모비스 |
| 병원 | 서울아산병원, 삼성서울병원, 서울대병원, 세브란스병원, 분당서울대병원, 강남세브란스병원, 아주대병원, 서울성모병원, 인하대병원, 경희대병원 |

**org_type별 job_name 옵션**

| 분야 | 직무 |
|------|------|
| 일반기업 | 개발엔지니어링, 마케팅영업, 고객서비스CS, 인사HR, 운영관리, 기획전략, 재무회계, 품질관리QA, 글로벌마케팅, 법무컴플라이언스, 해외영업 |
| 병원 | 간호사, 국제의료관광코디네이터 |

**Response (성공)**
```json
{
    "response": "네, 좋습니다. 먼저 간단히 자기소개를 해주시겠어요?",
    "llm_source": "LLM API (vllm-qwen3-30b-a3b)",
    "llm_params": {
        "model": "vllm-qwen3-30b-a3b",
        "max_tokens": 200,
        "temperature": 0.7,
        "system_prompt": "당신은 면접관입니다...",
        "history_count": 4,
        "org_type": "일반기업",
        "job_name": "개발엔지니어링",
        "company_name": "삼성전자",
        "question_set_rag_enabled": true
    }
}
```

**Response 필드**

| 필드 | 타입 | 설명 |
|------|------|------|
| response | string | LLM 응답 텍스트 |
| llm_source | string | 사용된 LLM 소스 (예: "LLM API (vllm-qwen3-30b-a3b)", "OpenAI (gpt-4o-mini)") |
| llm_params | object | LLM에 전송된 파라미터 정보 |

**llm_params 필드**

| 필드 | 타입 | 설명 |
|------|------|------|
| model | string | 사용된 LLM 모델명 |
| max_tokens | number | 최대 토큰 수 |
| temperature | number | 응답 다양성 (0.0 ~ 1.0) |
| system_prompt | string | 시스템 프롬프트 전체 내용 |
| history_count | number | 포함된 대화 기록 수 |
| org_type | string | 분야 (일반기업/병원) |
| job_name | string | null | 직무명 (없으면 null) |
| company_name | string | null | 기업/병원명 (없으면 null) |
| question_set_rag_enabled | boolean | 질문셋 RAG 활성화 여부 |

**Response (실패)**
```json
{
    "error": "메시지가 필요합니다"
}
```

---

### POST /api/chat/stream (스트리밍)

LLM 응답을 토큰 단위로 실시간 스트리밍합니다. Server-Sent Events(SSE) 형식으로 응답합니다.

**Request**
```bash
curl -X POST http://localhost:5000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "자기소개를 해주세요",
    "history": [],
    "org_type": "일반기업",
    "company_name": "삼성전자",
    "job_name": "개발엔지니어링"
  }'
```

**Request Body** (동일한 파라미터)

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| message | string | O | 현재 사용자 메시지 |
| history | array | X | 이전 대화 기록 (최대 50턴) |
| org_type | string | X | 분야 (기본값: "일반기업") |
| company_name | string | X | 기업/병원명 |
| job_name | string | X | 직무명 |

**Response (SSE 이벤트)**

스트리밍 응답은 `text/event-stream` 형식으로 전송됩니다.

```
data: {"type": "token", "content": "안녕"}

data: {"type": "token", "content": "하세요"}

data: {"type": "token", "content": "."}

data: {"type": "done", "response": "안녕하세요.", "llm_source": "LLM API (vllm-qwen3-30b-a3b)", "llm_params": {...}}
```

**이벤트 타입**

| type | 설명 |
|------|------|
| token | 토큰 단위 응답 (content 필드에 텍스트 포함) |
| done | 스트리밍 완료 (전체 response, llm_source, llm_params 포함) |
| error | 오류 발생 (message 필드에 오류 메시지) |

**JavaScript 사용 예시**
```javascript
const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: '자기소개를 해주세요',
        history: conversationHistory,
        org_type: '일반기업',
        company_name: '삼성전자',
        job_name: '개발엔지니어링'
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let fullResponse = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'token') {
                // 실시간으로 UI 업데이트
                fullResponse += data.content;
                updateUI(fullResponse);
            } else if (data.type === 'done') {
                // 스트리밍 완료
                console.log('LLM Source:', data.llm_source);
                console.log('LLM Params:', data.llm_params);
            } else if (data.type === 'error') {
                console.error('Error:', data.message);
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
    'http://localhost:5000/api/chat/stream',
    json={
        'message': '자기소개를 해주세요',
        'history': [],
        'org_type': '일반기업',
        'company_name': '삼성전자',
        'job_name': '개발엔지니어링'
    },
    stream=True
)

full_response = ''
for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = json.loads(line_str[6:])

            if data['type'] == 'token':
                print(data['content'], end='', flush=True)
                full_response += data['content']
            elif data['type'] == 'done':
                print(f"\n\n[완료] LLM: {data['llm_source']}")
            elif data['type'] == 'error':
                print(f"\n[오류] {data['message']}")
```

---

## 3. TTS 생성 API

### POST /api/v2/tts

텍스트를 음성으로 변환합니다.

**Request**
```bash
curl -X POST http://localhost:5000/api/v2/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 반갑습니다.",
    "engine": "elevenlabs",
    "voice": "Custom"
  }'
```

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| text | string | O | 합성할 텍스트 |
| engine | string | X | TTS 엔진 (기본값: "elevenlabs") |
| voice | string | X | 음성 종류 (기본값: "Custom") |

**Response (성공)**
```json
{
    "success": true,
    "audio_path": "assets/audio/tts_api_output.wav",
    "audio_url": "/assets/audio/tts_api_output.wav",
    "duration": 3.5,
    "elapsed": 2.1
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| audio_path | string | 생성된 오디오 파일 경로 |
| audio_url | string | 오디오 파일 URL |
| duration | number | 오디오 길이 (초) |
| elapsed | number | 처리 시간 (초) |

---

## 4. 립싱크 비디오 생성 API ⭐ 권장

클라이언트에서 LLM으로 생성한 텍스트를 보내면, 서버에서 TTS + 립싱크 비디오를 생성합니다.

### POST /api/v2/lipsync/stream (스트리밍 - 권장)

텍스트를 입력받아 TTS + 립싱크 비디오를 생성하면서 진행 상태를 실시간 스트리밍합니다.

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
| text | string | O | 립싱크로 만들 텍스트 (LLM 응답) |
| avatar | string | △ | 아바타 이름 (avatar 또는 avatar_path 중 하나 필수) |
| avatar_path | string | △ | 아바타 파일 직접 경로 (.pkl) |
| resolution | string | X | 해상도 ("720p", "480p", "360p", 기본값: "720p") |
| frame_skip | integer | X | 프레임 스킵 (기본값: 1) |
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

**JavaScript 사용 예시 (권장 워크플로우)**
```javascript
// 1. 클라이언트에서 LLM 호출 (예: OpenAI)
const llmResponse = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: conversationHistory
    })
});
const llmData = await llmResponse.json();
const aiText = llmData.choices[0].message.content;

// 2. 립싱크 서버로 텍스트 전송 (스트리밍)
const response = await fetch('/api/v2/lipsync/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        text: aiText,
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
                    updateProgressUI(data.stage, data.message);
                    if (data.stage === 'tts_done') {
                        // TTS 완료 시 오디오 미리 로드 가능
                        preloadAudio(data.audio_url);
                    }
                    break;

                case 'done':
                    // 립싱크 비디오 재생
                    playVideo(data.video_url);
                    console.log('소요 시간:', data.elapsed.total, '초');
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
import openai

# 1. 클라이언트에서 LLM 호출
client = openai.OpenAI(api_key="your-api-key")
llm_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=conversation_history
)
ai_text = llm_response.choices[0].message.content

# 2. 립싱크 서버로 텍스트 전송 (스트리밍)
response = requests.post(
    'http://localhost:5000/api/v2/lipsync/stream',
    json={
        'text': ai_text,
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

텍스트를 입력받아 립싱크 비디오를 생성합니다. (동기 방식, 응답 대기)

**Request**
```bash
curl -X POST http://localhost:5000/api/v2/lipsync \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 면접에 오신 것을 환영합니다.",
    "avatar": "talk_long",
    "resolution": "720p",
    "frame_skip": 1,
    "tts_engine": "elevenlabs",
    "tts_voice": "Custom"
  }'
```

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| text | string | O | 합성할 텍스트 |
| avatar | string | △ | 아바타 이름 (avatar 또는 avatar_path 중 하나 필수) |
| avatar_path | string | △ | 아바타 파일 직접 경로 (.pkl) |
| resolution | string | X | 해상도 ("720p", "480p", "360p", 기본값: "720p") |
| frame_skip | integer | X | 프레임 스킵 (1=없음, 2=절반추론, 3=1/3추론, 기본값: 1) |
| tts_engine | string | X | TTS 엔진 (기본값: "elevenlabs") |
| tts_voice | string | X | 음성 종류 (기본값: "Custom") |

**Response (성공)**
```json
{
    "success": true,
    "video_path": "results/realtime/output_with_audio.mp4",
    "video_url": "/video/output_with_audio.mp4",
    "elapsed": 25.3
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| video_path | string | 생성된 비디오 파일 경로 |
| video_url | string | 비디오 스트리밍 URL |
| elapsed | number | 처리 시간 (초) |

**Response (실패)**
```json
{
    "success": false,
    "error": "립싱크 엔진이 로드되지 않았습니다"
}
```

---

## 5. 채팅 + 립싱크 통합 API

### POST /api/v2/chat_and_lipsync

사용자 메시지를 받아 LLM 응답 생성 후 립싱크 비디오까지 한 번에 생성합니다.

**Request**
```bash
curl -X POST http://localhost:5000/api/v2/chat_and_lipsync \
  -H "Content-Type: application/json" \
  -d '{
    "message": "자기소개를 해주세요",
    "avatar": "talk_long",
    "resolution": "720p",
    "frame_skip": 1,
    "tts_engine": "elevenlabs",
    "tts_voice": "Custom"
  }'
```

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| message | string | O | 사용자 메시지 |
| avatar | string | △ | 아바타 이름 (avatar 또는 avatar_path 중 하나 필수) |
| avatar_path | string | △ | 아바타 파일 직접 경로 (.pkl) |
| resolution | string | X | 해상도 ("720p", "480p", "360p", 기본값: "720p") |
| frame_skip | integer | X | 프레임 스킵 (1=없음, 2=절반추론, 3=1/3추론, 기본값: 1) |
| tts_engine | string | X | TTS 엔진 (기본값: "elevenlabs") |
| tts_voice | string | X | 음성 종류 (기본값: "Custom") |

**Response (성공)**
```json
{
    "success": true,
    "response": "네, 좋습니다. 먼저 간단히 자기소개를 해주시겠어요?",
    "video_path": "results/realtime/output_with_audio.mp4",
    "video_url": "/video/output_with_audio.mp4",
    "elapsed": 28.5
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| response | string | LLM이 생성한 응답 텍스트 |
| video_path | string | 생성된 비디오 파일 경로 |
| video_url | string | 비디오 스트리밍 URL |
| elapsed | number | 전체 처리 시간 (초) |

---

### POST /api/v2/chat_and_lipsync/stream (스트리밍)

LLM 응답을 토큰 단위로 스트리밍하면서, TTS와 립싱크 생성 진행 상태도 실시간으로 전송합니다.

**Request**
```bash
curl -X POST http://localhost:5000/api/v2/chat_and_lipsync/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "자기소개를 해주세요",
    "history": [],
    "org_type": "일반기업",
    "company_name": "삼성전자",
    "job_name": "개발엔지니어링",
    "avatar": "talk_long",
    "resolution": "720p",
    "tts_engine": "elevenlabs"
  }'
```

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| message | string | O | 사용자 메시지 |
| history | array | X | 이전 대화 기록 (최대 50턴) |
| org_type | string | X | 분야 (기본값: "일반기업") |
| company_name | string | X | 기업/병원명 |
| job_name | string | X | 직무명 |
| avatar | string | △ | 아바타 이름 (avatar 또는 avatar_path 중 하나 필수) |
| avatar_path | string | △ | 아바타 파일 직접 경로 (.pkl) |
| resolution | string | X | 해상도 (기본값: "720p") |
| frame_skip | integer | X | 프레임 스킵 (기본값: 1) |
| tts_engine | string | X | TTS 엔진 (기본값: "elevenlabs") |
| tts_voice | string | X | 음성 종류 (기본값: "Custom") |

**Response (SSE 이벤트)**

```
data: {"type": "status", "stage": "llm", "message": "LLM 응답 생성 중..."}

data: {"type": "token", "content": "안녕"}

data: {"type": "token", "content": "하세요"}

data: {"type": "llm_done", "response": "안녕하세요...", "llm_source": "LLM API (...)", "elapsed": 1.2}

data: {"type": "status", "stage": "tts", "message": "TTS 음성 생성 중..."}

data: {"type": "status", "stage": "tts_done", "message": "TTS 완료 (3.5초)", "audio_url": "/assets/audio/tts_output.wav", "duration": 3.5, "elapsed": 2.1}

data: {"type": "status", "stage": "lipsync", "message": "립싱크 비디오 생성 중..."}

data: {"type": "done", "response": "안녕하세요...", "video_url": "/video/output_with_audio.mp4", "audio_url": "/assets/audio/tts_output.wav", "elapsed": {"llm": 1.2, "tts": 2.1, "lipsync": 15.3, "total": 18.6}}
```

**이벤트 타입**

| type | 설명 |
|------|------|
| status | 진행 상태 알림 (stage: llm, tts, tts_done, lipsync) |
| token | LLM 토큰 단위 응답 |
| llm_done | LLM 응답 완료 (전체 텍스트 포함) |
| done | 전체 처리 완료 (비디오 URL, 오디오 URL, 소요 시간 포함) |
| error | 오류 발생 |

**done 이벤트 필드**

| 필드 | 타입 | 설명 |
|------|------|------|
| response | string | LLM 응답 텍스트 |
| llm_source | string | 사용된 LLM 소스 |
| video_url | string | 립싱크 비디오 URL |
| audio_url | string | TTS 오디오 URL |
| audio_duration | number | 오디오 길이 (초) |
| elapsed | object | 단계별 소요 시간 (llm, tts, lipsync, total) |

**JavaScript 사용 예시**
```javascript
const response = await fetch('/api/v2/chat_and_lipsync/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: '자기소개를 해주세요',
        history: conversationHistory,
        avatar: 'talk_long',
        resolution: '720p'
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let llmResponse = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
                case 'token':
                    // LLM 응답 실시간 표시
                    llmResponse += data.content;
                    updateChatUI(llmResponse);
                    break;

                case 'llm_done':
                    // LLM 완료, TTS 시작 전
                    console.log('LLM 완료:', data.elapsed, '초');
                    break;

                case 'status':
                    // 진행 상태 표시
                    updateProgressUI(data.stage, data.message);
                    if (data.stage === 'tts_done') {
                        // TTS 완료 시 오디오 미리 로드 가능
                        preloadAudio(data.audio_url);
                    }
                    break;

                case 'done':
                    // 모든 처리 완료
                    playVideo(data.video_url);
                    console.log('총 소요 시간:', data.elapsed.total, '초');
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
    'http://localhost:5000/api/v2/chat_and_lipsync/stream',
    json={
        'message': '자기소개를 해주세요',
        'history': [],
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

            if data['type'] == 'token':
                print(data['content'], end='', flush=True)
            elif data['type'] == 'llm_done':
                print(f"\n[LLM 완료] {data['elapsed']}초")
            elif data['type'] == 'status':
                print(f"\n[상태] {data['message']}")
            elif data['type'] == 'done':
                print(f"\n[완료] 비디오: {data['video_url']}")
                print(f"       소요시간: {data['elapsed']['total']}초")
            elif data['type'] == 'error':
                print(f"\n[오류] {data['message']}")
```

---

## 6. 기타 API

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

### GET /api/tts_engines

사용 가능한 TTS 엔진 목록을 조회합니다.

**Response**
```json
{
    "engines": [
        {"id": "elevenlabs", "name": "ElevenLabs", "voices": ["Custom", "Rachel", "Domi", "Bella"]}
    ]
}
```

### GET /api/llm_status

LLM 서버 상태를 확인합니다.

**Response**
```json
{
    "status": "online",
    "model": "vllm-qwen3-30b-a3b",
    "fallback": "OpenAI (gpt-4o-mini)"
}
```

### POST /api/clear_history

대화 기록을 초기화합니다. (Legacy 모드에서만 필요, Stateless 모드에서는 클라이언트가 history를 관리)

**Response**
```json
{
    "status": "cleared"
}
```

### GET /api/queue_status

현재 생성 큐 상태를 조회합니다.

**Response**
```json
{
    "queue_length": 0,
    "processing_count": 1,
    "max_concurrent": 3,
    "processing": [
        {"request_id": "abc123", "sid": "session_id", "started_at": 1705900000.0}
    ],
    "queue": []
}
```

### POST /api/cancel

현재 진행 중인 생성을 취소합니다.

**Request Body**
```json
{
    "sid": "session_id"
}
```

**Response**
```json
{
    "status": "cancelled"
}
```

---

## 7. WebSocket 이벤트

WebSocket을 통해 실시간 진행 상황을 수신할 수 있습니다.

### 연결

```javascript
const socket = io('http://localhost:5000');
```

### 수신 이벤트

| 이벤트 | 설명 | 데이터 |
|--------|------|--------|
| progress | 생성 진행률 | `{progress: 50, message: "프레임 처리 중..."}` |
| generation_complete | 생성 완료 | `{video_url: "/video/output.mp4", elapsed: 25.3}` |
| error | 오류 발생 | `{error: "오류 메시지"}` |
| queue_position | 대기열 위치 | `{position: 2, estimated_wait: 60}` |

### 송신 이벤트

| 이벤트 | 설명 | 데이터 |
|--------|------|--------|
| generate | 립싱크 생성 요청 | `{text: "...", avatar: "...", resolution: "720p"}` |
| cancel | 생성 취소 | - |

---

## 8. 에러 코드

| HTTP 코드 | 설명 |
|-----------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 (필수 파라미터 누락) |
| 404 | 리소스를 찾을 수 없음 (아바타 파일 없음) |
| 500 | 서버 내부 오류 |
| 503 | 서비스 사용 불가 (모델 미로드) |

---

## 9. Python 예제 코드

### Stateless 모드 (권장) - 클라이언트가 대화 기록 관리

```python
import requests

BASE_URL = "http://localhost:5000"

# 대화 기록을 클라이언트에서 관리
conversation_history = []

def chat(message, org_type="일반기업", company_name="", job_name=""):
    """면접관과 대화 (Stateless 모드)"""
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": message,
            "history": conversation_history,  # 이전 대화 기록 전송
            "org_type": org_type,
            "company_name": company_name,
            "job_name": job_name
        }
    ).json()

    if "response" in response:
        # 대화 기록에 추가 (클라이언트에서 관리)
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": response["response"]})
        return response["response"]
    else:
        return None

# 1. 서버 상태 확인
status = requests.get(f"{BASE_URL}/api/v2/status").json()
print(f"모델 로드 상태: {status['models_loaded']}")

# 2. 면접 시작
response1 = chat(
    "면접을 시작합니다.",
    org_type="일반기업",
    company_name="삼성전자",
    job_name="개발엔지니어링"
)
print(f"면접관: {response1}")

# 3. 자기소개
response2 = chat("안녕하세요. 저는 컴퓨터공학을 전공한 홍길동입니다.")
print(f"면접관: {response2}")

# 4. 추가 질문에 답변 (이전 대화 컨텍스트 유지됨)
response3 = chat("네, 제가 지원한 이유는 반도체 기술에 관심이 많기 때문입니다.")
print(f"면접관: {response3}")

# 대화 기록 확인
print(f"\n총 대화 턴 수: {len(conversation_history) // 2}")

# 새 면접 시작 시 기록 초기화
conversation_history.clear()
```

### 립싱크 비디오 생성 예제

```python
# TTS 생성
tts_response = requests.post(
    f"{BASE_URL}/api/v2/tts",
    json={
        "text": "안녕하세요, 면접에 오신 것을 환영합니다.",
        "engine": "elevenlabs"
    }
).json()
print(f"오디오 URL: {tts_response['audio_url']}")

# 립싱크 비디오 생성
lipsync_response = requests.post(
    f"{BASE_URL}/api/v2/lipsync",
    json={
        "text": "안녕하세요, 면접에 오신 것을 환영합니다.",
        "avatar": "talk_long",
        "resolution": "720p"
    }
).json()
print(f"비디오 URL: {lipsync_response['video_url']}")
```

---

## 10. 환경 변수 설정

서버 실행 전 `.env` 파일에 다음 환경 변수를 설정합니다.

```env
# LLM API 설정
LLM_API_URL=https://api.mindprep.co.kr/v1/chat/completions
LLM_MODEL=vllm-qwen3-30b-a3b

# OpenAI API (fallback용)
OPENAI_API_KEY=sk-xxx

# ElevenLabs TTS
ELEVENLABS_API_KEY=xxx

# MuseTalk 경로
MUSETALK_PATH=c:/NewAvata/NewAvata/MuseTalk
```

---

## 11. 서버 실행

```bash
cd realtime-interview-avatar
python app.py
```

기본 포트: 5000

---

## 버전 정보

- API 버전: v2.0
- 최종 업데이트: 2025-01-22
