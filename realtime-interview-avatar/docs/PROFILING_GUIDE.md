# 성능 프로파일링 가이드

실시간 면접 아바타 시스템의 성능을 측정하고 최적화하는 방법을 안내합니다.

## 목차

- [개요](#개요)
- [빠른 시작](#빠른-시작)
- [측정 메트릭](#측정-메트릭)
- [성능 목표](#성능-목표)
- [프로파일링 결과 해석](#프로파일링-결과-해석)
- [최적화 가이드](#최적화-가이드)
- [고급 사용법](#고급-사용법)

---

## 개요

`scripts/profile.py`는 각 컴포넌트(STT, LLM, TTS, Avatar)의 성능을 종합적으로 측정하고 병목 구간을 식별하는 도구입니다.

### 주요 기능

- ⏱️ **레이턴시 벤치마크**: P50/P95/P99 레이턴시 측정
- 💾 **메모리 프로파일링**: RSS/VMS 추적, 메모리 누수 탐지
- 🎮 **GPU 모니터링**: VRAM 사용량, 활용률, 온도
- 📊 **시각화**: matplotlib 기반 차트 자동 생성
- 📝 **JSON 리포트**: 구조화된 성능 데이터 저장

---

## 빠른 시작

### 1. 의존성 설치

```bash
pip install numpy psutil matplotlib

# GPU 모니터링 (선택)
pip install nvidia-ml-py3
```

### 2. 전체 프로파일링 실행

```bash
python scripts/profile.py
```

실행 결과:
```
╔════════════════════════════════════════════════════════════╗
║            🔍 성능 프로파일링 시작                         ║
╚════════════════════════════════════════════════════════════╝

📊 STT 프로파일링 중...
  진행: 5/20 (평균: 145.23ms)
  진행: 10/20 (평균: 142.67ms)
  진행: 15/20 (평균: 140.12ms)
  진행: 20/20 (평균: 138.45ms)

📊 LLM 프로파일링 중...
  진행: 2/10 (TTFT 평균: 185.34ms)
  진행: 4/10 (TTFT 평균: 180.12ms)
  ...

============================================================
📊 프로파일링 결과
============================================================

【STT】
  레이턴시:
    • 평균:  138.45ms
    • 중앙값: 135.23ms
    • P95:   165.34ms
    • P99:   178.12ms
    • 표준편차: 12.34ms
  메모리:
    • 사용 전: 1250.5MB
    • 사용 후: 1275.3MB
    • 증가량:  24.8MB
  처리량: 7.2 ops/s

【LLM】
  ...
```

### 3. 결과 확인

프로파일링 완료 후 생성되는 파일들:

```
profile_results/
├── profile_20250105_120000.json     # JSON 리포트
├── latency_20250105_120000.png      # 레이턴시 차트
├── memory_20250105_120000.png       # 메모리 차트
└── gpu_20250105_120000.png          # GPU 차트
```

---

## 측정 메트릭

### 1. 레이턴시 메트릭

각 컴포넌트의 응답 시간을 측정합니다.

| 메트릭 | 설명 |
|--------|------|
| **평균 (Mean)** | 모든 샘플의 평균 레이턴시 |
| **중앙값 (Median)** | 50% 지점 레이턴시 (P50) |
| **P95** | 95% 지점 레이턴시 (상위 5% 제외) |
| **P99** | 99% 지점 레이턴시 (상위 1% 제외) |
| **표준편차 (Std)** | 레이턴시 변동 폭 |

**주요 지표**:
- **STT**: 오디오 입력부터 텍스트 출력까지
- **LLM TTFT**: Time To First Token (첫 토큰 생성까지)
- **TTS TTFB**: Time To First Byte (첫 오디오 청크까지)
- **Avatar**: 프레임당 렌더링 시간

### 2. 메모리 메트릭

| 메트릭 | 설명 |
|--------|------|
| **RSS** | Resident Set Size (물리 메모리) |
| **VMS** | Virtual Memory Size (가상 메모리) |
| **Delta** | 컴포넌트 실행 전후 메모리 증가량 |

**메모리 누수 판단**:
- Delta > 100MB: 의심
- Delta > 200MB: 확실

### 3. GPU 메트릭

| 메트릭 | 설명 |
|--------|------|
| **VRAM 사용량** | GPU 메모리 사용 (MB) |
| **VRAM 사용률** | 전체 VRAM 대비 사용 비율 (%) |
| **GPU 활용률** | GPU 코어 사용률 (%) |
| **온도** | GPU 온도 (°C) |

**병목 판단**:
- VRAM 사용률 > 90%: 메모리 부족
- GPU 활용률 < 30%: CPU 병목 가능성
- 온도 > 85°C: 쓰로틀링 가능성

---

## 성능 목표

### 레이턴시 목표

| 컴포넌트 | 목표 (P95) | 이유 |
|----------|-----------|------|
| **STT** | < 100ms | 실시간 대화 위해 빠른 인식 필수 |
| **LLM TTFT** | < 200ms | 첫 응답까지 지연 최소화 |
| **TTS TTFB** | < 200ms | 오디오 스트리밍 시작 지연 최소화 |
| **Avatar** | < 50ms/frame | 25 FPS 유지 (40ms + 여유) |

### 전체 파이프라인 목표

- **End-to-End 레이턴시**: < 500ms (TTS 스트리밍 제외)
- **처리량**: > 5 requests/sec (동시 세션)
- **메모리 사용량**: < 4GB (세션당)
- **GPU 메모리**: < 6GB (RTX 4090 기준)

---

## 프로파일링 결과 해석

### 예제 리포트

```json
{
  "timestamp": "2025-01-05T12:00:00",
  "duration_sec": 45.67,
  "components": {
    "STT": {
      "latency": {
        "mean": 138.45,
        "p95": 165.34,
        "p99": 178.12
      },
      "memory_delta_mb": 24.8,
      "throughput_per_sec": 7.2
    },
    "LLM": {
      "latency": {
        "mean": 180.67,
        "p95": 220.45,
        "p99": 250.12
      },
      "memory_delta_mb": 50.2,
      "throughput_per_sec": 5.5
    },
    ...
  },
  "bottlenecks": [
    "LLM: P95 220.45ms (목표 200ms 대비 10.2% 초과)"
  ],
  "recommendations": [
    "LLM: 더 빠른 모델 사용 (GPT-4o → GPT-4o-mini)",
    "TTS: 공통 질문 캐싱 활성화 (TTSCache)"
  ]
}
```

### 해석 가이드

#### ✅ 정상 (Good)

```
【STT】
  레이턴시:
    • P95: 95.23ms    ✓ 목표 100ms 달성
  메모리:
    • 증가량: 20.5MB  ✓ 합리적
```

**의미**: STT가 목표 레이턴시를 달성하고 메모리도 안정적

#### ⚠️ 주의 (Warning)

```
【LLM】
  레이턴시:
    • P95: 220.45ms   ⚠ 목표 200ms 초과
    • 표준편차: 85.3ms  ⚠ 높은 변동성
```

**의미**: LLM이 약간 느리고 레이턴시가 불안정함

#### ❌ 문제 (Critical)

```
【Avatar】
  레이턴시:
    • P95: 120.45ms   ❌ 목표 50ms 대폭 초과
  GPU:
    • VRAM: 95.2%     ❌ 메모리 부족
```

**의미**: Avatar 렌더링이 매우 느리고 GPU 메모리 부족

---

## 최적화 가이드

### STT 최적화

#### 문제: P95 > 100ms

**원인**:
- VAD 설정 부적절 (청크 크기 너무 큼)
- 네트워크 레이턴시
- API 서버 위치

**해결책**:

1. **VAD 설정 조정**
```python
# src/stt/vad_config.py
VAD_PRESETS = {
    "INTERVIEW_FAST": SileroVADConfig(
        threshold=0.4,          # 더 민감하게 (0.5 → 0.4)
        min_speech_duration_ms=200,  # 더 짧게 (250 → 200)
        max_speech_duration_ms=5000,  # 더 짧게 (10000 → 5000)
    )
}
```

2. **Deepgram 지역 선택**
```python
# .env
DEEPGRAM_REGION=us-west-1  # 가장 가까운 지역
```

3. **대안 STT 사용**
- Whisper (로컬): 네트워크 레이턴시 0ms, but GPU 필요
- AssemblyAI: Deepgram 대안

### LLM 최적화

#### 문제: TTFT > 200ms

**원인**:
- 모델 크기 (GPT-4o)
- 긴 프롬프트
- 높은 temperature

**해결책**:

1. **더 빠른 모델 사용**
```python
# src/llm/interviewer_agent.py
MODEL_CONFIGS = {
    "fast": "gpt-4o-mini",      # TTFT ~100ms
    "balanced": "gpt-4o",        # TTFT ~150ms
    "quality": "gpt-4-turbo",    # TTFT ~250ms
}
```

2. **프롬프트 최적화**
```python
# 긴 프롬프트 → 짧은 프롬프트
# Before (500 tokens)
prompt = f"You are an AI interviewer... [긴 설명]"

# After (200 tokens)
prompt = f"AI interviewer. Ask about: {topic}. Be concise."
```

3. **Streaming 최적화**
```python
# temperature 낮추기 (변동성 감소)
temperature=0.7  # → 0.5
```

### TTS 최적화

#### 문제: TTFB > 200ms

**원인**:
- ElevenLabs API 레이턴시
- 캐싱 미사용
- 긴 텍스트

**해결책**:

1. **캐싱 활성화**
```python
# src/tts/cache.py
cache = TTSCache(
    max_size=1000,
    enable_disk_cache=True,
)

# 공통 질문 prewarming
COMMON_QUESTIONS = [
    "자기소개 부탁드립니다",
    "경력에 대해 말씀해주세요",
    ...
]
await cache.prewarm(COMMON_QUESTIONS)
```

2. **청크 크기 최적화**
```python
# 긴 텍스트 → 문장 단위 스트리밍
async def stream_by_sentence(text: str):
    sentences = text.split('. ')
    for sentence in sentences:
        async for chunk in tts.stream_audio(sentence):
            yield chunk
```

3. **대안 TTS 사용**
- EdgeTTS (무료): TTFB ~50ms, but 품질 낮음
- Naver Clova: TTFB ~100ms, 한국어 최적

### Avatar 최적화

#### 문제: 프레임당 > 50ms

**원인**:
- Face enhancement 활성화
- GPU 메모리 부족
- 배치 사이즈 너무 큼

**해결책**:

1. **Face enhancement 비활성화**
```python
# src/avatar/musetalk_wrapper.py
config = MuseTalkConfig(
    enable_face_enhancement=False,  # True → False (30ms 절감)
)
```

2. **GPU 메모리 최적화**
```python
# 배치 사이즈 감소
batch_size=1  # 4 → 1

# Mixed precision
torch.set_default_dtype(torch.float16)
```

3. **GPU 업그레이드**
- RTX 3090 → RTX 4090: 2배 빠름
- RTX 4090 → A100: 1.5배 빠름

### 메모리 누수 해결

#### 문제: Delta > 100MB

**원인**:
- 캐시 무제한 증가
- GPU 텐서 미해제
- 순환 참조

**해결책**:

1. **캐시 크기 제한**
```python
# LRU 캐시
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_function(...):
    ...
```

2. **GPU 메모리 정리**
```python
# 매 N번째 요청마다
if request_count % 10 == 0:
    torch.cuda.empty_cache()
    gc.collect()
```

3. **메모리 프로파일링**
```bash
# memory_profiler 사용
pip install memory_profiler
python -m memory_profiler scripts/profile.py
```

---

## 고급 사용법

### 1. 특정 컴포넌트만 프로파일링

```bash
# STT와 LLM만
python scripts/profile.py --components stt llm

# Avatar만
python scripts/profile.py --components avatar

# TTS만 (샘플 50개)
python scripts/profile.py --components tts --samples 50
```

### 2. 출력 디렉토리 지정

```bash
python scripts/profile.py --output-dir results/performance
```

### 3. CI/CD 통합

```yaml
# .github/workflows/performance.yml
name: Performance Benchmark

on:
  schedule:
    - cron: '0 0 * * 0'  # 매주 일요일

jobs:
  benchmark:
    runs-on: ubuntu-latest-gpu
    steps:
      - uses: actions/checkout@v3

      - name: Run profiler
        run: |
          python scripts/profile.py --output-dir artifacts

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: artifacts/
```

### 4. 성능 회귀 탐지

```python
# scripts/compare_profiles.py
import json

def compare_profiles(baseline_path, current_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    for component in baseline['components']:
        baseline_p95 = baseline['components'][component]['latency']['p95']
        current_p95 = current['components'][component]['latency']['p95']

        regression = (current_p95 / baseline_p95 - 1) * 100

        if regression > 10:  # 10% 이상 느려짐
            print(f"⚠️ {component}: {regression:.1f}% 성능 저하")

# 사용
compare_profiles(
    'baseline_20250101.json',
    'profile_results/profile_20250105.json'
)
```

### 5. 연속 모니터링

```bash
# 1시간 동안 매 10분마다 프로파일링
while true; do
    python scripts/profile.py
    sleep 600  # 10분
done
```

---

## 베스트 프랙티스

### 1. 정기 프로파일링

- **주 1회**: 전체 프로파일링
- **배포 전**: 성능 회귀 확인
- **최적화 후**: 효과 측정

### 2. 목표 설정

각 컴포넌트의 목표 레이턴시를 명확히 하고, 초과 시 알림

### 3. 병목 우선순위

1. **Critical**: P95 > 목표의 2배
2. **High**: P95 > 목표의 1.5배
3. **Medium**: P95 > 목표의 1.2배

### 4. A/B 테스트

최적화 전후 성능을 비교하여 효과 검증

```bash
# Before
python scripts/profile.py --output-dir before/

# 최적화 적용

# After
python scripts/profile.py --output-dir after/

# 비교
python scripts/compare_profiles.py before/ after/
```

---

## 트러블슈팅

### 문제: GPU 메트릭이 수집되지 않음

**증상**:
```
⚠ GPU 메트릭이 없어 GPU 차트를 생성하지 않습니다.
```

**해결**:
```bash
# nvidia-ml-py3 설치
pip install nvidia-ml-py3

# NVIDIA 드라이버 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 문제: 차트가 생성되지 않음

**증상**:
```
⚠ matplotlib가 설치되어 있지 않습니다.
```

**해결**:
```bash
pip install matplotlib
```

### 문제: 프로파일링이 너무 느림

**해결**:
```bash
# 샘플 수 줄이기
python scripts/profile.py --samples 10

# 특정 컴포넌트만
python scripts/profile.py --components stt
```

---

## 참고 자료

- [Performance Optimization Guide](./OPTIMIZATION_GUIDE.md)
- [Memory Profiling](https://docs.python.org/3/library/profile.html)
- [NVIDIA Profiler](https://developer.nvidia.com/nsight-systems)

---

## 문의

성능 관련 문의는 이슈 트래커에 등록해주세요.
