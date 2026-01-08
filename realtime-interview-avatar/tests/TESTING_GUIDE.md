# 테스트 가이드

실시간 면접 아바타 시스템의 종합 테스트 가이드입니다.

## 목차

- [테스트 구조](#테스트-구조)
- [테스트 실행](#테스트-실행)
- [통합 테스트](#통합-테스트)
- [성능 테스트](#성능-테스트)
- [CI/CD 통합](#cicd-통합)
- [트러블슈팅](#트러블슈팅)

---

## 테스트 구조

### 디렉토리 구조

```
tests/
├── conftest.py              # Pytest 설정 및 공통 픽스처
├── fixtures/                # 테스트 데이터
│   ├── korean_sample.wav    # 한국어 샘플 오디오
│   └── test_avatar.jpg      # 테스트용 아바타 이미지
├── test_stt.py             # STT 단위 테스트
├── test_tts.py             # TTS 단위 테스트
├── test_integration.py     # 통합 테스트 ⭐
└── TESTING_GUIDE.md        # 이 파일
```

### 테스트 카테고리

| 카테고리 | 마커 | 설명 | 실행 시간 |
|---------|------|------|----------|
| **단위 테스트** | `unit` | 개별 컴포넌트 테스트 | < 1초 |
| **통합 테스트** | `integration` | 전체 파이프라인 테스트 | 5-30초 |
| **느린 테스트** | `slow` | 장시간 실행 테스트 | 30초+ |
| **벤치마크** | `benchmark` | 성능 측정 테스트 | 60초+ |
| **GPU 테스트** | `gpu` | GPU 필요 테스트 | 변동 |
| **비용 발생** | `expensive` | 실제 API 호출 테스트 | 변동 |

---

## 테스트 실행

### 기본 설치

```bash
# 개발 의존성 설치
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov pytest-benchmark

# 또는 개발용 requirements-dev.txt
pip install -r requirements-dev.txt
```

### 전체 테스트 실행

```bash
# 모든 테스트 실행
pytest

# 상세 출력
pytest -v

# 실시간 출력 (print 문 포함)
pytest -v -s
```

### 카테고리별 실행

```bash
# 단위 테스트만
pytest -m unit

# 통합 테스트만
pytest -m integration

# 통합 테스트 제외
pytest -m "not integration"

# 느린 테스트 제외
pytest -m "not slow"

# 비용 발생 테스트 제외 (기본 권장)
pytest -m "not expensive"

# 특정 테스트만
pytest tests/test_integration.py::TestIntegrationPipeline::test_full_pipeline
```

### 병렬 실행

```bash
# pytest-xdist 설치
pip install pytest-xdist

# 4개 워커로 병렬 실행
pytest -n 4

# 자동 워커 수 (CPU 코어 수)
pytest -n auto
```

### Coverage 리포트

```bash
# Coverage 측정
pytest --cov=src --cov-report=html

# HTML 리포트 열기
# Windows
start htmlcov/index.html

# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html
```

---

## 통합 테스트

### test_integration.py 개요

통합 테스트는 전체 파이프라인의 end-to-end 흐름을 검증합니다.

#### Test 1: 전체 파이프라인 테스트

**목적**: 오디오 입력부터 비디오 출력까지 전체 흐름 검증

**검증 항목**:
- STT 레이턴시 < 200ms
- LLM 레이턴시 < 2000ms
- Avatar 레이턴시 < 100ms
- 전체 파이프라인 레이턴시 < 500ms (TTS 제외)

**실행**:
```bash
pytest tests/test_integration.py::TestIntegrationPipeline::test_full_pipeline -v -s
```

**예상 출력**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         파이프라인 성능 메트릭
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전체 레이턴시:    450.23ms
  ├─ STT:         150.45ms
  ├─ LLM:         180.67ms
  ├─ TTS:         800.12ms
  └─ Avatar:      90.34ms
CPU 사용률:       45.2%
메모리 사용량:    1250.5MB
GPU 메모리:       2048.0MB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### Test 2: 한국어 대화 테스트

**목적**: 한국어 음성 인식 및 대화 흐름 검증

**검증 항목**:
- 한국어 STT 정확도
- 한국어 TTS 품질
- 면접 단계 전환 (greeting → self_introduction → technical)
- 대화 컨텍스트 유지

**실행**:
```bash
pytest tests/test_integration.py::TestIntegrationPipeline::test_korean_conversation -v -s
```

#### Test 3: 동시 세션 테스트

**목적**: 멀티 세션 처리 능력 검증

**검증 항목**:
- 2-3개 세션 동시 처리
- 세션 간 간섭 없음
- 리소스 사용량 합리적 (세션당 ~500MB)

**실행**:
```bash
pytest tests/test_integration.py::TestIntegrationPipeline::test_concurrent_sessions -v -s
```

**주의**: GPU 메모리 부족 시 세션 수 조정

#### Test 4: 장시간 세션 테스트

**목적**: 메모리 누수 및 안정성 검증

**검증 항목**:
- 30분 연속 실행 (테스트에서는 60초로 축소)
- 메모리 증가량 < 100MB
- 메모리 사용량 표준편차 < 50MB

**실행**:
```bash
pytest tests/test_integration.py::TestIntegrationPipeline::test_long_session -v -s -m slow
```

#### Test 5: 에러 복구 테스트

**목적**: 예외 상황 처리 및 복구 검증

**검증 항목**:
- 잘못된 입력 처리
- API 타임아웃 핸들링
- 그레이스풀 셧다운
- 재시작 가능성

**실행**:
```bash
pytest tests/test_integration.py::TestIntegrationPipeline::test_error_recovery -v -s
```

#### Test 6: 캐싱 효과 테스트

**목적**: TTS 캐싱 성능 검증

**검증 항목**:
- 캐시 히트 시 10배 이상 속도 향상
- 공통 질문 캐싱

**실행**:
```bash
pytest tests/test_integration.py::TestIntegrationPipeline::test_caching_performance -v -s
```

---

## 성능 테스트

### 벤치마크 테스트

**처리량 벤치마크**:
```bash
pytest tests/test_integration.py::TestPerformanceBenchmark::test_throughput_benchmark -v -s -m benchmark
```

**목표**: 최소 5 req/s 처리량

**예상 출력**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         처리량 벤치마크 결과
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
총 요청 수:       450
총 실행 시간:     60.00s
평균 처리량:      7.50 req/s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 레이턴시 프로파일링

```bash
# pytest-profiling 설치
pip install pytest-profiling

# 프로파일링 실행
pytest tests/test_integration.py --profile

# 결과 확인
snakeviz prof/combined.prof
```

---

## CI/CD 통합

### GitHub Actions 예제

`.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov

    - name: Run unit tests
      run: |
        pytest -m "unit and not gpu" --cov=src

    - name: Run integration tests (non-GPU)
      run: |
        pytest -m "integration and not gpu and not expensive" --cov=src --cov-append

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### GPU 테스트 (Self-hosted Runner)

```yaml
  test-gpu:
    runs-on: self-hosted-gpu
    steps:
    - uses: actions/checkout@v3

    - name: Run GPU tests
      run: |
        pytest -m "gpu" --cov=src
```

---

## 로컬 개발 워크플로우

### 개발 중 빠른 테스트

```bash
# 변경한 파일만 테스트
pytest --lf  # last-failed
pytest --ff  # failed-first

# 특정 패턴 매칭
pytest -k "test_korean"

# 첫 실패에서 중단
pytest -x

# 최대 3개 실패까지
pytest --maxfail=3
```

### TDD (Test-Driven Development)

```bash
# watch 모드 (pytest-watch 필요)
pip install pytest-watch

ptw -- -m "not slow"
```

### 테스트 커버리지 목표

- **전체**: 80% 이상
- **핵심 모듈** (pipeline, stt, tts, llm): 90% 이상
- **유틸리티**: 70% 이상

```bash
# 커버리지 80% 미만 시 실패
pytest --cov=src --cov-fail-under=80
```

---

## 트러블슈팅

### 1. GPU 메모리 부족

**증상**:
```
CUDA out of memory
```

**해결**:
```bash
# GPU 필요 테스트 제외
pytest -m "not gpu"

# 또는 배치 사이즈 감소
export TEST_BATCH_SIZE=1
pytest
```

### 2. API 키 오류

**증상**:
```
401 Unauthorized
```

**해결**:
```bash
# .env 파일 확인
cat .env

# 테스트용 Mock 사용
pytest -m "not expensive"
```

### 3. 느린 테스트 건너뛰기

```bash
# 빠른 테스트만
pytest -m "not slow"

# 타임아웃 설정
pytest --timeout=30
```

### 4. 병렬 실행 시 충돌

**증상**:
```
FAILED tests/test_integration.py::test_concurrent_sessions - RuntimeError: Event loop is closed
```

**해결**:
```bash
# 병렬 실행 비활성화
pytest -n 0

# 또는 scope 조정
# conftest.py에서 fixture scope를 "function"으로 변경
```

### 5. Import 오류

**증상**:
```
ModuleNotFoundError: No module named 'src'
```

**해결**:
```bash
# PYTHONPATH 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 또는 editable install
pip install -e .
```

---

## 테스트 작성 가이드

### 좋은 테스트 작성 원칙

1. **독립성**: 테스트는 독립적으로 실행 가능해야 함
2. **재현성**: 동일한 입력 → 동일한 결과
3. **빠른 피드백**: 단위 테스트는 1초 이내
4. **명확성**: 테스트 이름으로 의도 파악 가능
5. **최소성**: 하나의 테스트는 하나의 것만 검증

### 테스트 이름 규칙

```python
# Good
def test_stt_returns_korean_text_from_audio():
    pass

def test_llm_generates_response_within_2_seconds():
    pass

# Bad
def test_1():
    pass

def test_stuff():
    pass
```

### Fixture 활용

```python
@pytest.fixture
async def pipeline():
    """재사용 가능한 파이프라인 픽스처"""
    pipeline = await create_interview_pipeline()
    await pipeline.initialize()
    yield pipeline
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_something(pipeline):
    # pipeline fixture 사용
    result = await pipeline.process(...)
    assert result is not None
```

### 비동기 테스트

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

---

## 지속적 개선

### 테스트 메트릭 추적

- 커버리지 추이
- 평균 실행 시간
- 실패율
- 불안정한 테스트 (flaky tests) 식별

### 주기적 리뷰

- 매주: 실패한 테스트 분석
- 매월: 커버리지 리포트 리뷰
- 분기: 성능 벤치마크 비교

---

## 참고 자료

- [pytest 공식 문서](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

## 문의

테스트 관련 문의사항은 이슈 트래커에 등록해주세요.
