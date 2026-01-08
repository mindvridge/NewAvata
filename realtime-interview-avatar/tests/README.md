# 테스트 가이드

실시간 면접 아바타 시스템의 테스트 스위트입니다.

## 테스트 종류

### 1. 유닛 테스트 (Unit Tests)

개별 컴포넌트를 독립적으로 테스트합니다.

```bash
# 모든 유닛 테스트 실행
pytest tests/test_stt.py -v

# 특정 테스트 클래스만 실행
pytest tests/test_stt.py::TestDeepgramSTTService -v

# 특정 테스트 메서드만 실행
pytest tests/test_stt.py::TestDeepgramSTTService::test_initialization -v
```

### 2. 통합 테스트 (Integration Tests)

실제 API를 사용하는 테스트입니다. API 키가 필요합니다.

```bash
# 통합 테스트 실행 (API 키 필요)
pytest tests/test_stt.py -m integration -v

# 통합 테스트 제외
pytest tests/test_stt.py -m "not integration" -v
```

### 3. CLI 테스트

실시간으로 마이크 입력을 받아 테스트하는 인터랙티브 스크립트입니다.

```bash
# 인터랙티브 메뉴
python -m src.stt.test_cli

# 모든 테스트 자동 실행
python -m src.stt.test_cli --all
```

## 환경 설정

### API 키 설정

테스트를 실행하기 전에 `.env` 파일에 API 키를 설정하세요:

```bash
DEEPGRAM_API_KEY=your_actual_api_key_here
ELEVENLABS_API_KEY=your_actual_api_key_here
OPENAI_API_KEY=your_actual_api_key_here
```

### 의존성 설치

```bash
pip install pytest pytest-asyncio pyaudio
```

## 테스트 커버리지

커버리지 리포트 생성:

```bash
pytest tests/test_stt.py --cov=src/stt --cov-report=html
```

HTML 리포트는 `htmlcov/index.html`에서 확인할 수 있습니다.

## 테스트 시나리오

### test_stt.py

1. **Deepgram 연결 테스트**
   - WebSocket 연결/해제
   - 컨텍스트 매니저 사용
   - 재연결 로직

2. **한국어 음성 인식 테스트**
   - 샘플 오디오 파일 사용
   - 중간 결과 vs 최종 결과
   - 콜백 함수 동작

3. **실시간 스트리밍 테스트**
   - 마이크 입력 처리
   - 연속 오디오 스트리밍
   - 버퍼 관리

4. **VAD 동작 테스트**
   - 발화 시작/종료 감지
   - 모드 전환 (INTERVIEW_NORMAL, INTERVIEW_RELAXED, CONVERSATION)
   - 통계 추적

5. **레이턴시 측정 테스트**
   - 오디오 전송 레이턴시
   - 목표: 100ms 이하
   - 통계 분석

### test_cli.py

인터랙티브 테스트 메뉴:

1. **Deepgram 연결 테스트** - API 연결 확인
2. **한국어 음성 인식** - 10초 동안 실시간 인식
3. **실시간 스트리밍** - 5초 스트리밍 테스트
4. **VAD 기능** - 모드별 설정 확인
5. **레이턴시 측정** - 10회 측정 및 통계

## 예제 실행

### 기본 테스트

```bash
# 전체 테스트 (통합 테스트 제외)
pytest tests/test_stt.py -v -m "not integration"

# 빠른 테스트만
pytest tests/test_stt.py -v -m "not slow"

# 병렬 실행 (pytest-xdist 필요)
pytest tests/test_stt.py -n auto
```

### 인터랙티브 테스트

```bash
# CLI 테스트 실행
python -m src.stt.test_cli

# 메뉴에서 원하는 테스트 선택
# 1-5: 개별 테스트
# 6: 모든 테스트
# 0: 종료
```

### 실제 API 테스트

```bash
# 실제 Deepgram API로 테스트
pytest tests/test_stt.py::test_real_deepgram_connection -v -s

# 실제 VAD 모델 로딩 테스트
pytest tests/test_stt.py::test_real_vad_initialization -v -s
```

## 트러블슈팅

### PyAudio 설치 오류 (Windows)

```bash
# 미리 컴파일된 바이너리 사용
pip install pipwin
pipwin install pyaudio
```

### API 연결 오류

- `.env` 파일에 올바른 API 키가 있는지 확인
- 인터넷 연결 확인
- API 키의 권한 확인

### 오디오 장치 오류

- 마이크가 연결되어 있는지 확인
- 시스템 오디오 설정에서 마이크 권한 확인
- 다른 프로그램이 마이크를 사용 중인지 확인

## CI/CD 통합

GitHub Actions 예제:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: pytest tests/test_stt.py -v -m "not integration" --cov=src/stt
        env:
          DEEPGRAM_API_KEY: ${{ secrets.DEEPGRAM_API_KEY }}
```

## 참고

- pytest 문서: https://docs.pytest.org/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- Deepgram API: https://developers.deepgram.com/
