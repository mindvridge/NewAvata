# 시작하기 가이드 🚀

실시간 면접 아바타 시스템을 빠르게 시작하는 방법

---

## 📋 준비사항

### 필수

- **Python 3.10+**: [다운로드](https://www.python.org/downloads/)
- **OpenAI API 키**: [발급](https://platform.openai.com/api-keys)
- **GPU** (권장): NVIDIA GPU 8GB+ VRAM

### 선택사항

- **Deepgram API 키**: [발급](https://console.deepgram.com/) - STT (또는 로컬 Whisper 사용)
- **ElevenLabs API 키**: [발급](https://elevenlabs.io/) - TTS (또는 무료 EdgeTTS 사용)
- **Daily.co API 키**: [발급](https://dashboard.daily.co/) - WebRTC

---

## ⚡ 빠른 시작 (5분)

### 방법 1: 자동 설정 스크립트 (권장)

#### Linux/Mac:

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/realtime-interview-avatar.git
cd realtime-interview-avatar

# 2. 빠른 시작 (모든 것을 자동으로 설정)
./scripts/quick_start.sh
```

#### Windows:

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/realtime-interview-avatar.git
cd realtime-interview-avatar

# 2. 빠른 시작
scripts\quick_start.bat
```

스크립트가 자동으로:
- ✅ `.env` 파일 생성
- ✅ 가상환경 생성
- ✅ 필수 패키지 설치
- ✅ 환경 변수 검증
- ✅ 서버 시작

### 방법 2: 수동 설정

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/realtime-interview-avatar.git
cd realtime-interview-avatar

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
cp .env.example .env
nano .env  # API 키 입력

# 5. 환경 변수 검증
python src/utils/env_validator.py

# 6. 서버 시작
python -m src.server.main
```

---

## 🔑 API 키 설정

`.env` 파일에 다음 정보를 입력하세요:

```bash
# 필수: OpenAI API 키
OPENAI_API_KEY=sk-proj-your-actual-api-key-here

# 선택사항 (무료 대안 사용 가능)
DEEPGRAM_API_KEY=your-deepgram-key     # 또는 Whisper 로컬
ELEVENLABS_API_KEY=your-elevenlabs-key # 또는 EdgeTTS 무료
```

### 무료/저렴한 옵션

```bash
# .env 설정
TTS_PROVIDER=edge              # EdgeTTS (무료)
STT_PROVIDER=whisper           # Whisper 로컬 (무료)
LLM_MODEL=gpt-3.5-turbo       # GPT-3.5 (저렴)
```

**예상 비용**: $0.50~$2/월 (GPT-3.5만 사용)

---

## 🌐 서버 접속

서버가 시작되면:

1. **Swagger UI** (대화형 API 문서)
   - http://localhost:8000/docs
   - API를 직접 테스트할 수 있습니다

2. **ReDoc** (읽기용 문서)
   - http://localhost:8000/redoc
   - 깔끔한 문서 형식

3. **헬스 체크**
   - http://localhost:8000/health
   - 시스템 상태 확인

4. **웹 UI** (추후 추가 예정)
   - http://localhost:8000
   - 브라우저 기반 인터페이스

---

## 🧪 첫 API 호출

### 1. 헬스 체크

```bash
curl http://localhost:8000/health
```

**응답**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gpu_available": true,
  "services": {
    "stt": "operational",
    "llm": "operational",
    "tts": "operational",
    "avatar": "operational"
  }
}
```

### 2. 세션 생성

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_from_env" \
  -d '{
    "user_id": "test_user",
    "interview_type": "technical",
    "language": "ko"
  }'
```

**응답**:
```json
{
  "session_id": "sess_abc123",
  "websocket_url": "ws://localhost:8000/ws/sess_abc123",
  "status": "created"
}
```

### 3. Swagger UI에서 테스트

1. http://localhost:8000/docs 접속
2. "Authorize" 버튼 클릭
3. API 키 입력
4. 원하는 엔드포인트 클릭
5. "Try it out" 버튼 클릭
6. 파라미터 입력 후 "Execute"

---

## 🐛 문제 해결

### Python을 찾을 수 없음

```bash
# Python 설치 확인
python --version
# 또는
python3 --version

# 설치되지 않았다면
# Windows: https://www.python.org/downloads/
# Mac: brew install python
# Ubuntu: sudo apt install python3 python3-pip
```

### 패키지 설치 실패

```bash
# pip 업그레이드
pip install --upgrade pip

# 개별 설치
pip install fastapi
pip install uvicorn
pip install python-dotenv
pip install websockets
```

### GPU 관련 오류

```bash
# GPU가 없다면 CPU 모드로 실행
# .env 파일에서:
ENABLE_TENSORRT=false
CUDA_VISIBLE_DEVICES=-1
```

### 포트가 이미 사용 중

```bash
# .env 파일에서 포트 변경
SERVER_PORT=8001

# 또는 다른 포트 사용
python -m src.server.main
# 그리고 http://localhost:8001 접속
```

### .env 파일 오류

```bash
# 검증 실행
python src/utils/env_validator.py

# 다시 생성
rm .env
cp .env.example .env
nano .env
```

---

## 📚 다음 단계

### 1. API 문서 읽기

- [API 문서](docs/api.md) - 전체 API 레퍼런스
- [보안 가이드](docs/security.md) - 보안 설정

### 2. 설정 커스터마이징

`.env` 파일에서 원하는 설정 변경:

```bash
# 난이도 조절
DEFAULT_DIFFICULTY=easy        # easy | medium | hard

# 언어 변경
DEFAULT_LANGUAGE=en            # ko | en | ja

# 모델 변경
LLM_MODEL=gpt-4o              # 더 강력한 모델
TTS_PROVIDER=elevenlabs       # 더 나은 음성 품질
```

### 3. Python SDK 사용

```python
from src.client import InterviewAvatarClient

client = InterviewAvatarClient(
    api_url="http://localhost:8000",
    api_key="your_api_key"
)

# 세션 생성
session = await client.create_session(
    user_id="user123",
    interview_type="technical"
)

# WebSocket 연결
await client.connect(session.session_id)
```

### 4. 프로덕션 배포

- [배포 가이드](README.md#-배포-가이드) - Docker, 클라우드 배포
- [Docker Compose](docker-compose.yml) - 컨테이너 배포
- [클라우드 스크립트](scripts/deploy/) - RunPod, Vast.ai, Lambda Labs

---

## 💡 팁

### 개발 모드

```bash
# 핫 리로드 활성화
# .env 파일:
DEBUG=true

# 서버 재시작 없이 코드 변경사항 자동 반영
python -m src.server.main
```

### 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/app.log

# 에러만 확인
grep ERROR logs/app.log
```

### 성능 모니터링

```bash
# GPU 사용량 확인
nvidia-smi -l 1

# 프로파일링 실행
python scripts/profile.py --duration 60
```

---

## 🆘 도움말

### 문제가 있나요?

1. **문서 확인**:
   - [API 문서](docs/api.md)
   - [보안 가이드](docs/security.md)
   - [README](README.md)

2. **로그 확인**:
   ```bash
   tail -f logs/app.log
   docker-compose logs -f app
   ```

3. **환경 검증**:
   ```bash
   python src/utils/env_validator.py
   ```

4. **이슈 생성**:
   - [GitHub Issues](https://github.com/yourusername/realtime-interview-avatar/issues)

### 커뮤니티

- **Discord**: [참여하기](https://discord.gg/your-invite)
- **GitHub Discussions**: [질문하기](https://github.com/yourusername/realtime-interview-avatar/discussions)

---

## ✅ 체크리스트

시작하기 전 확인:

- [ ] Python 3.10+ 설치됨
- [ ] OpenAI API 키 발급
- [ ] `.env` 파일 생성 및 설정
- [ ] 필수 패키지 설치
- [ ] 환경 변수 검증 통과
- [ ] 서버가 정상적으로 시작됨
- [ ] http://localhost:8000/health 접속 가능
- [ ] http://localhost:8000/docs 접속 가능

모두 체크했다면 준비 완료! 🎉

---

**다음**: [API 문서 보기](docs/api.md) | [보안 가이드](docs/security.md)
