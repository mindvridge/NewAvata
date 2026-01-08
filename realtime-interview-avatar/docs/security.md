# 보안 가이드 🔒

실시간 면접 아바타 시스템의 보안 설정 및 모범 사례

---

## 📋 목차

- [환경 변수 관리](#환경-변수-관리)
- [API 키 보안](#api-키-보안)
- [Git 보안](#git-보안)
- [서버 보안](#서버-보안)
- [데이터 보호](#데이터-보호)
- [보안 체크리스트](#보안-체크리스트)

---

## 환경 변수 관리

### .env 파일 설정

**절대 하지 말아야 할 것** ❌:

```python
# 코드에 API 키 하드코딩 (절대 금지!)
api_key = "sk-1234567890abcdef"
openai_key = "your_actual_key_here"
```

**올바른 방법** ✅:

```bash
# .env 파일
OPENAI_API_KEY=sk-1234567890abcdef
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# Python 코드
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
```

### 환경 변수 검증

시작 전에 환경 변수를 검증하세요:

```bash
# 검증 스크립트 실행
python src/utils/env_validator.py

# 또는 서버 시작 시 자동 검증
python -m src.server.main
```

### 민감한 정보 분리

개발/스테이징/프로덕션 환경별로 다른 `.env` 파일 사용:

```bash
.env.development   # 개발 환경
.env.staging       # 스테이징 환경
.env.production    # 프로덕션 환경
```

---

## API 키 보안

### API 키 발급 및 관리

#### 1. OpenAI API 키

```bash
# 발급: https://platform.openai.com/api-keys

# 권장 설정:
- 프로젝트별로 별도 키 생성
- 사용량 제한 설정 ($100/월 등)
- 정기적으로 키 교체 (3개월마다)
```

#### 2. Deepgram API 키

```bash
# 발급: https://console.deepgram.com/

# 권장 설정:
- IP 화이트리스트 설정
- 크레딧 알림 활성화
- 테스트/프로덕션 키 분리
```

#### 3. ElevenLabs API 키

```bash
# 발급: https://elevenlabs.io/

# 권장 설정:
- 문자 수 제한 모니터링
- 할당량 초과 알림 설정
```

### API 키 교체 절차

1. **새 키 발급**
2. **스테이징 환경에서 테스트**
3. **프로덕션 환경 업데이트**
4. **구 키 비활성화** (1주일 후)

```bash
# .env 업데이트
OLD_OPENAI_API_KEY=sk-old...
OPENAI_API_KEY=sk-new...

# 서버 재시작
docker-compose restart app
```

### API 키 노출 대응

API 키가 노출된 경우 **즉시** 조치:

1. **키 비활성화** (API 제공자 대시보드)
2. **새 키 발급 및 교체**
3. **사용 내역 확인** (비정상 사용 여부)
4. **Git 히스토리에서 제거**:

```bash
# Git 히스토리에서 민감 정보 제거 (주의!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# 또는 BFG Repo-Cleaner 사용
bfg --replace-text passwords.txt
```

---

## Git 보안

### .gitignore 설정

**필수 항목**:

```gitignore
# 환경 변수
.env
.env.local
.env.production
.env.*.local

# API 키 및 인증 정보
*.pem
*.key
*.cert
credentials.json
gcs-credentials.json
service-account.json
api-keys.txt

# 비밀 정보
secrets/
.secrets/

# 개인 정보
*.db
*.sqlite
*.sqlite3
```

### .env.example 작성

실제 값을 제거한 예시 파일 제공:

```bash
# .env.example (안전)
OPENAI_API_KEY=sk-your-openai-api-key-here
DEEPGRAM_API_KEY=your-deepgram-api-key-here

# .env (절대 커밋 금지!)
OPENAI_API_KEY=sk-proj-1234567890abcdef
DEEPGRAM_API_KEY=abc123def456
```

### 커밋 전 검증

pre-commit 훅 설정:

```bash
# .git/hooks/pre-commit
#!/bin/bash

# .env 파일이 스테이징되어 있는지 확인
if git diff --cached --name-only | grep -q "^.env$"; then
    echo "❌ Error: .env file is staged!"
    echo "Please remove it: git reset HEAD .env"
    exit 1
fi

# API 키 패턴 검색
if git diff --cached | grep -E "sk-[a-zA-Z0-9]{48}"; then
    echo "❌ Error: Potential API key detected!"
    exit 1
fi

exit 0
```

```bash
chmod +x .git/hooks/pre-commit
```

---

## 서버 보안

### HTTPS 사용

프로덕션에서는 반드시 HTTPS:

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### CORS 설정

특정 도메인만 허용:

```python
# src/server/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-domain.com",  # 프로덕션
        "https://staging.your-domain.com",   # 스테이징
        # "http://localhost:3000",           # 개발 (프로덕션에서 제거)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

### Rate Limiting

요청 제한 설정:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/sessions")
@limiter.limit("10/minute")
async def create_session(request: Request, ...):
    ...
```

### API 키 인증

헤더 기반 인증:

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    expected_key = os.getenv("API_KEY")
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
```

### 보안 헤더

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["your-domain.com", "*.your-domain.com"]
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

---

## 데이터 보호

### 녹화 파일 보안

```python
# 녹화 파일 암호화
from cryptography.fernet import Fernet

def encrypt_file(file_path: str, key: bytes):
    fernet = Fernet(key)
    with open(file_path, 'rb') as f:
        data = f.read()
    encrypted = fernet.encrypt(data)
    with open(file_path + '.encrypted', 'wb') as f:
        f.write(encrypted)
```

### 데이터 삭제

```python
# 세션 종료 후 자동 삭제 (GDPR 준수)
@app.delete("/api/sessions/{session_id}")
async def terminate_session(session_id: str):
    # 세션 종료
    summary = await session_service.terminate(session_id)
    
    # 30일 후 자동 삭제 예약
    await schedule_deletion(session_id, days=30)
    
    return summary
```

### 개인정보 로깅 금지

```python
# ❌ 잘못된 예
logger.info(f"User {user_id} said: {transcript}")

# ✅ 올바른 예
logger.info(f"Session {session_id}: Transcript received")
```

---

## 보안 체크리스트

### 배포 전 체크리스트

- [ ] `.env` 파일이 `.gitignore`에 포함됨
- [ ] 모든 API 키를 환경 변수로 관리
- [ ] `DEBUG=false` 설정 (프로덕션)
- [ ] HTTPS 활성화
- [ ] CORS에 특정 도메인만 허용
- [ ] Rate limiting 설정
- [ ] API 키 인증 활성화
- [ ] 보안 헤더 설정
- [ ] 로그에 개인정보 미포함
- [ ] 데이터 자동 삭제 정책 수립

### 정기 점검 (월 1회)

- [ ] API 키 사용량 확인
- [ ] 비정상 접근 로그 확인
- [ ] 의존성 보안 업데이트
  ```bash
  pip list --outdated
  pip install --upgrade package-name
  ```
- [ ] 취약점 스캔
  ```bash
  pip install safety
  safety check
  ```

### 긴급 대응 절차

**API 키 노출 시**:
1. 즉시 키 비활성화
2. 새 키 발급 및 교체
3. 사용 내역 확인
4. Git 히스토리 정리

**서버 침해 의심 시**:
1. 서버 격리
2. 로그 백업 및 분석
3. 취약점 패치
4. 모든 키 교체
5. 사용자 알림

---

## 도구 및 리소스

### 보안 도구

```bash
# 의존성 취약점 검사
pip install safety
safety check

# 시크릿 검색
pip install detect-secrets
detect-secrets scan > .secrets.baseline

# Git 히스토리 검사
pip install truffleHog
truffleHog --regex --entropy=False .
```

### 추천 서비스

- **Secrets 관리**: AWS Secrets Manager, HashiCorp Vault
- **API 키 관리**: 1Password, LastPass
- **보안 모니터링**: Sentry, Datadog
- **취약점 스캔**: Snyk, Dependabot

---

## 참고 자료

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security.html)

---

**마지막 업데이트**: 2024-01-01

**문의**: security@your-domain.com
