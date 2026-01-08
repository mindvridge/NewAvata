# 배포 가이드

실시간 면접 아바타 시스템을 다양한 GPU 클라우드 플랫폼에 배포하는 가이드입니다.

## 목차

- [배포 옵션 비교](#배포-옵션-비교)
- [사전 준비](#사전-준비)
- [로컬 개발 환경](#로컬-개발-환경)
- [RunPod 배포](#runpod-배포)
- [Vast.ai 배포](#vastai-배포)
- [Lambda Labs 배포](#lambda-labs-배포)
- [비용 최적화 팁](#비용-최적화-팁)
- [트러블슈팅](#트러블슈팅)

---

## 배포 옵션 비교

| 플랫폼 | GPU 타입 | 시간당 비용 | 면접 1회 비용 | 안정성 | 관리 용이성 | 권장 용도 |
|--------|----------|------------|--------------|--------|------------|-----------|
| **Vast.ai** | RTX 4090 | $0.20-0.40 | $0.04-0.06 | ⭐⭐⭐ | ⭐⭐⭐ | 💰 가장 저렴 |
| **RunPod** | RTX 4090 | $0.34-0.54 | $0.06-0.10 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ 가장 균형잡힘 |
| **Lambda Labs** | A100 40GB | $1.10 | $0.18 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🏢 프로덕션 |

### 선택 가이드

- **가격 중심**: Vast.ai (RTX 4090, $0.25/hr)
- **안정성과 가격 균형**: RunPod (RTX 4090 Spot, $0.34/hr)
- **프로덕션 환경**: Lambda Labs (A100, $1.10/hr)

---

## 사전 준비

### 1. API 키 준비

다음 서비스의 API 키가 필요합니다:

- **Deepgram** (STT): https://console.deepgram.com/
- **ElevenLabs** (TTS): https://elevenlabs.io/
- **OpenAI** (LLM): https://platform.openai.com/
- **Daily.co** (WebRTC): https://dashboard.daily.co/

### 2. .env 파일 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 API 키 입력:

```bash
# STT
DEEPGRAM_API_KEY=your_deepgram_key

# TTS
ELEVENLABS_API_KEY=your_elevenlabs_key

# LLM
OPENAI_API_KEY=your_openai_key

# WebRTC
DAILY_API_KEY=your_daily_key

# Docker Hub (배포용)
DOCKER_USERNAME=your_docker_username
```

### 3. Docker 이미지 빌드 및 푸시

```bash
# Docker Hub 로그인
docker login

# 이미지 빌드
docker build -t your_username/interview-avatar:latest -f docker/Dockerfile .

# 이미지 푸시
docker push your_username/interview-avatar:latest
```

---

## 로컬 개발 환경

### 방법 1: 자동 스크립트 (권장)

```bash
./scripts/start_local.sh
```

대화형으로 실행 모드를 선택할 수 있습니다:
1. Docker Compose (완전한 환경, Redis 포함)
2. 직접 실행 (빠른 재시작, 디버깅 용이)
3. Docker만 (컨테이너만, Redis 제외)

### 방법 2: Docker Compose 직접 실행

**개발 모드 (핫 리로딩):**
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

**프로덕션 모드:**
```bash
docker-compose up -d
```

### 방법 3: Python 직접 실행

```bash
# 가상 환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 모델 다운로드
python scripts/setup_musetalk.py

# 서버 시작
uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 접속

- **메인 페이지**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **헬스체크**: http://localhost:8000/api/health

---

## RunPod 배포

### 특징
- RTX 4090 Spot: $0.34/hr (가장 균형잡힘)
- 안정적인 인프라
- Serverless 옵션 지원

### 배포 단계

#### 1. RunPod API 키 생성

https://www.runpod.io/console/user/settings 에서 API 키 생성

```bash
export RUNPOD_API_KEY=your_runpod_key
```

#### 2. 배포 스크립트 실행

```bash
./scripts/deploy/deploy_runpod.sh
```

#### 3. 대화형 선택

- GPU 타입 선택 (권장: RTX 4090 Spot)
- 배포 모드 선택 (Pod 또는 Serverless)
- Docker 이미지 빌드 여부

#### 4. 배포 확인

스크립트가 완료되면 다음 정보가 출력됩니다:
- Pod URL
- 인스턴스 ID
- 유용한 명령어

### 관리 명령어

```bash
# Pod 상태 확인
runpodctl get pod <pod_name>

# Pod 로그 확인
runpodctl logs <pod_name>

# Pod 중지
runpodctl stop pod <pod_name>

# Pod 삭제
runpodctl remove pod <pod_name>
```

### 비용 추정 (RTX 4090 Spot)

- 시간당: $0.34-0.54
- 면접 1회 (15분): $0.06-0.10
- 일일 10회 면접: $0.60-1.00
- 월간 (8시간/일): $81.60-129.60

---

## Vast.ai 배포

### 특징
- 가장 저렴한 가격 (RTX 4090: $0.20-0.40/hr)
- 실시간 가격 비교
- 신뢰도 기반 인스턴스 선택

### 배포 단계

#### 1. Vast.ai API 키 생성

https://cloud.vast.ai/api/ 에서 API 키 생성

```bash
export VAST_API_KEY=your_vast_key
```

#### 2. 배포 스크립트 실행

```bash
./scripts/deploy/deploy_vast.sh
```

#### 3. 대화형 선택

- GPU 타입 선택 (권장: RTX 4090)
- 인스턴스 자동 검색 (가격순 정렬)
- 최저가 인스턴스 자동 선택 또는 수동 선택

#### 4. 배포 확인

스크립트가 완료되면:
- 애플리케이션 URL
- 인스턴스 ID
- SSH 접속 정보

### 관리 명령어

```bash
# 인스턴스 상태 확인
vast show instances

# SSH 연결
vast ssh <instance_id>

# 인스턴스 중지
vast stop instance <instance_id>

# 인스턴스 삭제
vast destroy instance <instance_id>
```

### 비용 추정 (RTX 4090)

- 시간당: $0.20-0.40
- 면접 1회 (15분): $0.04-0.06
- 일일 10회 면접: $0.40-0.60
- 월간 (8시간/일): $48-76.80

### ⚠️ 주의사항

- 신뢰도(reliability) 95% 이상 인스턴스 선택 권장
- Spot 인스턴스는 중단될 수 있음
- 사용 후 반드시 인스턴스 삭제

---

## Lambda Labs 배포

### 특징
- 가장 안정적인 인프라 (99.9% 가동률)
- 빠른 네트워크 (10-100 Gbps)
- 관리하기 쉬운 대시보드

### 배포 단계

#### 1. Lambda Labs API 키 생성

https://cloud.lambdalabs.com/api-keys 에서 API 키 생성

```bash
export LAMBDA_API_KEY=your_lambda_key
```

#### 2. SSH 키 설정

스크립트가 자동으로 SSH 키를 생성하거나 기존 키를 사용합니다.

#### 3. 배포 스크립트 실행

```bash
./scripts/deploy/deploy_lambda.sh
```

#### 4. 대화형 선택

- GPU 타입 선택 (권장: A100 40GB)
- 지역 선택 (권장: us-west-1, 한국과 가까움)
- Docker 이미지 빌드 여부

#### 5. 배포 확인

스크립트가 완료되면:
- 애플리케이션 URL
- 인스턴스 ID 및 IP
- SSH 접속 정보

### 관리 명령어

```bash
# SSH 연결
ssh -i ~/.ssh/lambda_interview_avatar ubuntu@<instance_ip>

# 컨테이너 로그 확인
ssh -i ~/.ssh/lambda_interview_avatar ubuntu@<instance_ip> 'sudo docker logs interview-avatar'

# 인스턴스 종료
# Lambda Labs 대시보드에서 수동 종료
# https://cloud.lambdalabs.com/instances
```

### 비용 추정 (A100 40GB)

- 시간당: $1.10
- 면접 1회 (15분): $0.28
- 일일 10회 면접: $2.80
- 월간 (8시간/일): $211.20

---

## 비용 최적화 팁

### 1. Spot 인스턴스 활용

- RunPod: Spot 인스턴스 사용 시 20-40% 절감
- Vast.ai: 실시간 가격 비교로 최저가 선택
- 중단 위험 감수 필요 (자동 재시작 설정 권장)

### 2. Auto-scaling 설정

**RunPod Serverless:**
```yaml
workersMin: 0  # 사용하지 않을 때 0으로
workersMax: 3  # 최대 동시 세션 수
idleTimeout: 5  # 5분 유휴 후 자동 종료
```

### 3. TTS 캐싱 활성화

`.env` 파일에서:
```bash
ENABLE_TTS_CACHE=true
TTS_CACHE_SIZE=1000
```

공통 질문을 미리 캐싱하여 TTS API 호출 감소 (30-50% 비용 절감)

### 4. 대안 TTS 사용

- ElevenLabs (유료): 고품질, $0.30/1K chars
- EdgeTTS (무료): 중품질, 무료
- Naver Clova (유료): 한국어 최적화, 저렴

### 5. 사용 후 자동 종료

모든 스크립트는 배포 정보를 JSON 파일로 저장합니다:
```bash
deployment_info_<provider>_<timestamp>.json
```

cron job으로 자동 정리:
```bash
# 매일 밤 12시에 모든 인스턴스 정리
0 0 * * * /path/to/cleanup_instances.sh
```

### 6. 월간 예산 설정

각 플랫폼의 대시보드에서 예산 알림 설정:
- RunPod: Settings > Billing > Budget Alerts
- Lambda Labs: Billing > Budget Limits

---

## 트러블슈팅

### 1. GPU 메모리 부족

**증상:**
```
CUDA out of memory
```

**해결:**
- 더 큰 GPU 선택 (RTX 4090 → A100)
- 배치 사이즈 감소
- 얼굴 향상 비활성화: `ENABLE_FACE_ENHANCEMENT=false`

### 2. 모델 다운로드 실패

**증상:**
```
Failed to download model
```

**해결:**
```bash
# 수동 모델 다운로드
python scripts/setup_musetalk.py --model-dir ./models

# 또는 Docker 볼륨으로 마운트
docker run -v ./models:/app/models ...
```

### 3. API 키 오류

**증상:**
```
401 Unauthorized
```

**해결:**
- `.env` 파일의 API 키 확인
- API 키 유효성 테스트:
```bash
curl -H "Authorization: Bearer $DEEPGRAM_API_KEY" https://api.deepgram.com/v1/projects
```

### 4. 네트워크 연결 끊김

**증상:**
```
WebSocket connection closed
```

**해결:**
- 방화벽 확인 (포트 8000 개방)
- Daily.co 도메인 화이트리스트 추가
- 네트워크 품질 모니터링 활성화

### 5. 컨테이너 시작 실패

**증상:**
```
Container exited with code 1
```

**해결:**
```bash
# 로그 확인
docker logs <container_id>

# 대화형 디버깅
docker run -it --rm --entrypoint /bin/bash interview-avatar:latest
```

### 6. RunPod CLI 인증 실패

**증상:**
```
Unauthorized: Invalid API key
```

**해결:**
```bash
# API 키 재설정
runpodctl config --apiKey $RUNPOD_API_KEY

# 설정 확인
runpodctl config show
```

### 7. Vast.ai 인스턴스 시작 느림

**증상:**
인스턴스가 10분 이상 시작되지 않음

**해결:**
- 다른 인스턴스 선택 (신뢰도 높은 것)
- 지역 변경 (가까운 곳)
- 인스턴스 삭제 후 재생성

---

## 헬스체크 및 모니터링

### 기본 헬스체크

```bash
curl http://<your-instance-url>/api/health
```

응답:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-05T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "stt": "ok",
    "tts": "ok",
    "llm": "ok",
    "avatar": "ok"
  }
}
```

### 로그 모니터링

**Docker Compose:**
```bash
docker-compose logs -f app
```

**Docker:**
```bash
docker logs -f interview-avatar
```

**로컬:**
```bash
tail -f logs/app.log
```

### Prometheus + Grafana (프로덕션)

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile monitoring up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## 보안 권장사항

### 1. API 키 보호

- `.env` 파일을 절대 커밋하지 마세요
- 환경 변수로 주입:
```bash
docker run -e DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY ...
```

### 2. HTTPS 설정

프로덕션 환경에서는 반드시 HTTPS 사용:
```bash
# Let's Encrypt 인증서
certbot certonly --standalone -d your-domain.com
```

### 3. API 키 인증

`API_KEY_REQUIRED=true`로 설정하여 공개 접근 차단

### 4. 방화벽 설정

필요한 포트만 개방:
- 8000 (HTTP/WebSocket)
- 443 (HTTPS, 프로덕션)

### 5. 정기 업데이트

```bash
# 의존성 업데이트
pip install -U -r requirements.txt

# Docker 이미지 재빌드
docker-compose build --no-cache
```

---

## 지원 및 문의

- GitHub Issues: https://github.com/your-repo/issues
- 문서: https://your-docs-url.com
- 이메일: support@your-domain.com

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
