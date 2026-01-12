# Docker 배포 가이드

## 시스템 요구사항

### 필수 요구사항
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU (VRAM 8GB 이상 권장)
- NVIDIA Container Toolkit
- 디스크 공간: 최소 50GB

### NVIDIA Container Toolkit 설치 (Ubuntu)
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 빠른 시작

### 1. 프로젝트 복사
```bash
# 전체 프로젝트를 대상 서버로 복사
scp -r /path/to/NewAvata user@server:/home/user/
```

### 2. 환경 변수 설정
```bash
cd /home/user/NewAvata/realtime-interview-avatar

# .env 파일 생성
cp .env.example .env

# API 키 설정 (필수)
nano .env
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Docker 빌드 및 실행
```bash
# 빌드 (최초 1회, 약 20-30분 소요)
docker-compose build

# 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 4. 접속 확인
```bash
# 상태 확인
curl http://localhost:5000/api/v2/status

# 브라우저 접속
# http://서버IP:5000
```

## 필수 모델 파일

Docker 볼륨으로 마운트되는 모델 파일들이 필요합니다:

### MuseTalk 모델
```
MuseTalk/models/
├── dwpose/
│   └── dw-ll_ucoco_384.onnx
├── face-parse-bisent/
│   └── 79999_iter.pth
├── musetalk/
│   └── pytorch_model.bin
├── sd-vae-ft-mse/
│   └── diffusion_pytorch_model.bin
└── whisper/
    └── tiny.pt
```

### CosyVoice 모델
```
CosyVoice/pretrained_models/
└── CosyVoice2-0.5B/
    ├── cosyvoice2.yaml
    ├── llm.pt
    ├── flow.pt
    ├── hift.pt
    ├── campplus.onnx
    ├── speech_tokenizer_v2.onnx
    ├── spk2info.pt
    └── CosyVoice-BlankEN/
```

### 사전계산 아바타
```
realtime-interview-avatar/precomputed/
└── avatar_name/
    ├── config.json
    ├── coord_list_cycle.pkl
    ├── frame_list_cycle.pkl
    ├── input_latent_list_cycle.pkl
    └── mask_coords_list_cycle.pkl
```

## 문제 해결

### GPU 인식 안됨
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# Docker에서 GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 포트 충돌
```bash
# 5000 포트 사용중인 프로세스 확인
lsof -i :5000

# docker-compose.yml에서 포트 변경
# ports:
#   - "8080:5000"
```

### 메모리 부족
```bash
# Docker 메모리 제한 확인
docker stats

# Swap 추가 (필요시)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 로그 확인
```bash
# 실시간 로그
docker-compose logs -f avatar-server

# 최근 100줄
docker-compose logs --tail=100 avatar-server
```

## 관리 명령어

```bash
# 중지
docker-compose down

# 재시작
docker-compose restart

# 이미지 재빌드
docker-compose build --no-cache

# 컨테이너 쉘 접속
docker exec -it avatar-server bash

# 사용하지 않는 이미지 정리
docker system prune -a
```

## 포트포워딩 (원격 접속)

서버 방화벽에서 5000 포트 열기:
```bash
# Ubuntu UFW
sudo ufw allow 5000/tcp

# CentOS firewalld
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload
```
