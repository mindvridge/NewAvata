# MuseTalk 설치 스크립트

MuseTalk 모델과 의존성을 자동으로 다운로드하고 설치하는 스크립트입니다.

## 스크립트 종류

### 1. Bash 스크립트 (Linux/Mac)
```bash
bash scripts/setup_musetalk.sh
```

### 2. Python 스크립트 (크로스 플랫폼)
```bash
python scripts/setup_musetalk.py
```

Windows에서는 Python 스크립트를 사용하는 것을 권장합니다.

---

## 사용법

### 기본 사용법

```bash
# Bash (Linux/Mac)
bash scripts/setup_musetalk.sh

# Python (Windows/Linux/Mac)
python scripts/setup_musetalk.py
```

### 옵션

#### `--use-mirror`
중국 Hugging Face 미러 사용 (다운로드 속도 향상)

```bash
python scripts/setup_musetalk.py --use-mirror
```

#### `--skip-clone`
저장소 클론 건너뛰기 (이미 클론된 경우)

```bash
python scripts/setup_musetalk.py --skip-clone
```

#### `--models-only`
모델만 다운로드 (저장소 클론 안함)

```bash
python scripts/setup_musetalk.py --models-only
```

---

## 다운로드되는 모델

### 1. VAE (Variational AutoEncoder)
- **저장소**: `stabilityai/sd-vae-ft-mse`
- **크기**: ~300 MB
- **용도**: 이미지 인코딩/디코딩

### 2. Whisper (음성 인코더)
- **모델**: `tiny.pt`
- **크기**: ~39 MB
- **용도**: 오디오 특징 추출

### 3. DWPose (포즈 추정)
- **저장소**: `yzd-v/DWPose`
- **크기**: ~200 MB
- **용도**: 얼굴 및 신체 포즈 추정

### 4. Face Parse BiSeNet (얼굴 파싱)
- **저장소**: `jonathandinu/face-parsing`
- **크기**: ~50 MB
- **용도**: 얼굴 영역 분할

### 5. MuseTalk Checkpoint
- **저장소**: `TMElyralab/MuseTalk`
- **크기**: ~1.5 GB
- **용도**: MuseTalk 메인 모델

### 6. GFPGAN (선택사항)
- **URL**: GitHub Release
- **크기**: ~348 MB
- **용도**: 얼굴 품질 향상

**총 다운로드 크기**: 약 **2.5 GB** (GFPGAN 포함 시 ~2.9 GB)

---

## 설치 과정

### 1. 환경 확인
```
✓ Git 설치됨 (git version 2.39.0)
✓ Python 설치됨 (Python 3.10.0)
✓ pip 설치됨
```

### 2. MuseTalk 저장소 클론
```
ℹ MuseTalk 저장소 클론 중...
✓ 저장소 클론 완료
```

### 3. Hugging Face CLI 설치
```
ℹ huggingface_hub 설치 중...
✓ huggingface_hub 설치 완료
```

### 4. 모델 다운로드
각 모델이 순차적으로 다운로드됩니다.

```
========================================
1. VAE 모델 다운로드
========================================

ℹ 다운로드 중: stabilityai/sd-vae-ft-mse
✓ 다운로드 완료
```

### 5. 설정 파일 생성
```
✓ 설정 파일 생성: models/musetalk/model_paths.yaml
```

### 6. 모델 검증
```
✓ VAE: 335.2 MB
✓ Whisper: 39.1 MB
✓ DWPose: 설치됨
✓ Face Parse: 설치됨
✓ MuseTalk: 설치됨
```

### 7. 테스트 실행 (선택)
```
=== Python 환경 테스트 ===
✓ torch
✓ torchvision
✓ diffusers
✓ transformers
✓ opencv-python
✓ mediapipe
✓ whisper

=== PyTorch 정보 ===
PyTorch 버전: 2.0.1
CUDA 사용 가능: True
CUDA 버전: 11.8
GPU: NVIDIA GeForce RTX 3090
```

---

## 설치 후 디렉토리 구조

```
models/musetalk/
├── sd-vae-ft-mse/
│   └── diffusion_pytorch_model.safetensors
├── whisper/
│   └── tiny.pt
├── dwpose/
│   ├── yolox_l.onnx
│   └── dw-ll_ucoco_384.onnx
├── face-parse-bisent/
│   └── 79999_iter.pth
├── musetalk/
│   ├── musetalk_unet.pth
│   └── audio_processor.pth
├── gfpgan/
│   └── GFPGANv1.4.pth (선택사항)
└── model_paths.yaml
```

---

## 설정 파일 (model_paths.yaml)

```yaml
vae:
  path: models/musetalk/sd-vae-ft-mse
  checkpoint: diffusion_pytorch_model.safetensors

whisper:
  path: models/musetalk/whisper
  checkpoint: tiny.pt

dwpose:
  path: models/musetalk/dwpose
  det_checkpoint: yolox_l.onnx
  pose_checkpoint: dw-ll_ucoco_384.onnx

face_parse:
  path: models/musetalk/face-parse-bisent
  checkpoint: 79999_iter.pth

musetalk:
  path: models/musetalk/musetalk
  unet_checkpoint: musetalk_unet.pth
  audio_processor: audio_processor.pth

gfpgan:
  path: models/musetalk/gfpgan
  checkpoint: GFPGANv1.4.pth
  enabled: true
```

---

## 문제 해결

### 1. Hugging Face 다운로드 속도가 느림

**해결**: `--use-mirror` 옵션 사용

```bash
python scripts/setup_musetalk.py --use-mirror
```

### 2. Git 클론 실패

**해결**: `--models-only` 옵션으로 모델만 다운로드

```bash
python scripts/setup_musetalk.py --models-only
```

### 3. 디스크 공간 부족

**에러**: `No space left on device`

**해결**: 최소 **5 GB** 이상의 여유 공간 확보

### 4. Whisper 모델 다운로드 실패

**수동 다운로드**:
```bash
# 모델 URL
https://openaipublic.azureedge.net/main/whisper/models/tiny.pt

# 저장 위치
models/musetalk/whisper/tiny.pt
```

### 5. CUDA Out of Memory

**해결**: 설정에서 `use_fp16: true` 사용

```python
config = MuseTalkConfig(
    use_fp16=True,  # FP16 사용
    resolution=256   # 낮은 해상도
)
```

### 6. ImportError: No module named 'huggingface_hub'

**해결**:
```bash
pip install huggingface_hub
```

---

## 필수 패키지

스크립트 실행 전에 다음 패키지가 설치되어 있어야 합니다:

```bash
pip install -r requirements.txt
```

주요 패키지:
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `diffusers>=0.21.0`
- `transformers>=4.36.0`
- `opencv-python>=4.8.0`
- `mediapipe>=0.10.0`
- `whisper` (openai-whisper)
- `gfpgan>=1.3.8` (선택사항)

---

## 아바타 사용 시작하기

설치가 완료되면 다음 명령어로 예제를 실행할 수 있습니다:

```bash
# 예제 실행
python -m src.avatar.example_usage

# 또는 직접 사용
python
>>> from src.avatar import create_musetalk_avatar
>>> avatar = create_musetalk_avatar("./assets/avatar_source.jpg")
>>> # 오디오 처리
>>> import numpy as np
>>> audio_chunk = np.random.randn(640).astype(np.float32) * 0.1
>>> video_frame = avatar.process_audio_chunk(audio_chunk)
>>> print(video_frame.frame.shape)
(256, 256, 3)
```

---

## 추가 리소스

- **MuseTalk 공식 저장소**: https://github.com/TMElyralab/MuseTalk
- **Hugging Face**: https://huggingface.co/
- **Whisper 모델**: https://github.com/openai/whisper
- **GFPGAN**: https://github.com/TencentARC/GFPGAN

---

## 라이선스

이 스크립트는 MIT 라이선스를 따릅니다.

각 모델은 고유한 라이선스를 가질 수 있으므로, 상업적 사용 전에 해당 모델의 라이선스를 확인하세요.

---

## 기여

버그 리포트나 개선 제안은 이슈로 등록해주세요.
