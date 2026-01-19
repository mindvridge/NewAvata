# 립싱크 속도 향상 가이드

## 현재 구현된 최적화

1. ✅ **TensorRT 가속**: ONNX Runtime TensorRT EP 사용
2. ✅ **프레임 스킵**: frame_skip=2로 절반 프레임만 추론
3. ✅ **해상도 조정**: 720p/480p/360p 선택 가능
4. ✅ **GPU 인코딩**: NVENC 사용
5. ✅ **병렬 처리**: TTS와 프리컴퓨트 로드 병렬
6. ✅ **마스크 캐싱**: 블렌딩 시 마스크 재사용

## 추가 최적화 방안

### 1. 배치 크기 동적 조정 (우선순위: 높음)

**현재**: 배치 크기 32로 고정
**개선**: GPU 메모리에 따라 자동 조정

```python
# GPU 메모리에 따라 배치 크기 자동 조정
def get_optimal_batch_size(device, base_batch_size=32):
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        if free_memory > 8 * 1024**3:  # 8GB 이상
            return base_batch_size
        elif free_memory > 4 * 1024**3:  # 4GB 이상
            return base_batch_size // 2
        else:
            return base_batch_size // 4
    return base_batch_size
```

**예상 효과**: GPU 메모리 부족 시 안정성 향상, 메모리 여유 시 속도 향상

### 2. 프리컴퓨트 데이터 메모리 캐싱 (우선순위: 높음)

**현재**: 매번 디스크에서 로드
**개선**: 메모리에 캐싱하여 반복 로드 방지

```python
# 전역 프리컴퓨트 캐시
precomputed_cache = {}  # {path: data}

def load_precomputed_cached(path):
    if path in precomputed_cache:
        return precomputed_cache[path]
    with open(path, 'rb') as f:
        data = pickle.load(f)
    precomputed_cache[path] = data
    return data
```

**예상 효과**: 프리컴퓨트 로드 시간 0.5~1초 → 0초

### 3. 프레임 보간 최적화 (우선순위: 중간)

**현재**: 선형 보간 (cv2.addWeighted)
**개선**: 더 빠른 보간 방법 또는 보간 생략

```python
# 옵션 1: 보간 생략 (가장 빠름, 품질 약간 저하)
if frame_skip > 1 and skip_interpolation:
    # 보간 없이 추론된 프레임만 사용
    res_frame_list = inference_frame_list

# 옵션 2: 빠른 보간 (품질 유지)
# 현재 선형 보간이 이미 최적
```

**예상 효과**: 보간 시간 0.5~1초 절감 (skip_interpolation=True 시)

### 4. 블렌딩 최적화 (우선순위: 중간)

**현재**: ThreadPoolExecutor로 병렬 처리
**개선**: GPU 가속 블렌딩 또는 배치 블렌딩

```python
# 배치 블렌딩 (여러 프레임을 한 번에 처리)
def blend_frames_batch(frames_batch, coords_batch, mask_cache):
    # 배치 단위로 블렌딩하여 오버헤드 감소
    pass
```

**예상 효과**: 블렌딩 시간 10~20% 감소

### 5. 비디오 인코딩 최적화 (우선순위: 낮음)

**현재**: NVENC p1 (가장 빠른 프리셋)
**개선**: 추가 최적화 옵션

```python
ffmpeg_cmd.extend([
    '-c:v', 'h264_nvenc',
    '-preset', 'p1',           # 이미 최고 속도
    '-tune', 'ull',            # Ultra Low Latency
    '-rc', 'vbr',
    '-cq', '28',
    '-b:v', '2M',              # 비트레이트 제한 (인코딩 속도 향상)
    '-maxrate', '4M',
    '-bufsize', '8M',
    '-gpu', '0',               # GPU 0 사용
    '-delay', '0',             # 지연 최소화
])
```

**예상 효과**: 인코딩 시간 5~10% 감소

### 6. Whisper 특징 추출 최적화 (우선순위: 중간)

**현재**: 표준 Whisper 처리
**개선**: 배치 처리 또는 캐싱

```python
# 오디오 특징 추출 결과 캐싱 (동일 텍스트 재사용)
whisper_cache = {}

def get_whisper_features_cached(audio_input, text_hash):
    if text_hash in whisper_cache:
        return whisper_cache[text_hash]
    features = audio_processor.get_audio_feature(audio_input)
    whisper_cache[text_hash] = features
    return features
```

**예상 효과**: 동일 오디오 재사용 시 0.5~1초 절감

### 7. 스트리밍 모드 최적화 (우선순위: 높음)

**현재**: 스트리밍 모드에서도 전체 오디오 생성 후 립싱크
**개선**: 오디오 청크를 받는 대로 립싱크 생성 시작

```python
# TTS 청크를 받는 대로 립싱크 프레임 생성 시작
# 첫 오디오 청크만으로도 초기 프레임 생성 가능
```

**예상 효과**: 첫 프레임 표시 시간 50% 감소

## 성능 벤치마크

### 현재 성능 (RTX 5060 Ti 16GB 기준)
- 프레임 스킵 없음: ~2-3초/프레임 → 50-75초 (25프레임)
- 프레임 스킵=2: ~1-1.5초/프레임 → 12-18초 (13프레임)
- 프레임 스킵=3: ~0.7-1초/프레임 → 7-10초 (9프레임)

### 목표 성능
- 프레임 스킵=2: 8-12초 (50% 향상)
- 프레임 스킵=3: 5-7초 (30% 향상)

## 구현 우선순위

1. **프리컴퓨트 캐싱** (즉시 구현 가능, 효과 큼)
2. **배치 크기 동적 조정** (안정성 향상)
3. **스트리밍 모드 최적화** (사용자 경험 향상)
4. **프레임 보간 옵션** (속도 vs 품질 트레이드오프)
5. **블렌딩 최적화** (추가 개선)
