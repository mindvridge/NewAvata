"""
실시간 립싱크 테스트 웹 프론트엔드
Flask + WebSocket 기반 + 모델 프리로드
"""

import os
import sys

# TensorRT DLL 경로 추가 (Windows에서 ONNX Runtime TensorRT Provider 사용을 위해)
if sys.platform == 'win32':
    # TensorRT libs 경로
    trt_libs_path = os.path.join(os.path.dirname(__file__), 'venv', 'Lib', 'site-packages', 'tensorrt_libs')
    if os.path.exists(trt_libs_path):
        os.add_dll_directory(trt_libs_path)
        os.environ['PATH'] = trt_libs_path + os.pathsep + os.environ.get('PATH', '')

    # CUDA runtime 경로 (nvidia-cuda-runtime 패키지)
    cuda_runtime_path = os.path.join(os.path.dirname(__file__), 'venv', 'Lib', 'site-packages', 'nvidia', 'cuda_runtime', 'bin')
    if os.path.exists(cuda_runtime_path):
        os.add_dll_directory(cuda_runtime_path)
        os.environ['PATH'] = cuda_runtime_path + os.pathsep + os.environ.get('PATH', '')
import time
import json
import base64
import threading
import subprocess
import signal
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from tqdm import tqdm

# .env 파일 로드
load_dotenv()

# MuseTalk 경로 추가 (환경변수 또는 기본값)
MUSETALK_PATH = Path(os.getenv('MUSETALK_PATH', 'c:/NewAvata/NewAvata/MuseTalk'))
sys.path.insert(0, str(MUSETALK_PATH))

# 공식 MuseTalk 블렌딩 모듈
from musetalk.utils.blending import get_image, get_image_blending, get_image_prepare_material
from musetalk.utils.face_parsing import FaceParsing

# 전역 FaceParsing 인스턴스
face_parsing = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'realtime-lipsync-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 상태
lipsync_engine = None
precomputed_avatars = {}
precomputed_cache = {}  # 프리컴퓨트 데이터 메모리 캐시 {path: data}
current_process = None  # 현재 실행 중인 프로세스
models_loaded = False  # 모델 로드 상태

# TTS 오디오 캐시 (동일 텍스트 재요청 시 캐시 사용)
tts_audio_cache = {}  # {text_hash: (audio_numpy, sample_rate, timestamp)}
TTS_CACHE_MAX_SIZE = 50  # 최대 캐시 항목 수
TTS_CACHE_TTL = 3600  # 캐시 유효 시간 (1시간)

# 중앙 크롭 설정 (면접관 중앙 영역만 전송)
# 원본 영상: 1268x724, 3명 면접관 중 중앙만 크롭
CENTER_CROP_ENABLED = False  # 중앙 크롭 활성화 여부 (임시 비활성화 - UI 레이아웃 테스트)
CENTER_CROP_RATIO = (0.36, 0.64)  # 좌측 36% ~ 우측 64% (중앙 28%)
# 해상도별 크롭 좌표 (x_start, x_end, width, height)
CENTER_CROP_COORDS = {
    "720p": (456, 811, 355, 724),   # 1268 * 0.36 ~ 1268 * 0.64
    "480p": (305, 543, 238, 484),   # 848 * 0.36 ~ 848 * 0.64
    "360p": (228, 404, 176, 360),   # 632 * 0.36 ~ 632 * 0.64
}

# 클라이언트별 상태 관리 (멀티 브라우저 지원)
client_sessions = {}  # {sid: {'cancelled': False, 'start_time': None, 'generating': False}}
generation_lock = threading.Lock()  # 동시 생성 방지 락

# 큐 시스템
from queue import Queue
from collections import OrderedDict
import uuid

class ParallelGenerationQueue:
    """병렬 립싱크 생성 요청 큐 시스템"""

    MAX_CONCURRENT = 3  # 동시 처리 최대 수 (RTX 5060 Ti 16GB 기준, 안전 마진 포함)

    def __init__(self):
        self.queue = OrderedDict()  # {request_id: request_data}
        self.processing = {}  # {request_id: request_data} 현재 처리 중인 요청들
        self.lock = threading.Lock()

    def add_request(self, sid, request_data):
        """새 요청 추가, 요청 ID 반환"""
        with self.lock:
            request_id = str(uuid.uuid4())[:8]
            request_data['request_id'] = request_id
            request_data['sid'] = sid
            request_data['queued_at'] = time.time()
            request_data['status'] = 'queued'
            self.queue[request_id] = request_data

            # 대기 순서 계산 (처리 중인 것 이후)
            position = len(self.processing) + list(self.queue.keys()).index(request_id) + 1
            return request_id, position

    def get_next(self):
        """다음 요청 가져오기 (동시 처리 수 제한 확인)"""
        with self.lock:
            # 이미 최대 동시 처리 수에 도달했으면 None 반환
            if len(self.processing) >= self.MAX_CONCURRENT:
                return None
            if not self.queue:
                return None

            request_id, request_data = next(iter(self.queue.items()))
            del self.queue[request_id]
            request_data['status'] = 'processing'
            self.processing[request_id] = request_data
            return request_data

    def complete_request(self, request_id):
        """특정 요청 완료"""
        with self.lock:
            if request_id in self.processing:
                del self.processing[request_id]

    def cancel_request(self, sid):
        """특정 클라이언트의 요청 취소"""
        with self.lock:
            # 큐에서 해당 sid의 요청 제거
            to_remove = [rid for rid, data in self.queue.items() if data.get('sid') == sid]
            for rid in to_remove:
                del self.queue[rid]
            # 처리 중인 요청 중 해당 sid 취소 표시
            for rid, data in self.processing.items():
                if data.get('sid') == sid:
                    data['cancelled'] = True
            return len(to_remove) > 0

    def get_position(self, sid):
        """클라이언트의 대기 순서 반환 (0=처리중, -1=없음)"""
        with self.lock:
            # 처리 중인지 확인
            for rid, data in self.processing.items():
                if data.get('sid') == sid:
                    return 0  # 현재 처리 중

            # 대기열에서 위치 확인
            for i, (rid, data) in enumerate(self.queue.items()):
                if data.get('sid') == sid:
                    return i + 1  # 대기 순서 (1부터 시작)
            return -1  # 큐에 없음

    def get_status(self):
        """큐 상태 반환"""
        with self.lock:
            return {
                'queue_length': len(self.queue),
                'processing_count': len(self.processing),
                'max_concurrent': self.MAX_CONCURRENT,
                'processing': [
                    {
                        'request_id': rid,
                        'sid': data.get('sid'),
                        'started_at': data.get('queued_at')
                    }
                    for rid, data in self.processing.items()
                ],
                'queue': [
                    {
                        'request_id': rid,
                        'sid': data.get('sid'),
                        'queued_at': data.get('queued_at'),
                        'wait_time': time.time() - data.get('queued_at', time.time())
                    }
                    for rid, data in self.queue.items()
                ]
            }

    def can_process_more(self):
        """추가 요청 처리 가능 여부"""
        with self.lock:
            return len(self.processing) < self.MAX_CONCURRENT and len(self.queue) > 0

# 전역 큐 인스턴스
generation_queue = ParallelGenerationQueue()


class LipsyncEngine:
    """립싱크 엔진 - 모델 프리로드 및 추론 (TensorRT 지원)"""

    def __init__(self):
        self.device = None
        self.vae = None
        self.unet = None
        self.unet_trt = None  # TensorRT 엔진
        self.use_tensorrt = False  # TensorRT 사용 여부
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.timesteps = None
        self.weight_dtype = None
        self.loaded = False

    def load_onnx_tensorrt(self, onnx_path: str, use_fp16: bool = True):
        """ONNX Runtime TensorRT EP 로드"""
        try:
            import onnxruntime as ort

            print(f"  ONNX TensorRT 로드 중: {onnx_path}")

            # TensorRT EP 설정
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
                    'trt_fp16_enable': use_fp16,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': os.path.dirname(onnx_path),
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                }),
            ]

            # 세션 생성 (첫 실행시 TensorRT 엔진 빌드)
            print("  세션 생성 중 (첫 실행시 TensorRT 엔진 빌드, 수 분 소요)...")
            session = ort.InferenceSession(onnx_path, providers=providers)

            active_provider = session.get_providers()[0]
            print(f"  활성 Provider: {active_provider}")

            # ONNX Runtime UNet 래퍼 클래스
            class ONNXRuntimeUNet:
                def __init__(self, session, use_fp16):
                    self.session = session
                    self.use_fp16 = use_fp16

                def __call__(self, latent, timestep, encoder_hidden_states):
                    """ONNX Runtime 추론 실행"""
                    import torch

                    # PyTorch -> NumPy
                    dtype = np.float16 if self.use_fp16 else np.float32
                    latent_np = latent.cpu().numpy().astype(dtype)
                    timestep_np = timestep.cpu().numpy().astype(np.int64)
                    encoder_np = encoder_hidden_states.cpu().numpy().astype(dtype)

                    # 추론 실행
                    output = self.session.run(None, {
                        'latent': latent_np,
                        'timestep': timestep_np,
                        'encoder_hidden_states': encoder_np
                    })

                    # NumPy -> PyTorch
                    output_tensor = torch.from_numpy(output[0]).to(latent.device)

                    # diffusers 형식에 맞게 래핑
                    class UNetOutput:
                        def __init__(self, sample):
                            self.sample = sample

                    return UNetOutput(output_tensor)

            self.unet_trt = ONNXRuntimeUNet(session, use_fp16)
            self.use_tensorrt = True
            print("  ONNX TensorRT 로드 완료!")
            return True

        except Exception as e:
            print(f"  ONNX TensorRT 로드 실패: {e}")
            print("  PyTorch 모델을 사용합니다.")
            self.use_tensorrt = False
            return False

    def load_models(self, use_float16=True, use_tensorrt=True):
        """모델 로드 (서버 시작시 1회)"""
        import torch
        from musetalk.utils.blending import get_image_blending
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.models.vae import VAE
        from musetalk.models.unet import UNet, PositionalEncoding
        from transformers import WhisperModel

        print("\n" + "=" * 50)
        print("모델 프리로드 시작...")
        print("=" * 50)

        start_time = time.time()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.weight_dtype = torch.float16 if use_float16 else torch.float32

        # VAE 로드
        print("  [1/5] VAE 로드 중...")
        self.vae = VAE(model_path="./models/sd-vae", use_float16=use_float16)
        if use_float16:
            self.vae.vae = self.vae.vae.half()
        self.vae.vae = self.vae.vae.to(self.device)

        # UNet 로드 (ONNX TensorRT 또는 PyTorch)
        onnx_path = "./models/musetalkV15/unet_fp16.onnx"
        trt_loaded = False

        if use_tensorrt and os.path.exists(onnx_path):
            print("  [2/5] UNet ONNX TensorRT 로드 중...")
            trt_loaded = self.load_onnx_tensorrt(onnx_path, use_fp16=use_float16)

        if not trt_loaded:
            print("  [2/5] UNet PyTorch 로드 중...")
            self.unet = UNet(
                unet_config="./models/musetalkV15/musetalk.json",
                model_path="./models/musetalkV15/unet.pth",
                device=self.device
            )
            if use_float16:
                self.unet.model = self.unet.model.half()
            self.unet.model = self.unet.model.to(self.device)

        # PE 로드
        print("  [3/5] Positional Encoding 로드 중...")
        self.pe = PositionalEncoding(d_model=384)
        self.pe = self.pe.to(self.device)
        if use_float16:
            self.pe = self.pe.half()

        # Whisper 로드
        print("  [4/5] Whisper 로드 중...")
        self.whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
        self.whisper = self.whisper.to(self.device)
        if use_float16:
            self.whisper = self.whisper.half()
        self.whisper.eval()

        # AudioProcessor (경로에서 trailing slash 제거)
        self.audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")

        # Timesteps
        self.timesteps = torch.tensor([0], device=self.device)

        # CUDA 워밍업 (전체 파이프라인)
        print("  [5/5] CUDA 워밍업 중...")
        with torch.inference_mode():
            # 1. VAE 워밍업 (32x32 latent -> 256x256 이미지)
            dummy_latent = torch.randn(1, 4, 32, 32).to(self.device, dtype=self.weight_dtype)
            _ = self.vae.decode_latents(dummy_latent)

            # 2. UNet 워밍업 (모든 배치 크기에 대해 TensorRT 엔진 빌드)
            # TensorRT는 동적 배치에서 각 배치 크기별로 엔진을 빌드하므로
            # 자주 사용되는 배치 크기들을 미리 워밍업
            if self.use_tensorrt:
                warmup_batch_sizes = [32, 16, 8, 4, 2, 1]  # 자주 사용되는 배치 크기들
                print(f"    TensorRT 엔진 워밍업 (배치: {warmup_batch_sizes})...")
                for batch_size in warmup_batch_sizes:
                    dummy_latent_batch = torch.randn(batch_size, 8, 32, 32).to(self.device, dtype=self.weight_dtype)
                    dummy_audio = torch.randn(batch_size, 50, 384).to(self.device, dtype=self.weight_dtype)
                    dummy_audio_feature = self.pe(dummy_audio)
                    _ = self.unet_trt(
                        dummy_latent_batch,
                        self.timesteps,
                        dummy_audio_feature
                    ).sample
                    del dummy_latent_batch, dummy_audio, dummy_audio_feature
                    print(f"      배치 {batch_size}: 완료")
            else:
                dummy_latent_batch = torch.randn(16, 8, 32, 32).to(self.device, dtype=self.weight_dtype)
                dummy_audio = torch.randn(16, 50, 384).to(self.device, dtype=self.weight_dtype)
                dummy_audio_feature = self.pe(dummy_audio)
                _ = self.unet.model(
                    dummy_latent_batch,
                    self.timesteps,
                    encoder_hidden_states=dummy_audio_feature
                ).sample
                del dummy_latent_batch, dummy_audio, dummy_audio_feature

            # 3. Whisper 워밍업 (첫 실행 시 feature_extractor + CUDA 커널 컴파일 방지)
            print("    Whisper 워밍업...")
            import numpy as np
            # Feature extractor 첫 호출 워밍업 (mel 스펙트로그램 FFT 컴파일)
            # 30초 세그먼트 처리를 위해 실제와 유사한 길이의 오디오 사용
            for audio_len in [16000, 32000]:  # 1초, 2초
                dummy_audio_np = np.random.randn(audio_len).astype(np.float32)
                whisper_features, librosa_len = self.audio_processor.get_audio_feature((dummy_audio_np, 16000))
                # get_whisper_chunk에서 실제 Whisper 모델 실행
                _ = self.audio_processor.get_whisper_chunk(
                    whisper_features,
                    self.device,
                    self.weight_dtype,
                    self.whisper,
                    librosa_len,
                    fps=25,
                    audio_padding_length_left=2,
                    audio_padding_length_right=2,
                )
            print("      Whisper: 완료")

            # 4. librosa 워밍업 (파일 경로 모드에서 첫 호출 시 JIT 컴파일 방지)
            print("    librosa 워밍업...")
            import librosa
            import tempfile
            import soundfile as sf
            # 더미 오디오 파일 생성 후 librosa.load로 읽기
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                dummy_path = f.name
                sf.write(dummy_path, np.random.randn(16000).astype(np.float32), 16000)
            # librosa 내부 초기화 (soxr resampler, numba JIT 등)
            _ = librosa.load(dummy_path, sr=16000)
            # os는 파일 상단에서 이미 import됨
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
            print("      librosa: 완료")

            # 5. 메모리 정리
            del dummy_latent
            torch.cuda.empty_cache()

        # FaceParsing 미리 초기화
        print("  FaceParsing 초기화 중...")
        global face_parsing
        if face_parsing is None:
            face_parsing = FaceParsing(
                left_cheek_width=90,
                right_cheek_width=90
            )

        elapsed = time.time() - start_time
        engine_type = "TensorRT" if self.use_tensorrt else "PyTorch"
        print(f"\n모델 프리로드 완료! ({elapsed:.1f}초)")
        print(f"UNet 엔진: {engine_type}")
        print("=" * 50)

        self.loaded = True
        return True

    def generate_lipsync(self, precomputed_path, audio_input, output_dir="results/realtime", fps=25, sid=None, preloaded_data=None, frame_skip=1, resolution="720p", filename=None, text=None, output_format="mp4"):
        """
        프리컴퓨트된 아바타로 립싱크 생성 (배치 처리 + 공식 MuseTalk 블렌딩)

        Args:
            precomputed_path: 프리컴퓨트 데이터 경로
            audio_input: 오디오 파일 경로(str) 또는 (audio_numpy, sample_rate) 튜플
            output_dir: 출력 디렉토리
            fps: FPS (프리컴퓨트 데이터에서 자동 가져옴)
            sid: WebSocket 세션 ID (진행률 업데이트용)
            preloaded_data: 미리 로드된 프리컴퓨트 데이터 (병렬 처리용)
            frame_skip: 프레임 스킵 간격 (1=스킵없음, 2=2프레임당1추론, 3=3프레임당1추론)
            resolution: 출력 해상도 ("720p", "480p", "360p")
            filename: 출력 파일명 (확장자 제외, None이면 자동 생성)
            text: 대사 텍스트 (메타데이터 저장용)
            output_format: 출력 포맷 ("mp4" 또는 "webm")
        """
        import torch
        import copy
        from musetalk.utils.utils import datagen
        from concurrent.futures import ThreadPoolExecutor
        global face_parsing

        if not self.loaded:
            raise RuntimeError("모델이 로드되지 않았습니다")

        start_time = time.time()
        timings = {}  # 상세 타이밍 기록
        
        # 배치 크기 동적 조정 (GPU 메모리에 따라)
        import torch
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            if free_memory > 8 * 1024**3:  # 8GB 이상
                batch_size = 32
            elif free_memory > 4 * 1024**3:  # 4GB 이상
                batch_size = 16
            else:
                batch_size = 8
            print(f"GPU 메모리 기반 배치 크기: {batch_size} (여유 메모리: {free_memory / 1024**3:.1f}GB)")
        else:
            batch_size = 32  # CPU 모드

        # 프리컴퓨트 데이터 로드 (미리 로드된 데이터가 있으면 사용)
        t0 = time.time()
        if preloaded_data:
            print(f"프리컴퓨트 데이터 사용 (미리 로드됨): {precomputed_path}")
            coords_list = preloaded_data['coords_list']
            frames_list = preloaded_data['frames_list']
            input_latent_list = preloaded_data['input_latent_list']
            fps = preloaded_data['fps']
            precomputed = preloaded_data.get('_raw', {})  # 원본 객체 (마스크 캐싱용)
            timings['precompute_load'] = 0.0  # 이미 로드됨
        else:
            print(f"프리컴퓨트 데이터 로드 중: {precomputed_path}")
            if sid:
                emit_kwargs = {
                    'message': '프리컴퓨트 데이터 로드 중...',
                    'progress': 11,
                    'elapsed': time.time() - start_time
                }
                socketio.emit('status', emit_kwargs, to=sid)

            try:
                with open(precomputed_path, 'rb') as f:
                    precomputed = pickle.load(f)
            except FileNotFoundError:
                error_msg = f"프리컴퓨트 파일을 찾을 수 없습니다: {precomputed_path}"
                print(f"[ERROR] {error_msg}")
                raise FileNotFoundError(error_msg)
            except Exception as e:
                error_msg = f"프리컴퓨트 데이터 로드 실패: {e}"
                print(f"[ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
                raise

            # 프리컴퓨트 데이터 키 매핑 (새 형식 지원)
            coords_list = precomputed.get('coords_list', precomputed.get('coord_list_cycle'))
            frames_list = precomputed.get('frames', precomputed.get('frame_list_cycle'))
            input_latent_list = precomputed.get('input_latent_list', precomputed.get('input_latent_list_cycle'))

            # 필수 데이터 확인
            if not coords_list or not frames_list or not input_latent_list:
                error_msg = "프리컴퓨트 데이터에 필수 키가 없습니다. coords_list, frames, input_latent_list가 필요합니다."
                print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)

            # FPS 가져오기
            fps = precomputed.get('fps', 25)
            timings['precompute_load'] = time.time() - t0

        extra_margin = 10  # V1.5 bbox 하단 확장

        # 오디오 처리 (파일 경로 또는 numpy array)
        t0 = time.time()
        print("오디오 whisper 특징 추출 중...")
        if sid:
            emit_kwargs = {
                'message': '오디오 특징 추출 중...',
                'progress': 12,
                'elapsed': time.time() - start_time
            }
            socketio.emit('status', emit_kwargs, to=sid)
        
        try:
            t_audio = time.time()
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_input)
            t_audio_done = time.time()
            print(f"  - get_audio_feature: {t_audio_done - t_audio:.2f}초")
        except Exception as e:
            print(f"[ERROR] 오디오 특징 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
        t_chunk = time.time()
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        print(f"  - get_whisper_chunk: {time.time() - t_chunk:.2f}초")

        timings['whisper_feature'] = time.time() - t0

        num_frames = len(whisper_chunks)
        print(f"생성할 프레임 수: {num_frames}")
        
        if num_frames == 0:
            raise ValueError("오디오에서 프레임을 생성할 수 없습니다. 오디오 파일이 너무 짧거나 손상되었을 수 있습니다.")
        
        if sid:
            emit_kwargs = {
                'message': f'프레임 준비 완료: {num_frames}개',
                'progress': 15,
                'elapsed': time.time() - start_time
            }
            socketio.emit('status', emit_kwargs, to=sid)

        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 프레임 스킵 설정 (1=스킵없음, 2=절반 추론, 3=1/3 추론)
        frame_skip = max(1, min(frame_skip, 3))  # 1~3 사이로 제한

        # 스킵 적용: 추론할 프레임 인덱스 계산
        if frame_skip > 1:
            # 추론할 프레임 인덱스 (0, skip, 2*skip, ...)
            inference_indices = list(range(0, num_frames, frame_skip))
            # 마지막 프레임이 포함되지 않았으면 추가 (보간 품질 향상)
            if inference_indices[-1] != num_frames - 1:
                inference_indices.append(num_frames - 1)
            actual_inference_frames = len(inference_indices)
            print(f"[프레임 스킵] 전체 {num_frames}프레임 중 {actual_inference_frames}프레임만 추론 (skip={frame_skip})")
        else:
            inference_indices = list(range(num_frames))
            actual_inference_frames = num_frames

        # latent 리스트를 whisper_chunks 길이에 맞게 순환 확장 (추론할 프레임만)
        # CPU에서 로드된 latent를 GPU로 이동
        extended_latent_list = []
        selected_whisper_chunks = []
        for i in inference_indices:
            idx = i % len(input_latent_list)
            latent = input_latent_list[idx]
            # CPU 텐서를 GPU로 이동
            if isinstance(latent, torch.Tensor):
                latent = latent.to(self.device, dtype=self.weight_dtype)
            extended_latent_list.append(latent)
            selected_whisper_chunks.append(whisper_chunks[i])

        # 배치 처리를 위한 datagen 사용
        t0 = time.time()
        skip_info = f", skip={frame_skip}" if frame_skip > 1 else ""
        print(f"UNet 추론 시작 (batch_size={batch_size}{skip_info})...")
        if sid:
            emit_kwargs = {
                'message': f'UNet 추론 준비 중... (배치 크기: {batch_size}{skip_info})',
                'progress': 18,
                'elapsed': time.time() - start_time
            }
            socketio.emit('status', emit_kwargs, to=sid)

        try:
            gen = datagen(
                whisper_chunks=selected_whisper_chunks,
                vae_encode_latents=extended_latent_list,
                batch_size=batch_size,
                delay_frame=0,
                device=self.device,
            )
        except Exception as e:
            print(f"[ERROR] datagen 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

        inference_frame_list = []  # 추론된 프레임들 (스킵된 경우 일부만)
        total_batches = int(np.ceil(float(actual_inference_frames) / batch_size))

        # CUDA 최적화: inference_mode 사용 (더 빠름)
        engine_name = "TensorRT" if self.use_tensorrt else "PyTorch"
        with torch.inference_mode():
            for batch_idx, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total_batches, desc=f"UNet 추론 ({engine_name})")):
                # PE 적용
                audio_feature_batch = self.pe(whisper_batch)

                # 배치 UNet 추론 (TensorRT 또는 PyTorch)
                if self.use_tensorrt:
                    latent_batch = latent_batch.to(dtype=self.weight_dtype)
                    pred_latents = self.unet_trt(
                        latent_batch,
                        self.timesteps,
                        audio_feature_batch
                    ).sample
                else:
                    latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                    pred_latents = self.unet.model(
                        latent_batch,
                        self.timesteps,
                        encoder_hidden_states=audio_feature_batch
                    ).sample

                # 배치 VAE 디코딩
                recon_frames = self.vae.decode_latents(pred_latents)
                inference_frame_list.extend(recon_frames)  # extend 사용 (더 빠름)

                # 진행률 전송 (주기적으로 - 매 5배치마다 또는 마지막 배치)
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                    progress = 10 + int((batch_idx + 1) / total_batches * 60)  # 10~70%
                    elapsed = time.time() - start_time
                    emit_kwargs = {
                        'message': f'UNet 추론 중 ({engine_name})... {min((batch_idx+1)*batch_size, actual_inference_frames)}/{actual_inference_frames}',
                        'progress': progress,
                        'elapsed': elapsed
                    }
                    if sid:
                        socketio.emit('status', emit_kwargs, to=sid)
                    else:
                        socketio.emit('status', emit_kwargs)

                # 디버그 로그 출력 (매 10배치마다)
                if (batch_idx + 1) % 10 == 0:
                    print(f"  [{batch_idx+1}/{total_batches}] 배치 처리 중... ({int((batch_idx+1)/total_batches*100)}%)")

        timings['unet_inference'] = time.time() - t0

        # 프레임 보간 (스킵된 프레임 복원)
        if frame_skip > 1:
            # 프레임 보간 옵션 (환경변수로 제어, 기본값: false = 보간 사용)
            skip_interpolation = os.getenv('SKIP_INTERPOLATION', 'false').lower() == 'true'
            
            if skip_interpolation:
                # 보간 생략 (가장 빠름, 품질 약간 저하)
                print(f"[프레임 보간 생략] 추론된 {actual_inference_frames}프레임만 사용")
                res_frame_list = inference_frame_list
                timings['frame_interpolation'] = 0.0
            else:
                # 선형 보간 (기본, 품질 유지)
                t0_interp = time.time()
                print(f"[프레임 보간] {actual_inference_frames}프레임 -> {num_frames}프레임...")
                res_frame_list = [None] * num_frames

                # 추론된 프레임 배치
                for idx, frame_idx in enumerate(inference_indices):
                    res_frame_list[frame_idx] = inference_frame_list[idx]

                # 중간 프레임 보간 (선형 보간 - 최적화된 버전)
                for i in range(len(inference_indices) - 1):
                    start_idx = inference_indices[i]
                    end_idx = inference_indices[i + 1]
                    start_frame = res_frame_list[start_idx].astype(np.float32)
                    end_frame = res_frame_list[end_idx].astype(np.float32)
                    
                    # 중간 프레임들 보간 (벡터화된 연산)
                    num_interp = end_idx - start_idx - 1
                    if num_interp > 0:
                        alphas = np.linspace(0, 1, num_interp + 2)[1:-1]  # 시작/끝 제외
                        for j, alpha in enumerate(alphas, start=start_idx + 1):
                            interpolated = cv2.addWeighted(
                                start_frame, 1.0 - alpha,
                                end_frame, alpha, 0
                            ).astype(np.uint8)
                            res_frame_list[j] = interpolated

                timings['frame_interpolation'] = time.time() - t0_interp
                print(f"  프레임 보간 완료: {timings['frame_interpolation']:.2f}초")
        else:
            res_frame_list = inference_frame_list

        # GPU 메모리 정리
        torch.cuda.empty_cache()

        # 블렌딩 (GPU 최적화 - 마스크 프리컴퓨트)
        t0 = time.time()
        print("블렌딩 시작 (GPU 최적화)...")

        # 페이드 인/아웃 프레임 수 (약 0.3초)
        fade_frames = min(8, len(res_frame_list) // 4)
        total_frames = len(res_frame_list)

        # 마스크/crop_box 프리컴퓨트 (첫 프레임에서 1회만 계산 후 재사용)
        # 아바타 프레임은 동일한 얼굴 위치이므로 마스크 1개로 충분
        mask_cache = precomputed.get('mask_cache', None)
        crop_box_cache = precomputed.get('crop_box_cache', None)

        if mask_cache is None or crop_box_cache is None:
            print("  마스크 프리컴퓨트 중 (첫 실행시만)...")
            t_mask = time.time()
            # 첫 프레임에서만 마스크 계산 (모든 프레임 동일 위치)
            coord = coords_list[0]
            frame = frames_list[0]
            x1, y1, x2, y2 = [int(c) for c in coord]
            mask_cache, crop_box_cache = get_image_prepare_material(
                frame, [x1, y1, x2, y2],
                upper_boundary_ratio=0.5, expand=1.5,
                fp=face_parsing, mode="jaw"
            )
            # 캐시에 저장
            precomputed['mask_cache'] = mask_cache
            precomputed['crop_box_cache'] = crop_box_cache
            print(f"  마스크 프리컴퓨트 완료: {time.time() - t_mask:.2f}초")

        # 선명화 커널 프리컴퓨트 (모든 프레임에서 재사용)
        sharpen_alpha = 1.5
        sharpen_beta = -0.5

        def blend_single_frame_fast(args):
            """단일 프레임 블렌딩 함수 (마스크 재사용 - 최적화)"""
            i, res_frame = args
            idx = i % len(coords_list)
            coord = coords_list[idx]
            original_frame = frames_list[idx].copy()

            x1, y1, x2, y2 = [int(c) for c in coord]

            try:
                # 선명화 + 고품질 리사이즈 (최적화)
                res_frame_uint8 = res_frame.astype(np.uint8)

                # 언샤프 마스크 - 최적화된 버전 (더 작은 커널)
                gaussian = cv2.GaussianBlur(res_frame_uint8, (3, 3), 1.5)
                sharpened = cv2.addWeighted(res_frame_uint8, sharpen_alpha, gaussian, sharpen_beta, 0)

                # 빠른 리사이즈 (INTER_LINEAR이 LANCZOS4보다 4배 빠름, 품질 차이 미미)
                target_size = (x2 - x1, y2 - y1)
                if target_size[0] > 0 and target_size[1] > 0:
                    pred_frame_resized = cv2.resize(
                        sharpened,
                        target_size,
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    return (i, original_frame)
            except Exception as e:
                return (i, original_frame)

            # 캐시된 마스크 사용 (face_seg 재호출 없음 - 모든 프레임 동일 마스크)
            if mask_cache is not None and crop_box_cache is not None:
                # 빠른 블렌딩 (마스크 재계산 없음)
                result_frame = get_image_blending(
                    original_frame,
                    pred_frame_resized,
                    [x1, y1, x2, y2],
                    mask_cache,
                    crop_box_cache
                )
            else:
                # 폴백: 기존 방식
                result_frame = get_image(
                    original_frame,
                    pred_frame_resized,
                    [x1, y1, x2, y2],
                    mode="jaw",
                    fp=face_parsing
                )

            # 페이드 인/아웃 적용
            if i < fade_frames:
                alpha = i / fade_frames
                result_frame = cv2.addWeighted(
                    original_frame, 1 - alpha,
                    result_frame, alpha, 0
                )
            elif i >= total_frames - fade_frames:
                remaining = total_frames - i - 1
                alpha = remaining / fade_frames
                result_frame = cv2.addWeighted(
                    original_frame, 1 - alpha,
                    result_frame, alpha, 0
                )

            return (i, result_frame)

        # 병렬 블렌딩 실행 (CPU 코어 수에 맞춤)
        import os as os_module
        num_workers = min(16, os_module.cpu_count() or 8)
        generated_frames = [None] * total_frames
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(blend_single_frame_fast, enumerate(res_frame_list)),
                total=total_frames,
                desc="블렌딩"
            ))

        # 결과 정렬
        for idx, frame in results:
            generated_frames[idx] = frame

        timings['blending'] = time.time() - t0

        # ========== 아바타 영상 길이까지 원본 프레임 패딩 ==========
        # 립싱크 영상이 아바타 원본 영상보다 짧으면, 나머지는 원본 프레임으로 채움
        avatar_total_frames = len(frames_list)
        lipsync_frames = len(generated_frames)

        if lipsync_frames < avatar_total_frames:
            padding_frames = avatar_total_frames - lipsync_frames
            print(f"[프레임 패딩] 립싱크 {lipsync_frames}프레임 → 아바타 {avatar_total_frames}프레임 ({padding_frames}프레임 추가)")

            # 마지막 립싱크 프레임에서 원본으로 부드럽게 전환 (페이드 아웃)
            fade_out_frames = min(8, padding_frames)  # 페이드 아웃 프레임 수

            for i in range(padding_frames):
                frame_idx = lipsync_frames + i
                original_idx = frame_idx % avatar_total_frames
                original_frame = frames_list[original_idx].copy()

                # 해상도 조정 (필요시)
                if original_frame.shape[:2] != generated_frames[0].shape[:2]:
                    original_frame = cv2.resize(original_frame,
                        (generated_frames[0].shape[1], generated_frames[0].shape[0]))

                # 처음 몇 프레임은 립싱크 마지막 프레임과 블렌딩 (부드러운 전환)
                if i < fade_out_frames and len(generated_frames) > 0:
                    alpha = i / fade_out_frames  # 0 → 1
                    last_lipsync_frame = generated_frames[-1]
                    blended = cv2.addWeighted(last_lipsync_frame, 1 - alpha, original_frame, alpha, 0)
                    generated_frames.append(blended)
                else:
                    generated_frames.append(original_frame)

            print(f"[프레임 패딩] 완료: 총 {len(generated_frames)}프레임")

        # 진행률 전송
        emit_kwargs = {
            'message': f'블렌딩 완료',
            'progress': 80,
            'elapsed': time.time() - start_time
        }
        if sid:
            socketio.emit('status', emit_kwargs, to=sid)
        else:
            socketio.emit('status', emit_kwargs)

        # 비디오 저장 (파일명 생성)
        t0 = time.time()
        print("비디오 저장 중...")
        unique_id = str(uuid.uuid4())[:8]

        # 출력 포맷 확인 (mp4 또는 webm)
        output_format = output_format.lower() if output_format else "mp4"
        if output_format not in ["mp4", "webm"]:
            output_format = "mp4"
        file_ext = output_format

        # 파일명 생성: filename이 지정되면 사용, 아니면 text 기반 또는 uuid
        if filename:
            # 사용자 지정 파일명 사용 (특수문자 제거)
            import re
            safe_filename = re.sub(r'[\\/*?:"<>|]', '', filename)[:50]
            output_filename = f"{safe_filename}_{unique_id}"
        elif text:
            # 텍스트 기반 파일명 생성: 앞 10글자 + uuid
            import re
            # 특수문자 제거 및 공백을 언더스코어로
            safe_text = re.sub(r'[\\/*?:"<>|\n\r]', '', text)
            safe_text = re.sub(r'\s+', '_', safe_text)[:10]
            output_filename = f"{safe_text}_{unique_id}"
        else:
            output_filename = f"output_{unique_id}"

        temp_video = str(output_path / f"temp_{unique_id}.mp4")
        final_video = str(output_path / f"{output_filename}.{file_ext}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = generated_frames[0].shape[:2]
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

        for frame in tqdm(generated_frames, desc="프레임 저장"):
            writer.write(frame)
        writer.release()

        # 오디오 합성
        print("오디오 합성 중...")
        emit_kwargs = {
            'message': '오디오 합성 중...',
            'progress': 90,
            'elapsed': time.time() - start_time
        }
        if sid:
            socketio.emit('status', emit_kwargs, to=sid)
        else:
            socketio.emit('status', emit_kwargs)

        # audio_input이 튜플이면 임시 파일 생성 (고유 ID 사용)
        if isinstance(audio_input, tuple):
            import soundfile as sf
            temp_audio = str(output_path / f"temp_audio_{unique_id}.wav")
            audio_numpy, sample_rate = audio_input
            sf.write(temp_audio, audio_numpy, sample_rate)
            audio_file_path = temp_audio
        else:
            audio_file_path = audio_input

        # 해상도 스케일 필터 설정 (원본: 1268x724)
        # 모든 영상과 일치시키기 위해 1268x724 비율 유지
        scale_filters = {
            "720p": None,  # 원본 유지 (1268x724)
            "480p": "scale=848:484",  # 원본 비율 유지 (1268:724 ≈ 1.75:1)
            "360p": "scale=632:360"   # 원본 비율 유지
        }
        scale_filter = scale_filters.get(resolution)

        # 중앙 크롭 필터 (면접관 중앙 영역만 출력)
        center_crop = center_crop if 'center_crop' in dir() else CENTER_CROP_ENABLED
        crop_filter = None
        crop_coords = None
        if center_crop and resolution in CENTER_CROP_COORDS:
            x_start, x_end, crop_w, crop_h = CENTER_CROP_COORDS[resolution]
            crop_filter = f"crop={crop_w}:{crop_h}:{x_start}:0"
            crop_coords = CENTER_CROP_COORDS[resolution]
            print(f"  중앙 크롭: {resolution} → {crop_w}x{crop_h} (x={x_start})")

        if scale_filter:
            print(f"  해상도 스케일링: {resolution}")

        # 메타데이터 준비 (대사 텍스트 저장)
        metadata_args = []
        if text:
            # ffmpeg 메타데이터: title, comment, description
            # 특수문자 이스케이프 (ffmpeg용)
            safe_meta_text = text.replace('"', '\\"').replace("'", "\\'")
            metadata_args = [
                '-metadata', f'title={safe_meta_text[:100]}',  # 제목 (앞 100자)
                '-metadata', f'comment={safe_meta_text}',      # 코멘트 (전체 대사)
                '-metadata', f'description={safe_meta_text}',  # 설명
            ]
            print(f"  메타데이터 저장: {text[:30]}...")

        # 포맷에 따라 인코딩 방식 선택
        if output_format == "webm":
            # WebM 인코딩 - GPU (AV1 NVENC) 우선, CPU (VP9) 폴백
            # AV1 NVENC 시도 (RTX 40/50 시리즈 지원)
            print(f"  WebM 인코딩 시도 중 (AV1 NVENC)...")
            ffmpeg_cmd_av1 = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', audio_file_path,
            ]
            # 비디오 필터: 스케일 + 크롭 + 마지막 프레임 150ms 연장
            vf_parts = ['tpad=stop_mode=clone:stop_duration=0.15']
            if scale_filter:
                vf_parts.append(scale_filter)
            if crop_filter:
                vf_parts.append(crop_filter)
            ffmpeg_cmd_av1.extend(['-vf', ','.join(vf_parts)])
            # 오디오 필터: 앞에 120ms 무음 추가 (립싱크 동기화)
            ffmpeg_cmd_av1.extend(['-af', 'adelay=120|120'])
            ffmpeg_cmd_av1.extend(metadata_args)
            ffmpeg_cmd_av1.extend([
                '-c:v', 'av1_nvenc',     # AV1 NVENC (GPU 인코딩)
                '-preset', 'p1',          # 가장 빠른 프리셋
                '-cq', '30',              # 품질 수준
                '-c:a', 'libopus',        # Opus 오디오
                '-b:a', '128k',
                '-shortest',  # 비디오/오디오 중 짧은 것에 맞춤
                final_video
            ])
            av1_result = subprocess.run(ffmpeg_cmd_av1, capture_output=True)

            if av1_result.returncode != 0:
                # AV1 NVENC 실패 시 VP9 CPU 폴백
                print(f"  AV1 NVENC 실패, VP9 CPU 인코딩으로 폴백...")
                ffmpeg_cmd_vp9 = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', audio_file_path,
                ]
                # 비디오 필터: 스케일 + 크롭 + 마지막 프레임 150ms 연장
                vf_parts = ['tpad=stop_mode=clone:stop_duration=0.15']
                if scale_filter:
                    vf_parts.append(scale_filter)
                if crop_filter:
                    vf_parts.append(crop_filter)
                ffmpeg_cmd_vp9.extend(['-vf', ','.join(vf_parts)])
                # 오디오 필터: 앞에 120ms 무음 추가 (립싱크 동기화)
                ffmpeg_cmd_vp9.extend(['-af', 'adelay=120|120'])
                ffmpeg_cmd_vp9.extend(metadata_args)
                ffmpeg_cmd_vp9.extend([
                    '-c:v', 'libvpx-vp9',   # VP9 코덱
                    '-crf', '30',            # 품질
                    '-b:v', '0',             # CRF 모드
                    '-cpu-used', '4',        # 빠른 인코딩
                    '-row-mt', '1',          # 멀티스레드
                    '-c:a', 'libopus',
                    '-b:a', '128k',
                    '-shortest',  # 비디오/오디오 중 짧은 것에 맞춤
                    final_video
                ])
                vp9_result = subprocess.run(ffmpeg_cmd_vp9, capture_output=True)
                if vp9_result.returncode != 0:
                    print(f"  WebM 인코딩 오류: {vp9_result.stderr.decode()[:200]}")
                else:
                    print("  WebM (VP9 CPU) 인코딩 완료")
            else:
                print("  WebM (AV1 NVENC GPU) 인코딩 완료")
        else:
            # MP4 (H.264) 인코딩 - GPU 우선, CPU 폴백
            # GPU 인코딩 시도 (NVENC)
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', audio_file_path,
            ]

            # 비디오 필터: 스케일 + 크롭 + 마지막 프레임 150ms 연장
            vf_parts = ['tpad=stop_mode=clone:stop_duration=0.15']
            if scale_filter:
                vf_parts.append(scale_filter)
            if crop_filter:
                vf_parts.append(crop_filter)
            ffmpeg_cmd.extend(['-vf', ','.join(vf_parts)])

            # 오디오 필터: 앞에 120ms 무음 추가 (립싱크 동기화)
            ffmpeg_cmd.extend(['-af', 'adelay=120|120'])

            # 메타데이터 추가
            ffmpeg_cmd.extend(metadata_args)

            ffmpeg_cmd.extend([
                '-c:v', 'h264_nvenc',  # NVIDIA GPU 인코더
                '-preset', 'p1',       # 가장 빠른 프리셋 (p1=fastest, p7=slowest)
                '-tune', 'ull',        # Ultra Low Latency 튜닝
                '-rc', 'vbr',          # 가변 비트레이트
                '-cq', '28',           # 품질 수준 (낮을수록 고품질)
                '-c:a', 'aac',
                '-shortest',  # 비디오/오디오 중 짧은 것에 맞춤
                final_video
            ])

            nvenc_result = subprocess.run(ffmpeg_cmd, capture_output=True)

            # NVENC 실패시 CPU 인코딩으로 폴백
            if nvenc_result.returncode != 0:
                print("  NVENC 실패, CPU 인코딩 사용")
                ffmpeg_cmd_cpu = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', audio_file_path,
                ]
                # 비디오 필터: 스케일 + 마지막 프레임 150ms 연장
                vf_parts = ['tpad=stop_mode=clone:stop_duration=0.15']
                if scale_filter:
                    vf_parts.append(scale_filter)
                ffmpeg_cmd_cpu.extend(['-vf', ','.join(vf_parts)])
                # 오디오 필터: 앞에 120ms 무음 추가 (립싱크 동기화)
                ffmpeg_cmd_cpu.extend(['-af', 'adelay=120|120'])
                # 메타데이터 추가
                ffmpeg_cmd_cpu.extend(metadata_args)
                ffmpeg_cmd_cpu.extend([
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',  # 가장 빠른 CPU 프리셋
                    '-c:a', 'aac',
                    '-shortest',  # 비디오/오디오 중 짧은 것에 맞춤
                    final_video
                ])
                subprocess.run(ffmpeg_cmd_cpu, capture_output=True)
            else:
                print("  NVENC GPU 인코딩 사용")

        # 임시 오디오 파일 삭제
        if isinstance(audio_input, tuple) and os.path.exists(temp_audio):
            os.remove(temp_audio)

        # 임시 파일 삭제
        if os.path.exists(temp_video):
            os.remove(temp_video)

        timings['video_encoding'] = time.time() - t0

        # JSON 사이드카 파일 생성 (대사 및 메타정보 저장)
        if text and os.path.exists(final_video):
            import json
            from datetime import datetime
            json_path = final_video.replace('.mp4', '.json')
            metadata = {
                'text': text,
                'filename': os.path.basename(final_video),
                'created_at': datetime.now().isoformat(),
                'resolution': resolution,
                'duration_sec': round(num_frames / fps, 2),
                'fps': fps,
                'frames': num_frames
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"  JSON 메타데이터 저장: {json_path}")

        # 표준 파일명으로 복사 (기존 코드 호환성)
        standard_output = str(output_path / "output_with_audio.mp4")
        if os.path.exists(final_video):
            import shutil
            shutil.copy2(final_video, standard_output)
            print(f"표준 파일명으로 복사: {standard_output}")

        elapsed = time.time() - start_time
        timings['total'] = elapsed

        # 상세 타이밍 출력
        print(f"\n립싱크 생성 완료! ({elapsed:.1f}초)")
        print(f"  [타이밍 상세]")
        print(f"    - 프리컴퓨트 로드: {timings.get('precompute_load', 0):.2f}초")
        print(f"    - Whisper 특징추출: {timings.get('whisper_feature', 0):.2f}초")
        print(f"    - UNet 추론: {timings.get('unet_inference', 0):.2f}초")
        print(f"    - 블렌딩: {timings.get('blending', 0):.2f}초")
        print(f"    - 비디오 인코딩: {timings.get('video_encoding', 0):.2f}초")
        if crop_coords:
            print(f"    - 중앙 크롭: {crop_coords[2]}x{crop_coords[3]} (x={crop_coords[0]})")
        print(f"출력: {final_video}")
        print(f"표준 출력: {standard_output}")

        # 반환값: (비디오 경로, 크롭 좌표) 또는 비디오 경로만 (하위 호환)
        return (standard_output, crop_coords) if crop_coords else standard_output

    def generate_lipsync_streaming(self, precomputed_path, audio_input, sid, emit_callback, preloaded_data=None, frame_skip=1):
        """
        스트리밍 립싱크 생성 - 프레임 생성 즉시 WebSocket으로 전송

        Args:
            precomputed_path: 프리컴퓨트 데이터 경로
            audio_input: 오디오 파일 경로 또는 (audio_numpy, sample_rate) 튜플
            sid: WebSocket 세션 ID
            emit_callback: WebSocket emit 함수
            preloaded_data: 미리 로드된 프리컴퓨트 데이터
            frame_skip: 프레임 스킵 간격
        """
        import torch
        import base64
        from musetalk.utils.utils import datagen
        global face_parsing

        if not self.loaded:
            raise RuntimeError("모델이 로드되지 않았습니다")

        start_time = time.time()
        batch_size = 8  # 스트리밍용 작은 배치
        lipsync_timing = {}  # 립싱크 내부 타이밍

        # 프리컴퓨트 데이터 로드
        step_start = time.time()
        if preloaded_data:
            coords_list = preloaded_data['coords_list']
            frames_list = preloaded_data['frames_list']
            input_latent_list = preloaded_data['input_latent_list']
            fps = preloaded_data['fps']
        else:
            with open(precomputed_path, 'rb') as f:
                precomputed = pickle.load(f)
            coords_list = precomputed.get('coords_list', precomputed.get('coord_list_cycle'))
            frames_list = precomputed.get('frames', precomputed.get('frame_list_cycle'))
            input_latent_list = precomputed.get('input_latent_list', precomputed.get('input_latent_list_cycle'))
            fps = precomputed.get('fps', 25)

        lipsync_timing['data_load'] = time.time() - step_start
        extra_margin = 10

        # 오디오 처리
        step_start = time.time()
        emit_callback('status', {'message': '오디오 처리 중...', 'progress': 5, 'elapsed': time.time() - start_time}, to=sid)

        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_input)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )

        num_frames = len(whisper_chunks)
        lipsync_timing['audio_process'] = time.time() - step_start

        if num_frames == 0:
            raise ValueError("오디오에서 프레임을 생성할 수 없습니다")

        # 프레임 스킵 설정
        frame_skip = max(1, min(frame_skip, 3))
        if frame_skip > 1:
            inference_indices = list(range(0, num_frames, frame_skip))
            if inference_indices[-1] != num_frames - 1:
                inference_indices.append(num_frames - 1)
        else:
            inference_indices = list(range(num_frames))

        # latent 준비
        extended_latent_list = []
        selected_whisper_chunks = []
        for i in inference_indices:
            idx = i % len(input_latent_list)
            latent = input_latent_list[idx]
            if isinstance(latent, torch.Tensor):
                latent = latent.to(self.device, dtype=self.weight_dtype)
            extended_latent_list.append(latent)
            selected_whisper_chunks.append(whisper_chunks[i])

        # 스트리밍 시작 신호
        emit_callback('stream_start', {
            'total_frames': num_frames,
            'fps': fps,
            'elapsed': time.time() - start_time
        }, to=sid)

        # UNet 추론 시작
        step_start = time.time()

        # 페이드 프레임 수
        fade_frames = min(8, num_frames // 4)

        # 배치 단위로 처리하며 즉시 전송
        gen = datagen(
            whisper_chunks=selected_whisper_chunks,
            vae_encode_latents=extended_latent_list,
            batch_size=batch_size,
            delay_frame=0,
            device=self.device,
        )

        inference_results = {}  # 추론 결과 저장 (인덱스 -> 프레임)
        batch_idx = 0

        for whisper_batch, latent_batch in gen:
            # UNet 추론
            latent_batch = latent_batch.to(dtype=self.weight_dtype)

            with torch.no_grad():
                if self.use_tensorrt:
                    pred_latents = self.unet_session.run(
                        None,
                        {
                            "sample": latent_batch.cpu().numpy(),
                            "audio_embedding": whisper_batch.cpu().numpy()
                        }
                    )[0]
                    pred_latents = torch.from_numpy(pred_latents).to(self.device)
                else:
                    pred_latents = self.unet(latent_batch, whisper_batch)

            # VAE 디코딩
            recon_frames = self.vae.decode_latents(pred_latents)

            # 배치 내 각 프레임 처리 및 즉시 전송
            for i, res_frame in enumerate(recon_frames):
                global_idx = batch_idx * batch_size + i
                if global_idx >= len(inference_indices):
                    break

                original_frame_idx = inference_indices[global_idx]
                inference_results[original_frame_idx] = res_frame

            batch_idx += 1

            # 진행률 업데이트
            progress = min(80, 10 + int(70 * batch_idx * batch_size / len(inference_indices)))
            emit_callback('status', {
                'message': f'추론 중... ({min(batch_idx * batch_size, len(inference_indices))}/{len(inference_indices)})',
                'progress': progress,
                'elapsed': time.time() - start_time
            }, to=sid)

        lipsync_timing['unet_inference'] = time.time() - step_start

        # 프레임 보간 (frame_skip > 1인 경우)
        step_start = time.time()
        all_frames = {}
        if frame_skip > 1:
            for i in range(num_frames):
                if i in inference_results:
                    all_frames[i] = inference_results[i]
                else:
                    # 선형 보간
                    prev_idx = (i // frame_skip) * frame_skip
                    next_idx = min(prev_idx + frame_skip, num_frames - 1)
                    if next_idx in inference_results and prev_idx in inference_results:
                        alpha = (i - prev_idx) / (next_idx - prev_idx) if next_idx != prev_idx else 0
                        all_frames[i] = cv2.addWeighted(
                            inference_results[prev_idx].astype(np.float32), 1 - alpha,
                            inference_results[next_idx].astype(np.float32), alpha, 0
                        ).astype(np.uint8)
                    else:
                        all_frames[i] = inference_results.get(prev_idx, inference_results[min(inference_results.keys())])
        else:
            all_frames = inference_results

        lipsync_timing['interpolation'] = time.time() - step_start

        # 블렌딩 및 프레임 전송
        step_start = time.time()
        emit_callback('status', {'message': '프레임 전송 중...', 'progress': 85, 'elapsed': time.time() - start_time}, to=sid)

        for i in range(num_frames):
            res_frame = all_frames.get(i)
            if res_frame is None:
                continue

            idx = i % len(coords_list)
            coord = coords_list[idx]
            original_frame = frames_list[idx].copy()
            x1, y1, x2, y2 = [int(c) for c in coord]

            try:
                # 선명화 + 리사이즈
                res_frame_uint8 = res_frame.astype(np.uint8)
                gaussian = cv2.GaussianBlur(res_frame_uint8, (0, 0), 2.0)
                sharpened = cv2.addWeighted(res_frame_uint8, 1.5, gaussian, -0.5, 0)
                pred_frame_resized = cv2.resize(sharpened, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
            except Exception:
                continue

            # 블렌딩
            result_frame = get_image(original_frame, pred_frame_resized, [x1, y1, x2, y2], mode="jaw", fp=face_parsing)

            # 페이드 처리
            if i < fade_frames:
                alpha = i / fade_frames
                result_frame = cv2.addWeighted(original_frame, 1 - alpha, result_frame, alpha, 0)
            elif i >= num_frames - fade_frames:
                remaining = num_frames - i - 1
                alpha = remaining / fade_frames
                result_frame = cv2.addWeighted(original_frame, 1 - alpha, result_frame, alpha, 0)

            # JPEG 인코딩 및 Base64 변환
            _, buffer = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # 프레임 전송
            emit_callback('stream_frame', {
                'frame': frame_base64,
                'index': i,
                'total': num_frames,
                'fps': fps
            }, to=sid)

        lipsync_timing['blending_send'] = time.time() - step_start

        # 완료 신호
        elapsed = time.time() - start_time
        emit_callback('stream_complete', {
            'total_frames': num_frames,
            'elapsed': elapsed,
            'fps': fps
        }, to=sid)

        # ========== 립싱크 내부 성능 측정 결과 ==========
        print(f"\n[립싱크 성능측정] 총 {elapsed:.2f}초, {num_frames}프레임")
        print(f"  - 데이터 로드: {lipsync_timing.get('data_load', 0):.2f}초")
        print(f"  - 오디오 처리 (Whisper): {lipsync_timing.get('audio_process', 0):.2f}초")
        print(f"  - UNet 추론: {lipsync_timing.get('unet_inference', 0):.2f}초")
        print(f"  - 프레임 보간: {lipsync_timing.get('interpolation', 0):.2f}초")
        print(f"  - 블렌딩+전송: {lipsync_timing.get('blending_send', 0):.2f}초")

        return True

    def process_audio_chunk_to_frames(self, audio_chunk, sample_rate, preloaded_data, frame_offset=0):
        """
        오디오 청크를 프레임으로 변환 (파이프라인 병렬화용)

        Args:
            audio_chunk: 오디오 numpy 배열
            sample_rate: 샘플레이트
            preloaded_data: 프리로드된 아바타 데이터
            frame_offset: 시작 프레임 오프셋

        Returns:
            list: 생성된 프레임 리스트
        """
        import torch
        from musetalk.utils.utils import datagen

        if not self.loaded:
            raise RuntimeError("모델이 로드되지 않았습니다")

        fps = preloaded_data.get('fps', 25)
        input_latent_list = preloaded_data['input_latent_list']
        coords_list = preloaded_data['coords_list']
        frames_list = preloaded_data['frames_list']

        # Whisper 특징 추출
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
            (audio_chunk, sample_rate)
        )

        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )

        num_frames = len(whisper_chunks)
        if num_frames == 0:
            return []

        # latent 준비
        extended_latent_list = []
        for i in range(num_frames):
            idx = (frame_offset + i) % len(input_latent_list)
            latent = input_latent_list[idx]
            if isinstance(latent, torch.Tensor):
                latent = latent.to(self.device, dtype=self.weight_dtype)
            extended_latent_list.append(latent)

        # 배치 처리
        batch_size = 16
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=extended_latent_list,
            batch_size=batch_size,
            delay_frame=0,
            device=self.device,
        )

        generated_frames = []

        with torch.inference_mode():
            for whisper_batch, latent_batch in gen:
                audio_feature_batch = self.pe(whisper_batch)

                if self.use_tensorrt:
                    latent_batch = latent_batch.to(dtype=self.weight_dtype)
                    pred_latents = self.unet_trt(
                        latent_batch,
                        self.timesteps,
                        audio_feature_batch
                    ).sample
                else:
                    latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                    pred_latents = self.unet.model(
                        latent_batch,
                        self.timesteps,
                        encoder_hidden_states=audio_feature_batch
                    ).sample

                recon_frames = self.vae.decode_latents(pred_latents)
                generated_frames.extend(recon_frames)

        # 블렌딩
        blended_frames = []
        global face_parsing
        extra_margin = 10

        for i, res_frame in enumerate(generated_frames):
            idx = (frame_offset + i) % len(coords_list)
            coord = coords_list[idx]
            original_frame = frames_list[idx].copy()

            x1, y1, x2, y2 = [int(c) for c in coord]
            y2_extended = min(y2 + extra_margin, original_frame.shape[0])

            try:
                res_frame_uint8 = res_frame.astype(np.uint8)
                gaussian = cv2.GaussianBlur(res_frame_uint8, (0, 0), 2.0)
                sharpened = cv2.addWeighted(res_frame_uint8, 1.5, gaussian, -0.5, 0)
                pred_frame_resized = cv2.resize(
                    sharpened, (x2 - x1, y2_extended - y1),
                    interpolation=cv2.INTER_LANCZOS4
                )

                result_frame = get_image(
                    original_frame, pred_frame_resized,
                    [x1, y1, x2, y2_extended], mode="jaw", fp=face_parsing
                )
                blended_frames.append(result_frame)
            except Exception as e:
                blended_frames.append(original_frame)

        return blended_frames


# TTS 엔진 설정 - CosyVoice (기본) + ElevenLabs
TTS_ENGINES = {
    'cosyvoice': {
        'name': 'CosyVoice (로컬 서버)',
        'voices': ['default'],
        'default': True
    },
    'elevenlabs': {
        'name': 'ElevenLabs',
        'voices': ['Custom'],
        'default': False
    }
}

# 기본 TTS 엔진 - CosyVoice (로컬 서버)
DEFAULT_TTS_ENGINE = 'cosyvoice'
DEFAULT_TTS_VOICE = 'default'

# API 설정
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '')
COSYVOICE_API_URL = os.getenv('COSYVOICE_API_URL', 'http://172.16.10.200:5000')

# HTTP 세션 풀링 (Keep-Alive 연결 재사용)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_http_session(max_retries=3, pool_connections=10, pool_maxsize=20):
    """Keep-Alive 연결을 재사용하는 HTTP 세션 생성"""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# 전역 HTTP 세션 (TTS API용 - 연결 재사용)
tts_http_session = create_http_session()


def get_available_avatars():
    """사전 계산된 아바타 목록 조회"""
    precomputed_dir = Path("precomputed")
    if not precomputed_dir.exists():
        return []

    avatars = []
    for pkl_file in precomputed_dir.glob("*_precomputed.pkl"):
        name = pkl_file.stem.replace("_precomputed", "")

        # 미리보기 이미지/비디오 찾기
        preview = None
        for ext in ['.mp4', '.png', '.jpg']:
            preview_path = Path("assets/images") / f"{name}{ext}"
            if preview_path.exists():
                preview = str(preview_path)
                break

        avatars.append({
            "id": name,
            "name": name,
            "path": str(pkl_file),
            "preview": preview
        })
    return avatars


def get_available_audios():
    """사용 가능한 오디오 목록 조회"""
    audio_dir = Path("assets/audio")
    if not audio_dir.exists():
        return []

    audios = []
    for audio_file in audio_dir.glob("*.mp3"):
        audios.append({
            "id": audio_file.stem,
            "name": audio_file.name,
            "path": str(audio_file)
        })
    for audio_file in audio_dir.glob("*.wav"):
        audios.append({
            "id": audio_file.stem,
            "name": audio_file.name,
            "path": str(audio_file)
        })
    return audios


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/record')
def record_page():
    """영상 녹화 페이지"""
    return render_template('record.html')


@app.route('/api/avatars')
def api_avatars():
    """아바타 목록 API"""
    avatars = get_available_avatars()
    return jsonify(avatars)


@app.route('/api/audios')
def api_audios():
    """오디오 목록 API"""
    audios = get_available_audios()
    return jsonify(audios)


@app.route('/api/queue_status')
def api_queue_status():
    """큐 상태 확인 API (GPU 메모리 정보 포함)"""
    status = generation_queue.get_status()

    # GPU 메모리 정보 추가
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                status['gpu'] = {
                    'memory_total_mb': int(parts[0]),
                    'memory_used_mb': int(parts[1]),
                    'memory_free_mb': int(parts[2]),
                    'utilization_percent': int(parts[3])
                }
    except Exception as e:
        status['gpu'] = {'error': str(e)}

    return jsonify(status)


@app.route('/api/queue_position/<sid>')
def api_queue_position(sid):
    """특정 클라이언트의 대기 순서 확인"""
    position = generation_queue.get_position(sid)
    return jsonify({
        'sid': sid,
        'position': position,
        'status': 'processing' if position == 0 else ('queued' if position > 0 else 'not_in_queue')
    })


def load_precomputed_data(precomputed_path):
    """프리컴퓨트 데이터 로드 (병렬화용 + 메모리 캐싱)"""
    global precomputed_cache

    # 캐시 확인
    if precomputed_path in precomputed_cache:
        print(f"프리컴퓨트 캐시 사용: {precomputed_path}")
        precomputed = precomputed_cache[precomputed_path]
    else:
        with open(precomputed_path, 'rb') as f:
            precomputed = pickle.load(f)
        # 캐시에 저장
        precomputed_cache[precomputed_path] = precomputed
        print(f"프리컴퓨트 로드 및 캐시 저장: {precomputed_path}")

    coords_list = precomputed.get('coords_list', precomputed.get('coord_list_cycle'))
    frames_list = precomputed.get('frames', precomputed.get('frame_list_cycle'))
    input_latent_list = precomputed.get('input_latent_list', precomputed.get('input_latent_list_cycle'))
    fps = precomputed.get('fps', 25)

    return {
        'coords_list': coords_list,
        'frames_list': frames_list,
        'input_latent_list': input_latent_list,
        'fps': fps,
        '_raw': precomputed  # 원본 객체 (마스크 캐싱용)
    }


def get_elevenlabs_voice_id(voice_name):
    """ElevenLabs 음성 이름을 ID로 변환"""
    voice_ids = {
        'Custom': '2gbExjiWDnG1DMGr81Bx',
        'Rachel': '21m00Tcm4TlvDq8ikWAM',
        'Adam': 'pNInz6obpgDQGcFmaJgB',
        'Bella': 'EXAVITQu4vr4xnSDxMaL',
        'Antoni': 'ErXwobaYiN019PkySvjV',
    }
    return voice_ids.get(voice_name, voice_ids['Custom'])


def trim_audio_silence(audio_data, sample_rate=16000, threshold_db=-50, min_silence_duration=0.3):
    """
    오디오 끝부분의 무음 트리밍 (보수적 설정)
    
    Args:
        audio_data: numpy 오디오 배열
        sample_rate: 샘플레이트
        threshold_db: 무음 판정 임계값 (dB) - 더 낮을수록 보수적
        min_silence_duration: 연속 무음이 이 시간 이상이어야 트리밍 (초)
    
    Returns:
        트리밍된 오디오 배열
    """
    if len(audio_data) == 0:
        return audio_data
    
    # dB를 진폭으로 변환
    threshold = 10 ** (threshold_db / 20)
    
    # 윈도우 설정
    window_size = int(sample_rate * 0.05)  # 50ms 윈도우
    min_silence_samples = int(min_silence_duration * sample_rate)  # 최소 무음 지속 시간
    
    # 끝에서부터 검사하여 연속 무음 구간 찾기
    silence_start = len(audio_data)
    consecutive_silence = 0
    
    for i in range(len(audio_data) - window_size, 0, -window_size):
        window = audio_data[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        
        if rms < threshold:
            consecutive_silence += window_size
            silence_start = i
        else:
            # 소리가 있는 부분 발견
            if consecutive_silence >= min_silence_samples:
                # 충분한 무음이 있었으면 여기서 자름 (200ms 여유)
                end_idx = min(silence_start + int(sample_rate * 0.2), len(audio_data))
                trimmed = audio_data[:end_idx]
                if len(trimmed) < len(audio_data):
                    trim_duration = (len(audio_data) - len(trimmed)) / sample_rate
                    print(f"[오디오 트리밍] 끝부분 무음 {trim_duration:.2f}초 제거")
                return trimmed
            else:
                # 무음이 충분하지 않으면 리셋
                consecutive_silence = 0
                silence_start = len(audio_data)
    
    # 트리밍하지 않고 원본 반환
    return audio_data


def split_text_into_sentences(text, min_chunk_length=50, max_chunk_length=200):
    """
    텍스트를 문장 단위로 분할하고, 짧은 문장은 묶어서 반환
    
    Args:
        text: 입력 텍스트
        min_chunk_length: 최소 청크 길이 (이보다 짧으면 묶음)
        max_chunk_length: 최대 청크 길이 (이보다 길면 분할)
    
    Returns:
        List[str]: 분할된 텍스트 청크들
    """
    import re
    
    # 문장 종결 문자로 분할 (., !, ?, 。, ！, ？)
    sentence_endings = r'[.!?。！？]\s*'
    sentences = re.split(sentence_endings, text)
    
    # 빈 문장 제거 및 공백 정리
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 문장 끝에 구두점 추가 (분할 시 제거되었으므로)
        if not sentence.endswith(('.', '!', '?', '。', '！', '？')):
            sentence += '.'
        
        # 현재 청크에 추가했을 때 길이 확인
        potential_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        
        if len(potential_chunk) > max_chunk_length:
            # 현재 청크가 너무 길어지면 저장하고 새로 시작
            if current_chunk:
                chunks.append(current_chunk)
            # 단일 문장이 너무 길면 강제 분할
            if len(sentence) > max_chunk_length:
                # 긴 문장을 강제로 분할 (쉼표나 공백 기준)
                parts = re.split(r'[,，]\s*', sentence)
                temp_chunk = ""
                for part in parts:
                    if len(temp_chunk + part) > max_chunk_length:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = part
                    else:
                        temp_chunk = (temp_chunk + ", " + part).strip() if temp_chunk else part
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                current_chunk = sentence
        elif len(potential_chunk) < min_chunk_length:
            # 짧은 문장은 묶어서 처리
            current_chunk = potential_chunk
        else:
            # 적절한 길이면 저장하고 새로 시작
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]


def generate_tts_audio_streaming(text, engine, voice, chunk_callback, output_path=None):
    """
    TTS 오디오 스트리밍 생성 - 문장 단위로 처리, 청크를 받는 대로 콜백 호출

    Args:
        text: 합성할 텍스트
        engine: TTS 엔진 ('cosyvoice' 또는 'elevenlabs')
        voice: 음성 이름
        chunk_callback: 청크를 받을 때마다 호출되는 콜백 함수 (chunk_data, sample_rate, is_final)
        output_path: 저장 경로 (선택사항)

    Returns:
        tuple: (audio_numpy, sample_rate) - 성공 시 전체 오디오와 샘플레이트 반환
        None: 실패 시
    """
    import torch
    import torchaudio
    import numpy as np
    import soundfile as sf
    import io
    import hashlib

    print(f"[TTS Streaming] engine={engine}, voice={voice}, text={text[:50]}...")

    # 캐시 키 생성 (텍스트 + 엔진 + 음성)
    cache_key = hashlib.md5(f"{text}|{engine}|{voice}".encode()).hexdigest()

    # 캐시 확인
    global tts_audio_cache
    if cache_key in tts_audio_cache:
        cached_audio, cached_sr, cached_time = tts_audio_cache[cache_key]
        if time.time() - cached_time < TTS_CACHE_TTL:
            print(f"[TTS] 캐시 히트! (key={cache_key[:8]}..., 길이={len(cached_audio)/cached_sr:.2f}초)")

            # 캐시된 오디오를 청크로 분할하여 전송
            chunk_size = int(cached_sr * 2)  # 2초 단위
            for i in range(0, len(cached_audio), chunk_size):
                chunk = cached_audio[i:i + chunk_size]
                is_last = (i + chunk_size >= len(cached_audio))

                chunk_wav_buffer = io.BytesIO()
                sf.write(chunk_wav_buffer, chunk, cached_sr, format='WAV')
                chunk_wav_buffer.seek(0)
                import base64
                chunk_base64 = base64.b64encode(chunk_wav_buffer.read()).decode('utf-8')
                chunk_callback(chunk_base64, cached_sr, is_last)

            chunk_callback("", cached_sr, True)

            if output_path:
                sf.write(str(output_path), cached_audio, cached_sr)

            return (cached_audio, cached_sr)
        else:
            # 만료된 캐시 삭제
            del tts_audio_cache[cache_key]

    # 텍스트를 문장 단위로 분할
    text_chunks = split_text_into_sentences(text, min_chunk_length=50, max_chunk_length=200)
    print(f"[TTS Streaming] 텍스트를 {len(text_chunks)}개 청크로 분할")

    all_audio_chunks = []
    final_sample_rate = 16000

    if engine == 'cosyvoice':
        # CosyVoice TTS (병렬 처리 최적화 + Keep-Alive 연결 재사용)
        import librosa
        import base64
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = time.time()

        # 단일 청크 TTS 요청 함수 (세션 풀링 사용)
        def fetch_tts_chunk(chunk_idx, text_chunk):
            """단일 청크 TTS 요청 (병렬 처리용 - Keep-Alive 연결 재사용)"""
            try:
                # 전역 HTTP 세션 사용 (연결 재사용으로 오버헤드 감소)
                response = tts_http_session.post(
                    f"{COSYVOICE_API_URL}/api/tts/stream",
                    json={"text": text_chunk, "speed": 1.0},
                    stream=True,
                    timeout=120
                )

                if response.status_code == 200:
                    # 큰 청크로 데이터 수집 (64KB - 네트워크 오버헤드 감소)
                    wav_chunks = []
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            wav_chunks.append(chunk)
                    wav_data = b''.join(wav_chunks)

                    # numpy로 변환
                    wav_buffer = io.BytesIO(wav_data)
                    audio_data, sr = sf.read(wav_buffer)

                    # 스테레오 → 모노
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)

                    # 16kHz 리샘플링
                    if sr != 16000:
                        from scipy import signal
                        target_len = int(len(audio_data) * 16000 / sr)
                        audio_data = signal.resample(audio_data, target_len)

                    return (chunk_idx, audio_data.astype(np.float32), None)
                else:
                    return (chunk_idx, None, f"HTTP {response.status_code}")

            except Exception as e:
                return (chunk_idx, None, str(e))

        # 병렬 TTS 요청 (최대 3개 동시 요청 - 서버 부하 고려)
        max_parallel = min(3, len(text_chunks))
        results = [None] * len(text_chunks)
        failed = False

        print(f"[CosyVoice] {len(text_chunks)}개 청크 병렬 처리 시작 (max_parallel={max_parallel})")

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(fetch_tts_chunk, idx, chunk): idx
                for idx, chunk in enumerate(text_chunks)
            }

            for future in as_completed(futures):
                chunk_idx, audio_data, error = future.result()

                if error:
                    print(f"[CosyVoice] 청크 {chunk_idx + 1} 실패: {error}")
                    failed = True
                    break
                else:
                    results[chunk_idx] = audio_data
                    print(f"[CosyVoice] 청크 {chunk_idx + 1}/{len(text_chunks)} 완료 (길이: {len(audio_data)/16000:.2f}초)")

        if failed:
            print("[TTS] ElevenLabs로 폴백합니다...")
            return generate_tts_audio_streaming(text, 'elevenlabs', 'Custom', chunk_callback, output_path)

        # 순서대로 클라이언트에 전송 (순차적으로 스트리밍)
        for chunk_idx, audio_numpy in enumerate(results):
            if audio_numpy is None:
                continue

            is_last_chunk = (chunk_idx == len(results) - 1)
            all_audio_chunks.append(audio_numpy)

            # 클라이언트로 전송
            chunk_wav_buffer = io.BytesIO()
            sf.write(chunk_wav_buffer, audio_numpy, 16000)
            chunk_wav_buffer.seek(0)
            chunk_base64 = base64.b64encode(chunk_wav_buffer.read()).decode('utf-8')
            chunk_callback(chunk_base64, 16000, is_last_chunk)

        # 모든 청크를 합쳐서 최종 오디오 생성
        if all_audio_chunks:
            final_audio = np.concatenate(all_audio_chunks)
            final_sample_rate = 16000
            
            # 끝부분 무음/불필요한 소리 트리밍
            final_audio = trim_audio_silence(final_audio, final_sample_rate)

            # 최종 완료 신호
            chunk_callback("", final_sample_rate, True)

            if output_path:
                sf.write(str(output_path), final_audio, final_sample_rate)

            # 캐시에 저장 (LRU 방식)
            if len(tts_audio_cache) >= TTS_CACHE_MAX_SIZE:
                # 가장 오래된 항목 삭제
                oldest_key = min(tts_audio_cache.keys(), key=lambda k: tts_audio_cache[k][2])
                del tts_audio_cache[oldest_key]
            tts_audio_cache[cache_key] = (final_audio.copy(), final_sample_rate, time.time())
            print(f"[TTS] 캐시 저장 (key={cache_key[:8]}..., 현재 캐시 크기={len(tts_audio_cache)})")

            elapsed = time.time() - start_time
            print(f"[CosyVoice] 전체 TTS 완료 (총 {len(text_chunks)}개 청크, 병렬 처리, 시간: {elapsed:.2f}초, 길이: {len(final_audio)/final_sample_rate:.2f}초)")
            return (final_audio, final_sample_rate)

        return None

    elif engine == 'elevenlabs':
        # ElevenLabs TTS (병렬 처리 최적화 + Keep-Alive 연결 재사용)
        from pydub import AudioSegment
        import base64
        from concurrent.futures import ThreadPoolExecutor, as_completed

        voice_id = get_elevenlabs_voice_id(voice)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }

        start_time = time.time()

        # 단일 청크 TTS 요청 함수 (세션 풀링 사용)
        def fetch_elevenlabs_chunk(chunk_idx, text_chunk):
            """단일 청크 ElevenLabs TTS 요청 (병렬 처리용 - Keep-Alive 연결 재사용)"""
            try:
                data = {
                    "text": text_chunk,
                    "model_id": "eleven_flash_v2_5",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }

                # 전역 HTTP 세션 사용 (연결 재사용으로 오버헤드 감소)
                response = tts_http_session.post(url, json=data, headers=headers, stream=True, timeout=60)

                if response.status_code == 200:
                    # 큰 청크로 데이터 수집 (64KB)
                    mp3_chunks = []
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            mp3_chunks.append(chunk)
                    mp3_data = b''.join(mp3_chunks)

                    # numpy로 변환
                    mp3_bytes = io.BytesIO(mp3_data)
                    audio_segment = AudioSegment.from_mp3(mp3_bytes)
                    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

                    audio_numpy = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    audio_numpy = audio_numpy / 32768.0

                    return (chunk_idx, audio_numpy, None)
                else:
                    return (chunk_idx, None, f"HTTP {response.status_code}")

            except Exception as e:
                return (chunk_idx, None, str(e))

        # 병렬 TTS 요청 (최대 3개 동시 - API rate limit 고려)
        max_parallel = min(3, len(text_chunks))
        results = [None] * len(text_chunks)
        failed = False

        print(f"[ElevenLabs] {len(text_chunks)}개 청크 병렬 처리 시작 (max_parallel={max_parallel})")

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(fetch_elevenlabs_chunk, idx, chunk): idx
                for idx, chunk in enumerate(text_chunks)
            }

            for future in as_completed(futures):
                chunk_idx, audio_data, error = future.result()

                if error:
                    print(f"[ElevenLabs] 청크 {chunk_idx + 1} 실패: {error}")
                    failed = True
                    break
                else:
                    results[chunk_idx] = audio_data
                    print(f"[ElevenLabs] 청크 {chunk_idx + 1}/{len(text_chunks)} 완료 (길이: {len(audio_data)/16000:.2f}초)")

        if failed:
            return None

        # 순서대로 클라이언트에 전송
        for chunk_idx, audio_numpy in enumerate(results):
            if audio_numpy is None:
                continue

            is_last_chunk = (chunk_idx == len(results) - 1)
            all_audio_chunks.append(audio_numpy)

            # MP3로 변환하여 전송 (ElevenLabs 호환)
            audio_segment = AudioSegment(
                (audio_numpy * 32768).astype(np.int16).tobytes(),
                frame_rate=16000, sample_width=2, channels=1
            )
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3")
            mp3_buffer.seek(0)
            chunk_base64 = base64.b64encode(mp3_buffer.read()).decode('utf-8')
            chunk_callback(chunk_base64, 16000, is_last_chunk)

        # 모든 청크를 합쳐서 최종 오디오 생성
        if all_audio_chunks:
            final_audio = np.concatenate(all_audio_chunks)
            final_sample_rate = 16000
            
            # 끝부분 무음/불필요한 소리 트리밍
            final_audio = trim_audio_silence(final_audio, final_sample_rate)

            # 최종 완료 신호
            chunk_callback("", final_sample_rate, True)

            if output_path:
                sf.write(str(output_path), final_audio, final_sample_rate)

            # 캐시에 저장 (LRU 방식)
            if len(tts_audio_cache) >= TTS_CACHE_MAX_SIZE:
                oldest_key = min(tts_audio_cache.keys(), key=lambda k: tts_audio_cache[k][2])
                del tts_audio_cache[oldest_key]
            tts_audio_cache[cache_key] = (final_audio.copy(), final_sample_rate, time.time())
            print(f"[TTS] 캐시 저장 (key={cache_key[:8]}..., 현재 캐시 크기={len(tts_audio_cache)})")

            elapsed = time.time() - start_time
            print(f"[ElevenLabs] 전체 TTS 완료 (총 {len(text_chunks)}개 청크, 병렬 처리, 시간: {elapsed:.2f}초, 길이: {len(final_audio)/final_sample_rate:.2f}초)")
            return (final_audio, final_sample_rate)

        return None

    print(f"[TTS] 알 수 없는 엔진: {engine}")
    return None


def generate_tts_audio(text, engine, voice, output_path=None):
    """
    TTS 오디오 생성 (기존 함수 - 호환성 유지)

    Returns:
        tuple: (audio_numpy, sample_rate) - 성공 시 numpy array와 샘플레이트 반환
        None: 실패 시
    """
    import torch
    import torchaudio

    print(f"[TTS] engine={engine}, voice={voice}, text={text[:50]}...")

    if engine == 'cosyvoice':
        # CosyVoice TTS (로컬 서버 스트리밍)
        import requests
        import io
        import soundfile as sf
        import librosa

        start_time = time.time()

        try:
            # 스트리밍 TTS 요청
            response = requests.post(
                f"{COSYVOICE_API_URL}/api/tts/stream",
                json={"text": text, "speed": 1.0},
                stream=True,
                timeout=120
            )

            if response.status_code == 200:
                # 스트리밍 데이터 수집
                wav_data = b''
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        wav_data += chunk

                elapsed = time.time() - start_time
                print(f"[CosyVoice] 스트리밍 완료 (시간: {elapsed:.2f}초, 크기: {len(wav_data)} bytes)")

                # WAV 데이터를 numpy로 변환
                wav_buffer = io.BytesIO(wav_data)
                audio_data, sr = sf.read(wav_buffer)

                # 스테레오인 경우 모노로 변환
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)

                # 16kHz로 리샘플링 (CosyVoice는 24kHz 출력) - scipy가 더 빠름
                original_duration = len(audio_data) / sr
                print(f"[CosyVoice] 원본 오디오: {original_duration:.2f}초 ({sr}Hz, {len(audio_data)} samples)")

                if sr != 16000:
                    from scipy import signal
                    target_len = int(len(audio_data) * 16000 / sr)
                    audio_data = signal.resample(audio_data, target_len)
                    sr = 16000
                    print(f"[CosyVoice] 리샘플링 후: {len(audio_data)/16000:.2f}초 (16kHz, {len(audio_data)} samples)")

                audio_numpy = audio_data.astype(np.float32)

                # 끝부분 무음/불필요한 소리 트리밍
                before_trim = len(audio_numpy)
                audio_numpy = trim_audio_silence(audio_numpy, 16000)
                after_trim = len(audio_numpy)
                if before_trim != after_trim:
                    print(f"[CosyVoice] 트리밍: {before_trim/16000:.2f}초 → {after_trim/16000:.2f}초")

                if output_path:
                    sf.write(str(output_path), audio_numpy, 16000)

                print(f"[CosyVoice] TTS 최종 완료 (길이: {len(audio_numpy)/16000:.2f}초)")
                return (audio_numpy, 16000)

            else:
                print(f"[CosyVoice] 오류: {response.status_code} - {response.text}")
                # ElevenLabs로 폴백
                print("[TTS] ElevenLabs로 폴백합니다...")
                return generate_tts_audio(text, 'elevenlabs', 'Custom', output_path)

        except requests.exceptions.ConnectionError as e:
            print(f"[CosyVoice] 서버 연결 실패: {e}")
            print("[TTS] ElevenLabs로 폴백합니다...")
            return generate_tts_audio(text, 'elevenlabs', 'Custom', output_path)
        except Exception as e:
            print(f"[CosyVoice] 오류: {e}")
            import traceback
            traceback.print_exc()
            print("[TTS] ElevenLabs로 폴백합니다...")
            return generate_tts_audio(text, 'elevenlabs', 'Custom', output_path)

    elif engine == 'elevenlabs':
        # ElevenLabs TTS (스트리밍 + Flash v2.5 저지연 모델)
        import requests
        import io
        import soundfile as sf
        from pydub import AudioSegment

        voice_id = get_elevenlabs_voice_id(voice)
        # 스트리밍 엔드포인트 사용
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }

        data = {
            "text": text,
            "model_id": "eleven_flash_v2_5",  # 저지연 Flash 모델 (32개 언어, ~75ms)
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        # 스트리밍 요청
        start_time = time.time()
        response = requests.post(url, json=data, headers=headers, stream=True)

        if response.status_code == 200:
            # 스트리밍 데이터 수집
            audio_chunks = []
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    audio_chunks.append(chunk)

            mp3_data = b''.join(audio_chunks)
            elapsed = time.time() - start_time
            print(f"ElevenLabs 스트리밍 완료 (시간: {elapsed:.2f}초, 크기: {len(mp3_data)} bytes)")

            # MP3를 메모리에서 디코딩
            mp3_bytes = io.BytesIO(mp3_data)
            audio_segment = AudioSegment.from_mp3(mp3_bytes)

            # 16kHz로 리샘플링
            audio_segment = audio_segment.set_frame_rate(16000)
            audio_segment = audio_segment.set_channels(1)  # mono

            # numpy array로 변환
            audio_numpy = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            audio_numpy = audio_numpy / 32768.0  # int16 -> float32
            
            # 끝부분 무음/불필요한 소리 트리밍
            audio_numpy = trim_audio_silence(audio_numpy, 16000)

            # output_path가 제공되면 저장 (호환성)
            if output_path:
                sf.write(str(output_path), audio_numpy, 16000)

            print(f"ElevenLabs TTS 완료 (길이: {len(audio_numpy)/16000:.2f}초)")
            return (audio_numpy, 16000)

        print(f"ElevenLabs 오류: {response.status_code} - {response.text}")
        return None

    print(f"[TTS] 알 수 없는 엔진: {engine}")
    return None


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """립싱크 비디오 생성 API (큐 시스템)"""
    global current_process, lipsync_engine

    data = request.json
    avatar_path = data.get('avatar_path')
    text = data.get('text', '')
    tts_engine = data.get('tts_engine', DEFAULT_TTS_ENGINE)
    tts_voice = data.get('tts_voice', DEFAULT_TTS_VOICE)
    client_sid = data.get('sid')  # 클라이언트 SID
    frame_skip = data.get('frame_skip', 1)  # 프레임 스킵 (1=없음, 2=절반추론, 3=1/3추론)
    resolution = data.get('resolution', '720p')  # 출력 해상도 (720p, 480p, 360p)

    if not avatar_path:
        return jsonify({"error": "avatar_path가 필요합니다"}), 400

    if not text:
        return jsonify({"error": "text가 필요합니다"}), 400

    if not client_sid:
        return jsonify({"error": "sid가 필요합니다"}), 400

    # 이미 대기 중인지 확인
    existing_position = generation_queue.get_position(client_sid)
    if existing_position >= 0:
        return jsonify({
            "error": "이미 대기열에 있습니다",
            "position": existing_position
        }), 400

    # 요청을 큐에 추가
    request_data = {
        'avatar_path': avatar_path,
        'text': text,
        'tts_engine': tts_engine,
        'tts_voice': tts_voice,
        'frame_skip': frame_skip,
        'resolution': resolution
    }
    request_id, position = generation_queue.add_request(client_sid, request_data)

    # 대기 순서 알림
    if position > 1:
        socketio.emit('queue_update', {
            'position': position,
            'message': f'대기열 {position}번째 - 잠시만 기다려주세요',
            'request_id': request_id
        }, to=client_sid)
    else:
        socketio.emit('queue_update', {
            'position': position,
            'message': '곧 시작합니다...',
            'request_id': request_id
        }, to=client_sid)

    # 큐 처리 워커 시작 (이미 실행 중이면 무시)
    start_queue_worker()

    return jsonify({
        "status": "queued",
        "position": position,
        "request_id": request_id
    })


@app.route('/api/record', methods=['POST'])
def api_record():
    """영상 녹화 API - 텍스트 입력 → TTS → 립싱크 → 영상 생성"""
    global lipsync_engine

    data = request.json
    text = data.get('text', '')
    avatar_path = data.get('avatar_path')
    tts_engine = data.get('tts_engine', DEFAULT_TTS_ENGINE)
    tts_voice = data.get('tts_voice', DEFAULT_TTS_VOICE)
    resolution = data.get('resolution', '720p')  # 기본 루프 영상과 동일한 해상도
    filename = data.get('filename')  # 선택적 파일명
    output_format = data.get('output_format', 'mp4')  # 출력 포맷 (mp4/webm)
    frame_skip = data.get('frame_skip', 1)  # 프레임 스킵 (1=고품질, 2=빠름, 3=최고속)
    client_sid = data.get('sid')

    if not text:
        return jsonify({"success": False, "error": "텍스트가 필요합니다"}), 400

    if not avatar_path:
        return jsonify({"success": False, "error": "아바타가 필요합니다"}), 400

    if not client_sid:
        return jsonify({"success": False, "error": "sid가 필요합니다"}), 400

    # 텍스트 길이 제한
    if len(text) > 1000:
        return jsonify({"success": False, "error": "텍스트가 너무 깁니다 (최대 1000자)"}), 400

    # 이미 대기 중인지 확인
    existing_position = generation_queue.get_position(client_sid)
    if existing_position >= 0:
        return jsonify({
            "success": False,
            "error": "이미 대기열에 있습니다",
            "position": existing_position
        }), 400

    # 요청을 큐에 추가
    request_data = {
        'avatar_path': avatar_path,
        'text': text,
        'tts_engine': tts_engine,
        'tts_voice': tts_voice,
        'frame_skip': max(1, min(int(frame_skip), 3)),  # 1~3 범위로 제한
        'resolution': resolution,
        'filename': filename,
        'output_format': output_format,  # 출력 포맷 (mp4/webm)
        'is_record': True  # 녹화 모드 표시
    }
    request_id, position = generation_queue.add_request(client_sid, request_data)

    # 대기 순서 알림
    if position > 1:
        socketio.emit('queue_update', {
            'position': position,
            'message': f'대기열 {position}번째 - 잠시만 기다려주세요',
            'request_id': request_id
        }, to=client_sid)
    else:
        socketio.emit('queue_update', {
            'position': position,
            'message': '영상 생성을 시작합니다...',
            'request_id': request_id
        }, to=client_sid)

    # 큐 처리 워커 시작
    start_queue_worker()

    return jsonify({
        "success": True,
        "status": "queued",
        "position": position,
        "request_id": request_id
    })


# 병렬 큐 워커 관리
queue_manager_thread = None
queue_manager_running = False
active_worker_threads = []  # 현재 활성 워커 스레드들
worker_threads_lock = threading.Lock()

def start_queue_manager():
    """병렬 큐 관리자 시작 - 동시에 여러 워커 실행"""
    global queue_manager_thread, queue_manager_running

    if queue_manager_running:
        return  # 이미 실행 중

    def manager():
        global queue_manager_running
        queue_manager_running = True
        print(f"[Queue Manager] 시작 (최대 동시 처리: {ParallelGenerationQueue.MAX_CONCURRENT})")

        while queue_manager_running:
            # 대기열에 요청이 있고 처리 슬롯이 남아있으면 워커 시작
            while generation_queue.can_process_more():
                request_data = generation_queue.get_next()
                if request_data is None:
                    break

                # 새 워커 스레드 시작
                worker_thread = threading.Thread(
                    target=process_request_worker,
                    args=(request_data,),
                    daemon=True
                )
                with worker_threads_lock:
                    active_worker_threads.append(worker_thread)
                worker_thread.start()

                # 대기 중인 클라이언트들에게 순서 업데이트
                broadcast_queue_update()

            # 완료된 워커 스레드 정리
            with worker_threads_lock:
                active_worker_threads[:] = [t for t in active_worker_threads if t.is_alive()]

            # 짧은 대기 (CPU 부하 방지)
            time.sleep(0.1)

        print("[Queue Manager] 종료")

    queue_manager_thread = threading.Thread(target=manager, daemon=True)
    queue_manager_thread.start()


def process_request_worker(request_data):
    """개별 요청 처리 워커"""
    request_id = request_data.get('request_id')
    sid = request_data.get('sid')

    print(f"[Worker-{request_id}] 처리 시작: {sid}")

    try:
        process_generation_request(request_data)
    except Exception as e:
        print(f"[Worker-{request_id}] 오류: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': str(e)}, to=sid)
    finally:
        generation_queue.complete_request(request_id)
        print(f"[Worker-{request_id}] 완료")
        # 대기 중인 클라이언트들에게 순서 업데이트
        broadcast_queue_update()


# 기존 함수명 호환성 유지
def start_queue_worker():
    """기존 함수명 호환 - queue manager 시작"""
    start_queue_manager()


def broadcast_queue_update():
    """대기 중인 모든 클라이언트에게 순서 업데이트 전송"""
    status = generation_queue.get_status()
    for i, item in enumerate(status['queue']):
        position = i + 1
        socketio.emit('queue_update', {
            'position': position,
            'message': f'대기열 {position}번째',
            'request_id': item['request_id'],
            'wait_time': item['wait_time']
        }, to=item['sid'])


def select_avatar_by_audio_duration(audio_duration, avatar_path=None, frame_skip=1, resolution="720p"):
    """
    TTS 오디오 길이와 frame_skip 모드에 따라 적절한 아바타 선택

    모드별 전략:
    - Fast 모드 (frame_skip=3):
      - 10초 이하: 기존 로직 (≤5초→short, >5초→long)
      - 10초 초과: long 영상 1개 사용, 마지막 프레임→idle 페이드 전환

    - Balanced/Quality 모드 (frame_skip=1,2):
      - 10초 이하: 기존 로직 (≤5초→short, >5초→long)
      - 10초 초과: 영상 조합 사용
        - 10~15초: long + short (15초)
        - 15~20초: long × 2 (20초)
        - 20~25초: long × 2 + short (25초)
        - 25~30초: long × 3 (30초)
        - ...패턴 반복

    해상도별 프리컴퓨트:
    - 720p, 480p, 360p 해상도별 프리컴퓨트 데이터 사용
    - 해당 해상도 파일이 없으면 기본 경로 사용 (하위 호환)

    반환값:
    - 단일 아바타: 문자열 (경로)
    - 영상 조합: 리스트 [경로1, 경로2, ...]
    """
    # 해상도별 프리컴퓨트 경로 (talk_short, talk_long 사용)
    SHORT_AVATAR = _get_resolution_avatar("talk_short", resolution)
    LONG_AVATAR = _get_resolution_avatar("talk_long", resolution)
    SHORT_DURATION = 5.0  # short 영상 길이
    LONG_DURATION = 10.0  # long 영상 길이
    AUDIO_DURATION_THRESHOLD = 5.0  # 5초 기준

    # 명시적으로 아바타가 지정된 경우 그대로 사용
    if avatar_path and avatar_path != 'auto' and os.path.exists(avatar_path):
        return avatar_path

    # 10초 이하: 기존 로직
    if audio_duration <= LONG_DURATION:
        if audio_duration <= AUDIO_DURATION_THRESHOLD:
            selected = SHORT_AVATAR
            print(f"[아바타 자동 선택] 짧은 오디오 ({audio_duration:.2f}초 ≤ 5초) → talk_short (5초)")
        else:
            selected = LONG_AVATAR
            print(f"[아바타 자동 선택] 중간 오디오 ({audio_duration:.2f}초) → talk_long (10초)")

        if not os.path.exists(selected):
            return _get_fallback_avatar(avatar_path, LONG_AVATAR, SHORT_AVATAR)
        return selected

    # 10초 초과: 모드별 전략
    is_fast_mode = (frame_skip == 3)

    if is_fast_mode:
        # Fast 모드: long 영상 1개만 사용, 페이드 전환으로 처리
        print(f"[아바타 자동 선택] Fast 모드 - 긴 오디오 ({audio_duration:.2f}초 > 10초) → talk_long (10초) + 페이드 전환")
        selected = LONG_AVATAR
        if not os.path.exists(selected):
            return _get_fallback_avatar(avatar_path, LONG_AVATAR, SHORT_AVATAR)
        return selected
    else:
        # Balanced/Quality 모드: 영상 조합 사용
        mode_name = "Quality" if frame_skip == 1 else "Balanced"
        avatar_combo = _calculate_avatar_combination(audio_duration, SHORT_AVATAR, LONG_AVATAR, SHORT_DURATION, LONG_DURATION)

        # 조합 설명 생성
        combo_desc = []
        long_count = avatar_combo.count(LONG_AVATAR)
        short_count = avatar_combo.count(SHORT_AVATAR)
        if long_count > 0:
            combo_desc.append(f"long×{long_count}")
        if short_count > 0:
            combo_desc.append(f"short×{short_count}")

        total_duration = long_count * LONG_DURATION + short_count * SHORT_DURATION
        print(f"[아바타 자동 선택] {mode_name} 모드 - 긴 오디오 ({audio_duration:.2f}초) → {' + '.join(combo_desc)} ({total_duration}초)")

        return avatar_combo


def _get_resolution_avatar(base_name, resolution="720p"):
    """
    해상도별 프리컴퓨트 파일 경로 반환

    우선순위:
    1. precomputed/{resolution}/{base_name}_{resolution}_precomputed.pkl
    2. precomputed/{base_name}_precomputed.pkl (기본 경로, 하위 호환)
    """
    # 해상도별 경로
    res_path = f"precomputed/{resolution}/{base_name}_{resolution}_precomputed.pkl"
    if os.path.exists(res_path):
        return res_path

    # 기본 경로 (하위 호환)
    default_path = f"precomputed/{base_name}_precomputed.pkl"
    if os.path.exists(default_path):
        return default_path

    # 둘 다 없으면 해상도별 경로 반환 (나중에 생성될 수 있음)
    return res_path


def _get_fallback_avatar(avatar_path, long_avatar, short_avatar):
    """아바타 파일이 없을 때 대체 아바타 반환"""
    for fallback in [avatar_path, long_avatar, short_avatar]:
        if fallback and os.path.exists(fallback):
            return fallback
    # 모두 없으면 precomputed 폴더의 첫 번째 파일
    import glob
    pkls = glob.glob("precomputed/*_precomputed.pkl")
    if pkls:
        return pkls[0]
    return None


def _calculate_avatar_combination(audio_duration, short_avatar, long_avatar, short_dur, long_dur):
    """
    오디오 길이에 맞는 영상 조합 계산

    패턴:
    - 10~15초: long + short (15초)
    - 15~20초: long × 2 (20초)
    - 20~25초: long × 2 + short (25초)
    - 25~30초: long × 3 (30초)
    - ...5초 단위로 패턴 반복
    """
    # 필요한 영상 길이 계산 (5초 단위로 올림)
    import math
    target_duration = math.ceil(audio_duration / 5.0) * 5.0

    # long 영상 개수와 short 필요 여부 계산
    long_count = int(target_duration // long_dur)
    remaining = target_duration - (long_count * long_dur)
    need_short = remaining >= short_dur

    # 조합 생성
    combo = [long_avatar] * long_count
    if need_short:
        combo.append(short_avatar)

    return combo


def generate_lipsync_with_combo(lipsync_engine, avatar_combo, audio_input, frame_skip, output_dir="results/realtime", sid=None, preloaded_data=None, resolution="720p", filename=None, text=None, output_format="mp4"):
    """
    영상 조합을 사용한 립싱크 생성 (Balanced/Quality 모드)

    현재 구현: 첫 번째 아바타만 사용 (향후 확장 예정)

    향후 확장 시:
    1. 각 아바타의 프레임 연결 (long→short 또는 long→long)
    2. 오디오를 각 구간에 맞게 분할하여 처리
    3. 영상 간 페이드 전환 효과 적용

    Args:
        lipsync_engine: 립싱크 엔진
        avatar_combo: 아바타 경로 리스트 [path1, path2, ...]
        audio_input: 오디오 (audio_numpy, sample_rate) 튜플
        frame_skip: 프레임 스킵 값
        output_dir: 출력 디렉토리
        sid: 세션 ID
        preloaded_data: 프리로드된 데이터
        resolution: 해상도
        filename: 파일명
        text: 텍스트
        output_format: 출력 포맷
    """
    # 현재는 첫 번째 아바타만 사용
    # TODO: 영상 조합 처리 구현
    #   - 각 아바타별 프레임 수 계산
    #   - 오디오 분할
    #   - 각 구간 립싱크 생성
    #   - 프레임 연결 및 페이드 전환

    primary_avatar = avatar_combo[0]
    print(f"[영상 조합] 현재 첫 번째 아바타만 사용: {primary_avatar}")
    print(f"[영상 조합] 전체 조합: {avatar_combo}")

    # 첫 번째 아바타로 립싱크 생성
    return lipsync_engine.generate_lipsync(
        primary_avatar,
        audio_input,
        output_dir=output_dir,
        sid=sid,
        preloaded_data=preloaded_data,
        frame_skip=frame_skip,
        resolution=resolution,
        filename=filename,
        text=text,
        output_format=output_format
    )


def process_generation_request(request_data):
    """실제 립싱크 생성 처리"""
    global lipsync_engine

    sid = request_data.get('sid')
    avatar_path = request_data.get('avatar_path')
    text = request_data.get('text')
    tts_engine = request_data.get('tts_engine')
    tts_voice = request_data.get('tts_voice')
    frame_skip = request_data.get('frame_skip', 1)
    resolution = request_data.get('resolution', '720p')
    filename = request_data.get('filename')  # 사용자 지정 파일명
    output_format = request_data.get('output_format', 'mp4')  # 출력 포맷 (mp4/webm)

    start_time = time.time()

    # 취소 확인
    if request_data.get('cancelled'):
        socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
        return

    # 처리 시작 알림
    socketio.emit('queue_update', {
        'position': 0,
        'message': '처리 중...',
        'request_id': request_data.get('request_id')
    }, to=sid)

    # 1. TTS 생성
    engine_name = TTS_ENGINES.get(tts_engine, {}).get('name', tts_engine)
    socketio.emit('status', {'message': f'TTS 생성 중 ({engine_name})...', 'progress': 0, 'elapsed': 0}, to=sid)

    tts_result = generate_tts_audio(text, tts_engine, tts_voice)

    if tts_result is None:
        socketio.emit('error', {'message': f'TTS 생성 실패 ({engine_name})'}, to=sid)
        return

    audio_numpy, sample_rate = tts_result

    # TTS 생성 후 실제 오디오 길이로 아바타 자동 선택 (frame_skip 모드 + 해상도 기반)
    audio_duration = len(audio_numpy) / sample_rate
    avatar_selection = select_avatar_by_audio_duration(audio_duration, avatar_path, frame_skip, resolution)

    # 영상 조합 여부 확인
    is_avatar_combo = isinstance(avatar_selection, list)
    if is_avatar_combo:
        # Balanced/Quality 모드: 첫 번째 아바타로 시작 (TODO: 영상 조합 처리)
        print(f"[영상 조합] {len(avatar_selection)}개 영상 조합 사용")
        avatar_path = avatar_selection[0]  # 현재는 첫 번째만 사용
    else:
        avatar_path = avatar_selection

    # 취소 확인
    if request_data.get('cancelled'):
        socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
        return

    # 2. 프리컴퓨트 데이터 로드 (캐싱 적용)
    socketio.emit('status', {
        'message': '아바타 데이터 로드 중...',
        'progress': 8,
        'elapsed': time.time() - start_time
    }, to=sid)

    precomputed_data = load_precomputed_data(avatar_path)

    # 3. 립싱크 생성
    skip_info = f" (frame_skip={frame_skip})" if frame_skip > 1 else ""
    socketio.emit('status', {
        'message': f'립싱크 생성 시작...{skip_info}',
        'progress': 10,
        'elapsed': time.time() - start_time
    }, to=sid)

    lipsync_result = lipsync_engine.generate_lipsync(
        avatar_path,
        (audio_numpy, sample_rate),
        output_dir="results/realtime",
        sid=sid,
        preloaded_data=precomputed_data,  # 캐시된 데이터 전달
        frame_skip=frame_skip,
        resolution=resolution,
        filename=filename,
        text=text,  # 메타데이터 저장용
        output_format=output_format  # 출력 포맷 (mp4/webm)
    )

    # 반환값 처리 (튜플이면 크롭 좌표 포함)
    if isinstance(lipsync_result, tuple):
        output_video, crop_coords = lipsync_result
    else:
        output_video = lipsync_result
        crop_coords = None

    # 취소 확인
    if request_data.get('cancelled'):
        socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
        return

    if output_video:
        elapsed = time.time() - start_time
        socketio.emit('status', {'message': '완료!', 'progress': 100, 'elapsed': elapsed}, to=sid)
        # 실제 생성된 파일 경로에서 파일명 추출
        video_filename = os.path.basename(output_video)
        complete_data = {
            'video_path': output_video,
            'video_url': f'/video/{video_filename}',
            'elapsed': elapsed,
            'audio_duration': audio_duration  # 음성 길이 전달 (영상 유지 시간 계산용)
        }
        # 크롭 좌표가 있으면 추가 (클라이언트에서 오버레이 위치 계산용)
        if crop_coords:
            # 해상도별 원본 영상 크기 (크롭 전)
            resolution_sizes = {
                "720p": (1268, 724),
                "480p": (848, 484),
                "360p": (632, 360)
            }
            native_width, native_height = resolution_sizes.get(resolution, (1268, 724))
            complete_data['crop_coords'] = {
                'x': crop_coords[0],
                'width': crop_coords[2],
                'height': crop_coords[3],
                'native_width': native_width,    # 원본 영상 너비 (크롭 전)
                'native_height': native_height,  # 원본 영상 높이 (크롭 전)
                'center_crop': True
            }
        socketio.emit('complete', complete_data, to=sid)
    else:
        socketio.emit('error', {'message': '립싱크 생성 실패'}, to=sid)


@app.route('/api/cancel', methods=['POST'])
def api_cancel():
    """생성 취소 API"""
    data = request.json
    client_sid = data.get('sid') if data else None

    if client_sid:
        # 큐에서 취소
        removed = generation_queue.cancel_request(client_sid)

        # 기존 세션에서도 취소 표시
        if client_sid in client_sessions:
            client_sessions[client_sid]['cancelled'] = True

        return jsonify({
            "status": "cancelled",
            "removed_from_queue": removed
        })

    return jsonify({"status": "cancelled"})


# ============================================================
# REST API (동기적 - WebSocket 없이 사용 가능)
# ============================================================

@app.route('/api/v2/lipsync', methods=['POST'])
def api_v2_lipsync():
    """
    립싱크 비디오 생성 API (동기적)

    Request:
        {
            "text": "합성할 텍스트",
            "avatar": "avatar_name",      // precomputed 폴더의 아바타 이름
            "avatar_path": "path.pkl",    // 또는 직접 경로 지정
            "tts_engine": "elevenlabs",    // optional
            "tts_voice": "Custom"      // optional
        }

    Response:
        {
            "success": true,
            "video_path": "results/realtime/output_with_audio.mp4",
            "video_url": "/video/output_with_audio.mp4",
            "elapsed": 25.3
        }
    """
    global lipsync_engine

    data = request.json or {}
    text = data.get('text', '')
    avatar = data.get('avatar', '')
    avatar_path = data.get('avatar_path', '')
    tts_engine = data.get('tts_engine', DEFAULT_TTS_ENGINE)
    tts_voice = data.get('tts_voice', DEFAULT_TTS_VOICE)
    frame_skip = data.get('frame_skip', 1)  # 프레임 스킵 (1=없음, 2=절반추론, 3=1/3추론)
    resolution = data.get('resolution', '720p')  # 해상도 (720p, 480p, 360p)

    # 아바타 경로 결정
    if not avatar_path and avatar:
        avatar_path = f"precomputed/{avatar}_precomputed.pkl"

    if not avatar_path:
        return jsonify({"success": False, "error": "avatar 또는 avatar_path가 필요합니다"}), 400

    if not os.path.exists(avatar_path):
        return jsonify({"success": False, "error": f"아바타 파일을 찾을 수 없습니다: {avatar_path}"}), 404

    if not text:
        return jsonify({"success": False, "error": "text가 필요합니다"}), 400

    if not lipsync_engine or not lipsync_engine.loaded:
        return jsonify({"success": False, "error": "립싱크 엔진이 로드되지 않았습니다"}), 503

    try:
        start_time = time.time()

        # 병렬 처리: TTS 생성과 프리컴퓨트 로드를 동시에
        tts_output = Path("assets/audio/tts_output.wav")
        tts_result = None
        precomputed_data = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            # TTS와 프리컴퓨트 로드를 병렬로 실행
            tts_future = executor.submit(generate_tts_audio, text, tts_engine, tts_voice, tts_output)
            precompute_future = executor.submit(load_precomputed_data, avatar_path)

            # 결과 수집
            for future in as_completed([tts_future, precompute_future]):
                if future == tts_future:
                    tts_result = future.result()
                else:
                    precomputed_data = future.result()

        parallel_time = time.time() - start_time
        print(f"[병렬 처리] TTS + 프리컴퓨트 로드 완료: {parallel_time:.2f}초")

        if not tts_result:
            return jsonify({"success": False, "error": "TTS 생성 실패"}), 500

        if not precomputed_data:
            return jsonify({"success": False, "error": "프리컴퓨트 로드 실패"}), 500

        # TTS 생성 후 실제 오디오 길이로 아바타 자동 선택 (frame_skip 모드 + 해상도 기반)
        audio_numpy, sample_rate = tts_result
        audio_duration = len(audio_numpy) / sample_rate
        avatar_selection = select_avatar_by_audio_duration(audio_duration, avatar_path, frame_skip, resolution)

        # 영상 조합 여부 확인
        is_avatar_combo = isinstance(avatar_selection, list)
        if is_avatar_combo:
            print(f"[영상 조합] {len(avatar_selection)}개 영상 조합 사용")
            selected_avatar = avatar_selection[0]  # 현재는 첫 번째만 사용
        else:
            selected_avatar = avatar_selection

        # 선택된 아바타가 다르면 프리컴퓨트 데이터 다시 로드
        if selected_avatar != avatar_path:
            print(f"[아바타 변경] {avatar_path} → {selected_avatar}")
            avatar_path = selected_avatar
            precomputed_data = load_precomputed_data(avatar_path)

        # 립싱크 생성 (프리컴퓨트 데이터 전달 + 프레임 스킵)
        output_video = lipsync_engine.generate_lipsync(
            avatar_path,
            (audio_numpy, sample_rate),
            output_dir="results/realtime",
            preloaded_data=precomputed_data,  # 미리 로드된 데이터 전달
            frame_skip=frame_skip  # 프레임 스킵 적용
        )

        if output_video:
            elapsed = time.time() - start_time
            # 실제 생성된 파일 경로에서 파일명 추출
            video_filename = os.path.basename(output_video)
            return jsonify({
                "success": True,
                "video_path": output_video,
                "video_url": f"/video/{video_filename}",
                "elapsed": round(elapsed, 2)
            })
        else:
            return jsonify({"success": False, "error": "립싱크 생성 실패"}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v2/lipsync/stream', methods=['POST'])
def api_v2_lipsync_stream():
    """
    립싱크 비디오 생성 API (스트리밍 - Server-Sent Events)

    클라이언트가 LLM으로 생성한 텍스트를 보내면 TTS + 립싱크 비디오를 생성합니다.
    진행 상태를 실시간으로 스트리밍합니다.

    Request:
        {
            "text": "립싱크로 만들 텍스트 (LLM 응답)",
            "avatar": "avatar_name",
            "avatar_path": "path.pkl",  // optional
            "tts_engine": "elevenlabs",
            "tts_voice": "Custom",
            "resolution": "720p",
            "frame_skip": 1
        }

    Response (SSE):
        data: {"type": "status", "stage": "tts", "message": "TTS 생성 중..."}
        data: {"type": "status", "stage": "tts_done", "message": "TTS 완료", "audio_url": "...", "duration": 3.5}
        data: {"type": "status", "stage": "lipsync", "message": "립싱크 생성 중..."}
        data: {"type": "done", "video_url": "/video/...", "audio_url": "...", "elapsed": {...}}
        data: {"type": "error", "message": "오류 메시지"}
    """
    global lipsync_engine

    data = request.json or {}
    text = data.get('text', '')
    avatar = data.get('avatar', '')
    avatar_path = data.get('avatar_path', '')
    tts_engine = data.get('tts_engine', DEFAULT_TTS_ENGINE)
    tts_voice = data.get('tts_voice', DEFAULT_TTS_VOICE)
    frame_skip = data.get('frame_skip', 1)
    resolution = data.get('resolution', '720p')

    # 아바타 경로 결정
    if not avatar_path and avatar:
        avatar_path = f"precomputed/{avatar}_precomputed.pkl"

    if not text:
        return jsonify({"error": "text가 필요합니다"}), 400

    if not avatar_path:
        return jsonify({"error": "avatar 또는 avatar_path가 필요합니다"}), 400

    if not os.path.exists(avatar_path):
        return jsonify({"error": f"아바타 파일을 찾을 수 없습니다: {avatar_path}"}), 404

    if not lipsync_engine or not lipsync_engine.loaded:
        return jsonify({"error": "립싱크 엔진이 로드되지 않았습니다"}), 503

    def generate():
        nonlocal text, avatar_path, tts_engine, tts_voice, frame_skip, resolution

        try:
            start_time = time.time()

            # ===== 1단계: TTS 생성 =====
            yield f"data: {json.dumps({'type': 'status', 'stage': 'tts', 'message': 'TTS 음성 생성 중...'}, ensure_ascii=False)}\n\n"

            tts_start = time.time()
            tts_output = Path("assets/audio/tts_output.wav")
            tts_result = generate_tts_audio(text, tts_engine, tts_voice, tts_output)

            if not tts_result:
                yield f"data: {json.dumps({'type': 'error', 'message': 'TTS 생성 실패'}, ensure_ascii=False)}\n\n"
                return

            tts_time = time.time() - tts_start
            audio_numpy, sample_rate = tts_result
            audio_duration = len(audio_numpy) / sample_rate

            yield f"data: {json.dumps({'type': 'status', 'stage': 'tts_done', 'message': f'TTS 완료 ({audio_duration:.1f}초)', 'audio_url': '/assets/audio/tts_output.wav', 'duration': round(audio_duration, 2), 'elapsed': round(tts_time, 2)}, ensure_ascii=False)}\n\n"

            # ===== 2단계: 립싱크 생성 =====
            yield f"data: {json.dumps({'type': 'status', 'stage': 'lipsync', 'message': '립싱크 비디오 생성 중...'}, ensure_ascii=False)}\n\n"

            lipsync_start = time.time()

            # 아바타 자동 선택
            avatar_selection = select_avatar_by_audio_duration(audio_duration, avatar_path, frame_skip, resolution)
            if isinstance(avatar_selection, list):
                selected_avatar = avatar_selection[0]
            else:
                selected_avatar = avatar_selection

            if selected_avatar != avatar_path:
                print(f"[아바타 변경] {avatar_path} → {selected_avatar}")
                avatar_path = selected_avatar

            # 립싱크 생성
            output_video = lipsync_engine.generate_lipsync(
                avatar_path,
                (audio_numpy, sample_rate),
                output_dir="results/realtime",
                frame_skip=frame_skip
            )

            lipsync_time = time.time() - lipsync_start
            total_time = time.time() - start_time

            if output_video:
                video_filename = os.path.basename(output_video)
                yield f"data: {json.dumps({'type': 'done', 'text': text, 'video_path': output_video, 'video_url': f'/video/{video_filename}', 'audio_url': '/assets/audio/tts_output.wav', 'audio_duration': round(audio_duration, 2), 'elapsed': {'tts': round(tts_time, 2), 'lipsync': round(lipsync_time, 2), 'total': round(total_time, 2)}}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': '립싱크 생성 실패'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/v2/status', methods=['GET'])
def api_v2_status():
    """
    서버 상태 확인 API

    Response:
        {
            "status": "ok",
            "models_loaded": true,
            "available_avatars": ["avatar1", "avatar2"],
            "tts_engines": ["elevenlabs"]
        }
    """
    avatars = get_available_avatars()
    avatar_names = [a['id'] for a in avatars]

    available_engines = list(TTS_ENGINES.keys())

    return jsonify({
        "status": "ok",
        "models_loaded": models_loaded,
        "available_avatars": avatar_names,
        "tts_engines": available_engines
    })


@app.route('/api/generate_streaming', methods=['POST'])
def api_generate_streaming():
    """스트리밍 립싱크 비디오 생성 API (일반 생성과 동일하게 동작)"""
    global current_process, lipsync_engine

    data = request.json
    avatar_path = data.get('avatar_path')
    text = data.get('text', '')
    tts_engine = data.get('tts_engine', DEFAULT_TTS_ENGINE)
    tts_voice = data.get('tts_voice', DEFAULT_TTS_VOICE)
    client_sid = data.get('sid')  # 클라이언트 SID
    frame_skip = data.get('frame_skip', 1)  # 프레임 스킵 (1=없음, 2=절반추론, 3=1/3추론)
    resolution = data.get('resolution', '720p')  # 해상도 (720p, 480p, 360p)

    if not avatar_path:
        return jsonify({"error": "avatar_path가 필요합니다"}), 400

    if not text:
        return jsonify({"error": "text가 필요합니다"}), 400

    if not client_sid:
        return jsonify({"error": "sid가 필요합니다"}), 400

    # 동시 생성 방지
    with generation_lock:
        for sid, session in client_sessions.items():
            if session.get('generating'):
                if sid == client_sid:
                    return jsonify({"error": "이미 생성 중입니다"}), 400
                else:
                    return jsonify({"error": "다른 클라이언트가 생성 중입니다. 잠시 후 다시 시도해주세요."}), 400

        if client_sid in client_sessions:
            client_sessions[client_sid]['cancelled'] = False
            client_sessions[client_sid]['start_time'] = time.time()
            client_sessions[client_sid]['generating'] = True

    def generate_streaming_async(sid):
        try:
            if sid not in client_sessions:
                return

            start_time = client_sessions[sid]['start_time']
            import base64
            import soundfile as sf
            import io

            # ========== 성능 측정 시작 ==========
            timing_log = {}
            step_start = time.time()

            print(f"\n{'='*60}")
            print(f"[성능측정] 면접 스트리밍 시작 - TTS엔진: {tts_engine}")
            print(f"{'='*60}")

            # 1. TTS 스트리밍 생성 (청크를 받는 대로 전송)
            engine_name = TTS_ENGINES.get(tts_engine, {}).get('name', tts_engine)
            socketio.emit('status', {'message': f'TTS 스트리밍 시작 ({engine_name})...', 'progress': 0, 'elapsed': 0}, to=sid)

            # TTS 청크 콜백 함수
            audio_chunks = []
            sample_rate = 16000
            
            def tts_chunk_callback(chunk_base64, sr, is_final):
                """TTS 청크를 받을 때마다 호출되는 콜백"""
                nonlocal sample_rate
                sample_rate = sr
                
                # 빈 문자열이면 완료 신호만 전송
                if chunk_base64 == "":
                    socketio.emit('stream_audio_complete', {
                        'sample_rate': sr,
                        'total_length': 0,
                        'elapsed': time.time() - start_time
                    }, to=sid)
                    return
                
                # 청크를 클라이언트로 즉시 전송
                socketio.emit('stream_audio_chunk', {
                    'audio': chunk_base64,
                    'sample_rate': sr,
                    'is_final': is_final,
                    'elapsed': time.time() - start_time
                }, to=sid)
                
                # 전체 오디오 수집 (최종 반환용)
                if not is_final:
                    audio_chunks.append(chunk_base64)

            # TTS 스트리밍 생성
            tts_result = generate_tts_audio_streaming(text, tts_engine, tts_voice, tts_chunk_callback)

            timing_log['tts'] = time.time() - step_start
            print(f"[성능측정] 1. TTS 생성: {timing_log['tts']:.2f}초 ({tts_engine})")

            if tts_result is None:
                socketio.emit('error', {'message': f'TTS 생성 실패 ({engine_name})'}, to=sid)
                return

            audio_numpy, sample_rate = tts_result
            audio_duration = len(audio_numpy) / sample_rate
            print(f"[성능측정]    - 오디오 길이: {audio_duration:.2f}초")

            # TTS 생성 후 실제 오디오 길이로 아바타 자동 선택 (frame_skip 모드 + 해상도 기반)
            nonlocal avatar_path
            avatar_selection = select_avatar_by_audio_duration(audio_duration, avatar_path, frame_skip, resolution)

            # 영상 조합 여부 확인
            if isinstance(avatar_selection, list):
                print(f"[영상 조합] {len(avatar_selection)}개 영상 조합 사용")
                avatar_path = avatar_selection[0]  # 현재는 첫 번째만 사용
            else:
                avatar_path = avatar_selection

            if sid in client_sessions and client_sessions[sid]['cancelled']:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
                return

            # TTS 스트리밍 완료 신호
            socketio.emit('stream_audio_complete', {
                'sample_rate': sample_rate,
                'total_length': len(audio_numpy) / sample_rate,
                'elapsed': time.time() - start_time
            }, to=sid)

            # 3. 스트리밍 립싱크 생성
            step_start = time.time()
            skip_info = f" (frame_skip={frame_skip})" if frame_skip > 1 else ""
            socketio.emit('status', {
                'message': f'스트리밍 립싱크 시작...{skip_info}',
                'progress': 10,
                'elapsed': time.time() - start_time
            }, to=sid)

            result = lipsync_engine.generate_lipsync_streaming(
                avatar_path,
                (audio_numpy, sample_rate),
                sid=sid,
                emit_callback=socketio.emit,
                frame_skip=frame_skip
            )

            timing_log['lipsync'] = time.time() - step_start
            total_time = time.time() - start_time

            # ========== 성능 측정 결과 출력 ==========
            print(f"\n{'='*60}")
            print(f"[성능측정] 면접 스트리밍 완료 - 총 {total_time:.2f}초")
            print(f"{'='*60}")
            print(f"  1. TTS 생성 ({tts_engine}): {timing_log.get('tts', 0):.2f}초")
            print(f"  2. 립싱크 생성: {timing_log.get('lipsync', 0):.2f}초")
            print(f"{'='*60}")
            print(f"  오디오 길이: {audio_duration:.2f}초")
            print(f"  실시간 비율: {total_time/audio_duration:.2f}x (1.0 = 실시간)")
            print(f"{'='*60}\n")

            if sid in client_sessions and client_sessions[sid]['cancelled']:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
                return

        except Exception as e:
            import traceback
            traceback.print_exc()
            socketio.emit('error', {'message': str(e)}, to=sid)
        finally:
            if sid in client_sessions:
                client_sessions[sid]['generating'] = False

    thread = threading.Thread(target=generate_streaming_async, args=(client_sid,))
    thread.start()

    return jsonify({"status": "started"})


@app.route('/video/<path:filename>')
def serve_video(filename):
    """비디오 파일 서빙"""
    video_path = Path("results/realtime") / filename
    if video_path.exists():
        return send_file(str(video_path), mimetype='video/mp4')
    return "Not found", 404


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """에셋 파일 서빙"""
    asset_path = Path("assets") / filename
    if asset_path.exists():
        if filename.endswith('.mp4'):
            return send_file(str(asset_path), mimetype='video/mp4')
        elif filename.endswith('.mp3'):
            return send_file(str(asset_path), mimetype='audio/mpeg')
        elif filename.endswith('.wav'):
            return send_file(str(asset_path), mimetype='audio/wav')
        elif filename.endswith('.png'):
            return send_file(str(asset_path), mimetype='image/png')
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            return send_file(str(asset_path), mimetype='image/jpeg')
        return send_file(str(asset_path))
    return "Not found", 404


@socketio.on('connect')
def handle_connect():
    """WebSocket 연결"""
    from flask import request as flask_request
    sid = flask_request.sid
    client_sessions[sid] = {'cancelled': False, 'start_time': None, 'generating': False}
    print(f'Client connected: {sid}')
    emit('connected', {'status': 'ok', 'models_loaded': models_loaded, 'sid': sid})


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket 연결 해제"""
    from flask import request as flask_request
    sid = flask_request.sid
    if sid in client_sessions:
        del client_sessions[sid]
    print(f'Client disconnected: {sid}')


def kill_existing_server(port=5000):
    """기존 서버 종료"""
    try:
        if os.name == 'nt':
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            for line in result.stdout.strip().split('\n'):
                if 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
    except Exception as e:
        print(f"기존 서버 종료 중 오류: {e}")


if __name__ == '__main__':
    # Docker 환경 감지 및 작업 디렉토리 설정
    if os.path.exists("/app/realtime-interview-avatar"):
        # Docker 환경
        os.chdir("/app/realtime-interview-avatar")
    else:
        # 로컬 환경
        local_path = "c:/NewAvata/NewAvata/realtime-interview-avatar"
        if os.path.exists(local_path):
            os.chdir(local_path)

    # 기존 서버 종료
    kill_existing_server(5000)

    # 디렉토리 생성
    Path("templates").mkdir(exist_ok=True)
    Path("results/realtime").mkdir(parents=True, exist_ok=True)
    Path("assets/audio").mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("실시간 립싱크 테스트 서버 (프리로딩 버전)")
    print("=" * 50)

    # 모델 프리로드
    print("\n" + "=" * 50)
    print("모델 프리로드 시작...")
    print("=" * 50)

    lipsync_engine = LipsyncEngine()
    lipsync_engine.load_models(use_float16=True, use_tensorrt=True)  # TensorRT 활성화
    models_loaded = True

    print("\n프리로드 완료!")
    print(f"\nURL: http://localhost:5000")
    print("=" * 50)

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
