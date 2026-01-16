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
from flask import Flask, render_template, request, jsonify, send_file
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
current_process = None  # 현재 실행 중인 프로세스
models_loaded = False  # 모델 로드 상태

# 클라이언트별 상태 관리 (멀티 브라우저 지원)
client_sessions = {}  # {sid: {'cancelled': False, 'start_time': None, 'generating': False}}
generation_lock = threading.Lock()  # 동시 생성 방지 락


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

    def generate_lipsync(self, precomputed_path, audio_input, output_dir="results/realtime", fps=25, sid=None, preloaded_data=None, frame_skip=1, resolution="720p"):
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
        batch_size = 32  # 배치 크기 증가 (GPU 활용도 향상)

        # 프리컴퓨트 데이터 로드 (미리 로드된 데이터가 있으면 사용)
        t0 = time.time()
        if preloaded_data:
            print(f"프리컴퓨트 데이터 사용 (미리 로드됨): {precomputed_path}")
            coords_list = preloaded_data['coords_list']
            frames_list = preloaded_data['frames_list']
            input_latent_list = preloaded_data['input_latent_list']
            fps = preloaded_data['fps']
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
            t0_interp = time.time()
            print(f"[프레임 보간] {actual_inference_frames}프레임 -> {num_frames}프레임...")
            res_frame_list = [None] * num_frames

            # 추론된 프레임 배치
            for idx, frame_idx in enumerate(inference_indices):
                res_frame_list[frame_idx] = inference_frame_list[idx]

            # 중간 프레임 보간 (선형 보간)
            for i in range(len(inference_indices) - 1):
                start_idx = inference_indices[i]
                end_idx = inference_indices[i + 1]
                start_frame = res_frame_list[start_idx]
                end_frame = res_frame_list[end_idx]

                # 중간 프레임들 보간
                for j in range(start_idx + 1, end_idx):
                    alpha = (j - start_idx) / (end_idx - start_idx)
                    # 선형 보간 (빠름)
                    interpolated = cv2.addWeighted(
                        start_frame.astype(np.float32), 1.0 - alpha,
                        end_frame.astype(np.float32), alpha, 0
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

        def blend_single_frame_fast(args):
            """단일 프레임 블렌딩 함수 (마스크 재사용 - GPU 최적화)"""
            i, res_frame = args
            idx = i % len(coords_list)
            coord = coords_list[idx]
            original_frame = frames_list[idx].copy()

            x1, y1, x2, y2 = [int(c) for c in coord]

            try:
                # 선명화 + 고품질 리사이즈
                res_frame_uint8 = res_frame.astype(np.uint8)

                # 언샤프 마스크 (Unsharp Mask) - 선명화
                gaussian = cv2.GaussianBlur(res_frame_uint8, (0, 0), 2.0)
                sharpened = cv2.addWeighted(res_frame_uint8, 1.5, gaussian, -0.5, 0)

                # 고품질 리사이즈 (LANCZOS4)
                pred_frame_resized = cv2.resize(
                    sharpened,
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LANCZOS4
                )
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

        # 비디오 저장
        t0 = time.time()
        print("비디오 저장 중...")
        temp_video = str(output_path / "temp_output.mp4")
        final_video = str(output_path / "output_with_audio.mp4")

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

        # audio_input이 튜플이면 임시 파일 생성
        if isinstance(audio_input, tuple):
            import soundfile as sf
            temp_audio = str(output_path / "temp_audio.wav")
            audio_numpy, sample_rate = audio_input
            sf.write(temp_audio, audio_numpy, sample_rate)
            audio_file_path = temp_audio
        else:
            audio_file_path = audio_input

        # 해상도 스케일 필터 설정
        scale_filters = {
            "720p": None,  # 원본 유지
            "480p": "scale=854:480",
            "360p": "scale=640:360"
        }
        scale_filter = scale_filters.get(resolution)

        if scale_filter:
            print(f"  해상도 스케일링: {resolution}")

        # GPU 인코딩 시도 (NVENC), 실패시 CPU 인코딩
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_file_path,
        ]

        # 스케일 필터 추가
        if scale_filter:
            ffmpeg_cmd.extend(['-vf', scale_filter])

        ffmpeg_cmd.extend([
            '-c:v', 'h264_nvenc',  # NVIDIA GPU 인코더
            '-preset', 'p1',       # 가장 빠른 프리셋 (p1=fastest, p7=slowest)
            '-tune', 'ull',        # Ultra Low Latency 튜닝
            '-rc', 'vbr',          # 가변 비트레이트
            '-cq', '28',           # 품질 수준 (낮을수록 고품질)
            '-c:a', 'aac',
            '-shortest',
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
            if scale_filter:
                ffmpeg_cmd_cpu.extend(['-vf', scale_filter])
            ffmpeg_cmd_cpu.extend([
                '-c:v', 'libx264',
                '-preset', 'ultrafast',  # 가장 빠른 CPU 프리셋
                '-c:a', 'aac',
                '-shortest',
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
        print(f"출력: {final_video}")

        return final_video

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

        # 프리컴퓨트 데이터 로드
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

        extra_margin = 10

        # 오디오 처리
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

        # 프레임 보간 (frame_skip > 1인 경우)
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

        # 블렌딩 및 프레임 전송
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

        # 완료 신호
        elapsed = time.time() - start_time
        emit_callback('stream_complete', {
            'total_frames': num_frames,
            'elapsed': elapsed,
            'fps': fps
        }, to=sid)

        print(f"\n스트리밍 립싱크 완료! ({elapsed:.1f}초)")
        return True


# TTS 엔진 설정
TTS_ENGINES = {
    'elevenlabs': {
        'name': 'ElevenLabs',
        'voices': ['Custom'],
        'default': True
    },
    'edge': {
        'name': 'Edge TTS (무료)',
        'voices': ['ko-KR-SunHiNeural'],
        'default': False
    }
}

# 기본 TTS 엔진
DEFAULT_TTS_ENGINE = 'elevenlabs'
DEFAULT_TTS_VOICE = 'Custom'

# API 키 설정
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '')


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


@app.route('/api/tts_engines')
def api_tts_engines():
    """TTS 엔진 목록 API"""
    engines = []
    for engine_id, engine_info in TTS_ENGINES.items():
        engines.append({
            'id': engine_id,
            'name': engine_info['name'],
            'voices': engine_info['voices'],
            'available': True
        })
    return jsonify(engines)


def load_precomputed_data(precomputed_path):
    """프리컴퓨트 데이터 로드 (병렬화용)"""
    with open(precomputed_path, 'rb') as f:
        precomputed = pickle.load(f)

    coords_list = precomputed.get('coords_list', precomputed.get('coord_list_cycle'))
    frames_list = precomputed.get('frames', precomputed.get('frame_list_cycle'))
    input_latent_list = precomputed.get('input_latent_list', precomputed.get('input_latent_list_cycle'))
    fps = precomputed.get('fps', 25)

    return {
        'coords_list': coords_list,
        'frames_list': frames_list,
        'input_latent_list': input_latent_list,
        'fps': fps
    }


def get_elevenlabs_voice_id(voice_name):
    """ElevenLabs 음성 이름을 ID로 변환"""
    voice_ids = {
        'Custom': 'AFxUUThaQGYPjKmV6PhM',
        'Rachel': '21m00Tcm4TlvDq8ikWAM',
        'Adam': 'pNInz6obpgDQGcFmaJgB',
        'Bella': 'EXAVITQu4vr4xnSDxMaL',
        'Antoni': 'ErXwobaYiN019PkySvjV',
    }
    return voice_ids.get(voice_name, voice_ids['Custom'])


def generate_tts_audio(text, engine, voice, output_path=None):
    """
    TTS 오디오 생성

    Returns:
        tuple: (audio_numpy, sample_rate) - 성공 시 numpy array와 샘플레이트 반환
        None: 실패 시
    """
    import torch
    import torchaudio

    print(f"[TTS] engine={engine}, voice={voice}, text={text[:50]}...")

    if engine == 'elevenlabs':
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

            # output_path가 제공되면 저장 (호환성)
            if output_path:
                sf.write(str(output_path), audio_numpy, 16000)

            print(f"ElevenLabs TTS 완료 (길이: {len(audio_numpy)/16000:.2f}초)")
            return (audio_numpy, 16000)

        print(f"ElevenLabs 오류: {response.status_code} - {response.text}")
        return None

    elif engine == 'edge':
        # Microsoft Edge TTS (무료)
        try:
            import edge_tts
            import asyncio
            import io
            import soundfile as sf
            from pydub import AudioSegment

            start_time = time.time()

            # 음성 선택
            voice_name = voice if voice else 'ko-KR-SunHiNeural'

            async def generate_edge_tts():
                communicate = edge_tts.Communicate(text, voice_name)
                audio_data = b''
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                return audio_data

            # 새 이벤트 루프에서 실행 (스레드 안전)
            mp3_data = asyncio.run(generate_edge_tts())

            if not mp3_data:
                print("[TTS] Edge TTS: 오디오 데이터가 비어있습니다")
                return None

            # MP3를 AudioSegment로 변환
            mp3_buffer = io.BytesIO(mp3_data)
            audio_segment = AudioSegment.from_mp3(mp3_buffer)

            # 16kHz mono로 변환
            audio_segment = audio_segment.set_frame_rate(16000)
            audio_segment = audio_segment.set_channels(1)

            # numpy array로 변환
            audio_numpy = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            audio_numpy = audio_numpy / 32768.0

            elapsed = time.time() - start_time
            print(f"Edge TTS 완료 (시간: {elapsed:.2f}초, 길이: {len(audio_numpy)/16000:.2f}초)")

            if output_path:
                sf.write(str(output_path), audio_numpy, 16000)

            return (audio_numpy, 16000)

        except ImportError as e:
            print(f"[TTS] edge-tts가 설치되지 않았습니다. pip install edge-tts: {e}")
            return None
        except Exception as e:
            print(f"[TTS] Edge TTS 오류: {e}")
            import traceback
            traceback.print_exc()
            return None

    print(f"[TTS] 알 수 없는 엔진: {engine}")
    return None


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """립싱크 비디오 생성 API"""
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

    # 동시 생성 방지
    with generation_lock:
        # 이미 생성 중인 클라이언트가 있는지 확인
        for sid, session in client_sessions.items():
            if session.get('generating'):
                if sid == client_sid:
                    return jsonify({"error": "이미 생성 중입니다"}), 400
                else:
                    return jsonify({"error": "다른 클라이언트가 생성 중입니다. 잠시 후 다시 시도해주세요."}), 400

        # 클라이언트 세션 초기화
        if client_sid in client_sessions:
            client_sessions[client_sid]['cancelled'] = False
            client_sessions[client_sid]['start_time'] = time.time()
            client_sessions[client_sid]['generating'] = True

    def generate_async(sid):
        try:
            if sid not in client_sessions:
                print(f"[ERROR] SID {sid} not in client_sessions")
                return

            start_time = client_sessions[sid]['start_time']

            # 1. TTS 생성 (메모리에서 직접 처리)
            engine_name = TTS_ENGINES.get(tts_engine, {}).get('name', tts_engine)
            socketio.emit('status', {'message': f'TTS 생성 중 ({engine_name})...', 'progress': 0, 'elapsed': 0}, to=sid)

            tts_result = generate_tts_audio(text, tts_engine, tts_voice)

            if tts_result is None:
                socketio.emit('error', {'message': f'TTS 생성 실패 ({engine_name})'}, to=sid)
                return

            audio_numpy, sample_rate = tts_result

            if sid in client_sessions and client_sessions[sid]['cancelled']:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
                return

            # 2. 립싱크 생성 (파일 없이 numpy array 전달)
            skip_info = f" (frame_skip={frame_skip})" if frame_skip > 1 else ""
            socketio.emit('status', {
                'message': f'립싱크 생성 시작...{skip_info}',
                'progress': 10,
                'elapsed': time.time() - start_time
            }, to=sid)

            output_video = lipsync_engine.generate_lipsync(
                avatar_path,
                (audio_numpy, sample_rate),  # numpy array 전달
                output_dir="results/realtime",
                sid=sid,  # WebSocket 세션 ID 전달
                frame_skip=frame_skip,  # 프레임 스킵 적용
                resolution=resolution  # 출력 해상도
            )

            if sid in client_sessions and client_sessions[sid]['cancelled']:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
                return

            if output_video:
                elapsed = time.time() - start_time
                socketio.emit('status', {'message': '완료!', 'progress': 100, 'elapsed': elapsed}, to=sid)
                socketio.emit('complete', {
                    'video_path': 'results/realtime/output_with_audio.mp4',
                    'elapsed': elapsed
                }, to=sid)
            else:
                socketio.emit('error', {'message': '립싱크 생성 실패'}, to=sid)

        except Exception as e:
            import traceback
            traceback.print_exc()
            socketio.emit('error', {'message': str(e)}, to=sid)
        finally:
            # 생성 완료 표시
            if sid in client_sessions:
                client_sessions[sid]['generating'] = False

    thread = threading.Thread(target=generate_async, args=(client_sid,))
    thread.start()

    return jsonify({"status": "started"})


@app.route('/api/cancel', methods=['POST'])
def api_cancel():
    """생성 취소 API"""
    data = request.json
    client_sid = data.get('sid') if data else None

    if client_sid and client_sid in client_sessions:
        client_sessions[client_sid]['cancelled'] = True

    return jsonify({"status": "cancelled"})


# 대화 기록 저장
conversation_history = []

# LLM 시스템 프롬프트 (런타임에 수정 가능)
current_system_prompt = """당신은 친절하고 전문적인 AI 면접관입니다.
면접자의 답변에 대해 적절한 후속 질문을 하거나 피드백을 제공합니다.
응답은 2-3문장 정도로 간결하게 유지하세요."""

# 이전 코드 호환성을 위한 변수
SYSTEM_PROMPT = current_system_prompt


@app.route('/api/prompt', methods=['GET'])
def get_prompt():
    """현재 시스템 프롬프트 조회"""
    global current_system_prompt
    return jsonify({"prompt": current_system_prompt})


@app.route('/api/prompt', methods=['POST'])
def set_prompt():
    """시스템 프롬프트 수정"""
    global current_system_prompt, SYSTEM_PROMPT
    data = request.json
    new_prompt = data.get('prompt', '').strip()

    if not new_prompt:
        return jsonify({"error": "프롬프트가 필요합니다"}), 400

    current_system_prompt = new_prompt
    SYSTEM_PROMPT = new_prompt
    return jsonify({"success": True, "prompt": current_system_prompt})


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """LLM 채팅 API"""
    global conversation_history

    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({"error": "메시지가 필요합니다"}), 400

    try:
        # LLM API 설정
        llm_api_url = os.getenv('LLM_API_URL', 'https://api.mindprep.co.kr/v1/chat/completions')
        llm_model = os.getenv('LLM_MODEL', 'vllm-qwen3-30b-a3b')
        openai_api_key = os.getenv('OPENAI_API_KEY', '')

        import requests

        # 대화 기록에 사용자 메시지 추가
        conversation_history.append({"role": "user", "content": message})

        messages_payload = [
            {"role": "system", "content": current_system_prompt},
            *conversation_history[-10:]  # 최근 10개 대화만 유지
        ]

        assistant_message = None
        llm_source = None  # LLM 소스 추적

        # 1차 시도: LLM API Server (api.mindprep.co.kr - 인증 불필요)
        try:
            response = requests.post(
                llm_api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": llm_model,
                    "messages": messages_payload,
                    "max_tokens": 200,
                    "temperature": 0.7
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result['choices'][0]['message']['content']
                llm_source = f"LLM API ({llm_model})"
                print(f"LLM API Server 응답 성공")
            else:
                print(f"LLM API Server 실패: {response.status_code}, OpenAI로 fallback")
        except Exception as e:
            print(f"LLM API Server 연결 실패: {e}, OpenAI로 fallback")

        # 2차 시도: OpenAI API (fallback)
        if assistant_message is None and openai_api_key:
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": messages_payload,
                        "max_tokens": 200,
                        "temperature": 0.7
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    assistant_message = result['choices'][0]['message']['content']
                    llm_source = "OpenAI (gpt-4o-mini)"
                    print(f"OpenAI fallback 응답 성공")
                else:
                    print(f"OpenAI fallback 실패: {response.status_code}")
            except Exception as e:
                print(f"OpenAI fallback 연결 실패: {e}")

        if assistant_message:
            # LLM 특수 토큰 및 think 태그 제거
            import re
            assistant_message = re.sub(r'<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>', '', assistant_message)
            assistant_message = re.sub(r'<think>.*?</think>', '', assistant_message, flags=re.DOTALL).strip()

            conversation_history.append({"role": "assistant", "content": assistant_message})
            return jsonify({"response": assistant_message, "llm_source": llm_source})
        else:
            return jsonify({"error": "LLM API 및 OpenAI fallback 모두 실패"}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/clear_history', methods=['POST'])
def api_clear_history():
    """대화 기록 초기화 API"""
    global conversation_history
    conversation_history = []
    return jsonify({"status": "cleared"})


# ============================================================
# REST API (동기적 - WebSocket 없이 사용 가능)
# ============================================================

@app.route('/api/v2/tts', methods=['POST'])
def api_v2_tts():
    """
    TTS 오디오 생성 API (동기적)

    Request:
        {
            "text": "합성할 텍스트",
            "engine": "elevenlabs",  // optional, default: elevenlabs
            "voice": "Custom"    // optional
        }

    Response:
        {
            "success": true,
            "audio_path": "assets/audio/tts_output.wav",
            "audio_url": "/assets/audio/tts_output.wav",
            "duration": 3.5,
            "elapsed": 2.1
        }
    """
    data = request.json or {}
    text = data.get('text', '')
    engine = data.get('engine', DEFAULT_TTS_ENGINE)
    voice = data.get('voice', DEFAULT_TTS_VOICE)

    if not text:
        return jsonify({"success": False, "error": "text가 필요합니다"}), 400

    try:
        start_time = time.time()
        output_path = Path("assets/audio/tts_api_output.wav")

        # 메모리에서 생성 후 파일로 저장 (테스트 API용)
        tts_result = generate_tts_audio(text, engine, voice, output_path)

        if tts_result:
            audio_numpy, sample_rate = tts_result
            # 오디오 길이 계산 (메모리에서)
            duration = len(audio_numpy) / float(sample_rate)

            elapsed = time.time() - start_time
            return jsonify({
                "success": True,
                "audio_path": str(output_path),
                "audio_url": "/assets/audio/tts_api_output.wav",
                "duration": round(duration, 2),
                "elapsed": round(elapsed, 2)
            })
        else:
            return jsonify({"success": False, "error": "TTS 생성 실패"}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


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

        # 립싱크 생성 (프리컴퓨트 데이터 전달 + 프레임 스킵)
        output_video = lipsync_engine.generate_lipsync(
            avatar_path,
            str(tts_output),
            output_dir="results/realtime",
            preloaded_data=precomputed_data,  # 미리 로드된 데이터 전달
            frame_skip=frame_skip  # 프레임 스킵 적용
        )

        if output_video:
            elapsed = time.time() - start_time
            return jsonify({
                "success": True,
                "video_path": output_video,
                "video_url": "/video/output_with_audio.mp4",
                "elapsed": round(elapsed, 2)
            })
        else:
            return jsonify({"success": False, "error": "립싱크 생성 실패"}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v2/chat_and_lipsync', methods=['POST'])
def api_v2_chat_and_lipsync():
    """
    채팅 + 립싱크 통합 API (동기적)
    사용자 메시지를 받아 LLM 응답 생성 후 립싱크 비디오까지 생성

    Request:
        {
            "message": "사용자 메시지",
            "avatar": "avatar_name",
            "avatar_path": "path.pkl",    // 또는 직접 경로
            "tts_engine": "elevenlabs",
            "tts_voice": "Custom"
        }

    Response:
        {
            "success": true,
            "response": "LLM 응답 텍스트",
            "video_path": "results/realtime/output_with_audio.mp4",
            "video_url": "/video/output_with_audio.mp4",
            "elapsed": 28.5
        }
    """
    global conversation_history, lipsync_engine

    data = request.json or {}
    message = data.get('message', '')
    avatar = data.get('avatar', '')
    avatar_path = data.get('avatar_path', '')
    tts_engine = data.get('tts_engine', DEFAULT_TTS_ENGINE)
    tts_voice = data.get('tts_voice', DEFAULT_TTS_VOICE)
    frame_skip = data.get('frame_skip', 1)  # 프레임 스킵 (1=없음, 2=절반추론, 3=1/3추론)

    if not message:
        return jsonify({"success": False, "error": "message가 필요합니다"}), 400

    # 아바타 경로 결정
    if not avatar_path and avatar:
        avatar_path = f"precomputed/{avatar}_precomputed.pkl"

    if not avatar_path:
        return jsonify({"success": False, "error": "avatar 또는 avatar_path가 필요합니다"}), 400

    if not os.path.exists(avatar_path):
        return jsonify({"success": False, "error": f"아바타 파일을 찾을 수 없습니다: {avatar_path}"}), 404

    if not lipsync_engine or not lipsync_engine.loaded:
        return jsonify({"success": False, "error": "립싱크 엔진이 로드되지 않았습니다"}), 503

    try:
        start_time = time.time()

        # 1. LLM 응답 생성 (LLM API Server -> OpenAI fallback)
        llm_api_url = os.getenv('LLM_API_URL', 'https://api.mindprep.co.kr/v1/chat/completions')
        llm_model = os.getenv('LLM_MODEL', 'vllm-qwen3-30b-a3b')
        openai_api_key = os.getenv('OPENAI_API_KEY', '')

        import requests as req
        conversation_history.append({"role": "user", "content": message})

        messages_payload = [
            {"role": "system", "content": current_system_prompt},
            *conversation_history[-10:]
        ]

        llm_response = None

        # 1차 시도: LLM API Server
        try:
            response = req.post(
                llm_api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": llm_model,
                    "messages": messages_payload,
                    "max_tokens": 200,
                    "temperature": 0.7
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                llm_response = result['choices'][0]['message']['content']
                print(f"LLM API Server 응답 성공")
            else:
                print(f"LLM API Server 실패: {response.status_code}, OpenAI로 fallback")
        except Exception as e:
            print(f"LLM API Server 연결 실패: {e}, OpenAI로 fallback")

        # 2차 시도: OpenAI API (fallback)
        if llm_response is None and openai_api_key:
            try:
                response = req.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": messages_payload,
                        "max_tokens": 200,
                        "temperature": 0.7
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    llm_response = result['choices'][0]['message']['content']
                    print(f"OpenAI fallback 응답 성공")
                else:
                    print(f"OpenAI fallback 실패: {response.status_code}")
            except Exception as e:
                print(f"OpenAI fallback 연결 실패: {e}")

        if llm_response is None:
            return jsonify({"success": False, "error": "LLM API 및 OpenAI fallback 모두 실패"}), 500

        # LLM 특수 토큰 및 think 태그 제거
        import re
        llm_response = re.sub(r'<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>', '', llm_response)
        llm_response = re.sub(r'<think>.*?</think>', '', llm_response, flags=re.DOTALL).strip()

        conversation_history.append({"role": "assistant", "content": llm_response})

        # 2. TTS 생성
        tts_output = Path("assets/audio/tts_output.wav")
        tts_success = generate_tts_audio(llm_response, tts_engine, tts_voice, tts_output)

        if not tts_success:
            return jsonify({
                "success": False,
                "response": llm_response,
                "error": "TTS 생성 실패"
            }), 500

        # 3. 립싱크 생성 (프레임 스킵 적용)
        output_video = lipsync_engine.generate_lipsync(
            avatar_path,
            str(tts_output),
            output_dir="results/realtime",
            frame_skip=frame_skip
        )

        elapsed = time.time() - start_time

        if output_video:
            return jsonify({
                "success": True,
                "response": llm_response,
                "video_path": output_video,
                "video_url": "/video/output_with_audio.mp4",
                "elapsed": round(elapsed, 2)
            })
        else:
            return jsonify({
                "success": False,
                "response": llm_response,
                "error": "립싱크 생성 실패"
            }), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


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

            # 1. TTS 생성 (메모리에서 직접 처리)
            engine_name = TTS_ENGINES.get(tts_engine, {}).get('name', tts_engine)
            socketio.emit('status', {'message': f'TTS 생성 중 ({engine_name})...', 'progress': 0, 'elapsed': 0}, to=sid)

            tts_result = generate_tts_audio(text, tts_engine, tts_voice)

            if tts_result is None:
                socketio.emit('error', {'message': f'TTS 생성 실패 ({engine_name})'}, to=sid)
                return

            audio_numpy, sample_rate = tts_result

            if sid in client_sessions and client_sessions[sid]['cancelled']:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
                return

            # 2. 오디오를 Base64로 인코딩하여 즉시 전송
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_numpy, sample_rate, format='WAV')
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

            # 오디오 먼저 전송 (클라이언트가 오디오를 먼저 로드할 수 있음)
            socketio.emit('stream_audio', {
                'audio': audio_base64,
                'sample_rate': sample_rate,
                'elapsed': time.time() - start_time
            }, to=sid)

            # 3. 스트리밍 립싱크 생성
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

    # LLM 설정 정보 출력
    llm_api_url = os.getenv('LLM_API_URL', 'https://api.mindprep.co.kr/v1/chat/completions')
    llm_model = os.getenv('LLM_MODEL', 'vllm-qwen3-30b-a3b')
    openai_api_key = os.getenv('OPENAI_API_KEY', '')

    print("\n" + "=" * 50)
    print("LLM 설정:")
    print(f"  Primary: {llm_model} ({llm_api_url})")
    print(f"  Fallback: {'OpenAI gpt-4o-mini (설정됨)' if openai_api_key else 'OpenAI (미설정)'}")
    print("=" * 50)

    print(f"\nURL: http://localhost:5000")
    print("=" * 50)

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
