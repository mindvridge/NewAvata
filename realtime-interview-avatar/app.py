"""
실시간 립싱크 테스트 웹 프론트엔드
Flask + WebSocket 기반 + 모델 프리로드
"""

import os
import sys
import time
import json
import base64
import threading
import subprocess
import signal
import pickle
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

# CosyVoice 경로 추가 (환경변수 또는 기본값)
COSYVOICE_PATH = Path(os.getenv('COSYVOICE_PATH', 'c:/NewAvata/NewAvata/CosyVoice'))
sys.path.insert(0, str(COSYVOICE_PATH))

# 공식 MuseTalk 블렌딩 모듈
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing

# 전역 FaceParsing 인스턴스
face_parsing = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'realtime-lipsync-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 상태
lipsync_engine = None
cosyvoice_engine = None  # CosyVoice TTS 엔진
precomputed_avatars = {}
current_process = None  # 현재 실행 중인 프로세스
models_loaded = False  # 모델 로드 상태
cosyvoice_loaded = False  # CosyVoice 로드 상태

# 클라이언트별 상태 관리 (멀티 브라우저 지원)
client_sessions = {}  # {sid: {'cancelled': False, 'start_time': None, 'generating': False}}
generation_lock = threading.Lock()  # 동시 생성 방지 락


class LipsyncEngine:
    """립싱크 엔진 - 모델 프리로드 및 추론"""

    def __init__(self):
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.timesteps = None
        self.weight_dtype = None
        self.loaded = False

    def load_models(self, use_float16=True):
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
        print("  [1/4] VAE 로드 중...")
        self.vae = VAE(model_path="./models/sd-vae", use_float16=use_float16)
        if use_float16:
            self.vae.vae = self.vae.vae.half()
        self.vae.vae = self.vae.vae.to(self.device)

        # UNet 로드
        print("  [2/4] UNet 로드 중...")
        self.unet = UNet(
            unet_config="./models/musetalkV15/musetalk.json",
            model_path="./models/musetalkV15/unet.pth",
            device=self.device
        )
        if use_float16:
            self.unet.model = self.unet.model.half()
        self.unet.model = self.unet.model.to(self.device)

        # PE 로드
        print("  [3/4] Positional Encoding 로드 중...")
        self.pe = PositionalEncoding(d_model=384)
        self.pe = self.pe.to(self.device)
        if use_float16:
            self.pe = self.pe.half()

        # Whisper 로드
        print("  [4/4] Whisper 로드 중...")
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
        print("  CUDA 워밍업 중...")
        with torch.inference_mode():
            # 1. VAE 워밍업
            dummy_latent = torch.randn(1, 4, 64, 64).to(self.device, dtype=self.weight_dtype)
            _ = self.vae.decode_latents(dummy_latent)

            # 2. UNet 워밍업 (배치 크기 32로)
            dummy_latent_batch = torch.randn(32, 8, 64, 64).to(self.device, dtype=self.unet.model.dtype)
            dummy_audio = torch.randn(32, 50, 384).to(self.device, dtype=self.weight_dtype)
            dummy_audio_feature = self.pe(dummy_audio)
            _ = self.unet.model(
                dummy_latent_batch,
                self.timesteps,
                encoder_hidden_states=dummy_audio_feature
            ).sample

            # 3. 메모리 정리
            del dummy_latent, dummy_latent_batch, dummy_audio, dummy_audio_feature
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
        print(f"\n모델 프리로드 완료! ({elapsed:.1f}초)")
        print("=" * 50)

        self.loaded = True
        return True

    def generate_lipsync(self, precomputed_path, audio_path, output_dir="results/realtime", fps=25):
        """프리컴퓨트된 아바타로 립싱크 생성 (배치 처리 + 공식 MuseTalk 블렌딩)"""
        import torch
        import copy
        from musetalk.utils.utils import datagen
        from concurrent.futures import ThreadPoolExecutor
        global face_parsing

        if not self.loaded:
            raise RuntimeError("모델이 로드되지 않았습니다")

        start_time = time.time()
        batch_size = 16  # 배치 크기

        # 프리컴퓨트 데이터 로드
        with open(precomputed_path, 'rb') as f:
            precomputed = pickle.load(f)

        # 프리컴퓨트 데이터 키 매핑 (새 형식 지원)
        coords_list = precomputed.get('coords_list', precomputed.get('coord_list_cycle'))
        frames_list = precomputed.get('frames', precomputed.get('frame_list_cycle'))
        input_latent_list = precomputed.get('input_latent_list', precomputed.get('input_latent_list_cycle'))

        # FPS 가져오기
        fps = precomputed.get('fps', 25)
        extra_margin = 10  # V1.5 bbox 하단 확장

        # 오디오 처리
        print("오디오 whisper 특징 추출 중...")
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
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
        print(f"생성할 프레임 수: {num_frames}")

        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # latent 리스트를 whisper_chunks 길이에 맞게 순환 확장
        # CPU에서 로드된 latent를 GPU로 이동
        extended_latent_list = []
        for i in range(num_frames):
            idx = i % len(input_latent_list)
            latent = input_latent_list[idx]
            # CPU 텐서를 GPU로 이동
            if isinstance(latent, torch.Tensor):
                latent = latent.to(self.device, dtype=self.weight_dtype)
            extended_latent_list.append(latent)

        # 배치 처리를 위한 datagen 사용
        print(f"UNet 추론 시작 (batch_size={batch_size})...")
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=extended_latent_list,
            batch_size=batch_size,
            delay_frame=0,
            device=self.device,
        )

        res_frame_list = []
        total_batches = int(np.ceil(float(num_frames) / batch_size))

        # CUDA 최적화: inference_mode 사용 (더 빠름)
        with torch.inference_mode():
            for batch_idx, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total_batches, desc="UNet 추론")):
                # PE 적용
                audio_feature_batch = self.pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

                # 배치 UNet 추론
                pred_latents = self.unet.model(
                    latent_batch,
                    self.timesteps,
                    encoder_hidden_states=audio_feature_batch
                ).sample

                # 배치 VAE 디코딩
                recon_frames = self.vae.decode_latents(pred_latents)
                res_frame_list.extend(recon_frames)  # extend 사용 (더 빠름)

                # 진행률 전송
                progress = int((batch_idx + 1) / total_batches * 50)
                elapsed = time.time() - start_time
                socketio.emit('status', {
                    'message': f'UNet 추론 중... {min((batch_idx+1)*batch_size, num_frames)}/{num_frames}',
                    'progress': progress,
                    'elapsed': elapsed
                })

        # GPU 메모리 정리
        torch.cuda.empty_cache()

        # 블렌딩 (병렬 처리)
        print("블렌딩 시작 (병렬 처리)...")

        # 페이드 인/아웃 프레임 수 (약 0.3초)
        fade_frames = min(8, len(res_frame_list) // 4)
        total_frames = len(res_frame_list)

        def blend_single_frame(args):
            """단일 프레임 블렌딩 함수 (병렬 처리용) - 선명도 최적화"""
            i, res_frame = args
            idx = i % len(coords_list)
            coord = coords_list[idx]
            original_frame = frames_list[idx].copy()

            x1, y1, x2, y2 = [int(c) for c in coord]

            try:
                # 1. VAE 출력에 샤프닝 적용 (선명도 향상)
                res_frame_uint8 = res_frame.astype(np.uint8)

                # 언샤프 마스크 (Unsharp Mask) - 선명화
                gaussian = cv2.GaussianBlur(res_frame_uint8, (0, 0), 2.0)
                sharpened = cv2.addWeighted(res_frame_uint8, 1.5, gaussian, -0.5, 0)

                # 2. 고품질 리사이즈 (LANCZOS4 보간법)
                pred_frame_resized = cv2.resize(
                    sharpened,
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LANCZOS4
                )
            except Exception as e:
                return (i, original_frame)

            # 공식 get_image 블렌딩 (V1.5 jaw 모드)
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

        # 병렬 블렌딩 실행 (8개 스레드로 증가)
        generated_frames = [None] * total_frames
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(
                executor.map(blend_single_frame, enumerate(res_frame_list)),
                total=total_frames,
                desc="블렌딩"
            ))

        # 결과 정렬
        for idx, frame in results:
            generated_frames[idx] = frame

        # 진행률 전송
        socketio.emit('status', {
            'message': f'블렌딩 완료',
            'progress': 80,
            'elapsed': time.time() - start_time
        })

        # 비디오 저장
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
        socketio.emit('status', {
            'message': '오디오 합성 중...',
            'progress': 90,
            'elapsed': time.time() - start_time
        })

        subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            final_video
        ], capture_output=True)

        # 임시 파일 삭제
        if os.path.exists(temp_video):
            os.remove(temp_video)

        elapsed = time.time() - start_time
        print(f"\n립싱크 생성 완료! ({elapsed:.1f}초)")
        print(f"출력: {final_video}")

        return final_video


# TTS 엔진 설정 (CosyVoice 기본)
TTS_ENGINES = {
    'cosyvoice': {
        'name': 'CosyVoice 2.0',
        'voices': ['zero-shot'],
        'default': True
    },
    'elevenlabs': {
        'name': 'ElevenLabs',
        'voices': ['Custom']
    }
}

# 기본 TTS 엔진
DEFAULT_TTS_ENGINE = 'cosyvoice'
DEFAULT_TTS_VOICE = 'zero-shot'

# API 키 설정
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '')

# CosyVoice 설정
COSYVOICE_MODEL_PATH = os.getenv('COSYVOICE_MODEL_PATH', 'c:/NewAvata/NewAvata/CosyVoice/pretrained_models/CosyVoice2-0.5B')
COSYVOICE_PROMPT_AUDIO = 'assets/images/ElevenLabs_2026-01-12T04_48_56_여성 50대 면접관_gen_sp100_s50_sb75_se0_b_m2.mp3'
COSYVOICE_PROMPT_TEXT = "안녕하세요! 면접에 참여해 주셔서 감사합니다. 먼저, 본인에 대해 간단히 소개해 주시겠어요?"
cosyvoice_prompt_wav_path = None  # 캐싱된 프롬프트 오디오 경로 (24kHz WAV)


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
        # CosyVoice 가용성 확인
        if engine_id == 'cosyvoice':
            available = os.path.exists(COSYVOICE_MODEL_PATH)
        else:
            available = True

        engines.append({
            'id': engine_id,
            'name': engine_info['name'],
            'voices': engine_info['voices'],
            'available': available
        })
    return jsonify(engines)


def load_cosyvoice_model():
    """CosyVoice 모델 로드 (화자 프리로딩 없이 원본 방식 사용)"""
    global cosyvoice_engine, cosyvoice_loaded, cosyvoice_prompt_wav_path

    if cosyvoice_loaded and cosyvoice_engine is not None:
        return

    print("CosyVoice 모델 로드 중...")
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2

        cosyvoice_engine = CosyVoice2(COSYVOICE_MODEL_PATH, fp16=True)
        print(f"CosyVoice 모델 로드 완료! (샘플레이트: {cosyvoice_engine.sample_rate})")

        # 프롬프트 오디오 사전 변환 (24kHz WAV로 캐싱 - CosyVoice2 샘플레이트)
        print("프롬프트 오디오 사전 변환 중...")
        cosyvoice_prompt_wav_path = "assets/audio/cosyvoice_prompt_24k.wav"
        subprocess.run(
            f'ffmpeg -y -i "{COSYVOICE_PROMPT_AUDIO}" -ar 24000 -t 15 "{cosyvoice_prompt_wav_path}"',
            shell=True, capture_output=True
        )
        print(f"프롬프트 오디오 준비 완료: {cosyvoice_prompt_wav_path}")

        # NOTE: add_zero_shot_spk는 prompt_text 토큰까지 캐싱하여 TTS 품질 문제 발생
        # 원본 zero-shot 방식 사용 (매번 prompt_text + prompt_wav 전달)

        cosyvoice_loaded = True

    except Exception as e:
        print(f"CosyVoice 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        cosyvoice_loaded = False


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


def generate_tts_audio(text, engine, voice, output_path):
    """TTS 오디오 생성"""
    import torch
    import torchaudio

    print(f"[TTS] engine={engine}, voice={voice}, text={text[:50]}...")

    if engine == 'elevenlabs':
        # ElevenLabs TTS
        import requests

        voice_id = get_elevenlabs_voice_id(voice)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }

        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            mp3_path = str(output_path).replace('.wav', '.mp3')
            with open(mp3_path, 'wb') as f:
                f.write(response.content)

            # MP3 -> WAV 변환 (16kHz)
            subprocess.run(
                f'ffmpeg -y -i "{mp3_path}" -ar 16000 "{output_path}"',
                shell=True, capture_output=True
            )
            return os.path.exists(output_path)
        return False

    elif engine == 'cosyvoice':
        # CosyVoice 2.0 TTS (원본 zero-shot 방식 - 매번 prompt_text + prompt_wav 전달)
        global cosyvoice_engine, cosyvoice_loaded, cosyvoice_prompt_wav_path

        try:
            if not cosyvoice_loaded or cosyvoice_engine is None:
                print("CosyVoice 모델이 로드되지 않았습니다. 로드 시도 중...")
                load_cosyvoice_model()

                if not cosyvoice_loaded:
                    print(f"CosyVoice 로드 실패")
                    return False

            print(f"CosyVoice Zero-shot TTS 생성 중 (원본 방식)... (텍스트: {text[:30]}...)")
            start_time = time.time()

            # 원본 zero-shot 방식: prompt_text + prompt_wav를 매번 전달
            # 이 방식이 정확한 음성 복제를 수행함
            output_audio = None
            for result in cosyvoice_engine.inference_zero_shot(
                tts_text=text,
                prompt_text=COSYVOICE_PROMPT_TEXT,  # 원본 프롬프트 텍스트
                prompt_wav=cosyvoice_prompt_wav_path,  # 원본 프롬프트 오디오
                stream=False,
                speed=1.0,
                text_frontend=False  # 한국어: 텍스트 전처리 비활성화 (숫자 영어 변환 방지)
            ):
                if output_audio is None:
                    output_audio = result['tts_speech']
                else:
                    output_audio = torch.cat([output_audio, result['tts_speech']], dim=1)

            elapsed = time.time() - start_time
            if output_audio is not None:
                speech_len = output_audio.shape[1] / cosyvoice_engine.sample_rate
                rtf = elapsed / speech_len
                print(f"CosyVoice TTS 생성 완료 (길이: {speech_len:.2f}초, RTF: {rtf:.3f})")

                # WAV로 저장
                torchaudio.save(str(output_path), output_audio, cosyvoice_engine.sample_rate)

                # 16kHz로 리샘플링 (립싱크 호환)
                temp_path = str(output_path).replace('.wav', '_temp.wav')
                os.rename(str(output_path), temp_path)
                subprocess.run(
                    f'ffmpeg -y -i "{temp_path}" -ar 16000 "{output_path}"',
                    shell=True, capture_output=True
                )
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                print(f"CosyVoice TTS 완료: {output_path}")
                return os.path.exists(output_path)

            return False

        except Exception as e:
            print(f"CosyVoice TTS 오류: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"[TTS] 알 수 없는 엔진: {engine}")
    return False


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

            # 1. TTS 생성
            engine_name = TTS_ENGINES.get(tts_engine, {}).get('name', tts_engine)
            socketio.emit('status', {'message': f'TTS 생성 중 ({engine_name})...', 'progress': 0, 'elapsed': 0}, to=sid)
            tts_output = Path("assets/audio/tts_output.wav")

            success = generate_tts_audio(text, tts_engine, tts_voice, tts_output)

            if not success:
                socketio.emit('error', {'message': f'TTS 생성 실패 ({engine_name})'}, to=sid)
                return

            if sid in client_sessions and client_sessions[sid]['cancelled']:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
                return

            # 2. 립싱크 생성
            socketio.emit('status', {
                'message': '립싱크 생성 시작...',
                'progress': 10,
                'elapsed': time.time() - start_time
            }, to=sid)

            output_video = lipsync_engine.generate_lipsync(
                avatar_path,
                str(tts_output),
                output_dir="results/realtime"
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
        # OpenAI API 사용 (환경변수에서 키 가져오기)
        openai_api_key = os.getenv('OPENAI_API_KEY', '')

        if openai_api_key:
            import requests

            # 대화 기록에 사용자 메시지 추가
            conversation_history.append({"role": "user", "content": message})

            # API 호출
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": current_system_prompt},
                        *conversation_history[-10:]  # 최근 10개 대화만 유지
                    ],
                    "max_tokens": 200,
                    "temperature": 0.7
                }
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result['choices'][0]['message']['content']
                conversation_history.append({"role": "assistant", "content": assistant_message})
                return jsonify({"response": assistant_message})
            else:
                return jsonify({"error": f"OpenAI API 오류: {response.status_code}"}), 500
        else:
            # OpenAI 키가 없으면 간단한 응답
            responses = [
                f"'{message}'에 대해 더 자세히 설명해 주시겠어요?",
                "좋은 답변입니다. 다른 경험도 있으신가요?",
                "그 경험에서 무엇을 배우셨나요?",
                "흥미로운 관점이네요. 구체적인 예시를 들어주실 수 있나요?"
            ]
            import random
            response = random.choice(responses)
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": response})
            return jsonify({"response": response})

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
            "engine": "cosyvoice",  // optional, default: cosyvoice
            "voice": "zero-shot"    // optional
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

        success = generate_tts_audio(text, engine, voice, output_path)

        if success:
            # 오디오 길이 계산
            import wave
            with wave.open(str(output_path), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)

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
            "tts_engine": "cosyvoice",    // optional
            "tts_voice": "zero-shot"      // optional
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

        # 1. TTS 생성
        tts_output = Path("assets/audio/tts_output.wav")
        tts_success = generate_tts_audio(text, tts_engine, tts_voice, tts_output)

        if not tts_success:
            return jsonify({"success": False, "error": "TTS 생성 실패"}), 500

        # 2. 립싱크 생성
        output_video = lipsync_engine.generate_lipsync(
            avatar_path,
            str(tts_output),
            output_dir="results/realtime"
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
            "tts_engine": "cosyvoice",
            "tts_voice": "zero-shot"
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

        # 1. LLM 응답 생성
        openai_api_key = os.getenv('OPENAI_API_KEY', '')

        if openai_api_key:
            import requests as req
            conversation_history.append({"role": "user", "content": message})

            response = req.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": current_system_prompt},
                        *conversation_history[-10:]
                    ],
                    "max_tokens": 200,
                    "temperature": 0.7
                }
            )

            if response.status_code == 200:
                result = response.json()
                llm_response = result['choices'][0]['message']['content']
                conversation_history.append({"role": "assistant", "content": llm_response})
            else:
                return jsonify({"success": False, "error": f"OpenAI API 오류: {response.status_code}"}), 500
        else:
            # OpenAI 키가 없으면 간단한 응답
            import random
            responses = [
                f"'{message}'에 대해 더 자세히 설명해 주시겠어요?",
                "좋은 답변입니다. 다른 경험도 있으신가요?",
                "그 경험에서 무엇을 배우셨나요?",
            ]
            llm_response = random.choice(responses)
            conversation_history.append({"role": "user", "content": message})
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

        # 3. 립싱크 생성
        output_video = lipsync_engine.generate_lipsync(
            avatar_path,
            str(tts_output),
            output_dir="results/realtime"
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
            "cosyvoice_loaded": true,
            "available_avatars": ["avatar1", "avatar2"],
            "tts_engines": ["cosyvoice", "elevenlabs"]
        }
    """
    avatars = get_available_avatars()
    avatar_names = [a['id'] for a in avatars]

    available_engines = []
    for engine_id, engine_info in TTS_ENGINES.items():
        if engine_id == 'cosyvoice':
            if os.path.exists(COSYVOICE_MODEL_PATH):
                available_engines.append(engine_id)
        else:
            available_engines.append(engine_id)

    return jsonify({
        "status": "ok",
        "models_loaded": models_loaded,
        "cosyvoice_loaded": cosyvoice_loaded,
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

            # 1. TTS 생성
            engine_name = TTS_ENGINES.get(tts_engine, {}).get('name', tts_engine)
            socketio.emit('status', {'message': f'TTS 생성 중 ({engine_name})...', 'progress': 0, 'elapsed': 0}, to=sid)
            tts_output = Path("assets/audio/tts_output.wav")

            success = generate_tts_audio(text, tts_engine, tts_voice, tts_output)

            if not success:
                socketio.emit('error', {'message': f'TTS 생성 실패 ({engine_name})'}, to=sid)
                return

            if sid in client_sessions and client_sessions[sid]['cancelled']:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'}, to=sid)
                return

            # 2. 립싱크 생성
            socketio.emit('status', {
                'message': '립싱크 생성 시작...',
                'progress': 10,
                'elapsed': time.time() - start_time
            }, to=sid)

            output_video = lipsync_engine.generate_lipsync(
                avatar_path,
                str(tts_output),
                output_dir="results/realtime"
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
    lipsync_engine.load_models(use_float16=True)
    models_loaded = True

    # CosyVoice 모델 프리로드
    print("\n" + "=" * 50)
    print("CosyVoice 모델 프리로드 시작...")
    print("=" * 50)
    load_cosyvoice_model()

    print("\n프리로드 완료!")

    print(f"\nURL: http://localhost:5000")
    print("=" * 50)

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
