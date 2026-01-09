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

# MuseTalk 경로 추가
MUSETALK_PATH = Path("c:/NewAvata/NewAvata/MuseTalk")
sys.path.insert(0, str(MUSETALK_PATH))

# CosyVoice 경로 추가
COSYVOICE_PATH = Path("c:/NewAvata/NewAvata/CosyVoice")
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
generation_cancelled = False  # 취소 플래그
generation_start_time = None  # 생성 시작 시간
models_loaded = False  # 모델 로드 상태
cosyvoice_loaded = False  # CosyVoice 로드 상태


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

        # CUDA 워밍업
        print("  CUDA 워밍업 중...")
        with torch.no_grad():
            dummy_latent = torch.randn(1, 4, 64, 64).to(self.device, dtype=self.weight_dtype)
            _ = self.vae.decode_latents(dummy_latent)

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
        global face_parsing

        if not self.loaded:
            raise RuntimeError("모델이 로드되지 않았습니다")

        start_time = time.time()
        batch_size = 8  # 배치 크기 (GPU 메모리에 따라 조절)

        # FaceParsing 초기화 (최초 1회)
        if face_parsing is None:
            print("FaceParsing 초기화 중...")
            face_parsing = FaceParsing(
                left_cheek_width=90,
                right_cheek_width=90
            )

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
        extended_latent_list = []
        for i in range(num_frames):
            idx = i % len(input_latent_list)
            extended_latent_list.append(input_latent_list[idx])

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

        with torch.no_grad():
            for batch_idx, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total_batches, desc="UNet 추론")):
                if generation_cancelled:
                    print("생성 취소됨")
                    return None

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
                for res_frame in recon_frames:
                    res_frame_list.append(res_frame)

                # 진행률 전송
                progress = int((batch_idx + 1) / total_batches * 50)
                elapsed = time.time() - start_time
                socketio.emit('status', {
                    'message': f'UNet 추론 중... {min((batch_idx+1)*batch_size, num_frames)}/{num_frames}',
                    'progress': progress,
                    'elapsed': elapsed
                })

        # 블렌딩
        print("블렌딩 시작...")
        generated_frames = []

        for i, res_frame in enumerate(tqdm(res_frame_list, desc="블렌딩")):
            idx = i % len(coords_list)
            coord = coords_list[idx]
            original_frame = copy.deepcopy(frames_list[idx])

            x1, y1, x2, y2 = coord
            y2_extended = min(y2 + extra_margin, original_frame.shape[0])

            # 예측 프레임 리사이즈
            try:
                pred_frame_resized = cv2.resize(
                    res_frame.astype(np.uint8),
                    (x2 - x1, y2_extended - y1)
                )
            except Exception as e:
                print(f"리사이즈 오류: {e}")
                generated_frames.append(original_frame)
                continue

            # 공식 get_image 블렌딩 (V1.5 jaw 모드)
            result_frame = get_image(
                original_frame,
                pred_frame_resized,
                [x1, y1, x2, y2_extended],
                mode="jaw",
                fp=face_parsing
            )

            generated_frames.append(result_frame)

            # 진행률 전송
            progress = 50 + int((i + 1) / len(res_frame_list) * 30)
            elapsed = time.time() - start_time
            socketio.emit('status', {
                'message': f'블렌딩 중... {i+1}/{len(res_frame_list)}',
                'progress': progress,
                'elapsed': elapsed
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
COSYVOICE_PROMPT_AUDIO = 'assets/audio/ElevenLabs_2025-05-09T07_25_56_Psychological Consultant Woman_gen_sp100_s94_sb75_se0_b_m2.mp3'
COSYVOICE_PROMPT_TEXT = "안녕하세요. 저는 심리 상담사입니다."
COSYVOICE_SPK_ID = 'default_speaker'  # 프리로드된 화자 ID
cosyvoice_prompt_wav_path = None  # 캐싱된 프롬프트 오디오 경로


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
    """CosyVoice 모델 로드 및 화자 프리로딩"""
    global cosyvoice_engine, cosyvoice_loaded, cosyvoice_prompt_wav_path

    if cosyvoice_loaded and cosyvoice_engine is not None:
        return

    print("CosyVoice 모델 로드 중...")
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2

        cosyvoice_engine = CosyVoice2(COSYVOICE_MODEL_PATH, fp16=True)
        print(f"CosyVoice 모델 로드 완료! (샘플레이트: {cosyvoice_engine.sample_rate})")

        # 프롬프트 오디오 사전 변환 (16kHz WAV로 캐싱)
        print("프롬프트 오디오 사전 변환 중...")
        cosyvoice_prompt_wav_path = "assets/audio/cosyvoice_prompt_16k.wav"
        subprocess.run(
            f'ffmpeg -y -i "{COSYVOICE_PROMPT_AUDIO}" -ar 16000 -t 15 "{cosyvoice_prompt_wav_path}"',
            shell=True, capture_output=True
        )

        # 화자 정보 프리로딩 (add_zero_shot_spk로 speaker embedding 캐싱)
        print("화자 정보 프리로딩 중 (add_zero_shot_spk)...")
        cosyvoice_engine.add_zero_shot_spk(
            prompt_text=COSYVOICE_PROMPT_TEXT,
            prompt_wav=cosyvoice_prompt_wav_path,
            zero_shot_spk_id=COSYVOICE_SPK_ID
        )
        print(f"화자 '{COSYVOICE_SPK_ID}' 프리로딩 완료!")

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
        # CosyVoice 2.0 TTS (프리로드된 화자 사용으로 속도 최적화)
        global cosyvoice_engine, cosyvoice_loaded, cosyvoice_prompt_wav_path

        try:
            if not cosyvoice_loaded or cosyvoice_engine is None:
                print("CosyVoice 모델이 로드되지 않았습니다. 로드 시도 중...")
                load_cosyvoice_model()

                if not cosyvoice_loaded:
                    print(f"CosyVoice 로드 실패")
                    return False

            print(f"CosyVoice Zero-shot TTS 생성 중 (캐싱된 화자)... (텍스트: {text[:30]}...)")
            start_time = time.time()

            # 프리로드된 화자 ID 사용 시 prompt_text와 prompt_wav는 빈 문자열로 전달
            # (공식 예제: https://github.com/FunAudioLLM/CosyVoice example.py 48번 줄 참고)
            output_audio = None
            for result in cosyvoice_engine.inference_zero_shot(
                tts_text=text,
                prompt_text='',  # 캐싱된 화자 사용 시 빈 문자열
                prompt_wav='',   # 캐싱된 화자 사용 시 빈 문자열
                zero_shot_spk_id=COSYVOICE_SPK_ID,  # 캐싱된 화자 ID 사용
                stream=False,
                speed=1.0
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
    global current_process, generation_cancelled, lipsync_engine, generation_start_time

    data = request.json
    avatar_path = data.get('avatar_path')
    text = data.get('text', '')
    tts_engine = data.get('tts_engine', DEFAULT_TTS_ENGINE)
    tts_voice = data.get('tts_voice', DEFAULT_TTS_VOICE)

    if not avatar_path:
        return jsonify({"error": "avatar_path가 필요합니다"}), 400

    if not text:
        return jsonify({"error": "text가 필요합니다"}), 400

    generation_cancelled = False
    generation_start_time = time.time()

    def generate_async():
        global generation_cancelled, generation_start_time

        try:
            # 1. TTS 생성
            engine_name = TTS_ENGINES.get(tts_engine, {}).get('name', tts_engine)
            socketio.emit('status', {'message': f'TTS 생성 중 ({engine_name})...', 'progress': 0, 'elapsed': 0})
            tts_output = Path("assets/audio/tts_output.wav")

            success = generate_tts_audio(text, tts_engine, tts_voice, tts_output)

            if not success:
                socketio.emit('error', {'message': f'TTS 생성 실패 ({engine_name})'})
                return

            if generation_cancelled:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'})
                return

            # 2. 립싱크 생성
            socketio.emit('status', {
                'message': '립싱크 생성 시작...',
                'progress': 10,
                'elapsed': time.time() - generation_start_time
            })

            output_video = lipsync_engine.generate_lipsync(
                avatar_path,
                str(tts_output),
                output_dir="results/realtime"
            )

            if generation_cancelled:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'})
                return

            if output_video:
                elapsed = time.time() - generation_start_time
                socketio.emit('status', {'message': '완료!', 'progress': 100, 'elapsed': elapsed})
                socketio.emit('complete', {
                    'video_path': 'results/realtime/output_with_audio.mp4',
                    'elapsed': elapsed
                })
            else:
                socketio.emit('error', {'message': '립싱크 생성 실패'})

        except Exception as e:
            import traceback
            traceback.print_exc()
            socketio.emit('error', {'message': str(e)})

    thread = threading.Thread(target=generate_async)
    thread.start()

    return jsonify({"status": "started"})


@app.route('/api/cancel', methods=['POST'])
def api_cancel():
    """생성 취소 API"""
    global generation_cancelled
    generation_cancelled = True
    return jsonify({"status": "cancelled"})


# 대화 기록 저장
conversation_history = []

# LLM 시스템 프롬프트
SYSTEM_PROMPT = """당신은 친절하고 전문적인 AI 면접관입니다.
면접자의 답변에 대해 적절한 후속 질문을 하거나 피드백을 제공합니다.
응답은 2-3문장 정도로 간결하게 유지하세요."""


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
                        {"role": "system", "content": SYSTEM_PROMPT},
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


@app.route('/api/generate_streaming', methods=['POST'])
def api_generate_streaming():
    """스트리밍 립싱크 비디오 생성 API (일반 생성과 동일하게 동작)"""
    global current_process, generation_cancelled, lipsync_engine, generation_start_time

    data = request.json
    avatar_path = data.get('avatar_path')
    text = data.get('text', '')
    tts_engine = data.get('tts_engine', DEFAULT_TTS_ENGINE)
    tts_voice = data.get('tts_voice', DEFAULT_TTS_VOICE)

    if not avatar_path:
        return jsonify({"error": "avatar_path가 필요합니다"}), 400

    if not text:
        return jsonify({"error": "text가 필요합니다"}), 400

    generation_cancelled = False
    generation_start_time = time.time()

    def generate_streaming_async():
        global generation_cancelled, generation_start_time

        try:
            # 1. TTS 생성
            engine_name = TTS_ENGINES.get(tts_engine, {}).get('name', tts_engine)
            socketio.emit('status', {'message': f'TTS 생성 중 ({engine_name})...', 'progress': 0, 'elapsed': 0})
            tts_output = Path("assets/audio/tts_output.wav")

            success = generate_tts_audio(text, tts_engine, tts_voice, tts_output)

            if not success:
                socketio.emit('error', {'message': f'TTS 생성 실패 ({engine_name})'})
                return

            if generation_cancelled:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'})
                return

            # 2. 립싱크 생성
            socketio.emit('status', {
                'message': '립싱크 생성 시작...',
                'progress': 10,
                'elapsed': time.time() - generation_start_time
            })

            output_video = lipsync_engine.generate_lipsync(
                avatar_path,
                str(tts_output),
                output_dir="results/realtime"
            )

            if generation_cancelled:
                socketio.emit('cancelled', {'message': '생성이 취소되었습니다'})
                return

            if output_video:
                elapsed = time.time() - generation_start_time
                socketio.emit('status', {'message': '완료!', 'progress': 100, 'elapsed': elapsed})
                socketio.emit('complete', {
                    'video_path': 'results/realtime/output_with_audio.mp4',
                    'elapsed': elapsed
                })
            else:
                socketio.emit('error', {'message': '립싱크 생성 실패'})

        except Exception as e:
            import traceback
            traceback.print_exc()
            socketio.emit('error', {'message': str(e)})

    thread = threading.Thread(target=generate_streaming_async)
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
    print('Client connected')
    emit('connected', {'status': 'ok', 'models_loaded': models_loaded})


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket 연결 해제"""
    print('Client disconnected')


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
    os.chdir("c:/NewAvata/NewAvata/realtime-interview-avatar")

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
