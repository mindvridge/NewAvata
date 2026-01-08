"""
공식 MuseTalk inference.py의 핵심 로직을 그대로 사용

mmpose 의존성 없이 실행 가능하도록 얼굴 감지만 OpenCV로 대체
나머지 모든 코드는 공식 GitHub과 동일
"""

import os
import sys
import cv2
import copy
import math
import torch
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import subprocess

# MuseTalk 경로 추가
MUSETALK_PATH = Path("c:/NewAvata/NewAvata/MuseTalk")
sys.path.insert(0, str(MUSETALK_PATH))
os.chdir("c:/NewAvata/NewAvata/realtime-interview-avatar")

# 공식 MuseTalk 모듈 import (preprocessing 제외)
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import datagen
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding
from transformers import WhisperModel


# ========== 얼굴 감지 (OpenCV로 대체) ==========
def get_landmark_and_bbox_opencv(img_list, upperbondrange=0):
    """
    공식 MuseTalk의 get_landmark_and_bbox 대체 함수

    원본: mmpose + face_detection 사용
    대체: OpenCV Haar Cascade 사용
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    frames = []
    for img_path in img_list:
        frame = cv2.imread(img_path)
        frames.append(frame)

    coords_list = []

    for frame in tqdm(frames, desc="얼굴 bbox 추출"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        if len(faces) == 0:
            # 얼굴 미감지 시 이미지 중앙 사용
            h, w = frame.shape[:2]
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            coords_list.append([margin_x, margin_y, w - margin_x, h - margin_y])
            continue

        # 가장 큰 얼굴 선택
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        h, w = frame.shape[:2]

        # 공식 MuseTalk 스타일로 bbox 확장
        face_center_y = y + fh // 2
        face_bottom = y + fh + upperbondrange

        # upper_bond 계산
        half_face_dist = face_bottom - face_center_y
        upper_bond = max(0, face_center_y - half_face_dist)

        # bbox 계산
        x1 = max(0, x - int(fw * 0.15))
        y1 = int(upper_bond)
        x2 = min(w, x + fw + int(fw * 0.15))
        y2 = min(h, face_bottom)

        coords_list.append([x1, y1, x2, y2])

    return coords_list, frames


def run_official_inference():
    """공식 MuseTalk V1.5 inference 실행"""
    print("=" * 70)
    print("공식 MuseTalk V1.5 Inference 실행")
    print("=" * 70)

    # ===== 설정 (공식 기본값) =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version = "v15"
    use_float16 = False
    batch_size = 8
    fps = 25
    extra_margin = 10
    parsing_mode = "jaw"
    left_cheek_width = 90
    right_cheek_width = 90
    audio_padding_length_left = 2
    audio_padding_length_right = 2

    # 입출력 경로
    video_path = str(MUSETALK_PATH / "data" / "video" / "sun.mp4")
    audio_path = str(MUSETALK_PATH / "data" / "audio" / "sun.wav")
    output_dir = Path("./results/official_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"Video: {video_path}")
    print(f"Audio: {audio_path}")

    # ===== 1. 모델 로드 (공식 방식) =====
    print("\n[1/8] 모델 로드...")
    vae = VAE(model_path="./models/sd-vae", use_float16=use_float16)
    unet = UNet(
        unet_config="./models/musetalkV15/musetalk.json",
        model_path="./models/musetalkV15/unet.pth",
        device=device
    )
    pe = PositionalEncoding(d_model=384)
    timesteps = torch.tensor([0], device=device)

    weight_dtype = torch.float16 if use_float16 else torch.float32
    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    # Whisper 로드
    audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")
    whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # FaceParsing (V1.5)
    fp = FaceParsing(
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width
    )
    print("   모델 로드 완료")

    # ===== 2. 비디오 프레임 추출 =====
    print("\n[2/8] 비디오 프레임 추출...")
    temp_frames_dir = output_dir / "temp_frames"
    temp_frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(temp_frames_dir / f"{frame_count:08d}.png"), frame)
        frame_count += 1
    cap.release()

    print(f"   총 {frame_count} 프레임 추출 (FPS: {video_fps})")

    # 프레임 리스트
    input_img_list = sorted(
        [str(p) for p in temp_frames_dir.glob("*.png")]
    )

    # ===== 3. 얼굴 bbox 추출 =====
    print("\n[3/8] 얼굴 bbox 추출 (OpenCV)...")
    coord_list, frame_list = get_landmark_and_bbox_opencv(input_img_list)
    print(f"   bbox 추출 완료: {len(coord_list)} 개")

    # ===== 4. 오디오 특징 추출 (공식 AudioProcessor) =====
    print("\n[4/8] 오디오 특징 추출...")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=fps,
        audio_padding_length_left=audio_padding_length_left,
        audio_padding_length_right=audio_padding_length_right,
    )
    print(f"   whisper_chunks: {whisper_chunks.shape}")

    # ===== 5. VAE 인코딩 (공식 vae.get_latents_for_unet) =====
    print("\n[5/8] VAE 인코딩...")
    input_latent_list = []

    for i, (bbox, frame) in enumerate(tqdm(zip(coord_list, frame_list), total=len(coord_list), desc="   VAE 인코딩")):
        x1, y1, x2, y2 = bbox

        # V1.5: extra_margin 추가
        y2 = y2 + extra_margin
        y2 = min(y2, frame.shape[0])

        # 크롭 및 리사이즈
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # 공식 VAE 인코딩
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    print(f"   인코딩 완료: {len(input_latent_list)} 개")

    # ===== 6. UNet 추론 (공식 datagen 사용) =====
    print("\n[6/8] UNet 추론...")
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )

    res_frame_list = []
    total = int(np.ceil(float(len(whisper_chunks)) / batch_size))

    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total, desc="   UNet 추론")):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)

        pred_latents = unet.model(
            latent_batch,
            timesteps,
            encoder_hidden_states=audio_feature_batch
        ).sample

        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    print(f"   생성 완료: {len(res_frame_list)} 프레임")

    # ===== 7. 블렌딩 (공식 get_image) =====
    print("\n[7/8] 블렌딩...")
    output_frames_dir = output_dir / "output_frames"
    output_frames_dir.mkdir(exist_ok=True)

    for i, res_frame in enumerate(tqdm(res_frame_list, desc="   블렌딩")):
        bbox = coord_list[i % len(coord_list)]
        ori_frame = copy.deepcopy(frame_list[i % len(frame_list)])

        x1, y1, x2, y2 = bbox
        y2 = y2 + extra_margin
        y2 = min(y2, ori_frame.shape[0])

        # res_frame 리사이즈
        try:
            res_frame = cv2.resize(
                res_frame.astype(np.uint8),
                (x2 - x1, y2 - y1)
            )
        except:
            continue

        # 공식 V1.5 블렌딩
        combine_frame = get_image(
            ori_frame,
            res_frame,
            [x1, y1, x2, y2],
            mode=parsing_mode,
            fp=fp
        )

        # 프레임 저장
        cv2.imwrite(str(output_frames_dir / f"{i:08d}.png"), combine_frame)

    print(f"   블렌딩 완료")

    # ===== 8. 비디오 생성 (ffmpeg) =====
    print("\n[8/8] 비디오 생성...")
    output_video = str(output_dir / "output_video.mp4")
    output_video_with_audio = str(output_dir / "output_with_audio.mp4")

    # ffmpeg로 프레임 → 비디오
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(output_frames_dir / "%08d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    subprocess.run(ffmpeg_cmd, capture_output=True)

    # 오디오 추가
    ffmpeg_audio_cmd = [
        "ffmpeg", "-y",
        "-i", output_video,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_video_with_audio
    ]
    subprocess.run(ffmpeg_audio_cmd, capture_output=True)

    # 비교 이미지 생성
    print("\n비교 이미지 생성...")
    original = frame_list[0]
    generated_path = list(output_frames_dir.glob("*.png"))[0]
    generated = cv2.imread(str(generated_path))

    h = min(original.shape[0], generated.shape[0])
    w = min(original.shape[1], generated.shape[1])
    original_resized = cv2.resize(original, (w, h))
    generated_resized = cv2.resize(generated, (w, h))

    comparison = np.hstack([original_resized, generated_resized])
    cv2.imwrite(str(output_dir / "comparison.jpg"), comparison)

    # 정리
    print("\n정리 중...")
    shutil.rmtree(temp_frames_dir)

    print(f"\n" + "=" * 70)
    print(f"결과 저장:")
    print(f"  - 비디오: {output_video_with_audio}")
    print(f"  - 비교 이미지: {output_dir / 'comparison.jpg'}")
    print(f"  - 출력 프레임: {output_frames_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_official_inference()
