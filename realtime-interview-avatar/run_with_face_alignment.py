"""
face-alignment 라이브러리를 사용한 정밀한 얼굴 bbox 추출

공식 MuseTalk은 face_detection + mmpose를 사용하지만,
face-alignment 라이브러리로 68점 랜드마크를 추출할 수 있습니다.

핵심: 68점 랜드마크 기반으로 정확한 얼굴 bbox를 계산
"""

import os
import sys
import cv2
import copy
import math
import torch
import shutil
import numpy as np
import face_alignment
from pathlib import Path
from tqdm import tqdm
import subprocess

# MuseTalk 경로 추가
MUSETALK_PATH = Path("c:/NewAvata/NewAvata/MuseTalk")
sys.path.insert(0, str(MUSETALK_PATH))
os.chdir("c:/NewAvata/NewAvata/realtime-interview-avatar")

# 공식 MuseTalk 모듈 import
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import datagen
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding
from transformers import WhisperModel


def get_landmark_and_bbox_fa(img_list, upperbondrange=0):
    """
    face-alignment 라이브러리를 사용한 정밀 bbox 추출

    공식 MuseTalk의 get_landmark_and_bbox 함수를 모방
    68점 랜드마크 기반으로 정확한 bbox를 계산합니다.

    68점 랜드마크 인덱스:
    - 0-16: 턱 윤곽 (jaw)
    - 17-21: 왼쪽 눈썹
    - 22-26: 오른쪽 눈썹
    - 27-30: 코 다리
    - 31-35: 코 아래
    - 36-41: 왼쪽 눈
    - 42-47: 오른쪽 눈
    - 48-67: 입

    공식 코드에서 사용하는 face_land_mark[29]는 코 끝 (nose tip)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=device
    )

    frames = []
    for img_path in img_list:
        frame = cv2.imread(img_path)
        frames.append(frame)

    coords_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    print(f"   face-alignment으로 68점 랜드마크 추출 중...")

    for frame in tqdm(frames, desc="   랜드마크 추출"):
        # 68점 랜드마크 추출
        preds = fa.get_landmarks(frame)

        if preds is None or len(preds) == 0:
            # 랜드마크 감지 실패 시 이미지 중앙 사용
            h, w = frame.shape[:2]
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            coords_list.append([margin_x, margin_y, w - margin_x, h - margin_y])
            continue

        # 첫 번째 얼굴의 랜드마크 사용
        face_land_mark = preds[0].astype(np.int32)

        # 공식 MuseTalk 로직 그대로 구현
        # half_face_coord = face_land_mark[29] (코 끝)
        half_face_coord = face_land_mark[30].copy()  # 코 끝 (index 30 in 68-point)

        if upperbondrange != 0:
            half_face_coord[1] = upperbondrange + half_face_coord[1]

        # 얼굴 하단 최대 y값
        max_y = np.max(face_land_mark[:, 1])

        # half_face_dist 계산
        half_face_dist = max_y - half_face_coord[1]

        # upper_bond 계산
        min_upper_bond = 0
        upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

        # bbox 계산 (랜드마크 기반)
        x1 = np.min(face_land_mark[:, 0])
        y1 = int(upper_bond)
        x2 = np.max(face_land_mark[:, 0])
        y2 = int(max_y)

        h, w = frame.shape[:2]

        # 경계 체크
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            # 유효하지 않은 bbox
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            coords_list.append([margin_x, margin_y, w - margin_x, h - margin_y])
        else:
            coords_list.append([x1, y1, x2, y2])

    return coords_list, frames


def run_inference_with_face_alignment():
    """face-alignment 기반 정밀 inference 실행"""
    print("=" * 70)
    print("face-alignment 68점 랜드마크 기반 MuseTalk V1.5 Inference")
    print("=" * 70)

    # ===== 설정 =====
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
    output_dir = Path("./results/face_alignment_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"Video: {video_path}")
    print(f"Audio: {audio_path}")

    # ===== 1. 모델 로드 =====
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

    # ===== 3. face-alignment 랜드마크 기반 bbox 추출 =====
    print("\n[3/8] face-alignment 68점 랜드마크 기반 bbox 추출...")
    coord_list, frame_list = get_landmark_and_bbox_fa(input_img_list)
    print(f"   bbox 추출 완료: {len(coord_list)} 개")

    # bbox 시각화 저장
    if len(frame_list) > 0:
        debug_frame = frame_list[0].copy()
        x1, y1, x2, y2 = coord_list[0]
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 중심점 표시
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imwrite(str(output_dir / "debug_bbox.jpg"), debug_frame)
        print(f"   첫 번째 bbox: {coord_list[0]}")
        print(f"   bbox 중심: ({cx}, {cy})")

    # ===== 4. 오디오 특징 추출 =====
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

    # ===== 5. VAE 인코딩 =====
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

        # VAE 인코딩
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    print(f"   인코딩 완료: {len(input_latent_list)} 개")

    # ===== 6. UNet 추론 =====
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

    # ===== 7. 블렌딩 =====
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

        cv2.imwrite(str(output_frames_dir / f"{i:08d}.png"), combine_frame)

    print(f"   블렌딩 완료")

    # ===== 8. 비디오 생성 =====
    print("\n[8/8] 비디오 생성...")
    output_video = str(output_dir / "output_video.mp4")
    output_video_with_audio = str(output_dir / "output_with_audio.mp4")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(output_frames_dir / "%08d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    subprocess.run(ffmpeg_cmd, capture_output=True)

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

    # 크롭 비교
    x1, y1, x2, y2 = coord_list[0]
    y2_ext = min(y2 + extra_margin, original.shape[0])
    crop_original = original[y1:y2_ext, x1:x2]
    crop_original = cv2.resize(crop_original, (256, 256))

    if len(res_frame_list) > 0:
        crop_generated = res_frame_list[0]
        crop_comparison = np.hstack([crop_original, crop_generated])
        cv2.imwrite(str(output_dir / "crop_comparison.jpg"), crop_comparison)

    # 정리
    print("\n정리 중...")
    shutil.rmtree(temp_frames_dir)

    print(f"\n" + "=" * 70)
    print(f"결과 저장:")
    print(f"  - 비디오: {output_video_with_audio}")
    print(f"  - 비교 이미지: {output_dir / 'comparison.jpg'}")
    print(f"  - 크롭 비교: {output_dir / 'crop_comparison.jpg'}")
    print(f"  - bbox 디버그: {output_dir / 'debug_bbox.jpg'}")
    print("=" * 70)


if __name__ == "__main__":
    run_inference_with_face_alignment()
