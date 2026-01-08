"""
공식 MuseTalk 로직을 face_alignment 라이브러리만으로 구현

mmpose 대신 face_alignment의 68점 랜드마크를 사용하여
공식 코드와 동일한 bbox 계산 로직을 적용합니다.

68점 랜드마크 구조 (dlib 표준):
- 0-16: 턱 윤곽 (jaw)
- 17-21: 왼쪽 눈썹
- 22-26: 오른쪽 눈썹
- 27-30: 코 다리 (nose bridge)
- 31-35: 코 아래 (nose tip area)
- 36-41: 왼쪽 눈
- 42-47: 오른쪽 눈
- 48-59: 입 바깥 윤곽
- 60-67: 입 안쪽 윤곽

공식 MuseTalk의 face_land_mark[28], [29], [30]은:
- mmpose keypoints[23:91]의 28, 29, 30번
- 이는 코 다리/코 끝 영역에 해당
- 68점 랜드마크에서는 27-30번 (nose bridge)와 30-35번 (nose tip area)에 해당
"""

import os
import sys
import cv2
import copy
import torch
import shutil
import numpy as np
import face_alignment
from pathlib import Path
from tqdm import tqdm
import subprocess
import pickle

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


# ========== face_alignment으로 공식 로직 구현 ==========

# face_alignment 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device=device
)

coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def read_imgs(img_list):
    """공식 preprocessing.py의 read_imgs 함수 그대로"""
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    공식 preprocessing.py의 get_landmark_and_bbox 함수 로직을
    face_alignment 68점 랜드마크로 구현

    공식 코드 핵심:
    1. face_land_mark[29] = half_face_coord (코 끝 기준점)
    2. half_face_dist = max(face_land_mark[:,1]) - half_face_coord[1]
    3. upper_bond = half_face_coord[1] - half_face_dist
    4. bbox = (min_x, upper_bond, max_x, max_y)

    68점 랜드마크에서:
    - index 30 = 코 끝 (nose tip) - 공식 코드의 [29]와 유사
    - index 27-29 = 코 다리
    - index 31-35 = 코 아래 영역
    """
    frames = read_imgs(img_list)
    coords_list = []

    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')

    average_range_minus = []
    average_range_plus = []

    for frame in tqdm(frames, desc="   랜드마크 추출"):
        # face_alignment으로 68점 랜드마크 추출
        preds = fa.get_landmarks(frame)

        if preds is None or len(preds) == 0:
            coords_list.append(coord_placeholder)
            continue

        # 첫 번째 얼굴의 랜드마크
        face_land_mark = preds[0].astype(np.int32)

        # 공식 코드 로직 적용
        # half_face_coord = face_land_mark[29] (코 끝)
        # 68점 랜드마크에서 코 끝은 index 30
        half_face_coord = face_land_mark[30].copy()

        # range 계산 (공식 코드와 동일)
        # 68점에서: 30=코끝, 29=코다리아래, 28=코다리중간
        range_minus = (face_land_mark[33] - face_land_mark[30])[1]  # 코 아래 - 코 끝
        range_plus = (face_land_mark[30] - face_land_mark[29])[1]   # 코 끝 - 코 다리

        if range_minus != 0:
            average_range_minus.append(range_minus)
        if range_plus != 0:
            average_range_plus.append(range_plus)

        if upperbondrange != 0:
            half_face_coord[1] = upperbondrange + half_face_coord[1]

        # half_face_dist 계산 (공식 코드와 동일)
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]

        # upper_bond 계산 (공식 코드와 동일)
        min_upper_bond = 0
        upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

        # bbox 계산 (공식 코드와 동일)
        f_landmark = (
            np.min(face_land_mark[:, 0]),
            int(upper_bond),
            np.max(face_land_mark[:, 0]),
            np.max(face_land_mark[:, 1])
        )
        x1, y1, x2, y2 = f_landmark

        h, w = frame.shape[:2]

        # 유효성 검사 (공식 코드와 동일)
        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            # fallback: 전체 이미지 사용
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            coords_list.append((margin_x, margin_y, w - margin_x, h - margin_y))
            print(f"   invalid bbox, using fallback")
        else:
            coords_list.append(f_landmark)

    # 공식 코드 출력 메시지
    if average_range_minus and average_range_plus:
        print("*" * 100)
        print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
        print("*" * 100)

    return coords_list, frames


def run_official_inference():
    """공식 MuseTalk 로직으로 inference 실행"""
    print("=" * 70)
    print("공식 MuseTalk V1.5 Inference")
    print("- face_alignment 68점 랜드마크 사용")
    print("- 공식 preprocessing.py bbox 계산 로직 그대로 적용")
    print("=" * 70)

    # ===== 설정 (공식 기본값) =====
    inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    output_dir = Path("./results/official_fa_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice: {inference_device}")
    print(f"Video: {video_path}")
    print(f"Audio: {audio_path}")

    # ===== 1. 모델 로드 =====
    print("\n[1/8] 모델 로드...")
    vae = VAE(model_path="./models/sd-vae", use_float16=use_float16)
    unet = UNet(
        unet_config="./models/musetalkV15/musetalk.json",
        model_path="./models/musetalkV15/unet.pth",
        device=inference_device
    )
    pe = PositionalEncoding(d_model=384)
    timesteps = torch.tensor([0], device=inference_device)

    weight_dtype = torch.float16 if use_float16 else torch.float32
    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    pe = pe.to(inference_device)
    vae.vae = vae.vae.to(inference_device)
    unet.model = unet.model.to(inference_device)

    # Whisper 로드
    audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")
    whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper = whisper.to(device=inference_device, dtype=weight_dtype).eval()
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

    # ===== 3. 공식 bbox 추출 로직 =====
    print("\n[3/8] face_alignment 68점 랜드마크 기반 bbox 추출...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, upperbondrange=0)
    print(f"   bbox 추출 완료: {len(coord_list)} 개")

    # bbox 저장
    coord_path = output_dir / "coord_face.pkl"
    with open(coord_path, 'wb') as f:
        pickle.dump(coord_list, f)

    # bbox 시각화
    if len(frame_list) > 0 and coord_list[0] != coord_placeholder:
        debug_frame = frame_list[0].copy()
        x1, y1, x2, y2 = coord_list[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 얼굴 중심 표시
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)

        # 랜드마크 표시
        preds = fa.get_landmarks(frame_list[0])
        if preds is not None:
            for i, (px, py) in enumerate(preds[0]):
                color = (255, 0, 0) if i == 30 else (0, 255, 255)  # 코 끝은 빨간색
                cv2.circle(debug_frame, (int(px), int(py)), 2, color, -1)

        cv2.imwrite(str(output_dir / "debug_bbox_landmarks.jpg"), debug_frame)
        print(f"   첫 번째 bbox: ({x1}, {y1}, {x2}, {y2})")
        print(f"   bbox 크기: {x2-x1} x {y2-y1}")

    # ===== 4. 오디오 특징 추출 =====
    print("\n[4/8] 오디오 특징 추출...")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        inference_device,
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

    valid_indices = []
    for i, (bbox, frame) in enumerate(tqdm(zip(coord_list, frame_list), total=len(coord_list), desc="   VAE 인코딩")):
        if bbox == coord_placeholder:
            continue

        valid_indices.append(i)
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

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
        device=inference_device,
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

    valid_coords = [coord_list[i] for i in valid_indices]
    valid_frames = [frame_list[i] for i in valid_indices]

    for i, res_frame in enumerate(tqdm(res_frame_list, desc="   블렌딩")):
        bbox = valid_coords[i % len(valid_coords)]
        ori_frame = copy.deepcopy(valid_frames[i % len(valid_frames)])

        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
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
    original = valid_frames[0]
    generated_path = list(output_frames_dir.glob("*.png"))[0]
    generated = cv2.imread(str(generated_path))

    h = min(original.shape[0], generated.shape[0])
    w = min(original.shape[1], generated.shape[1])
    original_resized = cv2.resize(original, (w, h))
    generated_resized = cv2.resize(generated, (w, h))

    comparison = np.hstack([original_resized, generated_resized])
    cv2.imwrite(str(output_dir / "comparison.jpg"), comparison)

    # 크롭 비교
    x1, y1, x2, y2 = valid_coords[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
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
    print(f"  - bbox+랜드마크 디버그: {output_dir / 'debug_bbox_landmarks.jpg'}")
    print(f"  - bbox 좌표: {coord_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_official_inference()
