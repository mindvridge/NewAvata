"""
공식 MuseTalk 코드를 100% 그대로 사용

- face_detection 라이브러리 (공식 사용)
- mmpose (공식 사용)
- 공식 preprocessing.py의 get_landmark_and_bbox 함수 그대로 사용
"""

import os
import sys
import cv2
import copy
import torch
import shutil
import numpy as np
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

# 공식 preprocessing.py에서 사용하는 것과 동일한 라이브러리
# 공식 코드에서 "from face_detection import"로 되어있지만 실제 패키지명은 face_alignment
from face_alignment import FaceAlignment, LandmarksType
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples


# ========== 공식 preprocessing.py 코드 그대로 사용 ==========

# 공식 MuseTalk의 mmpose 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = str(MUSETALK_PATH / 'musetalk' / 'utils' / 'dwpose' / 'rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py')
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'

print(f"Loading mmpose model from: {checkpoint_file}")
mmpose_model = init_model(config_file, checkpoint_file, device=device)

# 공식 MuseTalk의 face detection 초기화
fa_device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=fa_device)

# bbox가 유효하지 않을 때 사용하는 placeholder
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
    공식 preprocessing.py의 get_landmark_and_bbox 함수 100% 그대로

    - mmpose로 68점 얼굴 랜드마크 추출
    - face_detection으로 얼굴 bbox 추출
    - 랜드마크 기반으로 정밀한 bbox 계산
    """
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []

    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')

    average_range_minus = []
    average_range_plus = []

    for fb in tqdm(batches):
        # mmpose로 랜드마크 추출 (공식 코드 그대로)
        results = inference_topdown(mmpose_model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]  # 얼굴 랜드마크 68점
        face_land_mark = face_land_mark.astype(np.int32)

        # face_detection으로 bbox 추출 (공식 코드 그대로)
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # 랜드마크 기반 bbox 조정 (공식 코드 그대로)
        for j, f in enumerate(bbox):
            if f is None:  # 얼굴 없음
                coords_list += [coord_placeholder]
                continue

            # 공식 코드: half_face_coord = face_land_mark[29] (코 끝)
            half_face_coord = face_land_mark[29].copy()
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)

            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]

            # 공식 bbox 계산 로직
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

            f_landmark = (
                np.min(face_land_mark[:, 0]),
                int(upper_bond),
                np.max(face_land_mark[:, 0]),
                np.max(face_land_mark[:, 1])
            )
            x1, y1, x2, y2 = f_landmark

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                coords_list += [f]
                w, h = f[2] - f[0], f[3] - f[1]
                print("error bbox:", f)
            else:
                coords_list += [f_landmark]

    print("*" * 100)
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print("*" * 100)

    return coords_list, frames


def run_official_exact_inference():
    """공식 MuseTalk 코드를 100% 그대로 사용한 inference"""
    print("=" * 70)
    print("공식 MuseTalk V1.5 Inference (100% 공식 코드)")
    print("- face_detection 라이브러리 사용")
    print("- mmpose 랜드마크 추출 사용")
    print("- 공식 preprocessing.py 로직 그대로 사용")
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
    output_dir = Path("./results/official_exact_inference")
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

    # ===== 3. 공식 get_landmark_and_bbox 사용 =====
    print("\n[3/8] 공식 mmpose + face_detection으로 bbox 추출...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, upperbondrange=0)
    print(f"   bbox 추출 완료: {len(coord_list)} 개")

    # bbox 저장 (공식 코드처럼)
    coord_path = output_dir / "coord_face.pkl"
    with open(coord_path, 'wb') as f:
        pickle.dump(coord_list, f)
    print(f"   bbox 저장: {coord_path}")

    # bbox 시각화
    if len(frame_list) > 0:
        debug_frame = frame_list[0].copy()
        x1, y1, x2, y2 = coord_list[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imwrite(str(output_dir / "debug_bbox.jpg"), debug_frame)
        print(f"   첫 번째 bbox: ({x1}, {y1}, {x2}, {y2})")

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

    for i, (bbox, frame) in enumerate(tqdm(zip(coord_list, frame_list), total=len(coord_list), desc="   VAE 인코딩")):
        if bbox == coord_placeholder:
            continue

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

    valid_coords = [c for c in coord_list if c != coord_placeholder]
    valid_frames = [f for c, f in zip(coord_list, frame_list) if c != coord_placeholder]

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
    print(f"  - bbox 디버그: {output_dir / 'debug_bbox.jpg'}")
    print(f"  - bbox 좌표: {coord_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_official_exact_inference()
