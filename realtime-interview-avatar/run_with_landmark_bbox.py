"""
MediaPipe FaceMesh를 사용한 정밀한 얼굴 bbox 추출

공식 MuseTalk은 mmpose + face_detection을 사용하지만,
MediaPipe FaceMesh로 유사한 정밀도를 달성할 수 있습니다.

핵심: 랜드마크 기반으로 정확한 얼굴 중심과 bbox를 계산
"""

import os
import sys
import cv2
import copy
import math
import torch
import shutil
import numpy as np
import mediapipe as mp
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


# ========== MediaPipe FaceMesh 기반 정밀 bbox 추출 ==========
class FaceLandmarkDetector:
    """
    MediaPipe FaceMesh를 사용한 정밀 얼굴 랜드마크 감지

    공식 MuseTalk의 mmpose 68점 랜드마크와 유사한 방식으로
    얼굴 bbox를 계산합니다.
    """

    # MediaPipe FaceMesh 랜드마크 인덱스 (468개 중 주요 포인트)
    # 코 끝: 1, 코 다리: 6, 턱 끝: 152
    # 왼쪽 눈: 33, 오른쪽 눈: 263
    # 입 상단: 13, 입 하단: 14
    # 얼굴 윤곽: 10(이마), 152(턱), 234(왼쪽), 454(오른쪽)

    NOSE_TIP = 1          # 코 끝 (공식 코드의 landmark[29]와 유사)
    NOSE_BRIDGE = 6       # 코 다리
    CHIN = 152            # 턱 끝
    FOREHEAD = 10         # 이마
    LEFT_CHEEK = 234      # 왼쪽 볼
    RIGHT_CHEEK = 454     # 오른쪽 볼
    UPPER_LIP = 13        # 윗입술
    LOWER_LIP = 14        # 아랫입술
    LEFT_EYE = 33         # 왼쪽 눈
    RIGHT_EYE = 263       # 오른쪽 눈

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, image):
        """이미지에서 얼굴 랜드마크 추출"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        h, w = image.shape[:2]
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])

        return np.array(landmarks)

    def get_bbox_from_landmarks(self, landmarks, image_shape, upperbondrange=0):
        """
        랜드마크 기반 bbox 계산 (공식 MuseTalk 방식 모방)

        공식 코드 핵심 로직:
        1. half_face_coord = face_land_mark[29] (코 끝)
        2. half_face_dist = max(face_land_mark[:,1]) - half_face_coord[1]
        3. upper_bond = half_face_coord[1] - half_face_dist
        4. bbox = (min_x, upper_bond, max_x, max_y)
        """
        h, w = image_shape[:2]

        # 코 끝을 기준점으로 사용 (공식 코드의 landmark[29])
        nose_tip = landmarks[self.NOSE_TIP]

        # 얼굴 윤곽 랜드마크들
        face_contour_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        face_contour = landmarks[face_contour_indices]

        # half_face_coord (코 끝 기준)
        half_face_coord = nose_tip.copy()
        if upperbondrange != 0:
            half_face_coord[1] += upperbondrange

        # 얼굴 하단 최대 y값
        max_y = np.max(face_contour[:, 1])

        # half_face_dist 계산
        half_face_dist = max_y - half_face_coord[1]

        # upper_bond 계산
        upper_bond = max(0, half_face_coord[1] - half_face_dist)

        # x 범위
        min_x = np.min(face_contour[:, 0])
        max_x = np.max(face_contour[:, 0])

        # bbox 계산
        x1 = max(0, int(min_x))
        y1 = max(0, int(upper_bond))
        x2 = min(w, int(max_x))
        y2 = min(h, int(max_y))

        return [x1, y1, x2, y2]

    def close(self):
        self.face_mesh.close()


def get_landmark_and_bbox_mediapipe(img_list, upperbondrange=0):
    """
    MediaPipe를 사용한 정밀 bbox 추출

    공식 MuseTalk의 get_landmark_and_bbox 함수를 모방
    """
    detector = FaceLandmarkDetector()

    frames = []
    for img_path in img_list:
        frame = cv2.imread(img_path)
        frames.append(frame)

    coords_list = []

    for frame in tqdm(frames, desc="얼굴 랜드마크 추출"):
        landmarks = detector.get_landmarks(frame)

        if landmarks is None:
            # 랜드마크 감지 실패 시 이미지 중앙 사용
            h, w = frame.shape[:2]
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            coords_list.append([margin_x, margin_y, w - margin_x, h - margin_y])
            continue

        bbox = detector.get_bbox_from_landmarks(landmarks, frame.shape, upperbondrange)
        coords_list.append(bbox)

    detector.close()
    return coords_list, frames


def run_inference_with_landmark():
    """MediaPipe 랜드마크 기반 정밀 inference 실행"""
    print("=" * 70)
    print("MediaPipe 랜드마크 기반 정밀 MuseTalk V1.5 Inference")
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
    output_dir = Path("./results/landmark_inference")
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

    # ===== 3. MediaPipe 랜드마크 기반 bbox 추출 =====
    print("\n[3/8] MediaPipe 랜드마크 기반 정밀 bbox 추출...")
    coord_list, frame_list = get_landmark_and_bbox_mediapipe(input_img_list)
    print(f"   bbox 추출 완료: {len(coord_list)} 개")

    # bbox 시각화 저장
    if len(frame_list) > 0:
        debug_frame = frame_list[0].copy()
        x1, y1, x2, y2 = coord_list[0]
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / "debug_bbox.jpg"), debug_frame)
        print(f"   첫 번째 bbox: {coord_list[0]}")

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
    run_inference_with_landmark()
