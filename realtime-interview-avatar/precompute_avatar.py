"""
아바타 Precompute 스크립트
비디오에서 프레임을 추출하고 VAE latent를 미리 계산하여 저장
FaceAlignment 68-point landmarks를 사용 (mmpose 대체)
"""

import os
import sys
import cv2
import torch
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# MuseTalk 경로 추가
MUSETALK_PATH = Path("c:/NewAvata/NewAvata/MuseTalk")
sys.path.insert(0, str(MUSETALK_PATH))
os.chdir("c:/NewAvata/NewAvata/realtime-interview-avatar")

from musetalk.models.vae import VAE
from musetalk.utils.face_parsing import FaceParsing

# FaceAlignment (face_detection 라이브러리 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from face_alignment import FaceAlignment, LandmarksType
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=str(device))


def get_landmark_and_bbox_for_frame(frame, bbox_shift=0, select_center=True):
    """
    단일 프레임에서 랜드마크와 bbox 추출 (FaceAlignment 사용)

    FaceAlignment 68-point landmark layout:
    - 0-16: 턱 윤곽선
    - 17-21: 왼쪽 눈썹
    - 22-26: 오른쪽 눈썹
    - 27-30: 코 중앙선
    - 31-35: 코 하단
    - 36-41: 왼쪽 눈
    - 42-47: 오른쪽 눈
    - 48-59: 바깥 입술
    - 60-67: 안쪽 입술

    Args:
        frame: BGR 이미지 (numpy array)
        bbox_shift: bbox 상단 조정값
        select_center: True면 이미지 중앙에 가장 가까운 얼굴 선택

    Returns:
        coord: (x1, y1, x2, y2) 또는 None
    """
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # FaceAlignment로 랜드마크 추출 (68개 포인트)
    try:
        all_landmarks = fa.get_landmarks(frame)
    except:
        return coord_placeholder

    if all_landmarks is None or len(all_landmarks) == 0:
        return coord_placeholder

    # 여러 사람이 감지된 경우 가운데 사람 선택
    if select_center and len(all_landmarks) > 1:
        best_idx = 0
        min_dist = float('inf')

        for i, landmarks in enumerate(all_landmarks):
            # 얼굴 중심점 계산 (모든 랜드마크의 평균)
            face_center = np.mean(landmarks, axis=0)
            dist = np.sqrt((face_center[0] - center_x)**2 + (face_center[1] - center_y)**2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        landmarks = all_landmarks[best_idx]
    else:
        landmarks = all_landmarks[0]

    landmarks = landmarks.astype(np.int32)

    # 원본 MuseTalk 스타일 bbox 계산 (preprocessing.py 그대로)
    # FaceAlignment 68-point와 mmpose 68-point 매핑 (동일한 표준 사용):
    # - face_land_mark[28] = 코 다리 3번째 (nose bridge)
    # - face_land_mark[29] = 코 다리 4번째 (nose bridge bottom) <-- 공식 코드 기준점
    # - face_land_mark[30] = 코 끝 (nose tip)

    # 원본 MuseTalk: half_face_coord = face_land_mark[29]
    # FaceAlignment에서도 동일하게 landmarks[29] 사용 (nose bridge bottom)
    half_face_coord = landmarks[29].copy()

    if bbox_shift != 0:
        half_face_coord[1] = bbox_shift + half_face_coord[1]

    # 원본 MuseTalk: half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
    half_face_dist = np.max(landmarks[:, 1]) - half_face_coord[1]

    # 원본 MuseTalk: upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
    min_upper_bond = 0
    upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

    # 원본 MuseTalk bbox 계산 (패딩/정사각형 조정 없음):
    # f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond),
    #               np.max(face_land_mark[:, 0]), np.max(face_land_mark[:,1]))
    x1 = np.min(landmarks[:, 0])
    y1 = int(upper_bond)
    x2 = np.max(landmarks[:, 0])
    y2 = np.max(landmarks[:, 1])

    if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
        return coord_placeholder

    return (x1, y1, x2, y2)


def precompute_avatar(video_path, output_path=None, use_float16=True, max_frames=250, bbox_shift=0, extra_margin=10):
    """
    아바타 비디오를 precompute하여 pkl로 저장

    Args:
        video_path: 입력 비디오 경로
        output_path: 출력 pkl 경로 (None이면 자동 생성)
        use_float16: FP16 사용 여부
        max_frames: 최대 프레임 수 (None이면 전체)
        bbox_shift: bbox 상단 조정값
        extra_margin: V1.5용 하단 여유 공간
    """
    print("=" * 60)
    print(f"아바타 Precompute 시작 (FaceAlignment 68-point 방식)")
    print(f"입력: {video_path}")
    print(f"bbox_shift: {bbox_shift}, extra_margin: {extra_margin}")
    print("=" * 60)

    weight_dtype = torch.float16 if use_float16 else torch.float32

    # 출력 경로 설정
    video_name = Path(video_path).stem
    if output_path is None:
        output_path = f"precomputed/{video_name}_precomputed.pkl"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # VAE 로드
    print("\n[1/3] VAE 모델 로드 중...")
    vae = VAE(model_path="./models/sd-vae", use_float16=use_float16)
    if use_float16:
        vae.vae = vae.vae.half()
    vae.vae = vae.vae.to(device)

    # 비디오에서 프레임 추출
    print("[2/3] 비디오 프레임 추출 및 랜드마크 분석 중...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"   FPS: {fps}, 총 프레임: {total_frames}")

    frames = []
    coords_list = []

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="   프레임 분석")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        # FaceAlignment로 랜드마크/bbox 추출
        coord = get_landmark_and_bbox_for_frame(frame, bbox_shift=bbox_shift, select_center=True)

        # V1.5: extra_margin 적용
        if coord != (0.0, 0.0, 0.0, 0.0) and extra_margin > 0:
            x1, y1, x2, y2 = coord
            y2 = min(y2 + extra_margin, frame.shape[0])
            coord = (x1, y1, x2, y2)

        frames.append(frame)
        coords_list.append(coord)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"   추출된 프레임: {len(frames)}")

    # 첫 번째 유효 coord 출력
    for i, coord in enumerate(coords_list):
        if coord != (0.0, 0.0, 0.0, 0.0):
            x1, y1, x2, y2 = coord
            print(f"   첫 번째 유효 bbox (프레임 {i}): {coord}, 크기: {x2-x1}x{y2-y1}")
            break

    # VAE latent 계산
    print("[3/3] VAE latent 계산 중...")
    input_latent_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    for i, (frame, coord) in enumerate(tqdm(zip(frames, coords_list),
                                             total=len(frames),
                                             desc="   VAE 인코딩")):
        if coord == coord_placeholder:
            continue

        x1, y1, x2, y2 = [int(c) for c in coord]

        # 얼굴 영역 추출 및 리사이즈
        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            continue

        # 256x256으로 리사이즈 (VAE 입력 크기)
        face_resized = cv2.resize(face_region, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # VAE 인코딩
        with torch.no_grad():
            latent = vae.get_latents_for_unet(face_resized)
            input_latent_list.append(latent.cpu())

    # pkl로 저장
    print(f"\n저장 중: {output_path}")
    precomputed_data = {
        'frames': frames,
        'coords_list': coords_list,
        'input_latent_list': input_latent_list,
        'fps': fps,
        'video_path': str(video_path)
    }

    with open(output_path, 'wb') as f:
        pickle.dump(precomputed_data, f)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"완료! 파일 크기: {file_size:.1f} MB")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="아바타 Precompute (FaceAlignment 방식)")
    parser.add_argument("--video", "-v", type=str,
                        default="assets/images/a24ca21b-0d02-49f0-99a7-d737c5ca6058.mp4",
                        help="입력 비디오 경로")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="출력 pkl 경로")
    parser.add_argument("--max-frames", "-m", type=int, default=250,
                        help="최대 프레임 수")
    parser.add_argument("--fp32", action="store_true",
                        help="FP32 사용 (기본: FP16)")
    parser.add_argument("--bbox-shift", "-b", type=int, default=0,
                        help="bbox 상단 조정값")
    parser.add_argument("--extra-margin", "-e", type=int, default=10,
                        help="V1.5용 하단 여유 공간")

    args = parser.parse_args()

    precompute_avatar(
        video_path=args.video,
        output_path=args.output,
        use_float16=not args.fp32,
        max_frames=args.max_frames,
        bbox_shift=args.bbox_shift,
        extra_margin=args.extra_margin
    )
