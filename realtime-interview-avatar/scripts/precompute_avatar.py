"""
아바타 사전 계산 스크립트

영상의 랜드마크, bbox, VAE latent, 블렌딩 마스크를 미리 계산하여 저장
실시간 추론 시에는 오디오만 받아서 UNet 추론만 수행

사용법:
    python precompute_avatar.py --video_path <영상경로> --face_index center
"""

import os
import sys
import cv2
import torch
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import time

# MuseTalk 경로 자동 탐지 (Linux/Windows 호환)
def find_musetalk_path():
    """MuseTalk 경로를 자동으로 찾습니다."""
    # 환경변수 우선
    if os.environ.get('MUSETALK_PATH'):
        return Path(os.environ['MUSETALK_PATH'])

    # 스크립트 기준 상대 경로들
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # NewAvata/

    possible_paths = [
        project_root / "MuseTalk",                    # NewAvata/MuseTalk
        script_dir.parent.parent.parent / "MuseTalk", # 상위 디렉토리
        Path.home() / "MuseTalk",                     # ~/MuseTalk
        Path("/home/elicer/MuseTalk"),                # 엘리스 클라우드 기본 경로
        Path("c:/NewAvata/NewAvata/MuseTalk"),        # Windows 로컬
    ]

    for path in possible_paths:
        if path.exists() and (path / "musetalk").exists():
            return path

    # 찾지 못한 경우 기본값 (에러 발생 시 명확한 메시지)
    print("[WARNING] MuseTalk 경로를 찾을 수 없습니다. MUSETALK_PATH 환경변수를 설정하세요.")
    return project_root / "MuseTalk"

MUSETALK_PATH = find_musetalk_path()
print(f"[INFO] MuseTalk 경로: {MUSETALK_PATH}")
sys.path.insert(0, str(MUSETALK_PATH))

import face_alignment
from musetalk.utils.blending import get_image_prepare_material
from musetalk.utils.face_parsing import FaceParsing
from musetalk.models.vae import VAE


def precompute_avatar(args):
    """아바타 영상 사전 계산"""
    print("=" * 70)
    print("아바타 사전 계산 시작")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Video: {args.video_path}")
    print(f"Face index: {args.face_index}")

    # ===== 1. face_alignment 초기화 =====
    print("\n[1/5] Face alignment 초기화...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=str(device)
    )

    # ===== 2. 비디오 프레임 추출 =====
    print("\n[2/5] 비디오 프레임 추출...")
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"   FPS: {fps}, 총 프레임: {total_frames}, 크기: {width}x{height}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"   {len(frames)} 프레임 로드 완료")

    # ===== 3. 랜드마크 & bbox 추출 =====
    print("\n[3/5] 랜드마크 & bbox 추출...")
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    coords_list = []

    start_time = time.time()
    for frame in tqdm(frames, desc="   랜드마크 추출"):
        preds = fa.get_landmarks(frame)

        if preds is None or len(preds) == 0:
            coords_list.append(coord_placeholder)
            continue

        # 얼굴 선택
        if args.face_index == 'center' and len(preds) > 1:
            frame_width = frame.shape[1]
            center_x = frame_width / 2
            face_centers = []
            for i, pred in enumerate(preds):
                face_center_x = np.mean(pred[:, 0])
                face_centers.append((i, abs(face_center_x - center_x)))
            face_centers.sort(key=lambda x: x[1])
            selected_idx = face_centers[0][0]
            face_land_mark = preds[selected_idx].astype(np.int32)
        elif isinstance(args.face_index, int) and args.face_index < len(preds):
            face_land_mark = preds[args.face_index].astype(np.int32)
        else:
            face_land_mark = preds[0].astype(np.int32)

        # bbox 계산 (공식 MuseTalk 로직)
        half_face_coord = face_land_mark[29].copy()
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        upper_bond = max(0, half_face_coord[1] - half_face_dist)

        f_landmark = (
            np.min(face_land_mark[:, 0]),
            int(upper_bond),
            np.max(face_land_mark[:, 0]),
            np.max(face_land_mark[:, 1])
        )
        x1, y1, x2, y2 = f_landmark

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            coords_list.append(coord_placeholder)
        else:
            coords_list.append(f_landmark)

    landmark_time = time.time() - start_time
    print(f"   랜드마크 추출 완료: {landmark_time:.1f}초 ({len(frames)/landmark_time:.1f} FPS)")

    # ===== 4. VAE 인코딩 =====
    print("\n[4/5] VAE 인코딩...")
    vae = VAE(model_path="./models/sd-vae", use_float16=args.use_float16)
    if args.use_float16:
        vae.vae = vae.vae.half()
    vae.vae = vae.vae.to(device)

    input_latent_list = []
    extra_margin = 10

    start_time = time.time()
    for bbox, frame in tqdm(zip(coords_list, frames), total=len(frames), desc="   VAE 인코딩"):
        if bbox == coord_placeholder:
            # placeholder인 경우 None 저장 (나중에 처리)
            input_latent_list.append(None)
            continue

        x1, y1, x2, y2 = bbox
        y2 = y2 + extra_margin
        y2 = min(y2, frame.shape[0])

        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    vae_time = time.time() - start_time
    print(f"   VAE 인코딩 완료: {vae_time:.1f}초 ({len(frames)/vae_time:.1f} FPS)")

    # ===== 5. 블렌딩 마스크 사전 계산 =====
    print("\n[5/5] 블렌딩 마스크 사전 계산...")
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

    # 첫 번째 유효한 bbox로 마스크 계산
    cached_mask = None
    cached_crop_box = None

    for bbox, frame in zip(coords_list, frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + extra_margin
        y2 = min(y2, frame.shape[0])

        start_time = time.time()
        cached_mask, cached_crop_box = get_image_prepare_material(
            frame, [x1, y1, x2, y2],
            mode="jaw",
            fp=fp
        )
        mask_time = time.time() - start_time
        print(f"   마스크 계산 완료: {mask_time*1000:.1f}ms")
        print(f"   마스크 shape: {cached_mask.shape}, crop_box: {cached_crop_box}")
        break

    # ===== 결과 저장 =====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(args.video_path).stem
    output_file = output_dir / f"{video_name}_precomputed.pkl"

    precomputed_data = {
        'video_path': args.video_path,
        'face_index': args.face_index,
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': len(frames),
        'coords_list': coords_list,
        'input_latent_list': input_latent_list,
        'cached_mask': cached_mask,
        'cached_crop_box': cached_crop_box,
        'extra_margin': extra_margin,
        'use_float16': args.use_float16,
        # 원본 프레임도 저장 (블렌딩에 필요)
        'frames': frames,
    }

    print(f"\n저장 중: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(precomputed_data, f)

    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"   파일 크기: {file_size:.1f} MB")

    # ===== 요약 =====
    print("\n" + "=" * 70)
    print("사전 계산 완료!")
    print("=" * 70)
    print(f"  - 영상: {args.video_path}")
    print(f"  - 프레임 수: {len(frames)}")
    print(f"  - 얼굴 선택: {args.face_index}")
    print(f"  - FP16: {args.use_float16}")
    print(f"  - 출력: {output_file}")
    print("\n실시간 추론 시 이 파일을 로드하여 사용하세요.")
    print("=" * 70)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="아바타 영상 사전 계산")
    parser.add_argument("--video_path", type=str, required=True, help="입력 영상 경로")
    parser.add_argument("--output_dir", type=str, default="./precomputed", help="출력 디렉토리")
    parser.add_argument("--face_index", type=str, default=None,
                        help="얼굴 선택: None=첫번째, 'center'=가운데, 숫자=인덱스")
    parser.add_argument("--use_float16", action="store_true", help="FP16 사용")

    args = parser.parse_args()

    # face_index 타입 변환
    if args.face_index is not None:
        if args.face_index.isdigit():
            args.face_index = int(args.face_index)

    # 스크립트 위치 기준으로 작업 디렉토리 설정 (Linux/Windows 호환)
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent  # realtime-interview-avatar/
    os.chdir(project_dir)
    print(f"[INFO] 작업 디렉토리: {project_dir}")

    precompute_avatar(args)
