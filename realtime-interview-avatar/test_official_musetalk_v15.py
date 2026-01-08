"""
공식 MuseTalk V1.5 GitHub 코드를 그대로 사용한 테스트

GitHub: https://github.com/TMElyralab/MuseTalk
공식 코드의 모듈을 직접 import하여 사용합니다.
"""

import os
import sys
import cv2
import copy
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# MuseTalk 경로 추가 (공식 GitHub 클론)
MUSETALK_PATH = Path("c:/NewAvata/NewAvata/MuseTalk")
sys.path.insert(0, str(MUSETALK_PATH))

# 작업 디렉토리를 realtime-interview-avatar로 설정 (모델 경로)
os.chdir("c:/NewAvata/NewAvata/realtime-interview-avatar")

# 공식 MuseTalk 모듈 import
from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel


def get_face_bbox_simple(image):
    """
    간단한 얼굴 bbox 추출 (OpenCV Haar Cascade)

    공식 MuseTalk은 mmpose + face_detection을 사용하지만,
    설치가 복잡하므로 OpenCV로 대체합니다.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        # 얼굴 감지 실패 시 이미지 중앙 80%를 얼굴로 가정
        h, w = image.shape[:2]
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        return [margin_x, margin_y, w - margin_x, h - margin_y]

    # 가장 큰 얼굴 선택
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    h, w = image.shape[:2]

    # 공식 MuseTalk 스타일로 bbox 확장
    # (랜드마크 기반 half_face_coord 계산을 시뮬레이션)
    face_center_y = y + fh // 2
    face_bottom = y + fh

    # upper_bond 계산 (얼굴 상단)
    half_face_dist = face_bottom - face_center_y
    upper_bond = max(0, face_center_y - half_face_dist)

    # bbox 계산
    x1 = max(0, x - int(fw * 0.1))
    y1 = int(upper_bond)
    x2 = min(w, x + fw + int(fw * 0.1))
    y2 = min(h, face_bottom + int(fh * 0.1))

    return [x1, y1, x2, y2]


def test_official_musetalk_v15():
    """
    공식 MuseTalk V1.5 테스트

    공식 inference.py의 로직을 그대로 따릅니다.
    """
    print("=" * 70)
    print("공식 MuseTalk V1.5 테스트 (GitHub 코드 직접 사용)")
    print("=" * 70)

    # ===== 설정 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # V1.5 설정 (공식 inference.py 기본값)
    version = "v15"
    use_float16 = False  # 더 정확한 결과를 위해 float32 사용
    batch_size = 8
    fps = 25
    extra_margin = 10  # V1.5 전용
    parsing_mode = "jaw"  # V1.5 권장
    left_cheek_width = 90
    right_cheek_width = 90
    audio_padding_length_left = 2
    audio_padding_length_right = 2

    # 모델 경로 (공식 MuseTalk 기본 경로 구조와 동일하게)
    # ./models/musetalkV15/unet.pth
    # ./models/sd-vae/
    # ./models/face-parse-bisent/
    unet_model_path = "./models/musetalkV15/unet.pth"
    unet_config = "./models/musetalkV15/musetalk.json"
    vae_path = "./models/sd-vae"
    whisper_dir = "openai/whisper-tiny"  # HuggingFace에서 로드

    print(f"\n모델 경로:")
    print(f"  UNet: {unet_model_path}")
    print(f"  UNet Config: {unet_config}")
    print(f"  VAE: {vae_path}")

    # ===== 1. 모델 로드 (공식 load_all_model 사용) =====
    print("\n1. 모델 로드 (공식 load_all_model)...")

    # VAE 경로 수정을 위해 직접 로드
    from musetalk.models.vae import VAE
    from musetalk.models.unet import UNet, PositionalEncoding

    vae = VAE(model_path=vae_path, use_float16=use_float16)
    unet = UNet(
        unet_config=unet_config,
        model_path=unet_model_path,
        device=device
    )
    pe = PositionalEncoding(d_model=384)

    timesteps = torch.tensor([0], device=device)

    # float16 변환 (공식 코드)
    weight_dtype = torch.float16 if use_float16 else torch.float32
    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    # 디바이스 이동 (공식 코드)
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    print("   VAE, UNet, PE 로드 완료")

    # ===== 2. Whisper 로드 (공식 코드) =====
    print("\n2. Whisper 및 AudioProcessor 로드...")
    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    print("   Whisper 로드 완료")

    # ===== 3. FaceParsing 로드 (V1.5) =====
    print("\n3. FaceParsing 로드 (V1.5)...")
    fp = FaceParsing(
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width
    )
    print("   FaceParsing 로드 완료")

    # ===== 4. 테스트 데이터 =====
    print("\n4. 테스트 데이터 로드...")
    video_path = str(MUSETALK_PATH / "data" / "video" / "sun.mp4")
    audio_path = str(MUSETALK_PATH / "data" / "audio" / "sun.wav")

    print(f"   Video: {video_path}")
    print(f"   Audio: {audio_path}")

    # 비디오에서 프레임 추출
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"   프레임 수: {len(frames)}")

    # ===== 5. 얼굴 bbox 추출 =====
    print("\n5. 얼굴 bbox 추출...")
    coord_list = []
    for frame in tqdm(frames[:10], desc="   bbox 추출"):  # 처음 10프레임만
        bbox = get_face_bbox_simple(frame)
        coord_list.append(bbox)
    print(f"   첫 번째 bbox: {coord_list[0]}")

    # ===== 6. 오디오 특징 추출 (공식 AudioProcessor) =====
    print("\n6. 오디오 특징 추출...")
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
    print(f"   whisper_chunks shape: {whisper_chunks.shape}")

    # ===== 7. VAE 인코딩 (공식 vae.get_latents_for_unet) =====
    print("\n7. VAE 인코딩...")
    input_latent_list = []
    for i, (bbox, frame) in enumerate(zip(coord_list, frames[:10])):
        x1, y1, x2, y2 = bbox

        # V1.5: extra_margin 추가
        if version == "v15":
            y2 = y2 + extra_margin
            y2 = min(y2, frame.shape[0])

        # 크롭 및 리사이즈
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # 공식 VAE 인코딩
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    print(f"   인코딩된 latent 수: {len(input_latent_list)}")
    print(f"   latent shape: {input_latent_list[0].shape}")

    # ===== 8. UNet 추론 (공식 datagen + 배치 처리) =====
    print("\n8. UNet 추론...")

    # 테스트용으로 처음 10개 프레임만 처리
    test_whisper_chunks = whisper_chunks[:10]

    # 공식 datagen 사용
    gen = datagen(
        whisper_chunks=test_whisper_chunks,
        vae_encode_latents=input_latent_list,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )

    res_frame_list = []
    total = int(np.ceil(float(len(test_whisper_chunks)) / batch_size))

    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total, desc="   추론")):
        # 공식 추론 코드
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

    print(f"   생성된 프레임 수: {len(res_frame_list)}")

    # ===== 9. 블렌딩 (공식 get_image) =====
    print("\n9. 블렌딩 (공식 get_image)...")
    output_frames = []

    for i, res_frame in enumerate(tqdm(res_frame_list, desc="   블렌딩")):
        bbox = coord_list[i % len(coord_list)]
        ori_frame = copy.deepcopy(frames[i % len(frames)])

        x1, y1, x2, y2 = bbox
        if version == "v15":
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

        # 공식 블렌딩 (V1.5)
        if version == "v15":
            combine_frame = get_image(
                ori_frame,
                res_frame,
                [x1, y1, x2, y2],
                mode=parsing_mode,
                fp=fp
            )
        else:
            combine_frame = get_image(
                ori_frame,
                res_frame,
                [x1, y1, x2, y2],
                fp=fp
            )

        output_frames.append(combine_frame)

    print(f"   출력 프레임 수: {len(output_frames)}")

    # ===== 10. 결과 저장 =====
    print("\n10. 결과 저장...")
    output_dir = Path("./results/official_v15_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 개별 프레임 저장
    for i, frame in enumerate(output_frames[:5]):  # 처음 5개만 저장
        cv2.imwrite(str(output_dir / f"frame_{i:04d}.jpg"), frame)

    # 비교 이미지 생성
    if len(output_frames) > 0:
        # 원본 vs 생성
        original = frames[0]
        generated = output_frames[0]

        # 크기 맞추기
        h = min(original.shape[0], generated.shape[0])
        w = min(original.shape[1], generated.shape[1])
        original_resized = cv2.resize(original, (w, h))
        generated_resized = cv2.resize(generated, (w, h))

        comparison = np.hstack([original_resized, generated_resized])
        cv2.imwrite(str(output_dir / "comparison.jpg"), comparison)

        # 크롭 영역 비교
        bbox = coord_list[0]
        x1, y1, x2, y2 = bbox
        y2_ext = min(y2 + extra_margin, original.shape[0])

        crop_original = original[y1:y2_ext, x1:x2]
        crop_original = cv2.resize(crop_original, (256, 256))

        crop_generated = res_frame_list[0] if len(res_frame_list) > 0 else crop_original

        crop_comparison = np.hstack([crop_original, crop_generated])
        cv2.imwrite(str(output_dir / "crop_comparison.jpg"), crop_comparison)

    print(f"\n결과 저장 완료: {output_dir}")
    print("\n" + "=" * 70)
    print("[테스트 완료]")
    print("=" * 70)


if __name__ == "__main__":
    test_official_musetalk_v15()
