"""
공식 MuseTalk 방식으로 얼굴 크롭 후 처리 테스트
MediaPipe로 얼굴 bbox 추출
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

# MuseTalk 경로 추가
musetalk_path = Path("c:/NewAvata/NewAvata/MuseTalk")
sys.path.insert(0, str(musetalk_path))

# 공식 MuseTalk 모듈 import
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel


def simple_blend(ori_frame, res_frame, bbox):
    """간단한 블렌딩 (FaceParsing 없이)"""
    x1, y1, x2, y2 = bbox
    h, w = ori_frame.shape[:2]

    # 결과 프레임 복사
    result = ori_frame.copy()

    # res_frame을 bbox 크기로 리사이즈
    res_resized = cv2.resize(res_frame, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)

    # 블렌딩 마스크 생성 (가장자리 페더링)
    mask_h, mask_w = res_resized.shape[:2]
    mask = np.ones((mask_h, mask_w), dtype=np.float32)

    # 가장자리 페더링 (15픽셀)
    feather = 15
    for i in range(feather):
        alpha = i / feather
        mask[i, :] *= alpha
        mask[mask_h - 1 - i, :] *= alpha
        mask[:, i] *= alpha
        mask[:, mask_w - 1 - i] *= alpha

    # 3채널로 확장
    mask = mask[:, :, np.newaxis]

    # 블렌딩
    blended = (ori_frame[y1:y2, x1:x2].astype(np.float32) * (1 - mask) +
               res_resized.astype(np.float32) * mask).astype(np.uint8)

    result[y1:y2, x1:x2] = blended
    return result


def mouth_only_blend(ori_frame, res_frame, bbox):
    """입 영역만 선택적으로 교체 (GitHub Issue #335 해결책)

    MuseTalk의 알려진 문제: 전체 하반부가 블러 처리됨
    해결책: 입 영역만 교체하고 나머지는 원본 유지
    """
    x1, y1, x2, y2 = bbox
    h, w = ori_frame.shape[:2]

    # 결과 프레임 복사
    result = ori_frame.copy()

    # res_frame을 bbox 크기로 리사이즈
    res_resized = cv2.resize(res_frame, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    mask_h, mask_w = res_resized.shape[:2]

    # 입 영역만 마스킹 (세로 기준 50%~75% 구간, 가로 기준 중앙 60%)
    # MuseTalk은 상반부를 마스킹하므로 하반부(50% 이하)만 생성함
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)

    # 입 영역 정의 (비율 기준)
    mouth_top = int(mask_h * 0.55)      # 입 상단 시작
    mouth_bottom = int(mask_h * 0.85)   # 입 하단 끝
    mouth_left = int(mask_w * 0.25)     # 입 좌측
    mouth_right = int(mask_w * 0.75)    # 입 우측

    # 타원형 마스크 생성 (입 모양에 가깝게)
    center_y = (mouth_top + mouth_bottom) // 2
    center_x = mask_w // 2
    radius_y = (mouth_bottom - mouth_top) // 2
    radius_x = (mouth_right - mouth_left) // 2

    for y in range(mask_h):
        for x in range(mask_w):
            # 타원 방정식: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
            dx = (x - center_x) / max(radius_x, 1)
            dy = (y - center_y) / max(radius_y, 1)
            dist = dx**2 + dy**2

            if dist <= 1.0:
                # 타원 내부: 중심에서 가장자리로 갈수록 페더링
                mask[y, x] = 1.0 - (dist ** 0.5) * 0.3  # 가장자리 30% 페더링
            elif dist <= 1.5:
                # 타원 외부 페더링 영역
                mask[y, x] = max(0, 1.0 - (dist - 1.0) * 2)

    # 3채널로 확장
    mask = mask[:, :, np.newaxis]

    # 블렌딩 (입 영역만)
    blended = (ori_frame[y1:y2, x1:x2].astype(np.float32) * (1 - mask) +
               res_resized.astype(np.float32) * mask).astype(np.uint8)

    result[y1:y2, x1:x2] = blended
    return result, mask[:,:,0]  # 마스크도 반환 (디버깅용)


def get_face_bbox_opencv(image):
    """OpenCV DNN으로 얼굴 bbox 추출"""
    # OpenCV의 기본 Haar Cascade 사용
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        # Haar가 실패하면 수동으로 중앙 영역 사용
        h, w = image.shape[:2]
        # 이미지 중앙의 80%를 얼굴로 가정
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        return (margin_x, margin_y, w - margin_x, h - margin_y)

    # 가장 큰 얼굴 선택
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    h, w = image.shape[:2]
    x1, y1 = x, y
    x2, y2 = x + fw, y + fh

    # 얼굴 영역 확장 (MuseTalk 스타일)
    face_w = x2 - x1
    face_h = y2 - y1

    # 가로 확장
    expand_w = int(face_w * 0.4)
    x1 = max(0, x1 - expand_w)
    x2 = min(w, x2 + expand_w)

    # 세로 확장 (위쪽 더 많이 - 머리카락 포함)
    expand_h_top = int(face_h * 0.6)
    expand_h_bottom = int(face_h * 0.4)
    y1 = max(0, y1 - expand_h_top)
    y2 = min(h, y2 + expand_h_bottom)

    # 정사각형으로 만들기
    new_w = x2 - x1
    new_h = y2 - y1
    if new_w > new_h:
        diff = new_w - new_h
        y1 = max(0, y1 - diff // 2)
        y2 = min(h, y2 + diff // 2)
    else:
        diff = new_h - new_w
        x1 = max(0, x1 - diff // 2)
        x2 = min(w, x2 + diff // 2)

    return (x1, y1, x2, y2)

def test_with_face_crop():
    """얼굴 크롭 후 처리 테스트"""
    print("="*60)
    print("공식 MuseTalk - 얼굴 크롭 방식 테스트")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 모델 경로
    model_dir = Path("./models/musetalk")

    # 1. 모델 로드 (V1 모델로 테스트)
    print("\n1. 모델 로드 (V1 모델)...")
    use_fp16 = False  # 공식 기본값은 float32
    weight_dtype = torch.float16 if use_fp16 else torch.float32
    use_v15 = False  # V1 모델 사용

    vae = VAE(model_path=str(model_dir / "sd-vae-ft-mse"), use_float16=use_fp16)

    if use_v15:
        unet_config = str(model_dir / "musetalk" / "musetalkV15" / "musetalk.json")
        unet_path = str(model_dir / "musetalk" / "musetalkV15" / "unet.pth")
    else:
        unet_config = str(model_dir / "musetalk" / "musetalk.json")
        unet_path = str(model_dir / "musetalk" / "pytorch_model.bin")

    print(f"   UNet config: {unet_config}")
    print(f"   UNet path: {unet_path}")

    unet = UNet(
        unet_config=unet_config,
        model_path=unet_path,
        device=device
    )
    if use_fp16:
        unet.model = unet.model.half()
    unet.model = unet.model.to(device)

    pe = PositionalEncoding(d_model=384)
    if use_fp16:
        pe = pe.half()
    pe = pe.to(device)

    # Whisper
    audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")
    whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    timesteps = torch.tensor([0], device=device)

    # 2. 비디오 프레임 로드
    print("\n2. 비디오 프레임 로드...")
    video_path = str(musetalk_path / "data" / "video" / "sun.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    print(f"   원본 프레임: {frame.shape}")

    # 3. 얼굴 bbox 추출 (OpenCV)
    print("\n3. 얼굴 bbox 추출 (OpenCV)...")
    os.makedirs("./results", exist_ok=True)

    bbox = get_face_bbox_opencv(frame)

    if bbox is None:
        print("얼굴 감지 실패!")
        return

    x1, y1, x2, y2 = bbox
    print(f"   얼굴 bbox: ({x1}, {y1}, {x2}, {y2})")
    print(f"   얼굴 크기: {x2-x1} x {y2-y1}")

    # V1.5: extra_margin 추가
    extra_margin = 10
    y2_extended = min(y2 + extra_margin, frame.shape[0])

    # 4. 얼굴 크롭
    print("\n4. 얼굴 크롭...")
    crop_frame = frame[y1:y2_extended, x1:x2]
    print(f"   크롭된 얼굴: {crop_frame.shape}")

    # 256x256으로 리사이즈
    crop_frame_256 = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    print(f"   리사이즈: {crop_frame_256.shape}")

    # 5. 오디오 특징 추출
    print("\n5. 오디오 특징 추출...")
    audio_path = str(musetalk_path / "data" / "audio" / "sun.wav")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper, librosa_length,
        fps=25, audio_padding_length_left=2, audio_padding_length_right=2
    )
    print(f"   whisper_chunks: {whisper_chunks.shape}, dtype: {whisper_chunks.dtype}")

    # 6. VAE 인코딩 (크롭된 얼굴)
    print("\n6. VAE 인코딩...")
    input_latent = vae.get_latents_for_unet(crop_frame_256)
    print(f"   input_latent: {input_latent.shape}")

    # 7. UNet 추론
    print("\n7. UNet 추론...")
    with torch.no_grad():
        audio_chunk = whisper_chunks[30:31]
        audio_feature = pe(audio_chunk)
        latent_batch = input_latent.to(dtype=unet.model.dtype)

        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature).sample
        print(f"   pred_latents: {pred_latents.shape}")

    # 8. VAE 디코딩
    print("\n8. VAE 디코딩...")
    recon = vae.decode_latents(pred_latents)
    recon_frame = recon[0]  # (256, 256, 3) BGR
    print(f"   recon_frame: {recon_frame.shape}")

    # 9. 원본에 합성 (입 영역만 교체 - GitHub Issue #335 해결책)
    print("\n9. 원본에 합성 (입 영역만 교체)...")
    combined_mouth, mouth_mask = mouth_only_blend(frame, recon_frame, (x1, y1, x2, y2_extended))
    combined_full = simple_blend(frame, recon_frame, (x1, y1, x2, y2_extended))
    print(f"   combined: {combined_mouth.shape}")

    # 10. 결과 저장
    print("\n10. 결과 저장...")
    output_dir = Path("./results/face_crop_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 원본
    cv2.imwrite(str(output_dir / "1_original.jpg"), frame)

    # 크롭된 얼굴
    cv2.imwrite(str(output_dir / "2_cropped_face.jpg"), crop_frame_256)

    # UNet 출력 (크롭)
    cv2.imwrite(str(output_dir / "3_unet_output_crop.jpg"), recon_frame)

    # 입 영역만 교체 결과
    cv2.imwrite(str(output_dir / "4_mouth_only_combined.jpg"), combined_mouth)

    # 전체 교체 결과 (비교용)
    cv2.imwrite(str(output_dir / "4b_full_combined.jpg"), combined_full)

    # 마스크 시각화
    mask_vis = (mouth_mask * 255).astype(np.uint8)
    mask_vis = cv2.resize(mask_vis, (x2 - x1, y2_extended - y1))
    cv2.imwrite(str(output_dir / "4c_mouth_mask.jpg"), mask_vis)

    # 크롭 비교
    crop_comparison = np.hstack([crop_frame_256, recon_frame])
    cv2.imwrite(str(output_dir / "5_crop_comparison.jpg"), crop_comparison)

    # 전체 비교 (원본 vs 입만 교체 vs 전체 교체)
    frame_resized = cv2.resize(frame, (400, 300))
    combined_mouth_resized = cv2.resize(combined_mouth, (400, 300))
    combined_full_resized = cv2.resize(combined_full, (400, 300))
    full_comparison = np.hstack([frame_resized, combined_mouth_resized, combined_full_resized])
    cv2.imwrite(str(output_dir / "6_comparison_3way.jpg"), full_comparison)

    # 통계
    crop_diff = np.abs(crop_frame_256.astype(float) - recon_frame.astype(float)).mean()
    print(f"\n크롭 영역 차이: {crop_diff:.2f}")
    print(f"  상반부: {np.abs(crop_frame_256[:128].astype(float) - recon_frame[:128].astype(float)).mean():.2f}")
    print(f"  하반부: {np.abs(crop_frame_256[128:].astype(float) - recon_frame[128:].astype(float)).mean():.2f}")

    print(f"\n결과 저장: {output_dir}")
    print("[완료]")


if __name__ == "__main__":
    test_with_face_crop()
