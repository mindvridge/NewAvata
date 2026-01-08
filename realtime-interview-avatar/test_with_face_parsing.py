"""
공식 MuseTalk 방식으로 FaceParsing을 사용한 테스트
문제: 현재 결과물이 공식 GitHub 결과물과 다름
원인: FaceParsing 모델을 사용하지 않아 정교한 마스킹이 안됨
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# MuseTalk 경로 추가
musetalk_path = Path("c:/NewAvata/NewAvata/MuseTalk")
sys.path.insert(0, str(musetalk_path))

# 공식 MuseTalk 모듈 import
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending import get_image, get_crop_box
from transformers import WhisperModel


def get_face_bbox_opencv(image):
    """OpenCV DNN으로 얼굴 bbox 추출"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        h, w = image.shape[:2]
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        return (margin_x, margin_y, w - margin_x, h - margin_y)

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    h, w = image.shape[:2]
    x1, y1 = x, y
    x2, y2 = x + fw, y + fh

    # 얼굴 영역 확장 (MuseTalk 스타일)
    face_w = x2 - x1
    face_h = y2 - y1

    expand_w = int(face_w * 0.4)
    x1 = max(0, x1 - expand_w)
    x2 = min(w, x2 + expand_w)

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


def test_with_face_parsing():
    """FaceParsing을 사용한 공식 방식 테스트"""
    print("="*60)
    print("공식 MuseTalk - FaceParsing 사용 테스트")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 모델 경로
    model_dir = Path("./models/musetalk")

    # 1. FaceParsing 모델 로드
    print("\n1. FaceParsing 모델 로드...")
    resnet_path = str(model_dir / "face-parse-bisent" / "resnet18-5c106cde.pth")
    bisenet_path = str(model_dir / "face-parse-bisent" / "79999_iter.pth")

    print(f"   resnet_path: {resnet_path}")
    print(f"   bisenet_path: {bisenet_path}")

    # FaceParsing 초기화 - 경로 수정 필요
    # 공식 코드의 model_init 메서드를 직접 호출
    from musetalk.utils.face_parsing.model import BiSeNet

    fp_net = BiSeNet(resnet_path)
    if torch.cuda.is_available():
        fp_net.cuda()
        fp_net.load_state_dict(torch.load(bisenet_path))
    else:
        fp_net.load_state_dict(torch.load(bisenet_path, map_location=torch.device('cpu')))
    fp_net.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def face_parsing(image, size=(512, 512), mode="raw"):
        """FaceParsing 함수"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        width, height = image.size
        with torch.no_grad():
            image_resized = image.resize(size, Image.BILINEAR)
            img = preprocess(image_resized)
            if torch.cuda.is_available():
                img = torch.unsqueeze(img, 0).cuda()
            else:
                img = torch.unsqueeze(img, 0)
            out = fp_net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # mode에 따른 마스크 생성
            if mode == "raw":
                # 피부(1), 입술(11, 12, 13) 영역만 선택
                parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                parsing[np.where(parsing != 255)] = 0
            elif mode == "jaw":
                # 턱/입 영역만 선택 (더 정교한 마스킹)
                parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                parsing[np.where(parsing != 255)] = 0

        return Image.fromarray(parsing.astype(np.uint8))

    print("   FaceParsing 로드 완료!")

    # 2. MuseTalk 모델 로드
    print("\n2. MuseTalk 모델 로드...")
    use_fp16 = False
    weight_dtype = torch.float16 if use_fp16 else torch.float32

    vae = VAE(model_path=str(model_dir / "sd-vae-ft-mse"), use_float16=use_fp16)

    unet_config = str(model_dir / "musetalk" / "musetalk.json")
    unet_path = str(model_dir / "musetalk" / "pytorch_model.bin")

    unet = UNet(
        unet_config=unet_config,
        model_path=unet_path,
        device=device
    )
    unet.model = unet.model.to(device)

    pe = PositionalEncoding(d_model=384)
    pe = pe.to(device)

    # Whisper
    audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")
    whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    timesteps = torch.tensor([0], device=device)

    print("   모델 로드 완료!")

    # 3. 비디오 프레임 로드
    print("\n3. 비디오 프레임 로드...")
    video_path = str(musetalk_path / "data" / "video" / "sun.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    print(f"   원본 프레임: {frame.shape}")

    # 4. 얼굴 bbox 추출
    print("\n4. 얼굴 bbox 추출...")
    bbox = get_face_bbox_opencv(frame)
    x1, y1, x2, y2 = bbox
    print(f"   얼굴 bbox: ({x1}, {y1}, {x2}, {y2})")

    # 5. 얼굴 크롭
    print("\n5. 얼굴 크롭...")
    crop_frame = frame[y1:y2, x1:x2]
    crop_frame_256 = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    print(f"   크롭된 얼굴: {crop_frame_256.shape}")

    # 6. 오디오 특징 추출
    print("\n6. 오디오 특징 추출...")
    audio_path = str(musetalk_path / "data" / "audio" / "sun.wav")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper, librosa_length,
        fps=25, audio_padding_length_left=2, audio_padding_length_right=2
    )
    print(f"   whisper_chunks: {whisper_chunks.shape}")

    # 7. VAE 인코딩
    print("\n7. VAE 인코딩...")
    input_latent = vae.get_latents_for_unet(crop_frame_256)
    print(f"   input_latent: {input_latent.shape}")

    # 8. UNet 추론
    print("\n8. UNet 추론...")
    with torch.no_grad():
        audio_chunk = whisper_chunks[30:31]
        audio_feature = pe(audio_chunk)
        latent_batch = input_latent.to(dtype=unet.model.dtype)

        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature).sample
        print(f"   pred_latents: {pred_latents.shape}")

    # 9. VAE 디코딩
    print("\n9. VAE 디코딩...")
    recon = vae.decode_latents(pred_latents)
    recon_frame = recon[0]  # (256, 256, 3) BGR
    print(f"   recon_frame: {recon_frame.shape}")

    # 10. FaceParsing으로 마스크 생성 및 블렌딩
    print("\n10. FaceParsing으로 마스크 생성...")

    # 공식 방식: crop_box 확장 영역에서 FaceParsing
    expand = 1.5
    crop_box, s = get_crop_box(bbox, expand)
    x_s, y_s, x_e, y_e = crop_box

    # 확장된 영역 크롭
    h, w = frame.shape[:2]
    x_s = max(0, x_s)
    y_s = max(0, y_s)
    x_e = min(w, x_e)
    y_e = min(h, y_e)

    face_large = frame[y_s:y_e, x_s:x_e]
    face_large_pil = Image.fromarray(cv2.cvtColor(face_large, cv2.COLOR_BGR2RGB))

    # FaceParsing 마스크 생성
    mask_image = face_parsing(face_large_pil, mode="raw")

    # upper_boundary_ratio 적용 (상반부 마스킹)
    upper_boundary_ratio = 0.5
    mask_array = np.array(mask_image)
    height_mask = mask_array.shape[0]
    top_boundary = int(height_mask * upper_boundary_ratio)
    mask_array[:top_boundary, :] = 0  # 상반부 마스크 제거

    # 가우시안 블러
    blur_kernel_size = int(0.05 * mask_array.shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(mask_array, (blur_kernel_size, blur_kernel_size), 0)

    print(f"   마스크 크기: {mask_array.shape}")
    print(f"   마스크 최대값: {mask_array.max()}")

    # 11. 블렌딩
    print("\n11. 블렌딩...")

    # recon_frame을 bbox 크기로 리사이즈
    recon_resized = cv2.resize(recon_frame, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)

    # 결과 프레임 복사
    result = frame.copy()

    # face_large에 recon_frame 삽입
    face_large_copy = face_large.copy()
    rel_x1 = x1 - x_s
    rel_y1 = y1 - y_s
    rel_x2 = rel_x1 + (x2 - x1)
    rel_y2 = rel_y1 + (y2 - y1)

    # 마스크 리사이즈
    mask_resized = cv2.resize(mask_array, (face_large.shape[1], face_large.shape[0]))

    # recon_frame을 face_large 내의 올바른 위치에 삽입
    face_large_copy[rel_y1:rel_y2, rel_x1:rel_x2] = recon_resized

    # 마스크로 블렌딩
    mask_3ch = mask_resized[:, :, np.newaxis] / 255.0
    blended_large = (face_large.astype(np.float32) * (1 - mask_3ch) +
                     face_large_copy.astype(np.float32) * mask_3ch).astype(np.uint8)

    result[y_s:y_e, x_s:x_e] = blended_large

    # 12. 결과 저장
    print("\n12. 결과 저장...")
    output_dir = Path("./results/face_parsing_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 원본
    cv2.imwrite(str(output_dir / "1_original.jpg"), frame)

    # 크롭된 얼굴
    cv2.imwrite(str(output_dir / "2_cropped_face.jpg"), crop_frame_256)

    # UNet 출력
    cv2.imwrite(str(output_dir / "3_unet_output.jpg"), recon_frame)

    # FaceParsing 마스크
    cv2.imwrite(str(output_dir / "4_face_parsing_mask.jpg"), mask_resized)

    # 최종 결과
    cv2.imwrite(str(output_dir / "5_final_result.jpg"), result)

    # 비교 이미지
    frame_resized = cv2.resize(frame, (400, 300))
    result_resized = cv2.resize(result, (400, 300))
    comparison = np.hstack([frame_resized, result_resized])
    cv2.imwrite(str(output_dir / "6_comparison.jpg"), comparison)

    # 크롭 비교
    crop_comparison = np.hstack([crop_frame_256, recon_frame])
    cv2.imwrite(str(output_dir / "7_crop_comparison.jpg"), crop_comparison)

    print(f"\n결과 저장: {output_dir}")
    print("[완료]")


if __name__ == "__main__":
    test_with_face_parsing()
