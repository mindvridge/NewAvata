"""
공식 MuseTalk 파이프라인을 그대로 따라서 테스트
V1.5 모델 + parsing_mode='jaw' + FaceParsing
"""

import os
import sys
import cv2
import copy
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# MuseTalk 경로 추가
musetalk_path = Path("c:/NewAvata/NewAvata/MuseTalk")
sys.path.insert(0, str(musetalk_path))

# 모델 경로를 realtime-interview-avatar로 설정
os.chdir("c:/NewAvata/NewAvata/realtime-interview-avatar")

# 공식 MuseTalk 모듈 import
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image, get_crop_box
from transformers import WhisperModel


class SimpleFaceParsing:
    """FaceParsing 클래스 (BiSeNet 사용)"""
    def __init__(self, resnet_path, model_path, left_cheek_width=90, right_cheek_width=90):
        from musetalk.utils.face_parsing.model import BiSeNet

        self.net = BiSeNet(resnet_path)
        if torch.cuda.is_available():
            self.net.cuda()
            self.net.load_state_dict(torch.load(model_path))
        else:
            self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.net.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # V1.5: jaw 모드용 커널 설정
        cone_height = 21
        tail_height = 12
        total_size = cone_height + tail_height

        kernel = np.zeros((total_size, total_size), dtype=np.uint8)
        center_x = total_size // 2

        for row in range(cone_height):
            if row < cone_height//2:
                continue
            width = int(2 * (row - cone_height//2) + 1)
            start = int(center_x - (width // 2))
            end = int(center_x + (width // 2) + 1)
            kernel[row, start:end] = 1

        if cone_height > 0:
            base_width = int(kernel[cone_height-1].sum())
        else:
            base_width = 1

        for row in range(cone_height, total_size):
            start = max(0, int(center_x - (base_width//2)))
            end = min(total_size, int(center_x + (base_width//2) + 1))
            kernel[row, start:end] = 1

        self.kernel = kernel
        self.cheek_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 3))
        self.cheek_mask = self._create_cheek_mask(left_cheek_width, right_cheek_width)

    def _create_cheek_mask(self, left_cheek_width=80, right_cheek_width=80):
        mask = np.zeros((512, 512), dtype=np.uint8)
        center = 512 // 2
        cv2.rectangle(mask, (0, 0), (center - left_cheek_width, 512), 255, -1)
        cv2.rectangle(mask, (center + right_cheek_width, 0), (512, 512), 255, -1)
        return mask

    def __call__(self, image, size=(512, 512), mode="raw"):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            image = Image.open(image)

        with torch.no_grad():
            image_resized = image.resize(size, Image.BILINEAR)
            img = self.preprocess(image_resized)
            if torch.cuda.is_available():
                img = torch.unsqueeze(img, 0).cuda()
            else:
                img = torch.unsqueeze(img, 0)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            if mode == "neck":
                parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
                parsing[np.where(parsing != 255)] = 0
            elif mode == "jaw":
                # V1.5 특수 마스킹
                face_region = np.isin(parsing, [1]) * 255
                face_region = face_region.astype(np.uint8)
                original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
                eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)
                face_region = cv2.bitwise_and(eroded, self.cheek_mask)
                face_region = cv2.bitwise_or(face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask))
                parsing[(face_region == 255) & (~np.isin(parsing, [10]))] = 255
                parsing[np.isin(parsing, [11, 12, 13])] = 255
                parsing[np.where(parsing != 255)] = 0
            else:
                parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                parsing[np.where(parsing != 255)] = 0

        return Image.fromarray(parsing.astype(np.uint8))


def get_face_bbox_opencv(image):
    """OpenCV로 얼굴 bbox 추출"""
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

    # 얼굴 영역 확장
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


def test_official_pipeline():
    """공식 파이프라인 테스트"""
    print("=" * 60)
    print("공식 MuseTalk V1.5 파이프라인 테스트")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_dir = Path("./models/musetalk")

    # V1.5 설정
    version = "v15"
    use_fp16 = False
    parsing_mode = "jaw"  # V1.5는 jaw 모드 사용!
    extra_margin = 10

    # 1. FaceParsing 로드
    print("\n1. FaceParsing 로드 (V1.5 jaw 모드)...")
    resnet_path = str(model_dir / "face-parse-bisent" / "resnet18-5c106cde.pth")
    bisenet_path = str(model_dir / "face-parse-bisent" / "79999_iter.pth")

    fp = SimpleFaceParsing(
        resnet_path=resnet_path,
        model_path=bisenet_path,
        left_cheek_width=90,
        right_cheek_width=90
    )
    print("   FaceParsing 로드 완료!")

    # 2. MuseTalk 모델 로드
    print("\n2. MuseTalk V1.5 모델 로드...")
    weight_dtype = torch.float16 if use_fp16 else torch.float32

    vae = VAE(model_path=str(model_dir / "sd-vae-ft-mse"), use_float16=use_fp16)

    # V1.5 모델 사용
    unet_config = str(model_dir / "musetalk" / "musetalkV15" / "musetalk.json")
    unet_path = str(model_dir / "musetalk" / "musetalkV15" / "unet.pth")

    print(f"   UNet config: {unet_config}")
    print(f"   UNet path: {unet_path}")

    unet = UNet(
        unet_config=unet_config,
        model_path=unet_path,
        device=device
    )

    pe = PositionalEncoding(d_model=384)

    if use_fp16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    timesteps = torch.tensor([0], device=device)
    print("   모델 로드 완료!")

    # 3. Whisper 로드
    print("\n3. Whisper 로드...")
    audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")
    whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    print("   Whisper 로드 완료!")

    # 4. 비디오 프레임 로드
    print("\n4. 비디오 프레임 로드...")
    video_path = str(musetalk_path / "data" / "video" / "sun.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    print(f"   원본 프레임: {frame.shape}")

    # 5. 얼굴 bbox 추출
    print("\n5. 얼굴 bbox 추출...")
    bbox = get_face_bbox_opencv(frame)
    x1, y1, x2, y2 = bbox
    print(f"   얼굴 bbox: ({x1}, {y1}, {x2}, {y2})")

    # V1.5: extra_margin 추가
    y2_ext = min(y2 + extra_margin, frame.shape[0])
    print(f"   V1.5 확장 bbox: ({x1}, {y1}, {x2}, {y2_ext})")

    # 6. 얼굴 크롭
    print("\n6. 얼굴 크롭...")
    crop_frame = frame[y1:y2_ext, x1:x2]
    crop_frame_256 = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    print(f"   크롭된 얼굴: {crop_frame_256.shape}")

    # 7. 오디오 특징 추출
    print("\n7. 오디오 특징 추출...")
    audio_path = str(musetalk_path / "data" / "audio" / "sun.wav")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper, librosa_length,
        fps=25, audio_padding_length_left=2, audio_padding_length_right=2
    )
    print(f"   whisper_chunks: {whisper_chunks.shape}")

    # 8. VAE 인코딩
    print("\n8. VAE 인코딩...")
    input_latent = vae.get_latents_for_unet(crop_frame_256)
    print(f"   input_latent: {input_latent.shape}")

    # 9. UNet 추론
    print("\n9. UNet 추론...")
    with torch.no_grad():
        audio_chunk = whisper_chunks[30:31]
        audio_feature = pe(audio_chunk)
        latent_batch = input_latent.to(dtype=unet.model.dtype)

        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature).sample
        print(f"   pred_latents: {pred_latents.shape}")

    # 10. VAE 디코딩
    print("\n10. VAE 디코딩...")
    recon = vae.decode_latents(pred_latents)
    res_frame = recon[0]
    print(f"   res_frame: {res_frame.shape}")

    # 11. 공식 블렌딩 (get_image 함수 사용)
    print("\n11. 공식 블렌딩 (V1.5 jaw 모드)...")

    # res_frame을 bbox 크기로 리사이즈
    res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2_ext - y1))

    # 공식 get_image 함수 사용
    ori_frame = copy.deepcopy(frame)
    combined = get_image(
        ori_frame,
        res_frame_resized,
        [x1, y1, x2, y2_ext],
        mode=parsing_mode,  # V1.5는 "jaw" 모드!
        fp=fp
    )
    print(f"   combined: {combined.shape}")

    # 12. 결과 저장
    print("\n12. 결과 저장...")
    output_dir = Path("./results/official_pipeline_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 원본
    cv2.imwrite(str(output_dir / "1_original.jpg"), frame)

    # 크롭된 얼굴
    cv2.imwrite(str(output_dir / "2_cropped_face.jpg"), crop_frame_256)

    # UNet 출력
    cv2.imwrite(str(output_dir / "3_unet_output.jpg"), res_frame)

    # 최종 결과
    cv2.imwrite(str(output_dir / "4_final_result.jpg"), combined)

    # 비교
    frame_resized = cv2.resize(frame, (400, 300))
    combined_resized = cv2.resize(combined, (400, 300))
    comparison = np.hstack([frame_resized, combined_resized])
    cv2.imwrite(str(output_dir / "5_comparison.jpg"), comparison)

    # 크롭 비교
    crop_comparison = np.hstack([crop_frame_256, res_frame])
    cv2.imwrite(str(output_dir / "6_crop_comparison.jpg"), crop_comparison)

    print(f"\n결과 저장: {output_dir}")
    print("[완료]")


if __name__ == "__main__":
    test_official_pipeline()
