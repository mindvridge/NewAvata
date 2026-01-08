"""
공식 MuseTalk 코드와 비교 테스트
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

def test_official_implementation():
    """공식 MuseTalk 구현으로 테스트"""
    print("="*60)
    print("공식 MuseTalk 구현 테스트")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 모델 경로
    model_dir = Path("./models/musetalk")

    # 1. VAE 로드 (공식 방식)
    print("\n1. VAE 로드...")
    vae = VAE(
        model_path=str(model_dir / "sd-vae-ft-mse"),
        use_float16=True
    )
    print(f"   VAE scaling_factor: {vae.scaling_factor}")

    # 2. UNet 로드 (공식 방식)
    print("\n2. UNet 로드...")
    unet = UNet(
        unet_config=str(model_dir / "musetalk" / "musetalkV15" / "musetalk.json"),
        model_path=str(model_dir / "musetalk" / "musetalkV15" / "unet.pth"),
        device=device
    )
    unet.model = unet.model.half().to(device)

    # 3. PositionalEncoding (공식 방식)
    print("\n3. PositionalEncoding 로드...")
    pe = PositionalEncoding(d_model=384)
    pe = pe.half().to(device)

    # 4. Whisper 로드 (공식 방식 - transformers)
    print("\n4. Whisper 로드...")
    audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")
    whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper = whisper.to(device=device, dtype=torch.float16).eval()
    whisper.requires_grad_(False)

    # 5. 테스트 비디오에서 프레임 추출 (공식 예제 사용)
    print("\n5. 공식 예제 비디오에서 프레임 추출...")
    test_video_path = str(musetalk_path / "data" / "video" / "sun.mp4")

    if not os.path.exists(test_video_path):
        print(f"공식 예제 비디오 없음: {test_video_path}")
        return

    print(f"   비디오: {test_video_path}")

    # 비디오에서 첫 프레임 추출
    cap = cv2.VideoCapture(test_video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("비디오 프레임 읽기 실패!")
        return

    print(f"   원본 프레임 크기: {frame.shape}")

    # 256x256으로 리사이즈
    img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    # 6. 공식 예제 오디오 로드
    print("\n6. 공식 예제 오디오 로드...")
    import librosa
    audio_path = str(musetalk_path / "data" / "audio" / "sun.wav")

    if not os.path.exists(audio_path):
        print(f"공식 예제 오디오 없음: {audio_path}")
        return

    print(f"   오디오: {audio_path}")

    # 7. 공식 방식으로 VAE latent 생성
    print("\n7. 공식 VAE latent 생성...")
    # 공식: vae.get_latents_for_unet(crop_frame)
    input_latent = vae.get_latents_for_unet(img)
    print(f"   input_latent shape: {input_latent.shape}")
    print(f"   input_latent range: [{input_latent.min():.3f}, {input_latent.max():.3f}]")

    # 8. 공식 방식으로 오디오 특징 추출
    print("\n8. 공식 오디오 특징 추출...")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    print(f"   whisper_input_features: {len(whisper_input_features)} segments")

    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        torch.float16,
        whisper,
        librosa_length,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2
    )
    print(f"   whisper_chunks shape: {whisper_chunks.shape}")

    # 9. 단일 프레임 추론 (공식 방식)
    print("\n9. UNet 추론 (공식 방식)...")
    timesteps = torch.tensor([0], device=device)

    with torch.no_grad():
        # 오디오 특징 가져오기 (프레임 30)
        audio_chunk = whisper_chunks[30:31]  # (1, 50, 384)
        print(f"   audio_chunk shape: {audio_chunk.shape}")

        # PositionalEncoding 적용 (공식)
        audio_feature = pe(audio_chunk)
        print(f"   audio_feature (after PE) shape: {audio_feature.shape}")

        # latent를 dtype 맞추기
        latent_batch = input_latent.to(dtype=unet.model.dtype)
        print(f"   latent_batch shape: {latent_batch.shape}")

        # UNet 추론 (공식)
        pred_latents = unet.model(
            latent_batch,
            timesteps,
            encoder_hidden_states=audio_feature
        ).sample
        print(f"   pred_latents shape: {pred_latents.shape}")
        print(f"   pred_latents range: [{pred_latents.min():.3f}, {pred_latents.max():.3f}]")

        # VAE 디코딩 (공식)
        recon = vae.decode_latents(pred_latents)
        print(f"   recon shape: {recon.shape}")
        print(f"   recon dtype: {recon.dtype}")

    # 10. 결과 저장
    print("\n10. 결과 저장...")
    output_dir = Path("./results/official_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 원본
    cv2.imwrite(str(output_dir / "1_original.jpg"), img)

    # 공식 출력
    cv2.imwrite(str(output_dir / "2_official_output.jpg"), recon[0])

    # 비교
    comparison = np.hstack([img, recon[0]])
    cv2.imwrite(str(output_dir / "3_comparison.jpg"), comparison)

    # 차이맵
    diff = cv2.absdiff(img, recon[0])
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "4_diff_amplified.jpg"), diff_amplified)

    # 통계
    print(f"\n차이 통계:")
    print(f"  전체 평균: {diff.mean():.2f}")
    print(f"  상반부 평균: {diff[:128].mean():.2f}")
    print(f"  하반부 평균: {diff[128:].mean():.2f}")

    print(f"\n결과 저장: {output_dir}")
    print("[완료]")


if __name__ == "__main__":
    test_official_implementation()
