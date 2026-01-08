"""
Debug script to visualize raw UNet output before blending
"""

import os
import sys
import cv2
import numpy as np
import torch
import librosa
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.avatar.musetalk_wrapper import MuseTalkConfig, MuseTalkAvatar, SourceType

def test_raw_output():
    """Test to see raw UNet output before blending"""
    print("="*60)
    print("MuseTalk Debug - Raw UNet Output")
    print("="*60)

    # Find face image
    face_image_path = None
    for img_path in Path("./assets/images").glob("*.png"):
        face_image_path = str(img_path)
        break
    for img_path in Path("./assets/images").glob("*.jpg"):
        if "comparison" not in str(img_path):
            face_image_path = str(img_path)
            break

    if face_image_path is None:
        print("No face image found")
        return

    print(f"Face image: {face_image_path}")

    # Find audio
    audio_path = None
    for aud_path in Path("./assets/audio").glob("*.mp3"):
        audio_path = str(aud_path)
        break

    if audio_path is None:
        print("No audio found")
        return

    print(f"Audio: {audio_path}")

    # Initialize avatar
    config = MuseTalkConfig(
        model_dir="./models/musetalk",
        unet_checkpoint="musetalk/musetalkV15/unet.pth",
        whisper_model="tiny",
        use_fp16=True,
        source_type=SourceType.IMAGE,
        source_image_path=face_image_path,
    )

    avatar = MuseTalkAvatar(config)
    avatar.load_models()

    if avatar.vae is None or avatar.unet is None:
        print("ERROR: Models not loaded!")
        return

    # Load source image
    source_img = cv2.imread(face_image_path)
    source_256 = cv2.resize(source_img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    avatar.source_image = source_256

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = audio[:48000].astype(np.float32)  # 3 seconds

    # Extract audio features
    print("\nExtracting audio features...")
    audio_features = avatar.audio_encoder.extract_full_audio(audio, fps=25)
    print(f"Audio features shape: {audio_features.shape}")

    # Generate raw output (without blending)
    print("\nGenerating raw UNet output...")

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=config.use_fp16):
            device = avatar.device
            dtype = avatar.dtype

            # Convert image to tensor
            img_rgb = cv2.cvtColor(source_256, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            img_tensor = img_tensor.unsqueeze(0).to(device, dtype=dtype)

            # Create masked image
            masked_tensor = img_tensor.clone()
            h = masked_tensor.shape[2]
            masked_tensor[:, :, h // 2:, :] = -1.0

            # VAE encode
            masked_latent = avatar.vae.encode(masked_tensor).latent_dist.mode() * 0.18215
            ref_latent = avatar.vae.encode(img_tensor).latent_dist.mode() * 0.18215

            # UNet input
            unet_input = torch.cat([masked_latent, ref_latent], dim=1)

            # Get audio feature for frame 30 (middle of clip)
            audio_feat = audio_features[30].unsqueeze(0)  # (1, 50, 384)

            print(f"UNet input shape: {unet_input.shape}")
            print(f"Audio feature shape: {audio_feat.shape}")

            # UNet inference
            generated_latent = avatar.unet(unet_input, audio_feat)
            print(f"Generated latent shape: {generated_latent.shape}")
            print(f"Generated latent range: [{generated_latent.min():.3f}, {generated_latent.max():.3f}]")

            # VAE decode
            decoded = avatar.vae.decode(generated_latent / 0.18215).sample
            print(f"Decoded shape: {decoded.shape}")
            print(f"Decoded range: [{decoded.min():.3f}, {decoded.max():.3f}]")

            # Convert to image
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded = decoded[0].permute(1, 2, 0).cpu().numpy()
            decoded = (decoded * 255).astype(np.uint8)
            decoded_bgr = cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)

    # Save results
    output_dir = Path("./results/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison
    print("\nSaving debug images...")

    # Original image
    cv2.imwrite(str(output_dir / "1_original.jpg"), source_256)

    # Masked image
    masked_img = source_256.copy()
    masked_img[128:, :] = 0
    cv2.imwrite(str(output_dir / "2_masked.jpg"), masked_img)

    # Raw UNet output
    cv2.imwrite(str(output_dir / "3_raw_unet_output.jpg"), decoded_bgr)

    # Difference map
    diff = cv2.absdiff(source_256, decoded_bgr)
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "4_diff_amplified.jpg"), diff_amplified)

    # Side by side comparison
    comparison = np.hstack([source_256, decoded_bgr, diff_amplified])
    cv2.imwrite(str(output_dir / "5_comparison.jpg"), comparison)

    # Calculate statistics
    diff_flat = diff.astype(float).mean(axis=2)
    print(f"\nDifference statistics:")
    print(f"  Overall mean: {diff.mean():.2f}")
    print(f"  Upper half mean: {diff[:128].mean():.2f}")
    print(f"  Lower half mean: {diff[128:].mean():.2f}")
    print(f"  Max diff location: row={np.unravel_index(diff_flat.argmax(), diff_flat.shape)[0]}")

    # Mouth region analysis (row 128-192)
    mouth_diff = diff[128:192].mean()
    print(f"  Mouth region (128-192) mean: {mouth_diff:.2f}")

    print(f"\nDebug images saved to: {output_dir}")
    print("[OK] Done")


if __name__ == "__main__":
    test_raw_output()
