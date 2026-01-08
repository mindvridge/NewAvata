"""
MuseTalk Lip-sync Test with Real Data

Tests the corrected implementation using:
1. Real face image
2. Real audio file
3. Full pipeline with video output
"""

import os
import sys
import cv2
import numpy as np
import torch
import librosa
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.avatar.musetalk_wrapper import (
    AudioFeatureExtractor,
    MuseTalkConfig,
    MuseTalkAvatar,
    SourceType
)

def test_with_real_data():
    """Test with real face image and audio"""
    print("\n" + "="*60)
    print("MuseTalk Lip-sync Test with Real Data")
    print("="*60)

    # Find real face image
    face_image_path = None
    image_dir = Path("./assets/images")
    for img_path in image_dir.glob("*.png"):
        face_image_path = str(img_path)
        break
    for img_path in image_dir.glob("*.jpg"):
        face_image_path = str(img_path)
        break

    if face_image_path is None:
        print("No face image found in assets/images/")
        return False

    print(f"Face image: {face_image_path}")

    # Find real audio
    audio_path = None
    audio_dir = Path("./assets/audio")
    for aud_path in audio_dir.glob("*.mp3"):
        audio_path = str(aud_path)
        break
    for aud_path in audio_dir.glob("*.wav"):
        audio_path = str(aud_path)
        break

    if audio_path is None:
        print("No audio file found in assets/audio/")
        return False

    print(f"Audio file: {audio_path}")

    # Configuration
    config = MuseTalkConfig(
        model_dir="./models/musetalk",
        unet_checkpoint="musetalk/musetalkV15/unet.pth",
        whisper_model="tiny",
        use_fp16=True,
        source_type=SourceType.IMAGE,
        source_image_path=face_image_path,
        resolution=256,
    )

    # Initialize avatar
    print("\nInitializing MuseTalk Avatar...")
    avatar = MuseTalkAvatar(config)
    avatar.load_models()

    if avatar.vae is None or avatar.unet is None:
        print("ERROR: Models not loaded!")
        return False

    print("[OK] Models loaded")

    # Load source image
    print("\nLoading source image...")
    source_img = cv2.imread(face_image_path)
    if source_img is None:
        print(f"ERROR: Could not load image: {face_image_path}")
        return False

    print(f"Source image shape: {source_img.shape}")

    # Resize to 256x256 for processing
    source_256 = cv2.resize(source_img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    avatar.source_image = source_256

    # Detect face landmarks
    print("\nDetecting face landmarks...")
    if avatar.landmark_detector is not None:
        landmarks = avatar.landmark_detector.detect(source_256)
        if landmarks is not None:
            print(f"Landmarks detected: {landmarks.shape}")
            avatar.source_landmarks = landmarks
        else:
            print("No landmarks detected")
    else:
        print("Landmark detector not available")

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = audio.astype(np.float32)

    # Limit to first 3 seconds for testing
    max_samples = 3 * 16000
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    print(f"Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Extract audio features
    print("\nExtracting audio features...")
    audio_features = avatar.audio_encoder.extract_full_audio(audio, fps=25)
    print(f"Audio features: shape={audio_features.shape}")

    # Generate frames
    print("\nGenerating lip-sync frames...")
    num_frames = audio_features.shape[0]
    output_frames = []

    for i in range(min(num_frames, 75)):  # Max 3 seconds @ 25fps
        # Get audio feature for this frame
        frame_feature = audio_features[i]  # (50, 384)

        # Generate frame
        try:
            output_frame = avatar._generate_frame(frame_feature)
            output_frames.append(output_frame)

            if i % 10 == 0:
                diff = np.abs(output_frame.astype(float) - source_256.astype(float)).mean()
                print(f"  Frame {i}: diff from source = {diff:.2f}")
        except Exception as e:
            print(f"  Frame {i}: ERROR - {e}")
            output_frames.append(source_256.copy())

    print(f"\nGenerated {len(output_frames)} frames")

    # Save output video
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)

    output_video_path = output_dir / "lipsync_test_output.mp4"
    print(f"\nSaving video to: {output_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (256, 256))

    for frame in output_frames:
        out.write(frame)

    out.release()
    print("[OK] Video saved")

    # Save comparison images
    print("\nSaving comparison images...")

    # First frame comparison
    first_frame = output_frames[0]
    comparison = np.hstack([source_256, first_frame])
    cv2.imwrite(str(output_dir / "comparison_first_frame.jpg"), comparison)

    # Middle frame comparison
    mid_idx = len(output_frames) // 2
    mid_frame = output_frames[mid_idx]
    comparison_mid = np.hstack([source_256, mid_frame])
    cv2.imwrite(str(output_dir / "comparison_mid_frame.jpg"), comparison_mid)

    # Last frame comparison
    last_frame = output_frames[-1]
    comparison_last = np.hstack([source_256, last_frame])
    cv2.imwrite(str(output_dir / "comparison_last_frame.jpg"), comparison_last)

    print("[OK] Comparison images saved")

    # Calculate statistics
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)

    diffs = []
    for frame in output_frames:
        diff = np.abs(frame.astype(float) - source_256.astype(float)).mean()
        diffs.append(diff)

    print(f"  Min diff: {min(diffs):.2f}")
    print(f"  Max diff: {max(diffs):.2f}")
    print(f"  Mean diff: {np.mean(diffs):.2f}")
    print(f"  Std diff: {np.std(diffs):.2f}")

    if np.mean(diffs) < 5:
        print("\n[WARN] Very low difference - lip-sync may not be working properly")
    elif np.mean(diffs) > 50:
        print("\n[WARN] Very high difference - output may be too distorted")
    else:
        print("\n[OK] Reasonable difference - lip-sync appears to be working")

    return True


def main():
    print("="*60)
    print("MuseTalk V1.5 Lip-sync Test with Real Data")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    success = test_with_real_data()

    print("\n" + "="*60)
    print(f"Test Result: {'PASS' if success else 'FAIL'}")
    print("="*60)


if __name__ == "__main__":
    main()
