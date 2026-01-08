"""
MuseTalk Lip-sync Test Script

Tests the corrected implementation using:
1. transformers WhisperModel with all hidden states (5 layers stacked -> 50 features)
2. MuseTalk V1.5 model
3. Official masking and blending approach
"""

import os
import sys
import cv2
import numpy as np
import torch
import librosa

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.avatar.musetalk_wrapper import (
    AudioFeatureExtractor,
    MuseTalkConfig,
    MuseTalkAvatar
)

def test_audio_feature_extraction():
    """Test the new audio feature extractor with transformers WhisperModel"""
    print("\n" + "="*60)
    print("Testing Audio Feature Extraction (transformers WhisperModel)")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize audio feature extractor
    audio_extractor = AudioFeatureExtractor(
        model_name="tiny",
        device=device,
        use_fp16=True,
        local_model_path="./models/musetalk/whisper"
    )

    if audio_extractor.whisper_model is None:
        print("ERROR: Whisper model not loaded!")
        return False

    # Load test audio
    test_audio_path = "./assets/test_audio.wav"
    if not os.path.exists(test_audio_path):
        # Try to find any audio file
        for ext in ['wav', 'mp3', 'flac']:
            candidates = [f for f in os.listdir('./assets') if f.endswith(f'.{ext}')]
            if candidates:
                test_audio_path = f"./assets/{candidates[0]}"
                break

    if not os.path.exists(test_audio_path):
        print(f"Test audio not found. Creating dummy audio...")
        # Create 2 seconds of dummy audio
        sample_rate = 16000
        duration = 2.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        audio = audio.astype(np.float32)
    else:
        print(f"Loading audio: {test_audio_path}")
        audio, sr = librosa.load(test_audio_path, sr=16000)
        audio = audio.astype(np.float32)

    print(f"Audio length: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Extract features
    print("\nExtracting audio features...")
    features = audio_extractor.extract_full_audio(audio, sample_rate=16000, fps=25)

    print(f"Audio features shape: {features.shape}")
    print(f"Expected shape: (num_frames, 50, 384)")

    # Verify shape
    if features.shape[1] == 50 and features.shape[2] == 384:
        print("[OK] Audio feature shape is CORRECT (50, 384)")
    else:
        print(f"[FAIL] Audio feature shape is WRONG! Expected (T, 50, 384), got {features.shape}")
        return False

    # Test slicing
    print("\nTesting feature slicing...")
    for frame_idx in [0, 10, 25]:
        if frame_idx < features.shape[0]:
            sliced = audio_extractor.get_sliced_feature(features, frame_idx)
            print(f"  Frame {frame_idx}: shape={sliced.shape}, range=[{sliced.min():.3f}, {sliced.max():.3f}]")

    return True


def test_full_pipeline():
    """Test the full MuseTalk pipeline with corrected audio features"""
    print("\n" + "="*60)
    print("Testing Full MuseTalk Pipeline")
    print("="*60)

    # Configuration
    config = MuseTalkConfig(
        model_dir="./models/musetalk",
        unet_checkpoint="musetalk/musetalkV15/unet.pth",
        whisper_model="tiny",
        use_fp16=True,
        source_image_path="./assets/test_image.jpg"
    )

    # Find a test image
    if not os.path.exists(config.source_image_path):
        for ext in ['jpg', 'png', 'jpeg']:
            candidates = [f for f in os.listdir('./assets') if f.lower().endswith(f'.{ext}')]
            if candidates:
                config.source_image_path = f"./assets/{candidates[0]}"
                break

    if not os.path.exists(config.source_image_path):
        print(f"No test image found in ./assets/")
        # Create a dummy image
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(dummy_img, (80, 80), (176, 200), (200, 180, 160), -1)  # Face
        cv2.circle(dummy_img, (110, 120), 10, (50, 50, 50), -1)  # Left eye
        cv2.circle(dummy_img, (146, 120), 10, (50, 50, 50), -1)  # Right eye
        cv2.ellipse(dummy_img, (128, 160), (25, 15), 0, 0, 180, (100, 80, 80), -1)  # Mouth
        config.source_image_path = "./assets/dummy_face.jpg"
        os.makedirs("./assets", exist_ok=True)
        cv2.imwrite(config.source_image_path, dummy_img)
        print(f"Created dummy image: {config.source_image_path}")

    print(f"Source image: {config.source_image_path}")
    print(f"UNet checkpoint: {config.unet_checkpoint}")

    # Initialize avatar
    print("\nInitializing MuseTalk Avatar...")
    avatar = MuseTalkAvatar(config)

    # Load models
    print("Loading models...")
    avatar.load_models()

    # Check model loading
    if avatar.vae is None:
        print("ERROR: VAE not loaded!")
        return False
    if avatar.unet is None:
        print("ERROR: UNet not loaded!")
        return False
    if avatar.audio_encoder is None or avatar.audio_encoder.whisper_model is None:
        print("ERROR: Audio encoder not loaded!")
        return False

    print("[OK] All models loaded successfully")

    # Load test audio
    test_audio_path = None
    for ext in ['wav', 'mp3', 'flac']:
        candidates = [f for f in os.listdir('./assets') if f.endswith(f'.{ext}')]
        if candidates:
            test_audio_path = f"./assets/{candidates[0]}"
            break

    if test_audio_path and os.path.exists(test_audio_path):
        print(f"\nLoading audio: {test_audio_path}")
        audio, sr = librosa.load(test_audio_path, sr=16000)
    else:
        print("\nUsing dummy audio (440Hz sine wave)")
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000)).astype(np.float32)

    # Extract audio features
    print("\nExtracting audio features...")
    audio_features = avatar.audio_encoder.extract_full_audio(audio, fps=25)
    print(f"Audio features: shape={audio_features.shape}")

    # Test frame generation
    print("\nGenerating lip-sync frame...")

    # Get single frame audio feature
    if audio_features.shape[0] > 0:
        single_frame_feature = audio_features[0]  # (50, 384)
        print(f"Single frame feature: shape={single_frame_feature.shape}")

        # Generate frame
        try:
            output_frame = avatar._generate_frame(single_frame_feature)
            print(f"Output frame: shape={output_frame.shape}, dtype={output_frame.dtype}")

            # Save output
            os.makedirs("./results", exist_ok=True)
            cv2.imwrite("./results/lipsync_test_output.jpg", output_frame)
            print("[OK] Output saved to ./results/lipsync_test_output.jpg")

            # Compare with original
            source_img = cv2.imread(config.source_image_path)
            if source_img is not None:
                source_resized = cv2.resize(source_img, (256, 256))
                diff = np.abs(output_frame.astype(float) - source_resized.astype(float)).mean()
                print(f"Difference from source: {diff:.2f}")

                if diff < 5:
                    print("[WARN] Output is nearly identical to source - lip-sync may not be working")
                elif diff > 50:
                    print("[WARN] Large difference - output may be too distorted")
                else:
                    print("[OK] Reasonable difference - lip-sync appears to be working")

            return True

        except Exception as e:
            print(f"ERROR during frame generation: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("ERROR: No audio features extracted!")
        return False


def main():
    print("="*60)
    print("MuseTalk Lip-sync Test")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Test 1: Audio feature extraction
    audio_ok = test_audio_feature_extraction()

    # Test 2: Full pipeline
    if audio_ok:
        pipeline_ok = test_full_pipeline()
    else:
        pipeline_ok = False
        print("\nSkipping pipeline test due to audio feature extraction failure")

    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    print(f"  Audio Feature Extraction: {'PASS' if audio_ok else 'FAIL'}")
    print(f"  Full Pipeline: {'PASS' if pipeline_ok else 'FAIL'}")
    print("="*60)


if __name__ == "__main__":
    main()
