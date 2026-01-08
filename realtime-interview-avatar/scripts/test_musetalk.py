#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuseTalk 모델 테스트 스크립트

다운로드한 모델이 올바르게 로드되는지 확인합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np


def print_header(title: str):
    """헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_status(name: str, status: bool, detail: str = ""):
    """상태 출력"""
    icon = "[OK]" if status else "[FAIL]"
    print(f"  {icon} {name}: {detail}")


def test_dependencies():
    """의존성 테스트"""
    print_header("의존성 확인")

    deps = {}

    # torch
    try:
        import torch
        deps["torch"] = (True, torch.__version__)
    except ImportError:
        deps["torch"] = (False, "Not installed")

    # diffusers
    try:
        import diffusers
        deps["diffusers"] = (True, diffusers.__version__)
    except ImportError:
        deps["diffusers"] = (False, "Not installed")

    # whisper
    try:
        import whisper
        deps["openai-whisper"] = (True, "installed")
    except ImportError:
        deps["openai-whisper"] = (False, "Not installed")

    # mediapipe
    try:
        import mediapipe
        deps["mediapipe"] = (True, mediapipe.__version__)
    except ImportError:
        deps["mediapipe"] = (False, "Not installed")

    # opencv
    try:
        import cv2
        deps["opencv-python"] = (True, cv2.__version__)
    except ImportError:
        deps["opencv-python"] = (False, "Not installed")

    # safetensors
    try:
        import safetensors
        deps["safetensors"] = (True, "installed")
    except ImportError:
        deps["safetensors"] = (False, "Not installed")

    for name, (status, version) in deps.items():
        print_status(name, status, version)

    return all(status for status, _ in deps.values())


def test_model_files():
    """모델 파일 존재 확인"""
    print_header("모델 파일 확인")

    model_dir = project_root / "models" / "musetalk"

    files = {
        "MuseTalk UNet config": model_dir / "musetalk" / "musetalk.json",
        "MuseTalk UNet checkpoint": model_dir / "musetalk" / "pytorch_model.bin",
        "VAE config": model_dir / "sd-vae-ft-mse" / "config.json",
        "VAE model": model_dir / "sd-vae-ft-mse" / "diffusion_pytorch_model.safetensors",
        "Whisper tiny": model_dir / "whisper" / "tiny.pt",
        "FaceParse model": model_dir / "face-parse-bisent" / "79999_iter.pth",
    }

    all_exist = True
    for name, path in files.items():
        exists = path.exists()
        size = ""
        if exists:
            size_mb = path.stat().st_size / (1024 * 1024)
            size = f"{size_mb:.1f} MB"
        print_status(name, exists, size if exists else "Not found")
        if not exists:
            all_exist = False

    return all_exist


def test_vae_loading():
    """VAE 로딩 테스트"""
    print_header("VAE 로딩 테스트")

    try:
        from diffusers import AutoencoderKL

        vae_path = project_root / "models" / "musetalk" / "sd-vae-ft-mse"

        print("  Loading VAE...")
        vae = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=torch.float32)

        # CPU에서 테스트
        vae = vae.to("cpu")
        vae.eval()

        # 테스트 입력
        test_input = torch.randn(1, 3, 256, 256)

        print("  Testing encode...")
        with torch.no_grad():
            latent = vae.encode(test_input).latent_dist.sample()
        print_status("VAE encode", True, f"latent shape: {latent.shape}")

        print("  Testing decode...")
        with torch.no_grad():
            decoded = vae.decode(latent).sample
        print_status("VAE decode", True, f"output shape: {decoded.shape}")

        del vae
        return True

    except Exception as e:
        print_status("VAE loading", False, str(e))
        return False


def test_unet_loading():
    """UNet 로딩 테스트"""
    print_header("UNet 로딩 테스트")

    try:
        from diffusers import UNet2DConditionModel
        import json

        config_path = project_root / "models" / "musetalk" / "musetalk" / "musetalk.json"
        checkpoint_path = project_root / "models" / "musetalk" / "musetalk" / "pytorch_model.bin"

        # Config 로드
        print("  Loading config...")
        with open(config_path, 'r') as f:
            unet_config = json.load(f)
        print_status("Config loaded", True, f"keys: {len(unet_config)}")

        # UNet 초기화
        print("  Initializing UNet...")
        unet = UNet2DConditionModel(**unet_config)
        print_status("UNet initialized", True, f"params: {sum(p.numel() for p in unet.parameters()):,}")

        # 체크포인트 로드
        print("  Loading checkpoint...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = unet.load_state_dict(state_dict, strict=False)
        print_status("Checkpoint loaded", True, f"missing: {len(missing)}, unexpected: {len(unexpected)}")

        # 테스트 추론
        print("  Testing forward pass...")
        unet.eval()

        # 입력 준비
        batch_size = 1
        sample = torch.randn(batch_size, 8, 32, 32)  # in_channels=8, sample_size=64 -> 32 for latent
        timestep = torch.tensor([0])
        encoder_hidden_states = torch.randn(batch_size, 1, 384)  # cross_attention_dim=384

        with torch.no_grad():
            output = unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states
            ).sample

        print_status("UNet forward", True, f"output shape: {output.shape}")

        del unet
        return True

    except Exception as e:
        print_status("UNet loading", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_whisper_loading():
    """Whisper 로딩 테스트"""
    print_header("Whisper 로딩 테스트")

    try:
        import whisper

        whisper_path = project_root / "models" / "musetalk" / "whisper" / "tiny.pt"

        print("  Loading Whisper...")
        model = whisper.load_model(str(whisper_path), device="cpu")

        print_status("Whisper loaded", True, "tiny model")

        # 테스트 오디오 - Whisper는 30초 오디오를 기대함
        print("  Testing audio encoding...")
        # 30초 분량의 오디오 (16kHz * 30 = 480000 샘플)
        test_audio = np.random.randn(16000 * 30).astype(np.float32) * 0.1

        # pad_or_trim을 사용하여 정확한 크기로 조정
        audio_padded = whisper.pad_or_trim(test_audio)
        print_status("Audio padded", True, f"samples: {len(audio_padded)}")

        mel = whisper.log_mel_spectrogram(audio_padded)
        print_status("Mel spectrogram", True, f"shape: {mel.shape}")

        # Encoder 테스트
        mel = mel.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model.encoder(mel)
        print_status("Whisper encoder", True, f"output shape: {features.shape}")

        del model
        return True

    except Exception as e:
        print_status("Whisper loading", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_musetalk_wrapper():
    """MuseTalk Wrapper 통합 테스트"""
    print_header("MuseTalk Wrapper 통합 테스트")

    try:
        from src.avatar.musetalk_wrapper import (
            MuseTalkAvatar,
            MuseTalkConfig,
            check_dependencies
        )

        # 의존성 확인
        print("  Checking dependencies...")
        deps = check_dependencies()
        for name, available in deps.items():
            print_status(f"  {name}", available, "")

        # Config 생성
        print("\n  Creating config...")
        config = MuseTalkConfig(
            model_dir=str(project_root / "models" / "musetalk"),
            resolution=256,
            use_fp16=False,  # CPU 테스트용
            device="cpu"
        )
        print_status("Config created", True, "")

        # Avatar 초기화
        print("  Initializing avatar...")
        avatar = MuseTalkAvatar(config)
        print_status("Avatar initialized", True, f"device: {avatar.device}")

        # 모델 로드
        print("  Loading models...")
        avatar.load_models()

        stats = avatar.get_stats()
        print_status("VAE loaded", stats["vae_loaded"], "")
        print_status("UNet loaded", stats["unet_loaded"], "")
        print_status("Audio encoder loaded", stats["audio_encoder_loaded"], "")

        # 테스트 오디오로 프레임 생성 (소스 이미지 없이)
        print("\n  Testing frame generation (without source image)...")
        avatar.source_image = np.zeros((256, 256, 3), dtype=np.uint8)

        test_audio = np.random.randn(640).astype(np.float32) * 0.1  # 40ms at 16kHz
        video_frame = avatar.process_audio_chunk(test_audio)

        print_status("Frame generated", True, f"shape: {video_frame.frame.shape}")
        print_status("State", True, video_frame.state.value)
        print_status("FPS", True, f"{video_frame.metadata['fps']:.1f}")

        # Cleanup
        avatar.cleanup()
        print_status("Cleanup", True, "")

        return True

    except Exception as e:
        print_status("Wrapper test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("  MuseTalk Model Test Script")
    print("=" * 70)

    results = {}

    # 1. 의존성 테스트
    results["dependencies"] = test_dependencies()

    # 2. 모델 파일 테스트
    results["model_files"] = test_model_files()

    # 3. VAE 로딩 테스트
    if results["dependencies"] and results["model_files"]:
        results["vae"] = test_vae_loading()
    else:
        results["vae"] = False
        print_header("VAE 로딩 테스트")
        print("  [SKIP] 의존성 또는 모델 파일 누락")

    # 4. UNet 로딩 테스트
    if results["dependencies"] and results["model_files"]:
        results["unet"] = test_unet_loading()
    else:
        results["unet"] = False
        print_header("UNet 로딩 테스트")
        print("  [SKIP] 의존성 또는 모델 파일 누락")

    # 5. Whisper 로딩 테스트
    if results["dependencies"] and results["model_files"]:
        results["whisper"] = test_whisper_loading()
    else:
        results["whisper"] = False
        print_header("Whisper 로딩 테스트")
        print("  [SKIP] 의존성 또는 모델 파일 누락")

    # 6. Wrapper 통합 테스트
    if all(results.values()):
        results["wrapper"] = test_musetalk_wrapper()
    else:
        results["wrapper"] = False
        print_header("MuseTalk Wrapper 통합 테스트")
        print("  [SKIP] 이전 테스트 실패")

    # 결과 요약
    print_header("테스트 결과 요약")

    all_passed = True
    for name, passed in results.items():
        print_status(name, passed, "PASSED" if passed else "FAILED")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  [SUCCESS] 모든 테스트 통과!")
    else:
        print("  [WARNING] 일부 테스트 실패")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
