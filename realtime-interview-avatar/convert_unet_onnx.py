#!/usr/bin/env python3
"""
UNet ONNX 변환 스크립트
MuseTalk UNet을 ONNX로 변환합니다.
ONNX Runtime TensorRT EP를 통해 TensorRT 가속을 사용합니다.

사용법:
    python convert_unet_onnx.py --batch_size 16

요구사항:
    - onnx
    - onnxruntime-gpu (with TensorRT EP)
"""

import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
import numpy as np

# MuseTalk 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MuseTalk'))

from diffusers import UNet2DConditionModel


class UNetWrapper(nn.Module):
    """ONNX 변환을 위한 UNet 래퍼"""

    def __init__(self, unet_config_path: str, unet_weights_path: str):
        super().__init__()

        # UNet 설정 로드
        with open(unet_config_path, 'r') as f:
            unet_config = json.load(f)

        # UNet 모델 생성
        self.unet = UNet2DConditionModel(**unet_config)

        # 가중치 로드
        weights = torch.load(unet_weights_path, map_location='cpu')
        self.unet.load_state_dict(weights)

        # 추론 모드
        self.unet.eval()

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, 8, H, W] - 입력 잠재 벡터 (동적 크기)
            timestep: [1] - 타임스텝 (항상 0)
            encoder_hidden_states: [B, 50, 384] - 오디오 특성

        Returns:
            [B, 4, H, W] - 예측된 잠재 벡터
        """
        output = self.unet(
            latent,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        )
        return output.sample


def export_to_onnx(model: nn.Module, onnx_path: str, batch_size: int = 16, use_fp16: bool = True, latent_size: int = 32, dynamic_mode: str = 'batch_only'):
    """PyTorch 모델을 ONNX로 변환

    Args:
        dynamic_mode:
            - 'none': 완전 정적 (배치/공간 모두 고정)
            - 'batch_only': 배치만 동적, 공간 고정 (TensorRT 최적화 권장)
            - 'all': 완전 동적 (배치/공간 모두)
    """

    mode_names = {'none': '완전 정적', 'batch_only': '배치 동적', 'all': '완전 동적'}
    print(f"[1/2] ONNX 변환 중 ({mode_names.get(dynamic_mode, dynamic_mode)}, batch_size={batch_size}, latent_size={latent_size}, FP16={use_fp16})...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_fp16:
        model = model.half()
    model = model.to(device)
    model.eval()

    dtype = torch.float16 if use_fp16 else torch.float32

    # 더미 입력 생성
    dummy_latent = torch.randn(batch_size, 8, latent_size, latent_size, device=device, dtype=dtype)
    dummy_timestep = torch.tensor([0], device=device, dtype=torch.long)
    dummy_encoder_hidden_states = torch.randn(batch_size, 50, 384, device=device, dtype=dtype)

    # ONNX 내보내기 설정
    export_kwargs = {
        'export_params': True,
        'opset_version': 17,
        'do_constant_folding': True,
        'input_names': ['latent', 'timestep', 'encoder_hidden_states'],
        'output_names': ['output'],
        'dynamo': False,  # 레거시 exporter 사용 (더 안정적)
    }

    # 동적 축 정의
    if dynamic_mode == 'batch_only':
        # 배치만 동적, 공간 크기는 고정 (TensorRT 최적화에 유리)
        export_kwargs['dynamic_axes'] = {
            'latent': {0: 'batch'},
            'encoder_hidden_states': {0: 'batch'},
            'output': {0: 'batch'}
        }
    elif dynamic_mode == 'all':
        # 완전 동적
        export_kwargs['dynamic_axes'] = {
            'latent': {0: 'batch', 2: 'height', 3: 'width'},
            'encoder_hidden_states': {0: 'batch'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    # dynamic_mode == 'none': dynamic_axes 설정 안함 (완전 정적)

    # ONNX 내보내기
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_latent, dummy_timestep, dummy_encoder_hidden_states),
            onnx_path,
            **export_kwargs
        )

    print(f"  ONNX 저장됨: {onnx_path}")

    # ONNX 검증
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX 검증 완료")

    # 파일 크기
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  파일 크기: {size_mb:.1f} MB")

    return onnx_path


def verify_onnx_tensorrt(onnx_path: str, batch_size: int = 16, use_fp16: bool = True, latent_size: int = 32):
    """ONNX Runtime TensorRT EP 검증"""

    print(f"\n[2/2] ONNX Runtime TensorRT 검증 중...")

    import onnxruntime as ort

    # TensorRT EP 설정
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
            'trt_fp16_enable': use_fp16,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': os.path.dirname(onnx_path),
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
        }),
    ]

    print("  세션 생성 중 (TensorRT 엔진 빌드, 수 분 소요)...")
    start = time.time()

    session = ort.InferenceSession(onnx_path, providers=providers)

    build_time = time.time() - start
    print(f"  세션 생성 완료: {build_time:.1f}초")

    # 실제 사용 중인 provider 확인
    active_provider = session.get_providers()[0]
    print(f"  활성 Provider: {active_provider}")

    # 입력 준비 (latent_size 사용)
    dtype = np.float16 if use_fp16 else np.float32
    latent = np.random.randn(batch_size, 8, latent_size, latent_size).astype(dtype)
    timestep = np.array([0], dtype=np.int64)
    encoder_hidden_states = np.random.randn(batch_size, 50, 384).astype(dtype)

    # 워밍업
    print("  워밍업 중...")
    for _ in range(3):
        _ = session.run(None, {
            'latent': latent,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states
        })

    # 벤치마크
    iterations = 20
    start = time.time()
    for _ in range(iterations):
        output = session.run(None, {
            'latent': latent,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states
        })
    elapsed = time.time() - start

    avg_time = (elapsed / iterations) * 1000
    fps = iterations * batch_size / elapsed

    print(f"\n  === 벤치마크 결과 ===")
    print(f"  평균 추론 시간: {avg_time:.2f}ms (batch_size={batch_size})")
    print(f"  처리량: {fps:.1f} frames/sec")
    print(f"  출력 형태: {output[0].shape}")
    print(f"  출력 범위: [{output[0].min():.4f}, {output[0].max():.4f}]")

    return session


def main():
    parser = argparse.ArgumentParser(description='UNet ONNX 변환')
    parser.add_argument('--unet_config', type=str,
                        default='./models/musetalkV15/musetalk.json',
                        help='UNet 설정 파일 경로')
    parser.add_argument('--unet_weights', type=str,
                        default='./models/musetalkV15/unet.pth',
                        help='UNet 가중치 파일 경로')
    parser.add_argument('--output_dir', type=str,
                        default='./models/musetalkV15',
                        help='출력 디렉토리')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='최적화 배치 크기')
    parser.add_argument('--latent_size', type=int, default=32,
                        help='Latent 크기 (기본: 32, 256px 이미지용)')
    parser.add_argument('--fp32', action='store_true',
                        help='FP32 사용 (기본: FP16)')
    parser.add_argument('--dynamic_mode', type=str, default='batch_only',
                        choices=['none', 'batch_only', 'all'],
                        help='동적 모드: none(완전정적), batch_only(배치만동적, 권장), all(완전동적)')
    parser.add_argument('--skip_verify', action='store_true',
                        help='검증 건너뛰기')

    args = parser.parse_args()
    use_fp16 = not args.fp32

    # 경로 설정
    suffix = 'fp16' if use_fp16 else 'fp32'
    onnx_path = os.path.join(args.output_dir, f'unet_{suffix}.onnx')

    mode_names = {'none': '완전 정적', 'batch_only': '배치만 동적 (권장)', 'all': '완전 동적'}

    print("=" * 60)
    print("UNet ONNX 변환 (ONNX Runtime TensorRT EP)")
    print("=" * 60)
    print(f"설정 파일: {args.unet_config}")
    print(f"가중치 파일: {args.unet_weights}")
    print(f"배치 크기: {args.batch_size}")
    print(f"Latent 크기: {args.latent_size}x{args.latent_size}")
    print(f"정밀도: {'FP16' if use_fp16 else 'FP32'}")
    print(f"동적 모드: {mode_names.get(args.dynamic_mode, args.dynamic_mode)}")
    print(f"출력 경로: {onnx_path}")
    print("=" * 60)

    # 1. ONNX 변환
    model = UNetWrapper(args.unet_config, args.unet_weights)
    export_to_onnx(model, onnx_path, args.batch_size, use_fp16, args.latent_size, args.dynamic_mode)

    # 2. TensorRT 검증
    if not args.skip_verify:
        verify_onnx_tensorrt(onnx_path, args.batch_size, use_fp16, args.latent_size)

    print("\n" + "=" * 60)
    print("변환 완료!")
    print(f"ONNX 파일: {onnx_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
