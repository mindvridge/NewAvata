#!/usr/bin/env python3
"""
UNet TensorRT 변환 스크립트
MuseTalk UNet을 TensorRT FP16 엔진으로 변환합니다.

사용법:
    python convert_unet_tensorrt.py --batch_size 16

요구사항:
    - tensorrt
    - torch_tensorrt 또는 torch2trt
    - onnx
    - onnxruntime-gpu
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn

# MuseTalk 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MuseTalk'))

from diffusers import UNet2DConditionModel


class UNetWrapper(nn.Module):
    """TensorRT 변환을 위한 UNet 래퍼"""

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
            latent: [B, 8, 64, 64] - 입력 잠재 벡터
            timestep: [1] - 타임스텝 (항상 0)
            encoder_hidden_states: [B, 50, 384] - 오디오 특성

        Returns:
            [B, 4, 64, 64] - 예측된 잠재 벡터
        """
        output = self.unet(
            latent,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        )
        return output.sample


def export_to_onnx(model: nn.Module, onnx_path: str, batch_size: int = 16):
    """PyTorch 모델을 ONNX로 변환"""

    print(f"[1/3] ONNX 변환 중 (batch_size={batch_size})...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).half()  # FP16

    # 더미 입력 생성
    dummy_latent = torch.randn(batch_size, 8, 64, 64, device=device, dtype=torch.float16)
    dummy_timestep = torch.tensor([0], device=device, dtype=torch.long)
    dummy_encoder_hidden_states = torch.randn(batch_size, 50, 384, device=device, dtype=torch.float16)

    # ONNX 내보내기
    torch.onnx.export(
        model,
        (dummy_latent, dummy_timestep, dummy_encoder_hidden_states),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['latent', 'timestep', 'encoder_hidden_states'],
        output_names=['output'],
        dynamic_axes={
            'latent': {0: 'batch_size'},
            'encoder_hidden_states': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"  ONNX 저장됨: {onnx_path}")

    # ONNX 검증
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX 검증 완료")

    return onnx_path


def convert_onnx_to_tensorrt(onnx_path: str, trt_path: str,
                              batch_size: int = 16, fp16: bool = True):
    """ONNX를 TensorRT 엔진으로 변환"""

    print(f"[2/3] TensorRT 변환 중 (FP16={fp16})...")

    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 빌더 생성
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ONNX 파싱
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"  ONNX 파싱 오류: {parser.get_error(error)}")
            raise RuntimeError("ONNX 파싱 실패")

    print("  ONNX 파싱 완료")

    # 빌더 설정
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)  # 4GB

    # FP16 활성화
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 모드 활성화")

    # 최적화 프로파일 설정 (동적 배치)
    profile = builder.create_optimization_profile()

    # latent: [B, 8, 64, 64]
    profile.set_shape("latent",
                      min=(1, 8, 64, 64),
                      opt=(batch_size, 8, 64, 64),
                      max=(batch_size * 2, 8, 64, 64))

    # encoder_hidden_states: [B, 50, 384]
    profile.set_shape("encoder_hidden_states",
                      min=(1, 50, 384),
                      opt=(batch_size, 50, 384),
                      max=(batch_size * 2, 50, 384))

    config.add_optimization_profile(profile)

    # 엔진 빌드
    print("  TensorRT 엔진 빌드 중... (수 분 소요)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT 엔진 빌드 실패")

    # 엔진 저장
    with open(trt_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"  TensorRT 엔진 저장됨: {trt_path}")

    return trt_path


def verify_tensorrt_engine(trt_path: str, batch_size: int = 16):
    """TensorRT 엔진 검증"""

    print("[3/3] TensorRT 엔진 검증 중...")

    import tensorrt as trt
    import numpy as np

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 엔진 로드
    with open(trt_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # 입력/출력 버퍼 생성
    import pycuda.driver as cuda
    import pycuda.autoinit

    # 입력 준비
    latent = np.random.randn(batch_size, 8, 64, 64).astype(np.float16)
    timestep = np.array([0], dtype=np.int64)
    encoder_hidden_states = np.random.randn(batch_size, 50, 384).astype(np.float16)
    output = np.zeros((batch_size, 4, 64, 64), dtype=np.float16)

    # GPU 메모리 할당
    d_latent = cuda.mem_alloc(latent.nbytes)
    d_timestep = cuda.mem_alloc(timestep.nbytes)
    d_encoder_hidden_states = cuda.mem_alloc(encoder_hidden_states.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    # 데이터 복사
    cuda.memcpy_htod(d_latent, latent)
    cuda.memcpy_htod(d_timestep, timestep)
    cuda.memcpy_htod(d_encoder_hidden_states, encoder_hidden_states)

    # 바인딩 설정
    context.set_input_shape("latent", (batch_size, 8, 64, 64))
    context.set_input_shape("encoder_hidden_states", (batch_size, 50, 384))

    bindings = [int(d_latent), int(d_timestep), int(d_encoder_hidden_states), int(d_output)]

    # 추론 실행
    import time

    # 워밍업
    for _ in range(3):
        context.execute_v2(bindings)

    # 벤치마크
    iterations = 10
    start = time.time()
    for _ in range(iterations):
        context.execute_v2(bindings)
    cuda.Context.synchronize()
    elapsed = time.time() - start

    # 결과 복사
    cuda.memcpy_dtoh(output, d_output)

    avg_time = (elapsed / iterations) * 1000
    print(f"  평균 추론 시간: {avg_time:.2f}ms (batch_size={batch_size})")
    print(f"  출력 형태: {output.shape}")
    print(f"  출력 범위: [{output.min():.4f}, {output.max():.4f}]")

    # 정리
    del d_latent, d_timestep, d_encoder_hidden_states, d_output

    print("  검증 완료!")

    return True


def main():
    parser = argparse.ArgumentParser(description='UNet TensorRT 변환')
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
    parser.add_argument('--skip_onnx', action='store_true',
                        help='ONNX 변환 건너뛰기 (기존 ONNX 사용)')
    parser.add_argument('--verify', action='store_true',
                        help='TensorRT 엔진 검증')

    args = parser.parse_args()

    # 경로 설정
    onnx_path = os.path.join(args.output_dir, 'unet_fp16.onnx')
    trt_path = os.path.join(args.output_dir, f'unet_fp16_b{args.batch_size}.trt')

    print("=" * 60)
    print("UNet TensorRT 변환")
    print("=" * 60)
    print(f"설정 파일: {args.unet_config}")
    print(f"가중치 파일: {args.unet_weights}")
    print(f"배치 크기: {args.batch_size}")
    print(f"출력 경로: {trt_path}")
    print("=" * 60)

    # 1. ONNX 변환
    if not args.skip_onnx:
        model = UNetWrapper(args.unet_config, args.unet_weights)
        export_to_onnx(model, onnx_path, args.batch_size)
    else:
        print("[1/3] ONNX 변환 건너뜀 (기존 파일 사용)")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 파일 없음: {onnx_path}")

    # 2. TensorRT 변환
    convert_onnx_to_tensorrt(onnx_path, trt_path, args.batch_size, fp16=True)

    # 3. 검증
    if args.verify:
        verify_tensorrt_engine(trt_path, args.batch_size)

    print("\n" + "=" * 60)
    print("변환 완료!")
    print(f"TensorRT 엔진: {trt_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
