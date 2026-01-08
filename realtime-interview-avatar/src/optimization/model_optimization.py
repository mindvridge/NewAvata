"""
모델 최적화 모듈

TensorRT, ONNX, 양자화 등을 통한 모델 추론 최적화
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class PrecisionMode(Enum):
    """추론 정밀도 모드"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


@dataclass
class OptimizationConfig:
    """최적화 설정"""
    precision: PrecisionMode = PrecisionMode.FP16
    use_tensorrt: bool = True
    use_onnx: bool = False
    max_batch_size: int = 4
    max_workspace_size: int = 1 << 30  # 1GB
    enable_cudnn_benchmark: bool = True
    cache_dir: Path = Path("models/optimized")


# ============================================================================
# TensorRT Optimizer
# ============================================================================

class TensorRTOptimizer:
    """
    TensorRT 모델 최적화

    PyTorch 모델을 TensorRT 엔진으로 변환하여 추론 속도 향상
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # TensorRT 사용 가능 여부 확인
        self.tensorrt_available = False
        try:
            import tensorrt as trt
            self.trt = trt
            self.tensorrt_available = True
            logger.info("TensorRT 사용 가능")
        except ImportError:
            logger.warning("TensorRT를 사용할 수 없습니다. pip install nvidia-tensorrt")

    def convert_to_tensorrt(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        model_name: str = "model",
    ) -> Optional[Any]:
        """
        PyTorch 모델을 TensorRT 엔진으로 변환

        Args:
            model: PyTorch 모델
            input_shape: 입력 텐서 shape (batch_size, channels, height, width)
            model_name: 모델 이름

        Returns:
            TensorRT 엔진 또는 None (실패 시)
        """
        if not self.tensorrt_available:
            logger.error("TensorRT를 사용할 수 없습니다.")
            return None

        try:
            # ONNX로 먼저 변환
            onnx_path = self.config.cache_dir / f"{model_name}.onnx"
            trt_path = self.config.cache_dir / f"{model_name}.trt"

            # 캐시된 TensorRT 엔진 확인
            if trt_path.exists():
                logger.info(f"캐시된 TensorRT 엔진 로드: {trt_path}")
                return self._load_tensorrt_engine(trt_path)

            # Step 1: PyTorch → ONNX
            logger.info(f"PyTorch 모델을 ONNX로 변환 중... ({onnx_path})")
            self._export_to_onnx(model, input_shape, onnx_path)

            # Step 2: ONNX → TensorRT
            logger.info(f"ONNX를 TensorRT 엔진으로 변환 중... ({trt_path})")
            engine = self._build_tensorrt_engine(onnx_path, trt_path)

            logger.info("TensorRT 변환 완료")
            return engine

        except Exception as e:
            logger.error(f"TensorRT 변환 실패: {e}")
            return None

    def _export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        onnx_path: Path,
    ):
        """PyTorch 모델을 ONNX로 내보내기"""
        model.eval()
        dummy_input = torch.randn(input_shape).cuda()

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        logger.info(f"ONNX 내보내기 완료: {onnx_path}")

    def _build_tensorrt_engine(self, onnx_path: Path, trt_path: Path) -> Any:
        """ONNX를 TensorRT 엔진으로 빌드"""
        TRT_LOGGER = self.trt.Logger(self.trt.Logger.WARNING)

        # Builder 생성
        builder = self.trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = self.trt.OnnxParser(network, TRT_LOGGER)

        # ONNX 파싱
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("ONNX 파싱 실패")

        # Builder 설정
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.max_workspace_size

        # Precision 설정
        if self.config.precision == PrecisionMode.FP16:
            if builder.platform_has_fast_fp16:
                config.set_flag(self.trt.BuilderFlag.FP16)
                logger.info("FP16 모드 활성화")
            else:
                logger.warning("FP16을 지원하지 않습니다. FP32로 대체")

        elif self.config.precision == PrecisionMode.INT8:
            if builder.platform_has_fast_int8:
                config.set_flag(self.trt.BuilderFlag.INT8)
                # INT8 calibration 필요
                logger.warning("INT8 모드는 calibration이 필요합니다.")
            else:
                logger.warning("INT8을 지원하지 않습니다. FP16으로 대체")
                config.set_flag(self.trt.BuilderFlag.FP16)

        # 엔진 빌드
        logger.info("TensorRT 엔진 빌드 중... (수 분 소요될 수 있음)")
        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("TensorRT 엔진 빌드 실패")

        # 엔진 저장
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())

        logger.info(f"TensorRT 엔진 저장: {trt_path}")
        return engine

    def _load_tensorrt_engine(self, trt_path: Path) -> Any:
        """저장된 TensorRT 엔진 로드"""
        TRT_LOGGER = self.trt.Logger(self.trt.Logger.WARNING)
        runtime = self.trt.Runtime(TRT_LOGGER)

        with open(trt_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        return engine


# ============================================================================
# ONNX Optimizer
# ============================================================================

class ONNXOptimizer:
    """
    ONNX 모델 최적화

    ONNX Runtime을 사용한 모델 추론 최적화
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # ONNX Runtime 사용 가능 여부 확인
        self.onnx_available = False
        try:
            import onnxruntime as ort
            self.ort = ort
            self.onnx_available = True
            logger.info(f"ONNX Runtime 사용 가능 (버전: {ort.__version__})")
        except ImportError:
            logger.warning("ONNX Runtime을 사용할 수 없습니다. pip install onnxruntime-gpu")

    def optimize_onnx_model(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        model_name: str = "model",
    ) -> Optional[str]:
        """
        PyTorch 모델을 최적화된 ONNX로 변환

        Returns:
            최적화된 ONNX 모델 경로
        """
        if not self.onnx_available:
            logger.error("ONNX Runtime을 사용할 수 없습니다.")
            return None

        try:
            onnx_path = self.config.cache_dir / f"{model_name}.onnx"
            optimized_path = self.config.cache_dir / f"{model_name}_optimized.onnx"

            # 캐시 확인
            if optimized_path.exists():
                logger.info(f"캐시된 최적화 ONNX 로드: {optimized_path}")
                return str(optimized_path)

            # Step 1: ONNX 내보내기
            logger.info("PyTorch를 ONNX로 변환 중...")
            self._export_to_onnx(model, input_shape, onnx_path)

            # Step 2: ONNX 최적화
            logger.info("ONNX 최적화 중...")
            self._optimize_onnx(onnx_path, optimized_path)

            return str(optimized_path)

        except Exception as e:
            logger.error(f"ONNX 최적화 실패: {e}")
            return None

    def _export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        onnx_path: Path,
    ):
        """PyTorch를 ONNX로 내보내기"""
        model.eval()
        dummy_input = torch.randn(input_shape)

        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            model = model.cuda()

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

    def _optimize_onnx(self, onnx_path: Path, optimized_path: Path):
        """ONNX 모델 최적화"""
        try:
            import onnx
            from onnxruntime.transformers import optimizer

            # ONNX 모델 로드
            model = onnx.load(str(onnx_path))

            # 최적화 수행
            optimized_model = optimizer.optimize_model(
                str(onnx_path),
                model_type='bert',  # 또는 적절한 모델 타입
                num_heads=0,
                hidden_size=0,
            )

            # 저장
            optimized_model.save_model_to_file(str(optimized_path))
            logger.info(f"ONNX 최적화 완료: {optimized_path}")

        except ImportError:
            # onnxruntime.transformers가 없으면 기본 최적화
            logger.warning("onnxruntime.transformers 없음, 기본 복사")
            import shutil
            shutil.copy(onnx_path, optimized_path)

    def create_inference_session(self, onnx_path: str) -> Optional[Any]:
        """ONNX Runtime 추론 세션 생성"""
        if not self.onnx_available:
            return None

        try:
            # 세션 옵션 설정
            sess_options = self.ort.SessionOptions()
            sess_options.graph_optimization_level = (
                self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4

            # Provider 설정 (GPU 우선)
            providers = []
            if self.ort.get_device() == 'GPU':
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')

            # 세션 생성
            session = self.ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=providers,
            )

            logger.info(f"ONNX 추론 세션 생성 완료 (Provider: {session.get_providers()})")
            return session

        except Exception as e:
            logger.error(f"ONNX 세션 생성 실패: {e}")
            return None


# ============================================================================
# Quantization
# ============================================================================

class ModelQuantizer:
    """
    모델 양자화

    FP16 또는 INT8로 양자화하여 메모리 및 추론 속도 개선
    """

    @staticmethod
    def quantize_to_fp16(model: torch.nn.Module) -> torch.nn.Module:
        """
        FP16 양자화

        메모리 사용량 50% 감소, 추론 속도 2배 향상 (GPU 의존)
        """
        logger.info("FP16 양자화 중...")

        if not torch.cuda.is_available():
            logger.warning("CUDA를 사용할 수 없어 FP16 양자화를 건너뜁니다.")
            return model

        model = model.half()  # FP32 → FP16
        model = model.cuda()

        logger.info("FP16 양자화 완료")
        return model

    @staticmethod
    def quantize_to_int8_dynamic(model: torch.nn.Module) -> torch.nn.Module:
        """
        동적 INT8 양자화

        메모리 사용량 75% 감소, CPU 추론에 효과적
        """
        logger.info("동적 INT8 양자화 중...")

        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},  # 양자화할 레이어
                dtype=torch.qint8
            )

            logger.info("동적 INT8 양자화 완료")
            return quantized_model

        except Exception as e:
            logger.error(f"INT8 양자화 실패: {e}")
            return model

    @staticmethod
    def quantize_to_int8_static(
        model: torch.nn.Module,
        calibration_data: torch.utils.data.DataLoader,
    ) -> torch.nn.Module:
        """
        정적 INT8 양자화 (Calibration 필요)

        더 정확한 양자화, 추론 속도 4배 향상 가능
        """
        logger.info("정적 INT8 양자화 중...")

        try:
            # Quantization 설정
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

            # Fuse 모듈
            torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)

            # Prepare
            torch.quantization.prepare(model, inplace=True)

            # Calibration
            model.eval()
            with torch.no_grad():
                for data, _ in calibration_data:
                    model(data)

            # Convert
            torch.quantization.convert(model, inplace=True)

            logger.info("정적 INT8 양자화 완료")
            return model

        except Exception as e:
            logger.error(f"정적 INT8 양자화 실패: {e}")
            return model


# ============================================================================
# Model Optimizer (통합)
# ============================================================================

class ModelOptimizer:
    """
    모델 최적화 통합 인터페이스

    TensorRT, ONNX, 양자화를 통합하여 관리
    """

    def __init__(self, config: OptimizationConfig = None):
        if config is None:
            config = OptimizationConfig()

        self.config = config
        self.tensorrt_optimizer = TensorRTOptimizer(config)
        self.onnx_optimizer = ONNXOptimizer(config)
        self.quantizer = ModelQuantizer()

    def optimize(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        model_name: str = "model",
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        모델 최적화 수행

        Returns:
            (optimized_model, metadata)
        """
        logger.info(f"모델 최적화 시작: {model_name}")
        logger.info(f"  - Precision: {self.config.precision.value}")
        logger.info(f"  - TensorRT: {self.config.use_tensorrt}")
        logger.info(f"  - ONNX: {self.config.use_onnx}")

        metadata = {
            "model_name": model_name,
            "input_shape": input_shape,
            "precision": self.config.precision.value,
            "optimization_applied": [],
        }

        optimized_model = model

        try:
            # 1. Precision 최적화
            if self.config.precision == PrecisionMode.FP16:
                optimized_model = self.quantizer.quantize_to_fp16(optimized_model)
                metadata["optimization_applied"].append("fp16")

            elif self.config.precision == PrecisionMode.INT8:
                optimized_model = self.quantizer.quantize_to_int8_dynamic(optimized_model)
                metadata["optimization_applied"].append("int8_dynamic")

            # 2. TensorRT 최적화
            if self.config.use_tensorrt and self.tensorrt_optimizer.tensorrt_available:
                trt_engine = self.tensorrt_optimizer.convert_to_tensorrt(
                    optimized_model,
                    input_shape,
                    model_name,
                )
                if trt_engine:
                    metadata["optimization_applied"].append("tensorrt")
                    metadata["tensorrt_engine_path"] = str(
                        self.config.cache_dir / f"{model_name}.trt"
                    )
                    return trt_engine, metadata

            # 3. ONNX 최적화
            if self.config.use_onnx and self.onnx_optimizer.onnx_available:
                onnx_path = self.onnx_optimizer.optimize_onnx_model(
                    optimized_model,
                    input_shape,
                    model_name,
                )
                if onnx_path:
                    metadata["optimization_applied"].append("onnx")
                    metadata["onnx_path"] = onnx_path

                    # ONNX 세션 생성
                    session = self.onnx_optimizer.create_inference_session(onnx_path)
                    if session:
                        return session, metadata

            # 4. PyTorch 최적화 (기본)
            # CuDNN Benchmark
            if self.config.enable_cudnn_benchmark and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                metadata["optimization_applied"].append("cudnn_benchmark")

            # JIT 컴파일
            try:
                dummy_input = torch.randn(input_shape)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                    optimized_model = optimized_model.cuda()

                optimized_model = torch.jit.trace(optimized_model, dummy_input)
                metadata["optimization_applied"].append("jit_trace")
            except Exception as e:
                logger.warning(f"JIT 컴파일 실패: {e}")

            logger.info(f"최적화 완료: {metadata['optimization_applied']}")
            return optimized_model, metadata

        except Exception as e:
            logger.error(f"모델 최적화 실패: {e}")
            return model, metadata


# ============================================================================
# Utility Functions
# ============================================================================

def compare_model_performance(
    original_model: torch.nn.Module,
    optimized_model: Any,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
) -> Dict[str, float]:
    """
    원본 모델과 최적화 모델의 성능 비교

    Returns:
        {
            "original_latency_ms": float,
            "optimized_latency_ms": float,
            "speedup": float,
            "original_memory_mb": float,
            "optimized_memory_mb": float,
            "memory_reduction": float,
        }
    """
    import time
    import torch.cuda

    results = {}

    # 원본 모델 성능
    original_model.eval()
    dummy_input = torch.randn(input_shape)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        original_model = original_model.cuda()
        torch.cuda.synchronize()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = original_model(dummy_input)

    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = original_model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    results["original_latency_ms"] = (elapsed / num_iterations) * 1000

    # 메모리
    if torch.cuda.is_available():
        results["original_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()

    # 최적화 모델 성능
    # (TensorRT, ONNX 등에 따라 추론 방법이 다르므로 구현 필요)
    # ... 생략 (실제 구현 시 추가)

    logger.info(f"성능 비교: 원본 {results['original_latency_ms']:.2f}ms")

    return results


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)

    # 더미 모델
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.fc = torch.nn.Linear(128 * 56 * 56, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = DummyModel()
    input_shape = (1, 3, 224, 224)

    # 최적화 수행
    config = OptimizationConfig(
        precision=PrecisionMode.FP16,
        use_tensorrt=False,  # TensorRT 환경이 없을 경우
        use_onnx=True,
    )

    optimizer = ModelOptimizer(config)
    optimized_model, metadata = optimizer.optimize(model, input_shape, "dummy_model")

    print(f"\n최적화 메타데이터: {metadata}")
