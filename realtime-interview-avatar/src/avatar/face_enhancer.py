"""
Face Enhancer Module
실시간 및 후처리 얼굴 품질 향상 모듈
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time

from loguru import logger

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    logger.warning("GFPGAN not available. Install: pip install gfpgan realesrgan")
    GFPGAN_AVAILABLE = False
    GFPGANer = None
    RealESRGANer = None

try:
    from basicsr.utils import img2tensor, tensor2img
    from basicsr.utils.download_util import load_file_from_url
    BASICSR_AVAILABLE = True
except ImportError:
    logger.warning("basicsr not available. Install: pip install basicsr")
    BASICSR_AVAILABLE = False


class EnhancementMode(Enum):
    """향상 모드"""
    DISABLED = "disabled"      # 향상 없음
    REALTIME = "realtime"      # 실시간 (빠름, ~30ms)
    QUALITY = "quality"        # 고품질 (느림, ~200ms)
    BALANCED = "balanced"      # 균형 (~100ms)


class EnhancementModel(Enum):
    """향상 모델"""
    GFPGAN_V1_4 = "gfpgan_v1.4"          # 빠름, 아이덴티티 보존
    GFPGAN_V1_3 = "gfpgan_v1.3"          # 이전 버전
    CODEFORMER = "codeformer"            # 고품질, 느림
    RESTOREFORMER = "restoreformer"      # 균형


@dataclass
class EnhancementConfig:
    """향상 설정"""
    # 모드 및 모델
    mode: EnhancementMode = EnhancementMode.REALTIME
    model: EnhancementModel = EnhancementModel.GFPGAN_V1_4

    # 업스케일
    upscale_factor: int = 1  # 1, 2, 4

    # 블렌딩
    face_weight: float = 0.5  # 0.0 (원본) ~ 1.0 (완전 향상)
    background_enhance: bool = False  # 배경도 향상

    # 모델 경로
    model_path: Optional[str] = None

    # 디바이스
    device: str = "cpu"  # cpu, cuda

    # 타일 설정 (메모리 절약)
    tile_size: int = 0  # 0 = 타일링 없음
    tile_pad: int = 10


@dataclass
class EnhancementResult:
    """향상 결과"""
    enhanced_image: np.ndarray
    original_image: np.ndarray
    processing_time_ms: float
    upscale_factor: int
    model_used: EnhancementModel


class FaceEnhancer:
    """얼굴 품질 향상기"""

    # 모델 URL
    MODEL_URLS = {
        EnhancementModel.GFPGAN_V1_4:
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        EnhancementModel.GFPGAN_V1_3:
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        EnhancementModel.CODEFORMER:
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        EnhancementModel.RESTOREFORMER:
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    }

    def __init__(self, config: Optional[EnhancementConfig] = None):
        """
        Args:
            config: 향상 설정
        """
        self.config = config or EnhancementConfig()

        # 모델
        self.enhancer = None
        self.background_enhancer = None

        # 성능 메트릭
        self.processing_times = []

        # 초기화
        if self.config.mode != EnhancementMode.DISABLED:
            self._initialize_models()

        logger.info(
            f"FaceEnhancer initialized: "
            f"mode={self.config.mode.value}, "
            f"model={self.config.model.value}, "
            f"device={self.config.device}"
        )

    def _initialize_models(self):
        """모델 초기화"""
        if not GFPGAN_AVAILABLE:
            logger.error("GFPGAN not available. Cannot initialize enhancer.")
            return

        try:
            # 모델 경로 결정
            if self.config.model_path:
                model_path = self.config.model_path
            else:
                # 기본 경로
                model_dir = Path("./models/face_enhance")
                model_dir.mkdir(parents=True, exist_ok=True)

                model_name = f"{self.config.model.value}.pth"
                model_path = str(model_dir / model_name)

                # 모델 다운로드 (없으면)
                if not Path(model_path).exists():
                    logger.info(f"Downloading {self.config.model.value}...")
                    model_url = self.MODEL_URLS.get(self.config.model)

                    if model_url and BASICSR_AVAILABLE:
                        from basicsr.utils.download_util import load_file_from_url
                        load_file_from_url(
                            url=model_url,
                            model_dir=str(model_dir),
                            file_name=model_name
                        )
                        logger.info(f"Model downloaded: {model_path}")
                    else:
                        logger.warning(f"Cannot download model. Please manually download from: {model_url}")

            # GFPGAN 모델 로드
            if self.config.model in [EnhancementModel.GFPGAN_V1_4, EnhancementModel.GFPGAN_V1_3]:
                self._load_gfpgan(model_path)

            # CodeFormer 모델 로드
            elif self.config.model == EnhancementModel.CODEFORMER:
                self._load_codeformer(model_path)

            # RestoreFormer 모델 로드
            elif self.config.model == EnhancementModel.RESTOREFORMER:
                self._load_restoreformer(model_path)

            logger.info(f"Face enhancer loaded: {self.config.model.value}")

        except Exception as e:
            logger.error(f"Failed to initialize face enhancer: {e}")
            self.enhancer = None

    def _load_gfpgan(self, model_path: str):
        """GFPGAN 로드"""
        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}")
            return

        # 배경 향상용 Real-ESRGAN
        bg_upsampler = None
        if self.config.background_enhance and self.config.upscale_factor > 1:
            try:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=self.config.upscale_factor
                )

                bg_model_path = f"./models/face_enhance/RealESRGAN_x{self.config.upscale_factor}plus.pth"

                if Path(bg_model_path).exists():
                    bg_upsampler = RealESRGANer(
                        scale=self.config.upscale_factor,
                        model_path=bg_model_path,
                        model=model,
                        tile=self.config.tile_size,
                        tile_pad=self.config.tile_pad,
                        pre_pad=0,
                        half=False,
                        device=self.config.device
                    )
                    logger.info("Background upsampler loaded")
            except Exception as e:
                logger.warning(f"Failed to load background upsampler: {e}")

        # GFPGAN 로드
        self.enhancer = GFPGANer(
            model_path=model_path,
            upscale=self.config.upscale_factor,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
            device=self.config.device
        )

    def _load_codeformer(self, model_path: str):
        """CodeFormer 로드 (실제 구현 필요)"""
        logger.warning("CodeFormer not yet implemented. Using GFPGAN fallback.")
        # CodeFormer는 별도 구현 필요
        # 여기서는 GFPGAN으로 폴백
        gfpgan_path = "./models/face_enhance/gfpgan_v1.4.pth"
        if Path(gfpgan_path).exists():
            self._load_gfpgan(gfpgan_path)

    def _load_restoreformer(self, model_path: str):
        """RestoreFormer 로드 (실제 구현 필요)"""
        logger.warning("RestoreFormer not yet implemented. Using GFPGAN fallback.")
        # RestoreFormer는 별도 구현 필요
        # 여기서는 GFPGAN으로 폴백
        gfpgan_path = "./models/face_enhance/gfpgan_v1.4.pth"
        if Path(gfpgan_path).exists():
            self._load_gfpgan(gfpgan_path)

    def enhance(
        self,
        image: np.ndarray,
        face_weight: Optional[float] = None
    ) -> EnhancementResult:
        """
        얼굴 품질 향상

        Args:
            image: 입력 이미지 (RGB 또는 BGR)
            face_weight: 향상 가중치 (None이면 config 사용)

        Returns:
            EnhancementResult: 향상 결과
        """
        start_time = time.time()

        # 모드 체크
        if self.config.mode == EnhancementMode.DISABLED or self.enhancer is None:
            return EnhancementResult(
                enhanced_image=image.copy(),
                original_image=image.copy(),
                processing_time_ms=0.0,
                upscale_factor=1,
                model_used=self.config.model
            )

        # face_weight 설정
        if face_weight is None:
            face_weight = self.config.face_weight

        try:
            # BGR 확인 (GFPGAN은 BGR 입력)
            # 이미지가 RGB면 BGR로 변환
            if image.shape[2] == 3:
                input_img = image.copy()
            else:
                input_img = image

            # GFPGAN 향상
            cropped_faces, restored_faces, restored_img = self.enhancer.enhance(
                input_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=face_weight
            )

            # 처리 시간
            elapsed = (time.time() - start_time) * 1000
            self.processing_times.append(elapsed)

            # 최근 100개 평균 유지
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]

            result = EnhancementResult(
                enhanced_image=restored_img,
                original_image=input_img,
                processing_time_ms=elapsed,
                upscale_factor=self.config.upscale_factor,
                model_used=self.config.model
            )

            return result

        except Exception as e:
            logger.error(f"Face enhancement failed: {e}")

            # 실패 시 원본 반환
            return EnhancementResult(
                enhanced_image=image.copy(),
                original_image=image.copy(),
                processing_time_ms=0.0,
                upscale_factor=1,
                model_used=self.config.model
            )

    def enhance_batch(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[EnhancementResult]:
        """
        배치 이미지 향상

        Args:
            images: 이미지 리스트
            progress_callback: 진행률 콜백 (current, total)

        Returns:
            List[EnhancementResult]: 향상 결과 리스트
        """
        results = []

        for i, image in enumerate(images):
            result = self.enhance(image)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(images))

        return results

    def enhance_video(
        self,
        video_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        save_comparison: bool = False
    ) -> bool:
        """
        비디오 파일 일괄 처리

        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 비디오 경로
            progress_callback: 진행률 콜백 (current_frame, total_frames)
            save_comparison: 원본/향상 비교 영상 저장

        Returns:
            bool: 성공 여부
        """
        try:
            # 비디오 열기
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return False

            # 비디오 정보
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(
                f"Video info: {width}x{height} @ {fps}fps, "
                f"{total_frames} frames"
            )

            # 출력 크기 (업스케일 고려)
            output_width = width * self.config.upscale_factor
            output_height = height * self.config.upscale_factor

            # 비교 영상이면 가로로 2배
            if save_comparison:
                output_width *= 2

            # VideoWriter 생성
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (output_width, output_height)
            )

            if not out.isOpened():
                logger.error(f"Cannot create video writer: {output_path}")
                cap.release()
                return False

            # 프레임별 처리
            frame_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # 향상
                result = self.enhance(frame)

                # 출력 프레임
                if save_comparison:
                    # 원본 리사이즈
                    original_resized = cv2.resize(
                        result.original_image,
                        (output_width // 2, output_height)
                    )

                    # 향상 리사이즈
                    enhanced_resized = cv2.resize(
                        result.enhanced_image,
                        (output_width // 2, output_height)
                    )

                    # 가로로 결합
                    output_frame = np.hstack([original_resized, enhanced_resized])
                else:
                    output_frame = result.enhanced_image

                # 프레임 쓰기
                out.write(output_frame)

                # 진행률 콜백
                if progress_callback:
                    progress_callback(frame_count, total_frames)

                # 로그 (100 프레임마다)
                if frame_count % 100 == 0:
                    logger.info(
                        f"Processed {frame_count}/{total_frames} frames "
                        f"({frame_count/total_frames*100:.1f}%)"
                    )

            # 정리
            cap.release()
            out.release()

            logger.info(f"Video enhancement completed: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Video enhancement failed: {e}")
            return False

    def get_average_processing_time(self) -> float:
        """평균 처리 시간 반환 (ms)"""
        if not self.processing_times:
            return 0.0

        return sum(self.processing_times) / len(self.processing_times)

    def is_realtime_capable(self, target_fps: int = 25) -> bool:
        """
        실시간 처리 가능 여부 확인

        Args:
            target_fps: 목표 FPS

        Returns:
            bool: 실시간 가능 여부
        """
        avg_time_ms = self.get_average_processing_time()

        if avg_time_ms == 0:
            return False

        # 프레임당 허용 시간 (ms)
        frame_time_ms = 1000.0 / target_fps

        return avg_time_ms < frame_time_ms

    def get_stats(self) -> dict:
        """통계 반환"""
        avg_time = self.get_average_processing_time()

        return {
            "mode": self.config.mode.value,
            "model": self.config.model.value,
            "device": self.config.device,
            "upscale_factor": self.config.upscale_factor,
            "face_weight": self.config.face_weight,
            "average_time_ms": avg_time,
            "realtime_25fps": self.is_realtime_capable(25),
            "realtime_30fps": self.is_realtime_capable(30),
            "total_processed": len(self.processing_times)
        }


# 헬퍼 함수
def create_realtime_enhancer(device: str = "cpu") -> FaceEnhancer:
    """
    실시간 향상기 생성 (빠름)

    Args:
        device: 디바이스

    Returns:
        FaceEnhancer: 실시간 향상기
    """
    config = EnhancementConfig(
        mode=EnhancementMode.REALTIME,
        model=EnhancementModel.GFPGAN_V1_4,
        upscale_factor=1,
        face_weight=0.5,
        background_enhance=False,
        device=device
    )

    return FaceEnhancer(config)


def create_quality_enhancer(
    upscale_factor: int = 2,
    device: str = "cpu"
) -> FaceEnhancer:
    """
    고품질 향상기 생성 (느림)

    Args:
        upscale_factor: 업스케일 배율
        device: 디바이스

    Returns:
        FaceEnhancer: 고품질 향상기
    """
    config = EnhancementConfig(
        mode=EnhancementMode.QUALITY,
        model=EnhancementModel.GFPGAN_V1_4,
        upscale_factor=upscale_factor,
        face_weight=0.8,
        background_enhance=True,
        device=device,
        tile_size=512  # 메모리 절약
    )

    return FaceEnhancer(config)


def create_balanced_enhancer(device: str = "cpu") -> FaceEnhancer:
    """
    균형 향상기 생성 (중간)

    Args:
        device: 디바이스

    Returns:
        FaceEnhancer: 균형 향상기
    """
    config = EnhancementConfig(
        mode=EnhancementMode.BALANCED,
        model=EnhancementModel.GFPGAN_V1_4,
        upscale_factor=1,
        face_weight=0.6,
        background_enhance=False,
        device=device
    )

    return FaceEnhancer(config)


# 사용 예시
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="얼굴 품질 향상 도구")

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='입력 파일 (이미지 또는 비디오)'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='출력 파일'
    )

    parser.add_argument(
        '--mode',
        choices=['realtime', 'quality', 'balanced'],
        default='balanced',
        help='향상 모드 (기본: balanced)'
    )

    parser.add_argument(
        '--upscale',
        type=int,
        choices=[1, 2, 4],
        default=1,
        help='업스케일 배율 (기본: 1)'
    )

    parser.add_argument(
        '--weight',
        type=float,
        default=0.5,
        help='얼굴 향상 가중치 0.0~1.0 (기본: 0.5)'
    )

    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='디바이스 (기본: cpu)'
    )

    parser.add_argument(
        '--comparison',
        action='store_true',
        help='비교 영상 저장 (비디오만)'
    )

    args = parser.parse_args()

    # 설정
    mode_map = {
        'realtime': EnhancementMode.REALTIME,
        'quality': EnhancementMode.QUALITY,
        'balanced': EnhancementMode.BALANCED
    }

    config = EnhancementConfig(
        mode=mode_map[args.mode],
        model=EnhancementModel.GFPGAN_V1_4,
        upscale_factor=args.upscale,
        face_weight=args.weight,
        background_enhance=(args.mode == 'quality'),
        device=args.device
    )

    # 향상기 생성
    enhancer = FaceEnhancer(config)

    # 입력 파일 타입 확인
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"입력 파일이 없습니다: {args.input}")
        exit(1)

    # 이미지 처리
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        print(f"이미지 향상 중: {args.input}")

        # 이미지 로드
        image = cv2.imread(str(input_path))

        if image is None:
            print(f"이미지를 로드할 수 없습니다: {args.input}")
            exit(1)

        # 향상
        result = enhancer.enhance(image)

        # 저장
        cv2.imwrite(args.output, result.enhanced_image)

        print(f"✓ 완료!")
        print(f"  - 처리 시간: {result.processing_time_ms:.1f}ms")
        print(f"  - 업스케일: {result.upscale_factor}x")
        print(f"  - 저장: {args.output}")

    # 비디오 처리
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        print(f"비디오 향상 중: {args.input}")

        def progress_callback(current, total):
            percent = current / total * 100
            print(f"\r진행: {current}/{total} ({percent:.1f}%)", end='', flush=True)

        success = enhancer.enhance_video(
            video_path=str(input_path),
            output_path=args.output,
            progress_callback=progress_callback,
            save_comparison=args.comparison
        )

        print()  # 줄바꿈

        if success:
            print(f"✓ 완료!")
            print(f"  - 저장: {args.output}")

            # 통계
            stats = enhancer.get_stats()
            print(f"  - 평균 처리 시간: {stats['average_time_ms']:.1f}ms/프레임")
            print(f"  - 실시간 가능 (25fps): {stats['realtime_25fps']}")
        else:
            print(f"✗ 비디오 처리 실패")
            exit(1)

    else:
        print(f"지원하지 않는 파일 형식: {input_path.suffix}")
        exit(1)
