"""
MuseTalk Avatar Wrapper
실시간 면접 아바타를 위한 MuseTalk 통합 모듈

공식 MuseTalk 구현 기반:
- https://github.com/TMElyralab/MuseTalk
- Latent Space Inpainting 방식 (Non-diffusion, Single-step)
- VAE (ft-mse-vae) + UNet (SD v1.4 기반) + Whisper (audio encoder)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
import time
from collections import deque
from contextlib import contextmanager

from loguru import logger

# MediaPipe for face landmark detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("MediaPipe not installed. Install: pip install mediapipe")
    mp = None
    MEDIAPIPE_AVAILABLE = False

# Whisper for audio encoding
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("Whisper not installed. Install: pip install openai-whisper")
    whisper = None
    WHISPER_AVAILABLE = False

# Diffusers for VAE
try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    logger.warning("Diffusers not installed. Install: pip install diffusers")
    AutoencoderKL = None
    DIFFUSERS_AVAILABLE = False

# Transformers for additional models
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model, WhisperModel, AutoFeatureExtractor
    from einops import rearrange
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not installed. Install: pip install transformers einops")
    WhisperModel = None
    AutoFeatureExtractor = None
    rearrange = None
    TRANSFORMERS_AVAILABLE = False

# Face enhancement (optional)
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    logger.warning("GFPGAN/RealESRGAN not installed. Face enhancement disabled.")
    GFPGANer = None
    GFPGAN_AVAILABLE = False

# safetensors support
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# torchvision transforms
try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    transforms = None

# PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


class AvatarState(Enum):
    """아바타 상태"""
    IDLE = "idle"  # 대기 중 (미세한 움직임)
    SPEAKING = "speaking"  # 말하는 중 (립싱크)
    THINKING = "thinking"  # 생각 중 (고개 끄덕임)
    LISTENING = "listening"  # 듣는 중 (집중)


class DeviceType(Enum):
    """디바이스 타입"""
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    CPU = "cpu"


class SourceType(Enum):
    """소스 타입"""
    IMAGE = "image"  # 단일 이미지
    VIDEO = "video"  # 영상 파일
    IMAGE_DIR = "image_dir"  # 이미지 디렉토리 (시퀀스)


@dataclass
class MuseTalkConfig:
    """MuseTalk 설정"""
    # 모델 경로
    model_dir: str = "./models/musetalk"
    vae_type: str = "sd-vae-ft-mse"  # stabilityai/sd-vae-ft-mse
    unet_checkpoint: str = "musetalk/musetalkV15/unet.pth"  # V1.5 모델 사용
    whisper_model: str = "tiny"  # tiny, base, small, medium

    # 소스 설정 (이미지 또는 영상)
    source_type: SourceType = SourceType.VIDEO  # 기본값을 영상으로 변경
    source_video_path: str = "./assets/avatar_source.mp4"  # 소스 영상 경로
    source_image_path: str = "./assets/avatar_source.jpg"  # 소스 이미지 경로 (fallback)
    source_image_dir: str = ""  # 이미지 시퀀스 디렉토리

    # 영상 설정
    video_fps: int = 25  # 소스 영상 FPS (MuseTalk 권장: 25fps)
    loop_video: bool = True  # 영상 루프 재생
    preload_frames: bool = True  # 프레임 미리 로드 (메모리 사용 증가, 속도 향상)
    max_preload_frames: int = 250  # 최대 미리 로드 프레임 수 (10초 @ 25fps)

    # 해상도 설정
    resolution: int = 256  # 256, 384, 512
    latent_size: int = 32  # resolution / 8 (VAE downscale factor)

    # 품질 설정
    use_face_enhance: bool = False
    smooth_factor: float = 0.3  # 0.0 ~ 1.0 (프레임 스무딩)

    # 성능 설정
    use_fp16: bool = True
    batch_size: int = 1
    buffer_size: int = 5  # 프레임 버퍼 크기

    # 오디오 설정
    audio_sample_rate: int = 16000
    audio_chunk_duration: float = 0.04  # 40ms (25 fps)
    whisper_feature_dim: int = 384  # whisper tiny output dimension

    # 얼굴 영역 설정
    face_expand_ratio: float = 1.2
    mouth_mask_expand: float = 1.5  # 입 영역 마스크 확장 비율

    # bbox_shift: 입 열림 정도 조절 (-7 ~ 7, 양수=더 열림)
    bbox_shift: int = 0

    # 얼굴 기반 크롭 설정 (영상에서 얼굴이 작을 때 사용)
    use_face_crop: bool = True  # 얼굴 영역만 크롭하여 처리
    face_crop_ratio: float = 2.5  # 얼굴 크기 대비 크롭 영역 비율 (2.5 = 얼굴의 2.5배 영역)
    face_crop_padding: float = 0.3  # 얼굴 위쪽 여백 비율 (이마/머리 포함)
    min_face_ratio: float = 0.25  # 얼굴이 이 비율 이하면 자동 크롭 활성화
    fixed_crop_region: bool = True  # 루프 영상용: 첫 프레임 크롭 영역을 고정 사용 (효율적)

    # 디바이스
    device: Optional[str] = None  # None이면 자동 감지

    # 캐싱
    cache_source_latent: bool = True  # 소스 이미지 latent 캐싱
    cache_video_frames: bool = True  # 영상 프레임 캐싱

    # FaceParsing 설정 (공식 MuseTalk V1.5 블렌딩)
    use_face_parsing: bool = True  # FaceParsing (BiSeNet) 사용 여부
    face_parsing_mode: str = "jaw"  # "raw", "jaw", "neck" - V1.5는 "jaw" 권장
    face_parsing_left_cheek_width: int = 90  # V1.5 파라미터
    face_parsing_right_cheek_width: int = 90  # V1.5 파라미터
    face_parsing_upper_boundary_ratio: float = 0.5  # 상반부 마스킹 비율
    face_parsing_expand: float = 1.5  # 블렌딩 확장 비율

    # V1.5 특수 설정
    extra_margin: int = 10  # V1.5 bbox 하단 확장 마진


@dataclass
class VideoFrame:
    """비디오 프레임 데이터"""
    frame: np.ndarray  # BGR 형식
    timestamp: float
    state: AvatarState
    metadata: Dict[str, Any]


class FaceLandmarkDetector:
    """얼굴 랜드마크 감지기 (MediaPipe 사용)"""

    def __init__(self):
        if mp is None:
            raise ImportError("MediaPipe is required. Install: pip install mediapipe")

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        logger.info("FaceLandmarkDetector initialized with MediaPipe")

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        얼굴 랜드마크 감지

        Args:
            image: BGR 이미지 (numpy array)

        Returns:
            landmarks: (468, 3) 배열 또는 None
        """
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 랜드마크 감지
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        # 첫 번째 얼굴의 랜드마크 추출
        face_landmarks = results.multi_face_landmarks[0]

        h, w = image.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            landmarks.append([x, y, z])

        return np.array(landmarks, dtype=np.float32)

    def get_face_bbox(
        self,
        landmarks: np.ndarray,
        expand_ratio: float = 1.2
    ) -> Tuple[int, int, int, int]:
        """
        얼굴 바운딩 박스 계산

        Args:
            landmarks: (468, 3) 랜드마크 배열
            expand_ratio: 확장 비율

        Returns:
            (x1, y1, x2, y2): 바운딩 박스 좌표
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        width = x_max - x_min
        height = y_max - y_min

        # 확장
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        new_width = width * expand_ratio
        new_height = height * expand_ratio

        x1 = int(x_center - new_width / 2)
        y1 = int(y_center - new_height / 2)
        x2 = int(x_center + new_width / 2)
        y2 = int(y_center + new_height / 2)

        return x1, y1, x2, y2


class OfficialFaceParsing:
    """
    공식 MuseTalk FaceParsing (BiSeNet)

    얼굴의 각 부위(피부, 입술, 코 등)를 정교하게 분할하여
    자연스러운 블렌딩 마스크를 생성합니다.

    V1.5에서는 'jaw' 모드를 사용하여 턱/입 영역만 선택적으로 교체합니다.
    """

    def __init__(
        self,
        resnet_path: str,
        model_path: str,
        device: str = "cuda",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90
    ):
        """
        Args:
            resnet_path: ResNet18 가중치 경로
            model_path: BiSeNet 가중치 경로 (79999_iter.pth)
            device: 디바이스
            left_cheek_width: 왼쪽 볼 너비 (V1.5 파라미터)
            right_cheek_width: 오른쪽 볼 너비 (V1.5 파라미터)
        """
        self.device = device
        self.net = None
        self.preprocess = None

        if not os.path.exists(resnet_path) or not os.path.exists(model_path):
            logger.warning(f"FaceParsing model files not found: {resnet_path}, {model_path}")
            return

        try:
            # BiSeNet 모델 import (MuseTalk의 face_parsing 모듈 사용)
            import sys
            musetalk_path = Path(__file__).parent.parent.parent.parent / "MuseTalk"
            if musetalk_path.exists():
                sys.path.insert(0, str(musetalk_path))
                from musetalk.utils.face_parsing.model import BiSeNet

                self.net = BiSeNet(resnet_path)
                if device == "cuda" and torch.cuda.is_available():
                    self.net.cuda()
                    self.net.load_state_dict(torch.load(model_path))
                else:
                    self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.net.eval()

                if transforms is not None:
                    self.preprocess = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])

                # V1.5 jaw 모드용 커널 설정
                self._setup_jaw_kernels(left_cheek_width, right_cheek_width)

                logger.info("OfficialFaceParsing initialized with BiSeNet")
            else:
                logger.warning(f"MuseTalk path not found: {musetalk_path}")

        except Exception as e:
            logger.error(f"Failed to initialize FaceParsing: {e}")
            self.net = None

    def _setup_jaw_kernels(self, left_cheek_width: int, right_cheek_width: int):
        """V1.5 jaw 모드용 커널 설정"""
        cone_height = 21
        tail_height = 12
        total_size = cone_height + tail_height

        kernel = np.zeros((total_size, total_size), dtype=np.uint8)
        center_x = total_size // 2

        for row in range(cone_height):
            if row < cone_height // 2:
                continue
            width = int(2 * (row - cone_height // 2) + 1)
            start = int(center_x - (width // 2))
            end = int(center_x + (width // 2) + 1)
            kernel[row, start:end] = 1

        if cone_height > 0:
            base_width = int(kernel[cone_height - 1].sum())
        else:
            base_width = 1

        for row in range(cone_height, total_size):
            start = max(0, int(center_x - (base_width // 2)))
            end = min(total_size, int(center_x + (base_width // 2) + 1))
            kernel[row, start:end] = 1

        self.kernel = kernel
        self.cheek_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 3))

        # 볼 영역 마스크
        cheek_mask = np.zeros((512, 512), dtype=np.uint8)
        center = 512 // 2
        cv2.rectangle(cheek_mask, (0, 0), (center - left_cheek_width, 512), 255, -1)
        cv2.rectangle(cheek_mask, (center + right_cheek_width, 0), (512, 512), 255, -1)
        self.cheek_mask = cheek_mask

    def __call__(
        self,
        image: Union[np.ndarray, "Image.Image"],
        size: Tuple[int, int] = (512, 512),
        mode: str = "jaw"
    ) -> Optional["Image.Image"]:
        """
        얼굴 파싱 마스크 생성

        Args:
            image: BGR 이미지 (numpy) 또는 PIL Image
            size: 출력 크기
            mode: 마스킹 모드 ("raw", "jaw", "neck")

        Returns:
            마스크 이미지 (PIL Image) 또는 None
        """
        if self.net is None or self.preprocess is None:
            return None

        try:
            # 이미지 변환
            if isinstance(image, np.ndarray):
                if Image is not None:
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    return None

            with torch.no_grad():
                image_resized = image.resize(size, Image.BILINEAR)
                img = self.preprocess(image_resized)

                if self.device == "cuda" and torch.cuda.is_available():
                    img = torch.unsqueeze(img, 0).cuda()
                else:
                    img = torch.unsqueeze(img, 0)

                out = self.net(img)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)

                # 모드에 따른 마스크 생성
                if mode == "neck":
                    parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
                    parsing[np.where(parsing != 255)] = 0
                elif mode == "jaw":
                    # V1.5 jaw 모드: 턱/입 영역만 정교하게 선택
                    face_region = np.isin(parsing, [1]) * 255
                    face_region = face_region.astype(np.uint8)
                    original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
                    eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)
                    face_region = cv2.bitwise_and(eroded, self.cheek_mask)
                    face_region = cv2.bitwise_or(face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask))
                    parsing[(face_region == 255) & (~np.isin(parsing, [10]))] = 255
                    parsing[np.isin(parsing, [11, 12, 13])] = 255
                    parsing[np.where(parsing != 255)] = 0
                else:  # raw
                    parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                    parsing[np.where(parsing != 255)] = 0

            return Image.fromarray(parsing.astype(np.uint8))

        except Exception as e:
            logger.error(f"FaceParsing failed: {e}")
            return None

    def is_available(self) -> bool:
        """FaceParsing 사용 가능 여부"""
        return self.net is not None and self.preprocess is not None


def official_get_image(
    image: np.ndarray,
    face: np.ndarray,
    face_box: Tuple[int, int, int, int],
    upper_boundary_ratio: float = 0.5,
    expand: float = 1.5,
    mode: str = "jaw",
    fp: Optional[OfficialFaceParsing] = None
) -> np.ndarray:
    """
    공식 MuseTalk 블렌딩 함수 (blending.py의 get_image)

    FaceParsing 마스크를 사용하여 정교하게 블렌딩합니다.

    Args:
        image: 원본 이미지 (BGR)
        face: 생성된 얼굴 이미지 (BGR)
        face_box: 얼굴 바운딩 박스 (x1, y1, x2, y2)
        upper_boundary_ratio: 상반부 마스킹 비율
        expand: 확장 비율
        mode: FaceParsing 모드
        fp: FaceParsing 인스턴스

    Returns:
        블렌딩된 이미지 (BGR)
    """
    if fp is None or not fp.is_available() or Image is None:
        # FaceParsing 불가능 시 간단한 블렌딩 사용
        return _simple_blend(image, face, face_box)

    try:
        # crop_box 계산 (확장된 영역)
        x, y, x1, y1 = face_box
        x_c, y_c = (x + x1) // 2, (y + y1) // 2
        w, h = x1 - x, y1 - y
        s = int(max(w, h) // 2 * expand)
        crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
        x_s, y_s, x_e, y_e = crop_box

        # 이미지 경계 처리
        img_h, img_w = image.shape[:2]
        x_s = max(0, x_s)
        y_s = max(0, y_s)
        x_e = min(img_w, x_e)
        y_e = min(img_h, y_e)

        # PIL 이미지로 변환
        body = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        face_pil = Image.fromarray(face[:, :, ::-1])

        # 확장된 얼굴 영역 크롭
        face_large = body.crop((x_s, y_s, x_e, y_e))
        ori_shape = face_large.size

        # FaceParsing 마스크 생성
        mask_image = fp(face_large, mode=mode)

        if mask_image is None:
            return _simple_blend(image, face, face_box)

        # 얼굴 영역만 마스크 크롭
        mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
        mask_image = Image.new('L', ori_shape, 0)
        mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

        # 상반부 마스크 제거
        width, height = mask_image.size
        top_boundary = int(height * upper_boundary_ratio)
        modified_mask_image = Image.new('L', ori_shape, 0)
        modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

        # 가우시안 블러로 경계 부드럽게
        blur_kernel_size = int(0.05 * ori_shape[0] // 2 * 2) + 1
        mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
        mask_image = Image.fromarray(mask_array)

        # 생성된 얼굴을 확장 영역에 삽입
        face_large.paste(face_pil, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

        # 원본에 블렌딩
        body.paste(face_large, (x_s, y_s), mask_image)

        # numpy 배열로 변환 (RGB -> BGR)
        result = np.array(body)[:, :, ::-1]

        return result

    except Exception as e:
        logger.error(f"official_get_image failed: {e}")
        return _simple_blend(image, face, face_box)


def _simple_blend(
    image: np.ndarray,
    face: np.ndarray,
    face_box: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    간단한 블렌딩 (FaceParsing 불가능 시 사용)

    타원형 마스크로 입 영역만 교체합니다.
    """
    x1, y1, x2, y2 = face_box
    h, w = image.shape[:2]

    result = image.copy()

    # face를 bbox 크기로 리사이즈
    face_resized = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)

    # 블렌딩 마스크 생성
    mask_h, mask_w = face_resized.shape[:2]
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)

    # 입 영역 정의 (타원형)
    center_y = int(mask_h * 0.65)
    center_x = mask_w // 2
    radius_y = int(mask_h * 0.15)
    radius_x = int(mask_w * 0.25)

    for y in range(mask_h):
        for x in range(mask_w):
            dx = (x - center_x) / max(radius_x, 1)
            dy = (y - center_y) / max(radius_y, 1)
            dist = dx**2 + dy**2

            if dist <= 1.0:
                mask[y, x] = 1.0 - (dist ** 0.5) * 0.3
            elif dist <= 1.8:
                mask[y, x] = max(0, 1.0 - (dist - 1.0) * 1.25)

    # 가우시안 블러로 부드럽게
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # 3채널로 확장
    mask_3ch = mask[:, :, np.newaxis]

    # 블렌딩
    blended = (image[y1:y2, x1:x2].astype(np.float32) * (1 - mask_3ch) +
               face_resized.astype(np.float32) * mask_3ch).astype(np.uint8)

    result[y1:y2, x1:x2] = blended

    return result


class VideoFrameLoader:
    """
    영상 프레임 로더

    소스 영상에서 프레임을 로드하고 관리합니다.
    루프 재생, 프레임 캐싱, 랜드마크 추출 등을 지원합니다.

    얼굴 기반 크롭 모드:
    - 영상에서 얼굴이 작을 경우 (예: 전신 영상)
    - 얼굴 위치를 추적하여 해당 영역만 크롭
    - 원본 영상은 유지하고, 얼굴 영역만 MuseTalk에 전달
    """

    def __init__(
        self,
        video_path: str,
        target_fps: int = 25,
        resolution: int = 256,
        loop: bool = True,
        preload: bool = True,
        max_frames: int = 250,
        use_face_crop: bool = True,
        face_crop_ratio: float = 2.5,
        face_crop_padding: float = 0.3,
        min_face_ratio: float = 0.25,
        fixed_crop_region: bool = False
    ):
        """
        Args:
            video_path: 영상 파일 경로
            target_fps: 목표 FPS
            resolution: 목표 해상도
            loop: 루프 재생 여부
            preload: 프레임 미리 로드 여부
            max_frames: 최대 로드 프레임 수
            use_face_crop: 얼굴 기반 크롭 사용 여부
            face_crop_ratio: 얼굴 크기 대비 크롭 영역 비율
            face_crop_padding: 얼굴 위쪽 여백 비율
            min_face_ratio: 자동 크롭 활성화 임계값
            fixed_crop_region: True면 첫 프레임 크롭 영역을 모든 프레임에 적용 (정적 영상),
                              False면 프레임마다 얼굴 검출 후 각각 기억 (동적 영상, 루프 시 재사용)
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.resolution = resolution
        self.loop = loop
        self.preload = preload
        self.max_frames = max_frames

        # 얼굴 기반 크롭 설정
        self.use_face_crop = use_face_crop
        self.face_crop_ratio = face_crop_ratio
        self.face_crop_padding = face_crop_padding
        self.min_face_ratio = min_face_ratio
        self.fixed_crop_region = fixed_crop_region

        # 영상 정보
        self.cap = None
        self.total_frames = 0
        self.original_fps = 0
        self.frame_width = 0
        self.frame_height = 0

        # 프레임 캐시 (크롭된 프레임)
        self.frames: List[np.ndarray] = []
        # 원본 프레임 (합성용)
        self.original_frames: List[np.ndarray] = []
        # 크롭 영역 정보 (x1, y1, x2, y2)
        self.crop_regions: List[Tuple[int, int, int, int]] = []
        self.landmarks_cache: List[Optional[np.ndarray]] = []
        self.latents_cache: List[Optional[torch.Tensor]] = []

        # 현재 상태
        self.current_frame_idx = 0
        self._is_loaded = False
        self._face_crop_enabled = False  # 실제 크롭 활성화 여부

        # 랜드마크 감지기
        self.landmark_detector = None

        # OpenCV 얼굴 검출기 (초기 분석용)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        logger.info(f"VideoFrameLoader initialized: {video_path}")

    def load(self, landmark_detector: Optional[FaceLandmarkDetector] = None) -> bool:
        """
        영상 로드 및 프레임 추출

        Args:
            landmark_detector: 랜드마크 감지기 (미리 추출할 경우)

        Returns:
            bool: 성공 여부
        """
        if not os.path.exists(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            return False

        self.landmark_detector = landmark_detector

        try:
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {self.video_path}")
                return False

            # 영상 정보 추출
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"Video loaded: {self.total_frames} frames, "
                f"{self.original_fps:.1f} fps, "
                f"{self.frame_width}x{self.frame_height}"
            )

            # 얼굴 크기 분석 (첫 프레임 기준)
            if self.use_face_crop:
                self._analyze_face_size()

            # FPS 변환 계산
            frame_skip = max(1, int(self.original_fps / self.target_fps))

            # 프레임 미리 로드
            if self.preload:
                self._preload_frames(frame_skip)

            self._is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            return False

    def _analyze_face_size(self):
        """
        첫 프레임에서 얼굴 크기 분석

        얼굴이 화면의 min_face_ratio 이하면 자동 크롭 활성화
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()

        if not ret:
            logger.warning("Failed to read first frame for face analysis")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        if len(faces) == 0:
            logger.warning("No face detected in first frame, disabling face crop")
            self._face_crop_enabled = False
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # 가장 큰 얼굴
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_area = w * h
        image_area = self.frame_width * self.frame_height
        face_ratio = face_area / image_area

        logger.info(f"Face analysis: {w}x{h}px, ratio={face_ratio*100:.1f}%")

        if face_ratio < self.min_face_ratio:
            self._face_crop_enabled = True
            logger.info(
                f"Face too small ({face_ratio*100:.1f}% < {self.min_face_ratio*100:.0f}%), "
                f"enabling face-based crop"
            )
        else:
            self._face_crop_enabled = False
            logger.info(f"Face size OK ({face_ratio*100:.1f}%), no crop needed")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _calculate_crop_region(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[int, int, int, int]:
        """
        얼굴 기반 크롭 영역 계산

        Args:
            frame: 원본 프레임
            face_bbox: 얼굴 바운딩 박스 (x, y, w, h) - None이면 자동 검출

        Returns:
            (x1, y1, x2, y2): 크롭 영역
        """
        h, w = frame.shape[:2]

        # 얼굴 검출
        if face_bbox is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                # 얼굴 없으면 중앙 크롭
                size = min(w, h)
                x1 = (w - size) // 2
                y1 = (h - size) // 2
                return x1, y1, x1 + size, y1 + size

            face_bbox = max(faces, key=lambda f: f[2] * f[3])

        fx, fy, fw, fh = face_bbox

        # 얼굴 중심
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2

        # 크롭 크기 계산 (얼굴 크기 * 비율)
        crop_size = int(max(fw, fh) * self.face_crop_ratio)

        # 정사각형 크롭 (최소 resolution 이상)
        crop_size = max(crop_size, self.resolution)

        # 위쪽 여백 추가 (이마/머리 포함)
        head_offset = int(fh * self.face_crop_padding)
        crop_cy = face_cy - head_offset

        # 크롭 영역 계산
        x1 = face_cx - crop_size // 2
        y1 = crop_cy - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # 경계 체크 및 조정
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h

        # 다시 경계 체크
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return x1, y1, x2, y2

    def _preload_frames(self, frame_skip: int = 1):
        """
        프레임 미리 로드 (얼굴 기반 크롭 지원)

        크롭 모드:
        - fixed_crop_region=True: 첫 프레임에서 얼굴 검출 후 모든 프레임에 동일 영역 적용 (효율적)
        - fixed_crop_region=False: 프레임마다 얼굴 검출하여 각각 기억, 루프 시 저장된 위치 재사용
        """
        logger.info(f"Preloading video frames... (fixed_crop={'enabled' if self.fixed_crop_region else 'per-frame'})")

        self.frames = []
        self.original_frames = []
        self.crop_regions = []
        self.landmarks_cache = []

        frame_count = 0
        loaded_count = 0

        # 고정 크롭 영역 (첫 프레임 기준)
        fixed_region = None
        # 이전 크롭 영역 (스무딩용 - per-frame 모드에서만 사용)
        prev_crop_region = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # FPS 맞추기 위한 프레임 스킵
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # 최대 프레임 수 제한
            if loaded_count >= self.max_frames:
                break

            # 원본 프레임 저장
            self.original_frames.append(frame.copy())

            # 얼굴 기반 크롭 적용
            if self._face_crop_enabled:
                if self.fixed_crop_region:
                    # 고정 크롭 모드: 첫 프레임에서만 얼굴 검출, 이후 동일 영역 사용
                    if fixed_region is None:
                        fixed_region = self._calculate_crop_region(frame)
                        logger.info(f"Fixed crop region set: {fixed_region}")
                    crop_region = fixed_region
                else:
                    # Per-frame 모드: 프레임마다 얼굴 검출하여 기억 (루프 시 재사용)
                    crop_region = self._calculate_crop_region(frame)

                    # 크롭 영역 스무딩 (급격한 변화 방지)
                    if prev_crop_region is not None:
                        crop_region = self._smooth_crop_region(prev_crop_region, crop_region, alpha=0.7)
                    prev_crop_region = crop_region

                self.crop_regions.append(crop_region)

                # 크롭 적용
                x1, y1, x2, y2 = crop_region
                cropped = frame[y1:y2, x1:x2]

                # 해상도 조정
                processed_frame = self._resize_cropped_frame(cropped)
            else:
                # 크롭 없이 전체 프레임 리사이즈
                self.crop_regions.append(None)
                processed_frame = self._resize_frame(frame)

            self.frames.append(processed_frame)

            # 랜드마크 추출 (처리된 프레임에서)
            if self.landmark_detector:
                landmarks = self.landmark_detector.detect(processed_frame)
                self.landmarks_cache.append(landmarks)
            else:
                self.landmarks_cache.append(None)

            loaded_count += 1
            frame_count += 1

        # 영상 처음으로 되감기
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 로드 완료 로그
        if self._face_crop_enabled:
            crop_mode = "fixed" if self.fixed_crop_region else "per-frame (cached)"
            unique_regions = len(set(self.crop_regions)) if self.crop_regions else 0
            logger.info(
                f"Preloaded {len(self.frames)} frames "
                f"(face_crop=enabled, mode={crop_mode}, unique_regions={unique_regions})"
            )
        else:
            logger.info(
                f"Preloaded {len(self.frames)} frames (face_crop=disabled)"
            )

    def _smooth_crop_region(
        self,
        prev_region: Tuple[int, int, int, int],
        curr_region: Tuple[int, int, int, int],
        alpha: float = 0.7
    ) -> Tuple[int, int, int, int]:
        """크롭 영역 스무딩 (급격한 변화 방지)"""
        x1 = int(prev_region[0] * alpha + curr_region[0] * (1 - alpha))
        y1 = int(prev_region[1] * alpha + curr_region[1] * (1 - alpha))
        x2 = int(prev_region[2] * alpha + curr_region[2] * (1 - alpha))
        y2 = int(prev_region[3] * alpha + curr_region[3] * (1 - alpha))
        return x1, y1, x2, y2

    def _resize_cropped_frame(self, cropped: np.ndarray) -> np.ndarray:
        """크롭된 프레임을 목표 해상도로 리사이즈"""
        # 정사각형으로 리사이즈 (크롭 영역이 이미 정사각형에 가까움)
        resized = cv2.resize(
            cropped,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_LANCZOS4
        )
        return resized

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 크기 조정"""
        h, w = frame.shape[:2]

        # 종횡비 유지하며 리사이즈
        if h > w:
            new_h = self.resolution
            new_w = int(w * (self.resolution / h))
        else:
            new_w = self.resolution
            new_h = int(h * (self.resolution / w))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 정사각형 캔버스에 중앙 배치
        canvas = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        y_offset = (self.resolution - new_h) // 2
        x_offset = (self.resolution - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def get_frame(self, index: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        프레임 가져오기

        Args:
            index: 프레임 인덱스 (None이면 다음 프레임)

        Returns:
            (frame, landmarks): 크롭/리사이즈된 프레임과 랜드마크
        """
        if not self._is_loaded:
            raise RuntimeError("Video not loaded. Call load() first.")

        if index is None:
            index = self.current_frame_idx
            self.current_frame_idx += 1

            # 루프 처리
            if self.current_frame_idx >= len(self.frames):
                if self.loop:
                    self.current_frame_idx = 0
                else:
                    self.current_frame_idx = len(self.frames) - 1

        # 인덱스 범위 체크
        index = index % len(self.frames) if self.loop else min(index, len(self.frames) - 1)

        frame = self.frames[index]
        landmarks = self.landmarks_cache[index] if index < len(self.landmarks_cache) else None

        return frame.copy(), landmarks

    def get_frame_with_info(
        self,
        index: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        프레임과 추가 정보 가져오기 (합성용)

        Args:
            index: 프레임 인덱스 (None이면 다음 프레임)

        Returns:
            (processed_frame, original_frame, crop_region, landmarks):
            - processed_frame: 크롭/리사이즈된 프레임 (MuseTalk 입력용)
            - original_frame: 원본 프레임 (최종 합성용)
            - crop_region: 크롭 영역 (x1, y1, x2, y2) - 크롭 미사용 시 None
            - landmarks: 랜드마크
        """
        if not self._is_loaded:
            raise RuntimeError("Video not loaded. Call load() first.")

        if index is None:
            index = self.current_frame_idx
            self.current_frame_idx += 1

            if self.current_frame_idx >= len(self.frames):
                if self.loop:
                    self.current_frame_idx = 0
                else:
                    self.current_frame_idx = len(self.frames) - 1

        index = index % len(self.frames) if self.loop else min(index, len(self.frames) - 1)

        processed_frame = self.frames[index].copy()
        original_frame = self.original_frames[index].copy() if self.original_frames else processed_frame
        crop_region = self.crop_regions[index] if index < len(self.crop_regions) else None
        landmarks = self.landmarks_cache[index] if index < len(self.landmarks_cache) else None

        return processed_frame, original_frame, crop_region, landmarks

    def get_next_frame(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """다음 프레임 가져오기"""
        return self.get_frame()

    def is_face_crop_enabled(self) -> bool:
        """얼굴 기반 크롭이 활성화되어 있는지 확인"""
        return self._face_crop_enabled

    def reset(self):
        """프레임 인덱스 초기화"""
        self.current_frame_idx = 0

    def get_frame_count(self) -> int:
        """총 프레임 수 반환"""
        return len(self.frames)

    def get_duration(self) -> float:
        """영상 길이 (초) 반환"""
        return len(self.frames) / self.target_fps if self.target_fps > 0 else 0

    def cache_latent(self, index: int, latent: torch.Tensor):
        """프레임의 VAE latent 캐싱"""
        # 캐시 리스트 확장
        while len(self.latents_cache) <= index:
            self.latents_cache.append(None)
        self.latents_cache[index] = latent

    def get_cached_latent(self, index: int) -> Optional[torch.Tensor]:
        """캐싱된 latent 가져오기"""
        if index < len(self.latents_cache):
            return self.latents_cache[index]
        return None

    def release(self):
        """리소스 해제"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.frames.clear()
        self.original_frames.clear()
        self.crop_regions.clear()
        self.landmarks_cache.clear()
        self.latents_cache.clear()
        self._is_loaded = False
        self._face_crop_enabled = False

        logger.info("VideoFrameLoader released")

    def __len__(self) -> int:
        return len(self.frames)

    def __del__(self):
        self.release()


class ImageSequenceLoader:
    """
    이미지 시퀀스 로더

    디렉토리 내의 이미지 파일들을 순차적으로 로드합니다.
    """

    def __init__(
        self,
        image_dir: str,
        target_fps: int = 25,
        resolution: int = 256,
        loop: bool = True,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        self.image_dir = image_dir
        self.target_fps = target_fps
        self.resolution = resolution
        self.loop = loop
        self.extensions = extensions

        self.image_paths: List[str] = []
        self.frames: List[np.ndarray] = []
        self.landmarks_cache: List[Optional[np.ndarray]] = []
        self.current_frame_idx = 0
        self._is_loaded = False

    def load(self, landmark_detector: Optional[FaceLandmarkDetector] = None) -> bool:
        """이미지 시퀀스 로드"""
        if not os.path.isdir(self.image_dir):
            logger.error(f"Directory not found: {self.image_dir}")
            return False

        # 이미지 파일 목록
        self.image_paths = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(self.extensions)
        ])

        if not self.image_paths:
            logger.error(f"No images found in: {self.image_dir}")
            return False

        logger.info(f"Found {len(self.image_paths)} images")

        # 이미지 로드
        for img_path in self.image_paths:
            frame = cv2.imread(img_path)
            if frame is not None:
                frame = self._resize_frame(frame)
                self.frames.append(frame)

                if landmark_detector:
                    landmarks = landmark_detector.detect(frame)
                    self.landmarks_cache.append(landmarks)
                else:
                    self.landmarks_cache.append(None)

        self._is_loaded = True
        logger.info(f"Loaded {len(self.frames)} images")
        return True

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 크기 조정 (VideoFrameLoader와 동일)"""
        h, w = frame.shape[:2]

        if h > w:
            new_h = self.resolution
            new_w = int(w * (self.resolution / h))
        else:
            new_w = self.resolution
            new_h = int(h * (self.resolution / w))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        y_offset = (self.resolution - new_h) // 2
        x_offset = (self.resolution - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def get_frame(self, index: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """프레임 가져오기"""
        if not self._is_loaded:
            raise RuntimeError("Images not loaded. Call load() first.")

        if index is None:
            index = self.current_frame_idx
            self.current_frame_idx += 1

            if self.current_frame_idx >= len(self.frames):
                if self.loop:
                    self.current_frame_idx = 0
                else:
                    self.current_frame_idx = len(self.frames) - 1

        index = index % len(self.frames) if self.loop else min(index, len(self.frames) - 1)

        frame = self.frames[index]
        landmarks = self.landmarks_cache[index] if index < len(self.landmarks_cache) else None

        return frame.copy(), landmarks

    def get_next_frame(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.get_frame()

    def reset(self):
        self.current_frame_idx = 0

    def get_frame_count(self) -> int:
        return len(self.frames)

    def release(self):
        self.frames.clear()
        self.landmarks_cache.clear()
        self._is_loaded = False

    def __len__(self) -> int:
        return len(self.frames)


class AudioFeatureExtractor:
    """
    오디오 특징 추출기 (MuseTalk 공식 방식 - transformers WhisperModel 사용)

    MuseTalk은 transformers의 WhisperModel을 사용하여 오디오에서 특징을 추출합니다.
    중요: 모든 hidden states를 stack하여 (T, 50, 384) 형태의 특징을 생성합니다.

    공식 MuseTalk 구현:
    - audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
    - audio_feats = torch.stack(audio_feats, dim=2)  # Stack all 5 hidden layers
    - audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')  # (T, 50, 384)
    """

    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cuda",
        use_fp16: bool = True,
        local_model_path: Optional[str] = None
    ):
        """
        Args:
            model_name: Whisper 모델 이름 (tiny, base, small, medium)
            device: 디바이스 (cuda/mps/cpu)
            use_fp16: FP16 사용 여부
            local_model_path: 로컬 모델 경로 (선택) - 예: models/musetalk/whisper
        """
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        self.whisper_model = None
        self.feature_extractor = None
        self.model_name = model_name
        self.feature_dim = 384  # Whisper tiny output dimension

        # Audio processing constants (from official MuseTalk)
        self.sample_rate = 16000
        self.audio_fps = 50  # Whisper outputs 50 features per second (30s -> 1500 frames)

        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available. Install: pip install transformers einops")
            return

        logger.info(f"Loading Whisper model (transformers): {model_name}")

        try:
            # transformers 형식의 로컬 모델 경로 확인 (preprocessor_config.json 필요)
            model_path = None

            # 1. 명시적으로 지정된 로컬 경로 확인
            if local_model_path and os.path.exists(local_model_path):
                config_file = os.path.join(local_model_path, "preprocessor_config.json")
                if os.path.exists(config_file):
                    model_path = local_model_path
                    logger.info(f"Loading Whisper from local path: {model_path}")

            # 2. 기본 로컬 경로 확인 (transformers 형식)
            if model_path is None:
                default_local_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "models", "musetalk", f"whisper-{model_name}"
                )
                config_file = os.path.join(default_local_path, "preprocessor_config.json")
                if os.path.exists(config_file):
                    model_path = default_local_path
                    logger.info(f"Loading Whisper from default local path: {model_path}")

            # 3. HuggingFace에서 다운로드
            if model_path is None:
                model_path = f"openai/whisper-{model_name}"
                logger.info(f"Loading Whisper from HuggingFace: {model_path}")

            # Feature extractor 로드
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

            # WhisperModel 로드 (transformers)
            self.whisper_model = WhisperModel.from_pretrained(model_path)

            # dtype 설정
            self.dtype = torch.float16 if self.use_fp16 else torch.float32

            # 모델을 디바이스로 이동
            self.whisper_model = self.whisper_model.to(device=device, dtype=self.dtype)

            # 평가 모드
            self.whisper_model.eval()

            # Freeze parameters
            self.whisper_model.requires_grad_(False)

            logger.info(f"AudioFeatureExtractor (transformers) initialized on {device} (dtype={self.dtype})")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            import traceback
            traceback.print_exc()
            self.whisper_model = None

    @torch.no_grad()
    def get_audio_features(self, audio: np.ndarray) -> Tuple[List[torch.Tensor], int]:
        """
        오디오에서 Whisper input features 추출 (MuseTalk 공식 방식)

        Args:
            audio: (n_samples,) 오디오 데이터 (16kHz, float32)

        Returns:
            features: List of input features (30s 청크 단위)
            librosa_length: 원본 오디오 길이
        """
        if self.feature_extractor is None:
            logger.error("Feature extractor not loaded")
            return [], 0

        audio = audio.astype(np.float32)
        librosa_length = len(audio)

        # 30초 청크로 분할
        segment_length = 30 * self.sample_rate
        segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]

        features = []
        for segment in segments:
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=self.sample_rate
            ).input_features
            features.append(audio_feature)

        return features, librosa_length

    @torch.no_grad()
    def get_whisper_chunks(
        self,
        whisper_input_features: List[torch.Tensor],
        librosa_length: int,
        fps: int = 25,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2
    ) -> torch.Tensor:
        """
        Whisper features를 비디오 프레임에 맞게 청크로 변환 (MuseTalk 공식 방식)

        중요: 모든 hidden states (5개 레이어)를 stack하여 (T, 50, 384) 형태로 반환

        Args:
            whisper_input_features: get_audio_features의 출력
            librosa_length: 원본 오디오 샘플 수
            fps: 비디오 FPS
            audio_padding_length_left: 좌측 패딩 프레임 수
            audio_padding_length_right: 우측 패딩 프레임 수

        Returns:
            audio_prompts: (num_frames, 50, 384) 오디오 특징 텐서
        """
        if self.whisper_model is None:
            logger.error("Whisper model not loaded")
            return torch.zeros(1, 50, self.feature_dim, device=self.device)

        # 각 프레임당 오디오 특징 길이: 2 * (left + right + 1) = 10
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)

        whisper_feature_list = []

        # 모든 30초 mel input features 처리
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(device=self.device, dtype=self.dtype)

            # 핵심: output_hidden_states=True로 모든 hidden states 추출
            encoder_output = self.whisper_model.encoder(
                input_feature,
                output_hidden_states=True
            )
            audio_feats = encoder_output.hidden_states  # Tuple of 5 tensors

            # 5개 hidden states를 stack: (1, 1500, 384) x 5 -> (1, 1500, 5, 384)
            audio_feats = torch.stack(audio_feats, dim=2)
            whisper_feature_list.append(audio_feats)

        # 모든 청크 결합
        whisper_feature = torch.cat(whisper_feature_list, dim=1)  # (1, total_frames, 5, 384)

        # 실제 오디오 길이에 맞게 자르기
        fps = int(fps)
        whisper_idx_multiplier = self.audio_fps / fps  # 50/25 = 2
        num_frames = int((librosa_length / self.sample_rate) * fps)
        actual_length = int((librosa_length / self.sample_rate) * self.audio_fps)
        whisper_feature = whisper_feature[:, :actual_length, ...]

        # 패딩 추가 (시작과 끝)
        padding_nums = int(np.ceil(whisper_idx_multiplier))
        whisper_feature = torch.cat([
            torch.zeros_like(whisper_feature[:, :padding_nums * audio_padding_length_left]),
            whisper_feature,
            # 추가 패딩으로 인덱스 초과 방지
            torch.zeros_like(whisper_feature[:, :padding_nums * 3 * audio_padding_length_right])
        ], dim=1)

        # 각 비디오 프레임에 해당하는 오디오 특징 추출
        audio_prompts = []
        for frame_index in range(num_frames):
            try:
                audio_index = int(frame_index * whisper_idx_multiplier)
                audio_clip = whisper_feature[:, audio_index:audio_index + audio_feature_length_per_frame]

                if audio_clip.shape[1] != audio_feature_length_per_frame:
                    # 패딩이 부족한 경우 zero padding
                    pad_size = audio_feature_length_per_frame - audio_clip.shape[1]
                    audio_clip = torch.cat([
                        audio_clip,
                        torch.zeros(1, pad_size, audio_clip.shape[2], audio_clip.shape[3],
                                   device=self.device, dtype=self.dtype)
                    ], dim=1)

                audio_prompts.append(audio_clip)
            except Exception as e:
                logger.error(f"Error at frame {frame_index}: {e}")
                # 에러 시 zero tensor 추가
                audio_prompts.append(torch.zeros(
                    1, audio_feature_length_per_frame, 5, self.feature_dim,
                    device=self.device, dtype=self.dtype
                ))

        # (num_frames, 1, 10, 5, 384) -> (num_frames, 10, 5, 384)
        audio_prompts = torch.cat(audio_prompts, dim=0)  # (T, 10, 5, 384)

        # rearrange: (T, 10, 5, 384) -> (T, 50, 384)
        # 'b c h w -> b (c h) w' where c=10, h=5, w=384
        if rearrange is not None:
            audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')
        else:
            # Manual reshape if einops not available
            T, c, h, w = audio_prompts.shape
            audio_prompts = audio_prompts.permute(0, 1, 2, 3).reshape(T, c * h, w)

        logger.info(f"Generated whisper chunks: shape={audio_prompts.shape}")
        return audio_prompts

    @torch.no_grad()
    def extract_full_audio(self, audio: np.ndarray, sample_rate: int = 16000, fps: int = 25) -> torch.Tensor:
        """
        전체 오디오를 처리하여 비디오 프레임에 맞는 특징 배열 생성 (MuseTalk 공식 방식)

        Args:
            audio: 전체 오디오 데이터 (float32, -1~1 범위)
            sample_rate: 샘플레이트 (기본 16kHz)
            fps: 비디오 FPS (기본 25)

        Returns:
            features: (num_frames, 50, 384) 오디오 특징 배열
        """
        if self.whisper_model is None:
            logger.warning("Whisper model not loaded")
            return torch.zeros(1, 50, self.feature_dim, device=self.device)

        try:
            # 1. 오디오에서 Whisper input features 추출
            whisper_input_features, librosa_length = self.get_audio_features(audio)

            # 2. Whisper chunks 생성 (5개 hidden states stack -> 50차원)
            audio_prompts = self.get_whisper_chunks(
                whisper_input_features,
                librosa_length,
                fps=fps,
                audio_padding_length_left=2,
                audio_padding_length_right=2
            )

            logger.info(f"Extracted {audio_prompts.shape[0]} audio features (shape: {audio_prompts.shape})")
            return audio_prompts

        except Exception as e:
            logger.error(f"Full audio feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(1, 50, self.feature_dim, device=self.device)

    @torch.no_grad()
    def extract(self, audio_chunk: np.ndarray, fps: int = 25) -> torch.Tensor:
        """
        단일 오디오 청크에서 특징 추출 (실시간 처리용)

        Args:
            audio_chunk: (n_samples,) 오디오 데이터 (16kHz, float32, -1~1 범위)
            fps: 비디오 FPS

        Returns:
            features: (1, 50, 384) 특징 벡터 (단일 프레임용)
        """
        # 전체 오디오 처리 후 첫 프레임 반환
        features = self.extract_full_audio(audio_chunk, fps=fps)
        if features.shape[0] > 0:
            return features[0:1]  # (1, 50, 384)
        return torch.zeros(1, 50, self.feature_dim, device=self.device)

    def get_sliced_feature(
        self,
        feature_array: torch.Tensor,
        vid_idx: int,
        fps: int = 25
    ) -> torch.Tensor:
        """
        비디오 프레임 인덱스에 해당하는 오디오 특징 반환

        Args:
            feature_array: (total_frames, 50, 384) 전체 오디오 특징
            vid_idx: 비디오 프레임 인덱스
            fps: 비디오 FPS

        Returns:
            sliced_features: (1, 50, 384) 해당 프레임의 오디오 특징
        """
        total_frames = feature_array.shape[0]

        # 인덱스 클램핑
        idx = max(0, min(vid_idx, total_frames - 1))

        return feature_array[idx:idx+1]  # (1, 50, 384)


class PositionalEncoding(nn.Module):
    """
    MuseTalk 공식 Positional Encoding

    오디오 특징에 위치 정보를 추가하여 시퀀스 내 위치를 인코딩합니다.
    Sinusoidal positional encoding 사용.
    """

    def __init__(self, d_model: int = 384, max_len: int = 5000):
        super().__init__()
        import math

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) 오디오 특징

        Returns:
            x + pe: 위치 인코딩이 추가된 특징
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MuseTalkUNet(nn.Module):
    """
    MuseTalk UNet 래퍼 클래스

    공식 MuseTalk은 diffusers의 UNet2DConditionModel을 사용합니다.
    musetalk.json 설정에 따라 UNet을 로드합니다.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        in_channels: int = 8,  # 4 (masked latent) + 4 (mask)
        out_channels: int = 4,
        audio_embed_dim: int = 384,  # Whisper tiny dimension
        cross_attention_dim: int = 384,
        use_diffusers_unet: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_diffusers_unet = use_diffusers_unet
        self.unet = None

        # Positional Encoding for audio features (MuseTalk official)
        self.pe = PositionalEncoding(d_model=audio_embed_dim)

        # Timestep embedding (MuseTalk uses t=0 for single-step inference)
        self.register_buffer('zero_timestep', torch.tensor([0], dtype=torch.long))

        if use_diffusers_unet and DIFFUSERS_AVAILABLE:
            try:
                from diffusers import UNet2DConditionModel

                if config_path and os.path.exists(config_path):
                    # Load from config file
                    import json
                    with open(config_path, 'r') as f:
                        unet_config = json.load(f)

                    logger.info(f"Loading UNet2DConditionModel from config: {config_path}")
                    self.unet = UNet2DConditionModel(**unet_config)
                else:
                    # Default config matching musetalk.json
                    logger.info("Creating UNet2DConditionModel with default MuseTalk config")
                    self.unet = UNet2DConditionModel(
                        sample_size=64,
                        in_channels=8,
                        out_channels=4,
                        center_input_sample=False,
                        flip_sin_to_cos=True,
                        freq_shift=0,
                        down_block_types=(
                            "CrossAttnDownBlock2D",
                            "CrossAttnDownBlock2D",
                            "CrossAttnDownBlock2D",
                            "DownBlock2D"
                        ),
                        up_block_types=(
                            "UpBlock2D",
                            "CrossAttnUpBlock2D",
                            "CrossAttnUpBlock2D",
                            "CrossAttnUpBlock2D"
                        ),
                        block_out_channels=(320, 640, 1280, 1280),
                        layers_per_block=2,
                        downsample_padding=1,
                        mid_block_scale_factor=1,
                        act_fn="silu",
                        norm_num_groups=32,
                        norm_eps=1e-5,
                        cross_attention_dim=384,
                        attention_head_dim=8
                    )

            except Exception as e:
                logger.error(f"Failed to create UNet2DConditionModel: {e}")
                self.unet = None
                self.use_diffusers_unet = False

        if not self.use_diffusers_unet or self.unet is None:
            # Fallback: simple UNet
            logger.warning("Using simplified fallback UNet")
            self._build_simple_unet(in_channels, out_channels, cross_attention_dim)

    def _build_simple_unet(self, in_ch: int, out_ch: int, cross_attn_dim: int):
        """Fallback simple UNet for testing"""
        base_ch = 64
        self.simple_unet = nn.Sequential(
            # Encoder
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            # Bottleneck
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.SiLU(),
            # Decoder
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, padding=1)
        )

    def load_checkpoint(self, checkpoint_path: str, device: str = "cuda"):
        """
        체크포인트 로드

        Args:
            checkpoint_path: 체크포인트 파일 경로
            device: 디바이스
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            logger.info(f"Loading UNet checkpoint: {checkpoint_path}")

            # Load state dict
            if checkpoint_path.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
                state_dict = load_safetensors(checkpoint_path, device=device)
            else:
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

            if self.unet is not None:
                # MuseTalk 체크포인트는 'unet.' 접두사 없이 저장됨
                # MuseTalkUNet 클래스의 self.unet (diffusers UNet)에 직접 로드
                # self는 MuseTalkUNet이고, self.unet은 UNet2DConditionModel

                # 체크포인트를 diffusers UNet에 직접 로드 (접두사 변환 없이)
                missing, unexpected = self.unet.load_state_dict(state_dict, strict=False)

                # 실제 로드된 키 수 계산
                loaded_keys = len(state_dict) - len(missing)
                logger.info(f"Loaded {loaded_keys}/{len(state_dict)} keys from checkpoint")

                if missing:
                    logger.warning(f"Missing keys: {len(missing)}")
                    for k in missing[:5]:
                        logger.warning(f"  - {k}")
                if unexpected:
                    logger.warning(f"Unexpected keys: {len(unexpected)}")
                    for k in unexpected[:5]:
                        logger.warning(f"  - {k}")

                if loaded_keys > 0:
                    logger.info("UNet checkpoint loaded successfully")
                    return True
                else:
                    logger.error("No keys were loaded from checkpoint!")
                    return False
            else:
                logger.warning("No UNet model to load checkpoint into")
                return False

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False

    def forward(
        self,
        x: torch.Tensor,
        audio_embeddings: torch.Tensor,
        timestep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (B, in_channels, H, W) - masked latent + ref latent concat
            audio_embeddings: (B, seq_len, audio_dim) - Whisper features (384-dim)
            timestep: Optional timestep (default: 0 for single-step)

        Returns:
            output: (B, out_channels, H, W) - generated latent
        """
        batch_size = x.shape[0]

        # MuseTalk: Whisper 384-dim == UNet cross_attention_dim 384
        # audio_embeddings shape: (B, seq_len, 384) or (seq_len, 384)
        if audio_embeddings.dim() == 2:
            audio_embeddings = audio_embeddings.unsqueeze(0)  # (1, seq_len, 384)

        # Positional Encoding 적용 (MuseTalk 공식 방식)
        audio_embeddings = self.pe(audio_embeddings)

        if self.unet is not None:
            # Use diffusers UNet
            if timestep is None:
                timestep = self.zero_timestep.expand(batch_size).to(x.device)

            # UNet2DConditionModel forward
            # encoder_hidden_states: (B, seq_len, cross_attention_dim=384)
            output = self.unet(
                sample=x,
                timestep=timestep,
                encoder_hidden_states=audio_embeddings  # 직접 전달
            ).sample

            return output
        else:
            # Use simple fallback UNet
            return self.simple_unet(x)


class MuseTalkAvatar:
    """
    MuseTalk 실시간 아바타

    MuseTalk 파이프라인:
    1. 소스 이미지를 VAE로 인코딩하여 latent 생성
    2. 입 영역을 마스킹
    3. Whisper로 오디오에서 특징 추출
    4. UNet으로 마스킹된 latent + 오디오 특징 → 새로운 latent 생성
    5. VAE 디코더로 latent → 이미지 복원
    """

    def __init__(self, config: Optional[MuseTalkConfig] = None):
        """
        Args:
            config: MuseTalk 설정
        """
        self.config = config or MuseTalkConfig()

        # 디바이스 자동 감지
        self.device = self._detect_device()
        self.dtype = torch.float16 if (self.config.use_fp16 and self.device == "cuda") else torch.float32
        logger.info(f"Using device: {self.device}, dtype: {self.dtype}")

        # 모델 초기화
        self.vae = None
        self.unet = None
        self.audio_encoder = None

        # 얼굴 랜드마크 감지기
        self.landmark_detector = None

        # 얼굴 향상 모델
        self.face_enhancer = None

        # FaceParsing 모델 (공식 MuseTalk V1.5 블렌딩)
        self.face_parsing: Optional[OfficialFaceParsing] = None

        # 소스 이미지 데이터 (단일 이미지 모드)
        self.source_image = None  # Original BGR image
        self.source_image_rgb = None  # RGB tensor
        self.source_landmarks = None
        self.face_bbox = None
        self.mouth_bbox = None

        # 영상/이미지 시퀀스 로더 (영상 모드)
        self.video_loader: Optional[VideoFrameLoader] = None
        self.image_sequence_loader: Optional[ImageSequenceLoader] = None
        self.current_frame_idx = 0  # 현재 프레임 인덱스

        # 캐시된 latent (성능 향상)
        self.source_latent = None
        self.mask_latent = None

        # 프레임 버퍼
        self.frame_buffer = deque(maxlen=self.config.buffer_size)

        # 현재 상태
        self.current_state = AvatarState.IDLE

        # 성능 메트릭
        self.fps_counter = deque(maxlen=30)

        # 모델 로드 상태
        self._models_loaded = False

        logger.info("MuseTalkAvatar initialized")

    def _detect_device(self) -> str:
        """디바이스 자동 감지"""
        if self.config.device:
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @contextmanager
    def _inference_mode(self):
        """추론 모드 컨텍스트 매니저"""
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast(enabled=self.config.use_fp16):
                    yield
            else:
                yield

    def load_models(self):
        """모델 로드"""
        model_dir = Path(self.config.model_dir)

        # 상대 경로일 경우 프로젝트 루트 기준으로 절대 경로 변환
        if not model_dir.is_absolute():
            # 현재 파일 기준으로 프로젝트 루트 찾기
            project_root = Path(__file__).parent.parent.parent  # src/avatar -> src -> project_root
            model_dir = project_root / model_dir
            logger.info(f"Model directory resolved to: {model_dir}")

        # 1. VAE 로드
        self._load_vae(model_dir)

        # 2. UNet 로드
        self._load_unet(model_dir)

        # 3. Whisper 오디오 인코더 로드
        self._load_audio_encoder()

        # 4. 얼굴 향상 모델 로드 (선택)
        if self.config.use_face_enhance:
            self._load_face_enhancer()

        # 5. FaceParsing 모델 로드 (공식 MuseTalk V1.5 블렌딩)
        if self.config.use_face_parsing:
            self._load_face_parsing(model_dir)

        self._models_loaded = True
        logger.info("All models loaded successfully")

    def _load_vae(self, model_dir: Path):
        """VAE 모델 로드"""
        logger.info("Loading VAE model...")

        if not DIFFUSERS_AVAILABLE:
            logger.error("Diffusers not available. VAE cannot be loaded.")
            return

        vae_path = model_dir / self.config.vae_type

        try:
            # HuggingFace에서 직접 로드 시도
            if vae_path.exists():
                self.vae = AutoencoderKL.from_pretrained(
                    str(vae_path),
                    torch_dtype=self.dtype
                )
                logger.info(f"VAE loaded from local: {vae_path}")
            else:
                # HuggingFace Hub에서 다운로드
                self.vae = AutoencoderKL.from_pretrained(
                    f"stabilityai/{self.config.vae_type}",
                    torch_dtype=self.dtype
                )
                logger.info(f"VAE loaded from HuggingFace: stabilityai/{self.config.vae_type}")

            self.vae = self.vae.to(self.device)
            self.vae.eval()

            # VAE는 freeze
            for param in self.vae.parameters():
                param.requires_grad = False

        except Exception as e:
            logger.error(f"Failed to load VAE: {e}")
            self.vae = None

    def _load_unet(self, model_dir: Path):
        """UNet 모델 로드"""
        logger.info("Loading UNet model...")

        unet_path = model_dir / self.config.unet_checkpoint
        # V1.5 모델이면 V1.5 config 사용, 아니면 기본 config 사용
        if "V15" in self.config.unet_checkpoint or "v15" in self.config.unet_checkpoint:
            config_path = model_dir / "musetalk" / "musetalkV15" / "musetalk.json"
            logger.info("Using MuseTalk V1.5 model")
        else:
            config_path = model_dir / "musetalk" / "musetalk.json"

        try:
            # UNet 초기화 (config 파일 사용)
            self.unet = MuseTalkUNet(
                config_path=str(config_path) if config_path.exists() else None,
                in_channels=8,  # 4 (masked latent) + 4 (mask)
                out_channels=4,
                audio_embed_dim=self.config.whisper_feature_dim,
                cross_attention_dim=self.config.whisper_feature_dim
            )

            # 체크포인트 로드 (새 메서드 사용)
            if unet_path.exists():
                self.unet.load_checkpoint(str(unet_path), device=self.device)
                logger.info(f"UNet loaded from: {unet_path}")
            else:
                logger.warning(f"UNet checkpoint not found: {unet_path}")
                logger.warning("Using randomly initialized UNet (for testing only)")

            self.unet = self.unet.to(self.device)

            if self.dtype == torch.float16:
                self.unet = self.unet.half()

            self.unet.eval()

            # Freeze UNet parameters
            for param in self.unet.parameters():
                param.requires_grad = False

        except Exception as e:
            logger.error(f"Failed to load UNet: {e}")
            import traceback
            traceback.print_exc()
            self.unet = None

    def _load_audio_encoder(self):
        """Whisper 오디오 인코더 로드"""
        logger.info("Loading Whisper audio encoder...")

        # 로컬 Whisper 모델 경로 확인
        model_dir = Path(self.config.model_dir)
        whisper_local_path = model_dir / "whisper" / f"{self.config.whisper_model}.pt"

        self.audio_encoder = AudioFeatureExtractor(
            model_name=self.config.whisper_model,
            device=self.device,
            use_fp16=self.config.use_fp16,
            local_model_path=str(whisper_local_path) if whisper_local_path.exists() else None
        )

    def _load_face_enhancer(self):
        """얼굴 향상 모델 로드 (GFPGAN)"""
        if not GFPGAN_AVAILABLE:
            logger.warning("GFPGAN not available. Skipping face enhancement.")
            return

        try:
            model_path = Path(self.config.model_dir) / "gfpgan" / "GFPGANv1.4.pth"

            if not model_path.exists():
                logger.warning(f"GFPGAN model not found: {model_path}")
                return

            self.face_enhancer = GFPGANer(
                model_path=str(model_path),
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )

            logger.info("Face enhancer (GFPGAN) loaded")
        except Exception as e:
            logger.warning(f"Failed to load face enhancer: {e}")
            self.face_enhancer = None

    def _load_face_parsing(self, model_dir: Path):
        """
        FaceParsing 모델 로드 (공식 MuseTalk V1.5 블렌딩용)

        BiSeNet 모델을 사용하여 얼굴의 각 부위를 정교하게 분할합니다.
        이를 통해 입/턱 영역만 선택적으로 교체할 수 있습니다.
        """
        logger.info("Loading FaceParsing model (BiSeNet)...")

        # FaceParsing 모델 경로
        resnet_path = model_dir / "face-parse-bisent" / "resnet18-5c106cde.pth"
        bisenet_path = model_dir / "face-parse-bisent" / "79999_iter.pth"

        if not resnet_path.exists() or not bisenet_path.exists():
            # 대체 경로 시도
            alt_resnet = model_dir / "face-parse-bisenet" / "resnet18-5c106cde.pth"
            alt_bisenet = model_dir / "face-parse-bisenet" / "79999_iter.pth"

            if alt_resnet.exists() and alt_bisenet.exists():
                resnet_path = alt_resnet
                bisenet_path = alt_bisenet
            else:
                logger.warning(
                    f"FaceParsing model files not found:\n"
                    f"  resnet: {resnet_path}\n"
                    f"  bisenet: {bisenet_path}\n"
                    f"FaceParsing will be disabled, using simple elliptical mask instead."
                )
                return

        try:
            self.face_parsing = OfficialFaceParsing(
                resnet_path=str(resnet_path),
                model_path=str(bisenet_path),
                device=self.device,
                left_cheek_width=self.config.face_parsing_left_cheek_width,
                right_cheek_width=self.config.face_parsing_right_cheek_width
            )

            if self.face_parsing.is_available():
                logger.info("FaceParsing (BiSeNet) loaded successfully")
            else:
                logger.warning("FaceParsing initialization failed, using simple mask")
                self.face_parsing = None

        except Exception as e:
            logger.warning(f"Failed to load FaceParsing: {e}")
            self.face_parsing = None

    def load_source_image(self, image_path: Optional[str] = None):
        """
        소스 이미지 로드 및 전처리

        Args:
            image_path: 이미지 경로 (None이면 config에서 가져옴)
        """
        path = image_path or self.config.source_image_path

        logger.info(f"Loading source image from {path}")

        # 이미지 로드
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        # 해상도 조정
        target_size = self.config.resolution
        image = self._resize_image(image, target_size)

        self.source_image = image

        # RGB 텐서로 변환
        self.source_image_rgb = self._image_to_tensor(image)

        # 얼굴 랜드마크 추출
        if self.landmark_detector is None:
            self.landmark_detector = FaceLandmarkDetector()

        self.source_landmarks = self.landmark_detector.detect(image)

        if self.source_landmarks is None:
            raise ValueError("No face detected in source image")

        # 얼굴 바운딩 박스 계산
        self.face_bbox = self.landmark_detector.get_face_bbox(
            self.source_landmarks,
            expand_ratio=self.config.face_expand_ratio
        )

        # 입 영역 바운딩 박스 계산
        self.mouth_bbox = self._get_mouth_bbox(self.source_landmarks)

        # 소스 이미지 latent 캐싱
        if self.config.cache_source_latent and self.vae is not None:
            self._cache_source_latent()

        logger.info(
            f"Source image loaded: {image.shape}, "
            f"Face bbox: {self.face_bbox}, "
            f"Mouth bbox: {self.mouth_bbox}"
        )

    def load_source(self, source_path: Optional[str] = None) -> bool:
        """
        소스 로드 (이미지, 영상, 이미지 시퀀스 자동 감지)

        config.source_type에 따라 적절한 로더를 사용합니다.
        - IMAGE: 단일 이미지 로드
        - VIDEO: 영상 파일 로드
        - IMAGE_DIR: 이미지 시퀀스 로드

        Args:
            source_path: 소스 경로 (None이면 config에서 가져옴)

        Returns:
            bool: 성공 여부
        """
        # 랜드마크 감지기 초기화
        if self.landmark_detector is None:
            self.landmark_detector = FaceLandmarkDetector()

        source_type = self.config.source_type

        if source_type == SourceType.VIDEO:
            return self._load_video_source(source_path)
        elif source_type == SourceType.IMAGE_DIR:
            return self._load_image_sequence_source(source_path)
        else:  # SourceType.IMAGE
            try:
                self.load_source_image(source_path)
                return True
            except Exception as e:
                logger.error(f"Failed to load source image: {e}")
                return False

    def _load_video_source(self, video_path: Optional[str] = None) -> bool:
        """
        영상 소스 로드

        Args:
            video_path: 영상 파일 경로

        Returns:
            bool: 성공 여부
        """
        path = video_path or self.config.source_video_path

        logger.info(f"Loading video source: {path}")

        # VideoFrameLoader 생성 (얼굴 기반 크롭 설정 포함)
        self.video_loader = VideoFrameLoader(
            video_path=path,
            target_fps=self.config.video_fps,
            resolution=self.config.resolution,
            loop=self.config.loop_video,
            preload=self.config.preload_frames,
            max_frames=self.config.max_preload_frames,
            # 얼굴 기반 크롭 설정
            use_face_crop=self.config.use_face_crop,
            face_crop_ratio=self.config.face_crop_ratio,
            face_crop_padding=self.config.face_crop_padding,
            min_face_ratio=self.config.min_face_ratio,
            fixed_crop_region=self.config.fixed_crop_region
        )

        # 영상 로드
        success = self.video_loader.load(landmark_detector=self.landmark_detector)

        if not success:
            logger.error(f"Failed to load video: {path}")
            self.video_loader = None
            return False

        # 첫 프레임을 기준 이미지로 설정 (호환성)
        first_frame, first_landmarks = self.video_loader.get_frame(0)
        self.source_image = first_frame
        self.source_image_rgb = self._image_to_tensor(first_frame)
        self.source_landmarks = first_landmarks

        if first_landmarks is not None:
            self.face_bbox = self.landmark_detector.get_face_bbox(
                first_landmarks,
                expand_ratio=self.config.face_expand_ratio
            )
            self.mouth_bbox = self._get_mouth_bbox(first_landmarks)
        else:
            logger.warning("No face detected in first frame via MediaPipe, using fallback")
            # Fallback: OpenCV Haar Cascade로 얼굴 검출 후 입 영역 추정
            self._set_mouth_bbox_fallback(first_frame)

        # 첫 프레임 latent 캐싱
        if self.config.cache_source_latent and self.vae is not None:
            self._cache_source_latent()

        # 영상 프레임 latent 미리 캐싱 (옵션)
        if self.config.cache_video_frames and self.vae is not None:
            self._cache_video_latents()

        # 얼굴 크롭 상태 로그
        if self.video_loader.is_face_crop_enabled():
            logger.info(
                f"Video source loaded: {len(self.video_loader)} frames, "
                f"Duration: {self.video_loader.get_duration():.1f}s, "
                f"Face crop: ENABLED (auto-detected small face)"
            )
        else:
            logger.info(
                f"Video source loaded: {len(self.video_loader)} frames, "
                f"Duration: {self.video_loader.get_duration():.1f}s"
            )
        return True

    def _load_image_sequence_source(self, image_dir: Optional[str] = None) -> bool:
        """
        이미지 시퀀스 소스 로드

        Args:
            image_dir: 이미지 디렉토리 경로

        Returns:
            bool: 성공 여부
        """
        path = image_dir or self.config.source_image_dir

        if not path:
            logger.error("Image directory not specified")
            return False

        logger.info(f"Loading image sequence source: {path}")

        # ImageSequenceLoader 생성
        self.image_sequence_loader = ImageSequenceLoader(
            image_dir=path,
            target_fps=self.config.video_fps,
            resolution=self.config.resolution,
            loop=self.config.loop_video
        )

        # 이미지 시퀀스 로드
        success = self.image_sequence_loader.load(landmark_detector=self.landmark_detector)

        if not success:
            logger.error(f"Failed to load image sequence: {path}")
            self.image_sequence_loader = None
            return False

        # 첫 프레임을 기준 이미지로 설정
        first_frame, first_landmarks = self.image_sequence_loader.get_frame(0)
        self.source_image = first_frame
        self.source_image_rgb = self._image_to_tensor(first_frame)
        self.source_landmarks = first_landmarks

        if first_landmarks is not None:
            self.face_bbox = self.landmark_detector.get_face_bbox(
                first_landmarks,
                expand_ratio=self.config.face_expand_ratio
            )
            self.mouth_bbox = self._get_mouth_bbox(first_landmarks)

        logger.info(
            f"Image sequence loaded: {len(self.image_sequence_loader)} frames"
        )
        return True

    def _cache_video_latents(self):
        """
        비디오 프레임의 VAE latent 미리 캐싱

        프레임 수가 많으면 시간이 오래 걸릴 수 있습니다.
        """
        if self.video_loader is None or self.vae is None:
            return

        frame_count = len(self.video_loader)
        logger.info(f"Caching video latents for {frame_count} frames...")

        with self._inference_mode():
            for i in range(frame_count):
                # 이미 캐싱된 latent가 있으면 스킵
                if self.video_loader.get_cached_latent(i) is not None:
                    continue

                frame, _ = self.video_loader.get_frame(i)
                frame_tensor = self._image_to_tensor(frame)

                # VAE 인코딩
                latent_dist = self.vae.encode(frame_tensor).latent_dist
                latent = latent_dist.sample() * 0.18215

                # 캐싱
                self.video_loader.cache_latent(i, latent)

                if (i + 1) % 50 == 0:
                    logger.info(f"  Cached {i + 1}/{frame_count} latents")

        logger.info(f"Video latents cached: {frame_count} frames")

    def get_current_source_frame(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[torch.Tensor]]:
        """
        현재 소스 프레임 가져오기 (영상 모드에서 사용)

        Returns:
            (frame, landmarks, cached_latent): 프레임, 랜드마크, 캐시된 latent
        """
        if self.video_loader is not None:
            # 영상 모드
            idx = self.current_frame_idx
            frame, landmarks = self.video_loader.get_frame(idx)
            cached_latent = self.video_loader.get_cached_latent(idx)

            # 다음 프레임으로 이동
            self.current_frame_idx += 1
            if self.current_frame_idx >= len(self.video_loader):
                if self.config.loop_video:
                    self.current_frame_idx = 0
                else:
                    self.current_frame_idx = len(self.video_loader) - 1

            return frame, landmarks, cached_latent

        elif self.image_sequence_loader is not None:
            # 이미지 시퀀스 모드
            idx = self.current_frame_idx
            frame, landmarks = self.image_sequence_loader.get_frame(idx)

            self.current_frame_idx += 1
            if self.current_frame_idx >= len(self.image_sequence_loader):
                if self.config.loop_video:
                    self.current_frame_idx = 0
                else:
                    self.current_frame_idx = len(self.image_sequence_loader) - 1

            return frame, landmarks, None

        else:
            # 단일 이미지 모드
            return self.source_image, self.source_landmarks, self.source_latent

    def _resize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        이미지 크기 조정 (종횡비 유지)

        Args:
            image: 입력 이미지
            target_size: 목표 크기

        Returns:
            resized_image: 리사이즈된 이미지
        """
        h, w = image.shape[:2]

        # 종횡비 계산
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))

        # 리사이즈
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 패딩 (정사각형으로)
        if new_h != target_size or new_w != target_size:
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            resized = canvas

        return resized

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        BGR 이미지를 RGB 텐서로 변환

        Args:
            image: BGR numpy array (H, W, 3), 0-255

        Returns:
            tensor: RGB tensor (1, 3, H, W), -1~1 범위
        """
        # BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # numpy -> tensor
        tensor = torch.from_numpy(rgb).float()

        # (H, W, 3) -> (3, H, W)
        tensor = tensor.permute(2, 0, 1)

        # 0-255 -> -1~1
        tensor = (tensor / 127.5) - 1.0

        # 배치 차원 추가
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device, dtype=self.dtype)

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        RGB 텐서를 BGR 이미지로 변환

        Args:
            tensor: RGB tensor (1, 3, H, W) or (3, H, W), -1~1 범위

        Returns:
            image: BGR numpy array (H, W, 3), 0-255
        """
        # 배치 차원 제거
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # -1~1 -> 0-255
        tensor = ((tensor + 1.0) * 127.5).clamp(0, 255)

        # (3, H, W) -> (H, W, 3)
        tensor = tensor.permute(1, 2, 0)

        # tensor -> numpy
        image = tensor.cpu().numpy().astype(np.uint8)

        # RGB -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def _get_mouth_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        MediaPipe 랜드마크에서 입 영역 바운딩 박스 계산

        MediaPipe Face Mesh 입술 랜드마크 인덱스:
        - 외부 입술: 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37
        - 내부 입술: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95

        Args:
            landmarks: (468, 3) 랜드마크 배열

        Returns:
            (x1, y1, x2, y2): 입 영역 바운딩 박스
        """
        # 입술 관련 주요 랜드마크 인덱스
        mouth_indices = [
            # 외부 입술
            0, 267, 269, 270, 409, 291, 375, 321, 405, 314,
            17, 84, 181, 91, 146, 61, 185, 40, 39, 37,
            # 내부 입술
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95
        ]

        # 입술 랜드마크 추출
        mouth_landmarks = landmarks[mouth_indices]

        # 바운딩 박스 계산
        x_min = int(mouth_landmarks[:, 0].min())
        x_max = int(mouth_landmarks[:, 0].max())
        y_min = int(mouth_landmarks[:, 1].min())
        y_max = int(mouth_landmarks[:, 1].max())

        # 확장
        width = x_max - x_min
        height = y_max - y_min
        expand = self.config.mouth_mask_expand

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        new_width = int(width * expand)
        new_height = int(height * expand)

        # bbox_shift 적용 (입 열림 정도 조절)
        y_center += self.config.bbox_shift

        x1 = max(0, x_center - new_width // 2)
        y1 = max(0, y_center - new_height // 2)
        x2 = min(self.config.resolution, x_center + new_width // 2)
        y2 = min(self.config.resolution, y_center + new_height // 2)

        return x1, y1, x2, y2

    def _set_mouth_bbox_fallback(self, frame: np.ndarray):
        """
        OpenCV Haar Cascade를 사용하여 입 영역 추정 (MediaPipe 실패 시 fallback)

        얼굴을 검출하고, 얼굴 하단 영역을 입으로 추정합니다.

        Args:
            frame: BGR 이미지 (numpy array)
        """
        # OpenCV 얼굴 검출
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            # 얼굴도 검출되지 않으면 이미지 중앙 하단을 입 영역으로 가정
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            # 이미지 중앙 하단 영역
            mouth_width = w // 3
            mouth_height = h // 6
            x1 = center_x - mouth_width // 2
            y1 = center_y + h // 8  # 중앙보다 약간 아래
            x2 = x1 + mouth_width
            y2 = y1 + mouth_height
            self.mouth_bbox = (x1, y1, x2, y2)
            logger.warning(f"No face detected, using center-bottom as mouth bbox: {self.mouth_bbox}")
            return

        # 가장 큰 얼굴 선택
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # 얼굴 바운딩 박스 설정
        self.face_bbox = (x, y, x + w, y + h)

        # 입 영역 추정: 얼굴 하단 1/3 영역
        # 입은 보통 얼굴 높이의 65-85% 위치에 있음
        mouth_y_start = y + int(h * 0.60)
        mouth_y_end = y + int(h * 0.90)
        mouth_x_start = x + int(w * 0.20)
        mouth_x_end = x + int(w * 0.80)

        # 확장 비율 적용
        expand = self.config.mouth_mask_expand
        mouth_cx = (mouth_x_start + mouth_x_end) // 2
        mouth_cy = (mouth_y_start + mouth_y_end) // 2
        mouth_w = int((mouth_x_end - mouth_x_start) * expand)
        mouth_h = int((mouth_y_end - mouth_y_start) * expand)

        # bbox_shift 적용
        mouth_cy += self.config.bbox_shift

        x1 = max(0, mouth_cx - mouth_w // 2)
        y1 = max(0, mouth_cy - mouth_h // 2)
        x2 = min(self.config.resolution, mouth_cx + mouth_w // 2)
        y2 = min(self.config.resolution, mouth_cy + mouth_h // 2)

        self.mouth_bbox = (x1, y1, x2, y2)
        logger.info(f"Fallback mouth bbox set via Haar Cascade: {self.mouth_bbox}")

    def _cache_source_latent(self):
        """소스 이미지의 VAE latent 캐싱"""
        if self.vae is None or self.source_image_rgb is None:
            return

        logger.info("Caching source image latent...")

        with self._inference_mode():
            # VAE 인코딩
            latent_dist = self.vae.encode(self.source_image_rgb).latent_dist
            self.source_latent = latent_dist.sample() * 0.18215  # VAE scaling factor

        logger.info(f"Source latent cached: {self.source_latent.shape}")

    def _create_mouth_mask(self) -> torch.Tensor:
        """
        입 영역 마스크 생성 (latent space)

        Returns:
            mask: (1, 1, latent_h, latent_w) 마스크 텐서
                  입 영역 = 1, 나머지 = 0
        """
        if self.mouth_bbox is None:
            # 전체 마스크 (fallback)
            return torch.ones(1, 1, self.config.latent_size, self.config.latent_size,
                            device=self.device, dtype=self.dtype)

        x1, y1, x2, y2 = self.mouth_bbox

        # 이미지 공간에서 latent 공간으로 변환 (8x downscale)
        scale = self.config.resolution / self.config.latent_size
        lx1, ly1 = int(x1 / scale), int(y1 / scale)
        lx2, ly2 = int(x2 / scale), int(y2 / scale)

        # 마스크 생성
        mask = torch.zeros(1, 1, self.config.latent_size, self.config.latent_size,
                          device=self.device, dtype=self.dtype)
        mask[:, :, ly1:ly2, lx1:lx2] = 1.0

        # Gaussian blur로 부드럽게
        if mask.sum() > 0:
            # 간단한 box blur
            kernel_size = 3
            mask = F.avg_pool2d(
                F.pad(mask, (kernel_size//2,)*4, mode='replicate'),
                kernel_size, stride=1
            )

        return mask

    def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        timestamp: Optional[float] = None
    ) -> VideoFrame:
        """
        오디오 청크를 처리하여 비디오 프레임 생성

        Args:
            audio_chunk: PCM 오디오 데이터 (16kHz, mono, float32)
            timestamp: 타임스탬프 (초)

        Returns:
            VideoFrame: 생성된 비디오 프레임
        """
        start_time = time.time()

        # 오디오 특징 추출
        if self.audio_encoder and self.audio_encoder.model is not None:
            audio_features = self.audio_encoder.extract(audio_chunk)
        else:
            # 더미 특징
            audio_features = torch.zeros(1, self.config.whisper_feature_dim,
                                        device=self.device, dtype=self.dtype)

        # 상태 결정 (오디오 에너지 기반)
        audio_energy = np.abs(audio_chunk).mean() if len(audio_chunk) > 0 else 0
        if audio_energy > 0.01:
            self.current_state = AvatarState.SPEAKING
        else:
            self.current_state = AvatarState.IDLE

        # 프레임 생성
        frame = self._generate_frame(audio_features)

        # 얼굴 향상
        if self.config.use_face_enhance and self.face_enhancer:
            frame = self._enhance_face(frame)

        # 프레임 스무딩
        if self.config.smooth_factor > 0 and len(self.frame_buffer) > 0:
            frame = self._smooth_frame(frame)

        # 프레임 버퍼에 추가
        self.frame_buffer.append(frame)

        # FPS 계산
        elapsed = time.time() - start_time
        self.fps_counter.append(1.0 / elapsed if elapsed > 0 else 0)

        current_fps = np.mean(self.fps_counter) if self.fps_counter else 0

        # VideoFrame 생성
        video_frame = VideoFrame(
            frame=frame,
            timestamp=timestamp or time.time(),
            state=self.current_state,
            metadata={
                "fps": current_fps,
                "audio_energy": float(audio_energy),
                "processing_time_ms": elapsed * 1000
            }
        )

        return video_frame

    def _generate_frame(self, audio_features: torch.Tensor) -> np.ndarray:
        """
        오디오 특징으로부터 프레임 생성 (MuseTalk 파이프라인)

        영상 모드에서는 현재 프레임을 기준으로 립싱크를 생성합니다.
        원본 영상의 움직임(몸, 머리)은 보존되고, 입 영역만 교체됩니다.

        Args:
            audio_features: (seq_len, feature_dim) 오디오 특징 벡터

        Returns:
            frame: BGR 프레임 (numpy array)
        """
        # 영상 모드: 현재 프레임 가져오기
        if self.video_loader is not None or self.image_sequence_loader is not None:
            current_frame, current_landmarks, cached_latent = self.get_current_source_frame()

            if current_frame is None:
                return np.zeros(
                    (self.config.resolution, self.config.resolution, 3),
                    dtype=np.uint8
                )

            # 현재 프레임의 랜드마크로 입 영역 업데이트
            if current_landmarks is not None:
                self.mouth_bbox = self._get_mouth_bbox(current_landmarks)
        else:
            # 단일 이미지 모드
            current_frame = self.source_image
            cached_latent = self.source_latent

        # 소스 이미지가 없으면 검은 프레임 반환
        if current_frame is None:
            return np.zeros(
                (self.config.resolution, self.config.resolution, 3),
                dtype=np.uint8
            )

        # 모델이 로드되지 않았으면 현재 프레임 반환
        if self.vae is None or self.unet is None:
            logger.warning("Models not loaded, returning current frame")
            return current_frame.copy()

        try:
            with self._inference_mode():
                # MuseTalk 공식 추론 방식:
                # 1. 이미지 공간에서 하반부를 -1로 마스킹
                # 2. 마스킹된 이미지와 원본 이미지를 각각 VAE 인코딩
                # 3. [masked_latent, ref_latent] concat → UNet
                # 4. UNet 출력을 VAE 디코딩

                # 1. 현재 프레임을 텐서로 변환
                frame_tensor = self._image_to_tensor(current_frame)  # (1, 3, H, W), range [-1, 1]

                # 2. 마스킹된 이미지 생성 (이미지 공간에서 하반부 = -1)
                masked_frame_tensor = frame_tensor.clone()
                h = masked_frame_tensor.shape[2]
                masked_frame_tensor[:, :, h // 2:, :] = -1.0  # 하반부를 -1로 마스킹

                # 3. VAE 인코딩
                # 마스킹된 프레임 인코딩
                masked_latent_dist = self.vae.encode(masked_frame_tensor).latent_dist
                masked_latent = masked_latent_dist.mode() * 0.18215

                # 참조 프레임 인코딩 (원본)
                if cached_latent is not None:
                    ref_latent = cached_latent
                else:
                    ref_latent_dist = self.vae.encode(frame_tensor).latent_dist
                    ref_latent = ref_latent_dist.mode() * 0.18215

                # 4. UNet 입력 준비: [masked_latent, ref_latent] concat
                unet_input = torch.cat([masked_latent, ref_latent], dim=1)  # (1, 8, H, W)

                # 5. 오디오 특징 준비
                if audio_features.dim() == 2:
                    audio_features = audio_features.unsqueeze(0)  # (1, seq_len, dim)

                # 6. UNet 추론 - 전체 프레임 생성
                generated_latent = self.unet(unet_input, audio_features)

                # 7. VAE 디코딩
                generated_latent = generated_latent / 0.18215  # VAE scaling factor 역변환
                decoded = self.vae.decode(generated_latent).sample

                # 8. 후처리: 공식 FaceParsing 블렌딩 또는 타원형 마스킹
                # MuseTalk의 알려진 이슈: 전체 하반부가 블러 처리됨 (GitHub Issue #335)
                # 공식 해결책: FaceParsing으로 입/턱 영역만 정교하게 교체
                h, w = 256, 256

                # UNet 출력을 이미지로 변환
                decoded_img = self._tensor_to_image(decoded)

                # FaceParsing이 사용 가능하면 공식 블렌딩 사용
                if self.face_parsing is not None and self.face_parsing.is_available():
                    # 공식 MuseTalk V1.5 블렌딩 사용
                    # face_box는 전체 256x256 영역
                    face_box = (0, 0, w, h)

                    frame = official_get_image(
                        image=current_frame,
                        face=decoded_img,
                        face_box=face_box,
                        upper_boundary_ratio=self.config.face_parsing_upper_boundary_ratio,
                        expand=self.config.face_parsing_expand,
                        mode=self.config.face_parsing_mode,
                        fp=self.face_parsing
                    )
                else:
                    # FaceParsing 불가능 시 간단한 타원형 마스크 사용
                    frame = _simple_blend(
                        image=current_frame,
                        face=decoded_img,
                        face_box=(0, 0, w, h)
                    )

                return frame

        except Exception as e:
            logger.error(f"Frame generation failed: {e}")
            import traceback
            traceback.print_exc()
            return current_frame.copy()

    def composite_to_original(
        self,
        lipsync_frame: np.ndarray,
        frame_idx: int
    ) -> Optional[np.ndarray]:
        """
        립싱크된 256x256 프레임을 원본 영상에 합성

        MuseTalk은 256x256 크롭 영역에서만 립싱크를 수행합니다.
        이 메서드는 생성된 프레임을 원본 영상에 다시 합성합니다.

        Args:
            lipsync_frame: 립싱크된 256x256 프레임 (BGR)
            frame_idx: 원본 프레임 인덱스

        Returns:
            합성된 원본 해상도 프레임 또는 None (합성 불가 시)
        """
        if self.video_loader is None:
            logger.warning("No video loader available for compositing")
            return lipsync_frame

        # 원본 프레임과 크롭 영역 가져오기
        if frame_idx >= len(self.video_loader.original_frames):
            frame_idx = frame_idx % len(self.video_loader.original_frames)

        original_frame = self.video_loader.original_frames[frame_idx].copy()
        crop_region = self.video_loader.crop_regions[frame_idx] if frame_idx < len(self.video_loader.crop_regions) else None

        if crop_region is None:
            # 크롭 없이 처리된 경우, 립싱크 프레임을 원본 크기로 리사이즈
            h, w = original_frame.shape[:2]
            resized = cv2.resize(lipsync_frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
            return resized

        x1, y1, x2, y2 = crop_region
        crop_w = x2 - x1
        crop_h = y2 - y1

        # 립싱크 프레임을 크롭 영역 크기로 리사이즈
        resized_lipsync = cv2.resize(
            lipsync_frame,
            (crop_w, crop_h),
            interpolation=cv2.INTER_LANCZOS4
        )

        # 원본 프레임에 합성 (경계 블렌딩 적용)
        composite = self._blend_paste(
            original_frame,
            resized_lipsync,
            (x1, y1, x2, y2),
            blend_margin=20  # 경계 블렌딩 픽셀
        )

        return composite

    def _blend_paste(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        region: Tuple[int, int, int, int],
        blend_margin: int = 20
    ) -> np.ndarray:
        """
        블렌딩을 적용하여 전경을 배경에 합성

        경계 부분을 부드럽게 블렌딩하여 자연스러운 합성을 수행합니다.

        Args:
            background: 배경 이미지 (원본)
            foreground: 전경 이미지 (립싱크된 크롭)
            region: 합성 영역 (x1, y1, x2, y2)
            blend_margin: 경계 블렌딩 여백 (픽셀)

        Returns:
            합성된 이미지
        """
        x1, y1, x2, y2 = region
        h, w = foreground.shape[:2]

        # 결과 이미지 복사
        result = background.copy()

        # 블렌드 마스크 생성 (중앙은 1.0, 경계는 0.0~1.0 그라데이션)
        mask = np.ones((h, w), dtype=np.float32)

        if blend_margin > 0:
            # 각 경계에 그라데이션 적용
            for i in range(blend_margin):
                alpha = i / blend_margin
                # 상단
                if i < h:
                    mask[i, :] = min(mask[i, 0], alpha)
                # 하단
                if h - 1 - i >= 0:
                    mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)
                # 좌측
                if i < w:
                    mask[:, i] = np.minimum(mask[:, i], alpha)
                # 우측
                if w - 1 - i >= 0:
                    mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)

        # 3채널 마스크로 확장
        mask_3ch = np.stack([mask, mask, mask], axis=-1)

        # 경계 체크
        bg_h, bg_w = background.shape[:2]
        paste_x1 = max(0, x1)
        paste_y1 = max(0, y1)
        paste_x2 = min(bg_w, x2)
        paste_y2 = min(bg_h, y2)

        # 전경에서 사용할 영역 계산
        fg_x1 = paste_x1 - x1
        fg_y1 = paste_y1 - y1
        fg_x2 = fg_x1 + (paste_x2 - paste_x1)
        fg_y2 = fg_y1 + (paste_y2 - paste_y1)

        # 블렌딩 적용
        fg_region = foreground[fg_y1:fg_y2, fg_x1:fg_x2]
        bg_region = background[paste_y1:paste_y2, paste_x1:paste_x2]
        mask_region = mask_3ch[fg_y1:fg_y2, fg_x1:fg_x2]

        # 알파 블렌딩
        blended = (fg_region.astype(np.float32) * mask_region +
                   bg_region.astype(np.float32) * (1 - mask_region))

        result[paste_y1:paste_y2, paste_x1:paste_x2] = blended.astype(np.uint8)

        return result

    def process_audio_chunk_full_frame(
        self,
        audio_chunk: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Tuple[VideoFrame, Optional[np.ndarray]]:
        """
        오디오 청크를 처리하여 원본 해상도 프레임 생성

        process_audio_chunk와 동일하지만, 추가로 원본 해상도에 합성된
        프레임도 반환합니다.

        Args:
            audio_chunk: PCM 오디오 데이터 (16kHz, mono, float32)
            timestamp: 타임스탬프 (초)

        Returns:
            (VideoFrame, full_frame): 크롭 프레임과 원본 해상도 합성 프레임
        """
        # 현재 프레임 인덱스 저장 (process_audio_chunk에서 증가하기 전)
        current_idx = self.current_frame_idx

        # 기존 립싱크 처리
        video_frame = self.process_audio_chunk(audio_chunk, timestamp)

        # 원본 해상도에 합성
        if video_frame.frame is not None and self.video_loader is not None:
            full_frame = self.composite_to_original(video_frame.frame, current_idx)
        else:
            full_frame = None

        return video_frame, full_frame

    def _enhance_face(self, frame: np.ndarray) -> np.ndarray:
        """
        얼굴 향상 (GFPGAN)

        Args:
            frame: 입력 프레임

        Returns:
            enhanced_frame: 향상된 프레임
        """
        if self.face_enhancer is None:
            return frame

        try:
            _, _, enhanced = self.face_enhancer.enhance(
                frame,
                has_aligned=False,
                only_center_face=True,
                paste_back=True
            )
            return enhanced
        except Exception as e:
            logger.warning(f"Face enhancement failed: {e}")
            return frame

    def _smooth_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임 스무딩 (시간적 일관성)

        Args:
            frame: 현재 프레임

        Returns:
            smoothed_frame: 스무딩된 프레임
        """
        if len(self.frame_buffer) == 0:
            return frame

        # 이전 프레임과 블렌딩
        prev_frame = self.frame_buffer[-1]
        alpha = 1.0 - self.config.smooth_factor

        smoothed = cv2.addWeighted(
            frame, alpha,
            prev_frame, self.config.smooth_factor,
            0
        )

        return smoothed.astype(np.uint8)

    def set_state(self, state: AvatarState):
        """
        아바타 상태 설정

        Args:
            state: 새로운 상태
        """
        if self.current_state != state:
            logger.info(f"Avatar state changed: {self.current_state.value} -> {state.value}")
            self.current_state = state

    def get_idle_frame(self) -> np.ndarray:
        """
        대기 상태 프레임 생성 (미세한 움직임)

        Returns:
            frame: BGR 프레임
        """
        self.set_state(AvatarState.IDLE)

        # 무음 오디오로 프레임 생성
        silent_audio = np.zeros(
            int(self.config.audio_sample_rate * self.config.audio_chunk_duration),
            dtype=np.float32
        )

        video_frame = self.process_audio_chunk(silent_audio)
        return video_frame.frame

    def get_thinking_frame(self) -> np.ndarray:
        """
        생각 중 프레임 생성 (고개 끄덕임 등)

        Returns:
            frame: BGR 프레임
        """
        self.set_state(AvatarState.THINKING)

        # 무음 오디오로 프레임 생성
        silent_audio = np.zeros(
            int(self.config.audio_sample_rate * self.config.audio_chunk_duration),
            dtype=np.float32
        )

        video_frame = self.process_audio_chunk(silent_audio)
        return video_frame.frame

    def reset(self):
        """상태 초기화"""
        self.frame_buffer.clear()
        self.fps_counter.clear()
        self.current_state = AvatarState.IDLE
        self.current_frame_idx = 0  # 프레임 인덱스 초기화

        # 비디오/이미지 시퀀스 로더 리셋
        if self.video_loader is not None:
            self.video_loader.reset()
        if self.image_sequence_loader is not None:
            self.image_sequence_loader.reset()

        logger.info("Avatar state reset")

    def get_stats(self) -> Dict[str, Any]:
        """
        성능 통계 반환

        Returns:
            stats: 통계 정보
        """
        stats = {
            "device": self.device,
            "dtype": str(self.dtype),
            "fps": float(np.mean(self.fps_counter)) if self.fps_counter else 0,
            "buffer_size": len(self.frame_buffer),
            "current_state": self.current_state.value,
            "resolution": self.config.resolution,
            "latent_size": self.config.latent_size,
            "use_fp16": self.config.use_fp16,
            "use_face_enhance": self.config.use_face_enhance,
            "models_loaded": self._models_loaded,
            "vae_loaded": self.vae is not None,
            "unet_loaded": self.unet is not None,
            "audio_encoder_loaded": self.audio_encoder is not None and self.audio_encoder.model is not None,
            "source_image_loaded": self.source_image is not None,
            "source_latent_cached": self.source_latent is not None,
            # 소스 타입 정보
            "source_type": self.config.source_type.value,
            "video_loader_loaded": self.video_loader is not None,
            "image_sequence_loader_loaded": self.image_sequence_loader is not None,
        }

        # 영상 모드 추가 정보
        if self.video_loader is not None:
            stats["video_frame_count"] = len(self.video_loader)
            stats["video_duration"] = self.video_loader.get_duration()
            stats["current_frame_idx"] = self.current_frame_idx
        elif self.image_sequence_loader is not None:
            stats["image_sequence_count"] = len(self.image_sequence_loader)
            stats["current_frame_idx"] = self.current_frame_idx

        return stats

    def generate_video_stream(
        self,
        audio_stream: Union[np.ndarray, List[np.ndarray]],
        fps: int = 25
    ) -> List[VideoFrame]:
        """
        오디오 스트림에서 비디오 프레임 스트림 생성

        Args:
            audio_stream: 전체 오디오 데이터 또는 청크 리스트
            fps: 목표 FPS

        Returns:
            frames: VideoFrame 리스트
        """
        chunk_samples = int(self.config.audio_sample_rate / fps)
        frames = []

        # 단일 오디오 배열인 경우 청크로 분할
        if isinstance(audio_stream, np.ndarray):
            audio_chunks = []
            for i in range(0, len(audio_stream), chunk_samples):
                chunk = audio_stream[i:i + chunk_samples]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                audio_chunks.append(chunk)
        else:
            audio_chunks = audio_stream

        # 각 청크에 대해 프레임 생성
        for i, chunk in enumerate(audio_chunks):
            timestamp = i / fps
            frame = self.process_audio_chunk(chunk, timestamp=timestamp)
            frames.append(frame)

        return frames

    def save_video(
        self,
        frames: List[VideoFrame],
        output_path: str,
        fps: int = 25,
        codec: str = "mp4v"
    ):
        """
        프레임 리스트를 비디오 파일로 저장

        Args:
            frames: VideoFrame 리스트
            output_path: 출력 파일 경로
            fps: FPS
            codec: 비디오 코덱
        """
        if not frames:
            logger.warning("No frames to save")
            return

        # 첫 프레임에서 크기 가져오기
        h, w = frames[0].frame.shape[:2]

        # VideoWriter 초기화
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        try:
            for video_frame in frames:
                writer.write(video_frame.frame)

            logger.info(f"Video saved: {output_path} ({len(frames)} frames, {fps} fps)")

        finally:
            writer.release()

    @torch.no_grad()
    def warmup(self, iterations: int = 3):
        """
        모델 워밍업 (첫 추론 지연 방지)

        Args:
            iterations: 워밍업 반복 횟수
        """
        if not self._models_loaded:
            logger.warning("Models not loaded, skipping warmup")
            return

        logger.info(f"Warming up models ({iterations} iterations)...")

        dummy_audio = np.zeros(int(self.config.audio_sample_rate * 0.1), dtype=np.float32)

        for i in range(iterations):
            _ = self.process_audio_chunk(dummy_audio)

        # 버퍼 초기화
        self.frame_buffer.clear()
        self.fps_counter.clear()

        logger.info("Warmup complete")

    def cleanup(self):
        """리소스 정리"""
        logger.info("Cleaning up MuseTalkAvatar resources...")

        # 모델 해제
        if self.vae is not None:
            del self.vae
            self.vae = None

        if self.unet is not None:
            del self.unet
            self.unet = None

        if self.audio_encoder is not None:
            if hasattr(self.audio_encoder, 'model') and self.audio_encoder.model is not None:
                del self.audio_encoder.model
            del self.audio_encoder
            self.audio_encoder = None

        if self.face_enhancer is not None:
            del self.face_enhancer
            self.face_enhancer = None

        # 비디오/이미지 시퀀스 로더 해제
        if self.video_loader is not None:
            self.video_loader.release()
            self.video_loader = None

        if self.image_sequence_loader is not None:
            self.image_sequence_loader.release()
            self.image_sequence_loader = None

        # 캐시 정리
        self.source_latent = None
        self.source_image = None
        self.source_image_rgb = None
        self.current_frame_idx = 0

        # 버퍼 정리
        self.frame_buffer.clear()
        self.fps_counter.clear()

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._models_loaded = False
        logger.info("Cleanup complete")

    def __enter__(self):
        """Context manager 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.cleanup()
        return False


# 헬퍼 함수
def create_musetalk_avatar(
    source_path: str,
    model_dir: str = "./models/musetalk",
    resolution: int = 256,
    use_face_enhance: bool = False,
    device: Optional[str] = None,
    source_type: Optional[SourceType] = None
) -> MuseTalkAvatar:
    """
    MuseTalk 아바타 생성 및 초기화

    소스 타입은 자동 감지됩니다:
    - .mp4, .avi, .mov, .mkv -> VIDEO
    - 디렉토리 -> IMAGE_DIR
    - 그 외 -> IMAGE

    Args:
        source_path: 소스 경로 (이미지, 영상, 또는 이미지 디렉토리)
        model_dir: 모델 디렉토리
        resolution: 해상도
        use_face_enhance: 얼굴 향상 사용 여부
        device: 디바이스
        source_type: 소스 타입 (None이면 자동 감지)

    Returns:
        MuseTalkAvatar: 초기화된 아바타
    """
    # 소스 타입 자동 감지
    if source_type is None:
        if os.path.isdir(source_path):
            source_type = SourceType.IMAGE_DIR
        elif source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            source_type = SourceType.VIDEO
        else:
            source_type = SourceType.IMAGE

    # 설정 생성
    config = MuseTalkConfig(
        model_dir=model_dir,
        source_type=source_type,
        resolution=resolution,
        use_face_enhance=use_face_enhance,
        device=device
    )

    # 소스 타입에 따라 경로 설정
    if source_type == SourceType.VIDEO:
        config.source_video_path = source_path
    elif source_type == SourceType.IMAGE_DIR:
        config.source_image_dir = source_path
    else:
        config.source_image_path = source_path

    avatar = MuseTalkAvatar(config)
    avatar.load_models()

    # 통합 소스 로드 메서드 사용
    if not avatar.load_source():
        raise ValueError(f"Failed to load source: {source_path}")

    return avatar


def create_musetalk_avatar_from_video(
    video_path: str,
    model_dir: str = "./models/musetalk",
    resolution: int = 256,
    video_fps: int = 25,
    loop: bool = True,
    preload_frames: bool = True,
    cache_latents: bool = True,
    use_face_enhance: bool = False,
    device: Optional[str] = None
) -> MuseTalkAvatar:
    """
    영상 기반 MuseTalk 아바타 생성 및 초기화

    Args:
        video_path: 소스 영상 파일 경로
        model_dir: 모델 디렉토리
        resolution: 해상도
        video_fps: 목표 FPS (MuseTalk 권장: 25)
        loop: 영상 루프 재생
        preload_frames: 프레임 미리 로드
        cache_latents: VAE latent 캐싱
        use_face_enhance: 얼굴 향상 사용 여부
        device: 디바이스

    Returns:
        MuseTalkAvatar: 초기화된 아바타
    """
    config = MuseTalkConfig(
        model_dir=model_dir,
        source_type=SourceType.VIDEO,
        source_video_path=video_path,
        video_fps=video_fps,
        loop_video=loop,
        preload_frames=preload_frames,
        cache_video_frames=cache_latents,
        resolution=resolution,
        use_face_enhance=use_face_enhance,
        device=device
    )

    avatar = MuseTalkAvatar(config)
    avatar.load_models()

    if not avatar.load_source():
        raise ValueError(f"Failed to load video source: {video_path}")

    return avatar


def check_dependencies() -> Dict[str, bool]:
    """
    의존성 확인

    Returns:
        dict: 의존성 이름 -> 설치 여부
    """
    return {
        "torch": True,  # 필수
        "mediapipe": MEDIAPIPE_AVAILABLE,
        "whisper": WHISPER_AVAILABLE,
        "diffusers": DIFFUSERS_AVAILABLE,
        "gfpgan": GFPGAN_AVAILABLE,
        "safetensors": SAFETENSORS_AVAILABLE
    }


# 사용 예시
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuseTalk Avatar Test")
    parser.add_argument("--source", type=str, default="./assets/avatar_source.jpg",
                       help="Source path (image, video, or image directory)")
    parser.add_argument("--source-type", type=str, choices=["image", "video", "image_dir"],
                       default=None, help="Source type (auto-detect if not specified)")
    parser.add_argument("--model-dir", type=str, default="./models/musetalk",
                       help="Model directory")
    parser.add_argument("--resolution", type=int, default=256,
                       help="Output resolution (256, 384, 512)")
    parser.add_argument("--video-fps", type=int, default=25,
                       help="Video source FPS (default: 25)")
    parser.add_argument("--no-loop", action="store_true",
                       help="Disable video loop playback")
    parser.add_argument("--no-preload", action="store_true",
                       help="Disable video frame preloading")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda, mps, cpu)")
    parser.add_argument("--no-fp16", action="store_true",
                       help="Disable FP16")
    parser.add_argument("--face-enhance", action="store_true",
                       help="Enable face enhancement (GFPGAN)")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check dependencies only")
    args = parser.parse_args()

    # 의존성 확인
    print("\n=== 의존성 확인 ===")
    deps = check_dependencies()
    for name, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")

    if args.check_deps:
        exit(0)

    # 소스 타입 결정
    if args.source_type:
        source_type = SourceType(args.source_type)
    elif os.path.isdir(args.source):
        source_type = SourceType.IMAGE_DIR
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        source_type = SourceType.VIDEO
    else:
        source_type = SourceType.IMAGE

    # 설정
    config = MuseTalkConfig(
        source_type=source_type,
        source_image_path=args.source if source_type == SourceType.IMAGE else "",
        source_video_path=args.source if source_type == SourceType.VIDEO else "",
        source_image_dir=args.source if source_type == SourceType.IMAGE_DIR else "",
        model_dir=args.model_dir,
        resolution=args.resolution,
        video_fps=args.video_fps,
        loop_video=not args.no_loop,
        preload_frames=not args.no_preload,
        use_face_enhance=args.face_enhance,
        use_fp16=not args.no_fp16,
        device=args.device
    )

    print(f"\n=== 설정 ===")
    print(f"  Source: {args.source}")
    print(f"  Source type: {source_type.value}")
    print(f"  Model dir: {config.model_dir}")
    print(f"  Resolution: {config.resolution}")
    if source_type == SourceType.VIDEO:
        print(f"  Video FPS: {config.video_fps}")
        print(f"  Loop: {config.loop_video}")
        print(f"  Preload: {config.preload_frames}")
    print(f"  Device: {config.device or 'auto'}")
    print(f"  FP16: {config.use_fp16}")
    print(f"  Face enhance: {config.use_face_enhance}")

    # Context manager 사용
    with MuseTalkAvatar(config) as avatar:
        # 모델 로드
        print("\n=== 모델 로드 ===")
        avatar.load_models()

        # 소스 로드 (통합 메서드)
        print(f"\n=== 소스 로드 ({source_type.value}) ===")
        try:
            success = avatar.load_source()
            if not success:
                print("Failed to load source")
                print("Creating test with black image...")
                avatar.source_image = np.zeros((config.resolution, config.resolution, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error loading source: {e}")
            print("Creating test with black image...")
            avatar.source_image = np.zeros((config.resolution, config.resolution, 3), dtype=np.uint8)

        # 워밍업
        print("\n=== 워밍업 ===")
        avatar.warmup(iterations=2)

        # 시뮬레이션: 오디오 청크 처리
        print("\n=== 실시간 처리 시뮬레이션 ===")

        for i in range(10):
            # 더미 오디오 생성 (16kHz, 40ms)
            audio_chunk = np.random.randn(640).astype(np.float32) * 0.1

            # 프레임 생성
            video_frame = avatar.process_audio_chunk(audio_chunk, timestamp=time.time())

            # 프레임 인덱스 표시 (영상 모드)
            frame_info = ""
            if avatar.video_loader is not None:
                frame_info = f", VideoFrame={avatar.current_frame_idx:3d}/{len(avatar.video_loader)}"
            elif avatar.image_sequence_loader is not None:
                frame_info = f", ImgSeq={avatar.current_frame_idx:3d}/{len(avatar.image_sequence_loader)}"

            print(
                f"Frame {i+1:2d}: "
                f"State={video_frame.state.value:10s}, "
                f"FPS={video_frame.metadata['fps']:5.1f}, "
                f"Time={video_frame.metadata['processing_time_ms']:6.1f}ms"
                f"{frame_info}"
            )

            # 25 FPS 시뮬레이션
            time.sleep(0.04)

        # 통계 출력
        print("\n=== 성능 통계 ===")
        stats = avatar.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 대기 프레임 생성
        print("\n=== 대기 프레임 생성 ===")
        idle_frame = avatar.get_idle_frame()
        print(f"  Idle frame shape: {idle_frame.shape}")

        # 생각 중 프레임 생성
        print("\n=== 생각 중 프레임 생성 ===")
        thinking_frame = avatar.get_thinking_frame()
        print(f"  Thinking frame shape: {thinking_frame.shape}")

        # 프레임 저장 테스트
        print("\n=== 프레임 저장 ===")
        output_path = "./test_output.jpg"
        cv2.imwrite(output_path, idle_frame)
        print(f"  Saved: {output_path}")

    print("\n=== 완료 ===")
    print("Context manager가 자동으로 리소스를 정리했습니다.")
