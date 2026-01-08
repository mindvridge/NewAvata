"""
Avatar Image Processor
아바타 소스 이미지 전처리 및 최적화 모듈
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import argparse

from loguru import logger

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    logger.warning("InsightFace not available. Install: pip install insightface")
    INSIGHTFACE_AVAILABLE = False

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    logger.warning("rembg not available. Install: pip install rembg")
    REMBG_AVAILABLE = False

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    logger.warning("Real-ESRGAN not available. Install: pip install realesrgan")
    REALESRGAN_AVAILABLE = False


class ProcessingQuality(Enum):
    """처리 품질 등급"""
    EXCELLENT = "excellent"  # 90-100점
    GOOD = "good"  # 70-89점
    FAIR = "fair"  # 50-69점
    POOR = "poor"  # 0-49점


class FaceAngle(Enum):
    """얼굴 각도"""
    FRONTAL = "frontal"  # 정면 (±15도)
    SLIGHT_LEFT = "slight_left"  # 약간 왼쪽 (15-30도)
    SLIGHT_RIGHT = "slight_right"  # 약간 오른쪽
    PROFILE = "profile"  # 측면 (30도 이상)


@dataclass
class FaceInfo:
    """얼굴 정보"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # (5, 2) or (68, 2)
    score: float  # 감지 신뢰도

    # 얼굴 속성
    angle_yaw: float  # 좌우 회전 (도)
    angle_pitch: float  # 상하 회전 (도)
    angle_roll: float  # 기울기 (도)

    # 크기 정보
    face_area: float  # 얼굴 영역 비율 (0~1)
    face_size: Tuple[int, int]  # (width, height)

    # 품질
    sharpness: float  # 선명도 (0~1)
    brightness: float  # 밝기 (0~1)
    occlusion: float  # 가려짐 (0~1, 낮을수록 좋음)


@dataclass
class ProcessingResult:
    """이미지 처리 결과"""
    # 처리된 이미지
    image: np.ndarray

    # 얼굴 정보
    face_info: FaceInfo

    # 처리 메타데이터
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]

    # 품질 점수
    quality_score: float  # 0~100
    quality_grade: ProcessingQuality

    # 처리 단계 플래그
    face_detected: bool
    face_aligned: bool
    background_removed: bool
    upscaled: bool

    # 경고 메시지
    warnings: List[str]


class AvatarImageProcessor:
    """아바타 이미지 전처리기"""

    def __init__(
        self,
        target_size: int = 512,
        use_face_align: bool = True,
        use_background_removal: bool = False,
        use_upscaling: bool = False,
        device: str = "cpu"
    ):
        """
        Args:
            target_size: 목표 이미지 크기 (정사각형)
            use_face_align: 얼굴 정렬 사용
            use_background_removal: 배경 제거 사용
            use_upscaling: 업스케일링 사용 (Real-ESRGAN)
            device: 디바이스 (cpu/cuda)
        """
        self.target_size = target_size
        self.use_face_align = use_face_align
        self.use_background_removal = use_background_removal
        self.use_upscaling = use_upscaling
        self.device = device

        # 얼굴 분석기
        self.face_analyzer = None

        # 업스케일러
        self.upscaler = None

        # 초기화
        self._initialize()

        logger.info(
            f"AvatarImageProcessor initialized: "
            f"size={target_size}, align={use_face_align}, "
            f"bg_removal={use_background_removal}, upscale={use_upscaling}"
        )

    def _initialize(self):
        """컴포넌트 초기화"""
        # InsightFace 초기화
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_analyzer = FaceAnalysis(
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_analyzer.prepare(ctx_id=0 if self.device == "cuda" else -1)
                logger.info("InsightFace initialized")
            except Exception as e:
                logger.warning(f"InsightFace initialization failed: {e}")
                self.face_analyzer = None

        # Real-ESRGAN 초기화
        if self.use_upscaling and REALESRGAN_AVAILABLE:
            try:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2
                )

                self.upscaler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    device=self.device
                )
                logger.info("Real-ESRGAN initialized")
            except Exception as e:
                logger.warning(f"Real-ESRGAN initialization failed: {e}")
                self.upscaler = None

    def process(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        save_intermediate: bool = False
    ) -> ProcessingResult:
        """
        이미지 전처리 파이프라인

        Args:
            image_path: 입력 이미지 경로
            output_path: 출력 이미지 경로 (None이면 저장 안함)
            save_intermediate: 중간 단계 저장 여부

        Returns:
            ProcessingResult: 처리 결과
        """
        logger.info(f"Processing image: {image_path}")

        warnings = []

        # 1. 이미지 로드 및 검증
        image = self._load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_size = (image.shape[1], image.shape[0])
        logger.info(f"Original size: {original_size}")

        # 2. 얼굴 감지 및 분석
        face_info = self._detect_face(image)

        if face_info is None:
            raise ValueError("No face detected in image")

        logger.info(
            f"Face detected: bbox={face_info.bbox}, "
            f"score={face_info.score:.2f}, "
            f"yaw={face_info.angle_yaw:.1f}°"
        )

        # 3. 얼굴 검증
        validation_warnings = self._validate_face(face_info, image.shape)
        warnings.extend(validation_warnings)

        # 4. 얼굴 정렬
        face_aligned = False
        if self.use_face_align:
            image = self._align_face(image, face_info)
            face_aligned = True
            logger.info("Face aligned")

            if save_intermediate:
                self._save_intermediate(image, image_path, "aligned")

        # 5. 크롭 및 리사이즈
        image = self._crop_and_resize(image, face_info)
        logger.info(f"Cropped and resized to {self.target_size}x{self.target_size}")

        if save_intermediate:
            self._save_intermediate(image, image_path, "cropped")

        # 6. 조명 정규화
        image = self._normalize_lighting(image)
        logger.info("Lighting normalized")

        # 7. 배경 제거 (선택적)
        background_removed = False
        if self.use_background_removal and REMBG_AVAILABLE:
            image = self._remove_background(image)
            background_removed = True
            logger.info("Background removed")

            if save_intermediate:
                self._save_intermediate(image, image_path, "no_bg")

        # 8. 업스케일링 (선택적)
        upscaled = False
        if self.use_upscaling and self.upscaler is not None:
            image = self._upscale_image(image)
            upscaled = True
            logger.info("Image upscaled")

        # 9. 최종 품질 점수 계산
        quality_score = self._calculate_quality_score(face_info, warnings)
        quality_grade = self._get_quality_grade(quality_score)

        logger.info(f"Quality score: {quality_score:.1f} ({quality_grade.value})")

        # 10. 결과 저장
        if output_path:
            self._save_image(image, output_path)
            logger.info(f"Saved to: {output_path}")

        # 결과 생성
        result = ProcessingResult(
            image=image,
            face_info=face_info,
            original_size=original_size,
            processed_size=(image.shape[1], image.shape[0]),
            quality_score=quality_score,
            quality_grade=quality_grade,
            face_detected=True,
            face_aligned=face_aligned,
            background_removed=background_removed,
            upscaled=upscaled,
            warnings=warnings
        )

        return result

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """이미지 로드"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            # BGR -> RGB 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None

    def _detect_face(self, image: np.ndarray) -> Optional[FaceInfo]:
        """얼굴 감지 및 분석"""
        if self.face_analyzer is None:
            logger.warning("Face analyzer not available. Using fallback detection.")
            return self._detect_face_fallback(image)

        try:
            # InsightFace로 얼굴 감지
            faces = self.face_analyzer.get(image)

            if len(faces) == 0:
                return None

            # 가장 큰 얼굴 선택
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])

            # FaceInfo 생성
            bbox = face.bbox.astype(int)
            landmarks = face.kps  # (5, 2)

            # 얼굴 각도 계산
            yaw, pitch, roll = self._estimate_pose(landmarks)

            # 얼굴 영역 비율
            img_h, img_w = image.shape[:2]
            face_w = bbox[2] - bbox[0]
            face_h = bbox[3] - bbox[1]
            face_area = (face_w * face_h) / (img_w * img_h)

            # 품질 메트릭
            sharpness = self._calculate_sharpness(image, bbox)
            brightness = self._calculate_brightness(image, bbox)

            face_info = FaceInfo(
                bbox=bbox,
                landmarks=landmarks,
                score=face.det_score,
                angle_yaw=yaw,
                angle_pitch=pitch,
                angle_roll=roll,
                face_area=face_area,
                face_size=(face_w, face_h),
                sharpness=sharpness,
                brightness=brightness,
                occlusion=0.0  # InsightFace는 occlusion 제공 안함
            )

            return face_info

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None

    def _detect_face_fallback(self, image: np.ndarray) -> Optional[FaceInfo]:
        """폴백: OpenCV Haar Cascade로 얼굴 감지"""
        logger.info("Using fallback face detection (Haar Cascade)")

        try:
            # Haar Cascade 로드
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)

            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # 얼굴 감지
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )

            if len(faces) == 0:
                return None

            # 가장 큰 얼굴
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            bbox = np.array([x, y, x + w, y + h])

            # 더미 랜드마크 (5 points)
            landmarks = np.array([
                [x + w * 0.3, y + h * 0.4],  # 왼쪽 눈
                [x + w * 0.7, y + h * 0.4],  # 오른쪽 눈
                [x + w * 0.5, y + h * 0.6],  # 코
                [x + w * 0.3, y + h * 0.8],  # 왼쪽 입
                [x + w * 0.7, y + h * 0.8],  # 오른쪽 입
            ])

            img_h, img_w = image.shape[:2]
            face_area = (w * h) / (img_w * img_h)

            face_info = FaceInfo(
                bbox=bbox,
                landmarks=landmarks,
                score=0.9,  # 더미 점수
                angle_yaw=0.0,
                angle_pitch=0.0,
                angle_roll=0.0,
                face_area=face_area,
                face_size=(w, h),
                sharpness=0.7,
                brightness=0.5,
                occlusion=0.0
            )

            return face_info

        except Exception as e:
            logger.error(f"Fallback face detection failed: {e}")
            return None

    def _estimate_pose(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        랜드마크로부터 얼굴 포즈 추정

        Returns:
            (yaw, pitch, roll): 각도 (도)
        """
        if landmarks.shape[0] < 5:
            return 0.0, 0.0, 0.0

        # 5-point 랜드마크: 눈2, 코1, 입2
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        # Roll (기울기)
        eye_delta = right_eye - left_eye
        roll = np.degrees(np.arctan2(eye_delta[1], eye_delta[0]))

        # Yaw (좌우 회전) - 간단한 추정
        eye_center = (left_eye + right_eye) / 2
        nose_offset = nose[0] - eye_center[0]
        eye_width = np.linalg.norm(eye_delta)
        yaw = (nose_offset / eye_width) * 30  # 대략적인 추정

        # Pitch (상하 회전) - 간단한 추정
        nose_eye_distance = nose[1] - eye_center[1]
        pitch = (nose_eye_distance / eye_width) * 20

        return yaw, pitch, roll

    def _calculate_sharpness(self, image: np.ndarray, bbox: np.ndarray) -> float:
        """얼굴 영역 선명도 계산 (Laplacian variance)"""
        try:
            x1, y1, x2, y2 = bbox
            face_region = image[y1:y2, x1:x2]

            # 그레이스케일 변환
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)

            # Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            # 0~1 정규화 (100이 좋은 값)
            sharpness = min(variance / 100, 1.0)

            return float(sharpness)
        except:
            return 0.5

    def _calculate_brightness(self, image: np.ndarray, bbox: np.ndarray) -> float:
        """얼굴 영역 밝기 계산"""
        try:
            x1, y1, x2, y2 = bbox
            face_region = image[y1:y2, x1:x2]

            # 평균 밝기
            brightness = np.mean(face_region) / 255.0

            return float(brightness)
        except:
            return 0.5

    def _validate_face(
        self,
        face_info: FaceInfo,
        image_shape: Tuple[int, ...]
    ) -> List[str]:
        """얼굴 검증"""
        warnings = []

        # 1. 얼굴 크기 (30-70%)
        if face_info.face_area < 0.3:
            warnings.append("얼굴이 너무 작습니다 (전체의 30% 미만)")
        elif face_info.face_area > 0.7:
            warnings.append("얼굴이 너무 큽니다 (전체의 70% 초과)")

        # 2. 정면 각도 (±15도)
        if abs(face_info.angle_yaw) > 15:
            warnings.append(f"얼굴이 정면이 아닙니다 (yaw: {face_info.angle_yaw:.1f}°)")

        if abs(face_info.angle_pitch) > 15:
            warnings.append(f"얼굴이 위/아래를 향하고 있습니다 (pitch: {face_info.angle_pitch:.1f}°)")

        # 3. 해상도 (최소 256x256)
        min_size = min(face_info.face_size)
        if min_size < 256:
            warnings.append(f"얼굴 해상도가 낮습니다 ({min_size}px < 256px)")

        # 4. 선명도
        if face_info.sharpness < 0.3:
            warnings.append("이미지가 흐릿합니다")

        # 5. 밝기
        if face_info.brightness < 0.3:
            warnings.append("이미지가 너무 어둡습니다")
        elif face_info.brightness > 0.8:
            warnings.append("이미지가 너무 밝습니다")

        return warnings

    def _align_face(self, image: np.ndarray, face_info: FaceInfo) -> np.ndarray:
        """얼굴 정렬 (정면 보정)"""
        try:
            landmarks = face_info.landmarks

            if landmarks.shape[0] < 5:
                logger.warning("Not enough landmarks for alignment")
                return image

            # 눈 위치
            left_eye = landmarks[0]
            right_eye = landmarks[1]

            # 회전 각도 계산
            eye_delta = right_eye - left_eye
            angle = np.degrees(np.arctan2(eye_delta[1], eye_delta[0]))

            # 회전 중심 (두 눈 중간)
            eye_center = ((left_eye + right_eye) / 2).astype(int)

            # 회전 행렬
            M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)

            # 이미지 회전
            h, w = image.shape[:2]
            aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

            return aligned

        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            return image

    def _crop_and_resize(
        self,
        image: np.ndarray,
        face_info: FaceInfo
    ) -> np.ndarray:
        """얼굴 중심으로 크롭 및 리사이즈"""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = face_info.bbox

            # 얼굴 중심
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 얼굴 크기의 1.5배 영역 확보
            face_w = x2 - x1
            face_h = y2 - y1
            crop_size = int(max(face_w, face_h) * 1.5)

            # 크롭 영역 계산
            crop_x1 = max(0, center_x - crop_size // 2)
            crop_y1 = max(0, center_y - crop_size // 2)
            crop_x2 = min(w, center_x + crop_size // 2)
            crop_y2 = min(h, center_y + crop_size // 2)

            # 크롭
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

            # 정사각형으로 패딩
            crop_h, crop_w = cropped.shape[:2]
            max_dim = max(crop_h, crop_w)

            canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            y_offset = (max_dim - crop_h) // 2
            x_offset = (max_dim - crop_w) // 2
            canvas[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped

            # 목표 크기로 리사이즈
            resized = cv2.resize(
                canvas,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LANCZOS4
            )

            return resized

        except Exception as e:
            logger.error(f"Crop and resize failed: {e}")
            # 폴백: 전체 이미지 리사이즈
            return cv2.resize(
                image,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LANCZOS4
            )

    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """조명 정규화 (CLAHE)"""
        try:
            # LAB 색공간 변환
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # LAB 병합 후 RGB 변환
            lab = cv2.merge([l, a, b])
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            return normalized

        except Exception as e:
            logger.warning(f"Lighting normalization failed: {e}")
            return image

    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """배경 제거 (rembg)"""
        if not REMBG_AVAILABLE:
            logger.warning("rembg not available. Skipping background removal.")
            return image

        try:
            # RGB -> BGR (rembg expects BGR)
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 배경 제거
            output = remove(bgr_image)

            # BGR -> RGB
            rgb_output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            return rgb_output

        except Exception as e:
            logger.warning(f"Background removal failed: {e}")
            return image

    def _upscale_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 업스케일링 (Real-ESRGAN)"""
        if self.upscaler is None:
            logger.warning("Upscaler not available. Skipping upscaling.")
            return image

        try:
            # RGB -> BGR
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 업스케일
            output, _ = self.upscaler.enhance(bgr_image, outscale=2)

            # BGR -> RGB
            rgb_output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            # 목표 크기로 다운샘플
            if rgb_output.shape[0] != self.target_size:
                rgb_output = cv2.resize(
                    rgb_output,
                    (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LANCZOS4
                )

            return rgb_output

        except Exception as e:
            logger.warning(f"Upscaling failed: {e}")
            return image

    def _calculate_quality_score(
        self,
        face_info: FaceInfo,
        warnings: List[str]
    ) -> float:
        """품질 점수 계산 (0~100)"""
        score = 100.0

        # 감점 요소

        # 1. 얼굴 크기 (최대 -20점)
        if face_info.face_area < 0.3:
            score -= (0.3 - face_info.face_area) * 50
        elif face_info.face_area > 0.7:
            score -= (face_info.face_area - 0.7) * 50

        # 2. 얼굴 각도 (최대 -30점)
        angle_penalty = (abs(face_info.angle_yaw) + abs(face_info.angle_pitch)) / 2
        score -= min(angle_penalty, 30)

        # 3. 선명도 (최대 -20점)
        if face_info.sharpness < 0.5:
            score -= (0.5 - face_info.sharpness) * 40

        # 4. 밝기 (최대 -15점)
        brightness_diff = abs(face_info.brightness - 0.5)
        score -= brightness_diff * 30

        # 5. 경고 개수 (각 -5점)
        score -= len(warnings) * 5

        # 6. 감지 신뢰도 (최대 -15점)
        if face_info.score < 0.9:
            score -= (0.9 - face_info.score) * 150

        return max(0.0, min(100.0, score))

    def _get_quality_grade(self, score: float) -> ProcessingQuality:
        """품질 등급 결정"""
        if score >= 90:
            return ProcessingQuality.EXCELLENT
        elif score >= 70:
            return ProcessingQuality.GOOD
        elif score >= 50:
            return ProcessingQuality.FAIR
        else:
            return ProcessingQuality.POOR

    def _save_image(self, image: np.ndarray, output_path: str):
        """이미지 저장"""
        try:
            # RGB -> BGR
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 저장
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, bgr_image)

        except Exception as e:
            logger.error(f"Failed to save image: {e}")

    def _save_intermediate(self, image: np.ndarray, original_path: str, suffix: str):
        """중간 결과 저장"""
        try:
            original = Path(original_path)
            intermediate_path = original.parent / f"{original.stem}_{suffix}{original.suffix}"
            self._save_image(image, str(intermediate_path))

        except Exception as e:
            logger.warning(f"Failed to save intermediate: {e}")


# CLI 도구
def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(
        description="아바타 소스 이미지 전처리 도구"
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='입력 이미지 경로'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='출력 이미지 경로'
    )

    parser.add_argument(
        '--size', '-s',
        type=int,
        default=512,
        help='출력 이미지 크기 (기본: 512)'
    )

    parser.add_argument(
        '--no-align',
        action='store_true',
        help='얼굴 정렬 비활성화'
    )

    parser.add_argument(
        '--remove-bg',
        action='store_true',
        help='배경 제거 활성화'
    )

    parser.add_argument(
        '--upscale',
        action='store_true',
        help='업스케일링 활성화 (Real-ESRGAN)'
    )

    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='중간 결과 저장'
    )

    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='디바이스 (기본: cpu)'
    )

    args = parser.parse_args()

    # 프로세서 생성
    processor = AvatarImageProcessor(
        target_size=args.size,
        use_face_align=not args.no_align,
        use_background_removal=args.remove_bg,
        use_upscaling=args.upscale,
        device=args.device
    )

    # 처리
    try:
        result = processor.process(
            image_path=args.input,
            output_path=args.output,
            save_intermediate=args.save_intermediate
        )

        # 결과 출력
        print("\n" + "=" * 70)
        print("이미지 처리 완료!")
        print("=" * 70)
        print(f"\n입력: {args.input}")
        print(f"출력: {args.output}")
        print(f"\n원본 크기: {result.original_size}")
        print(f"처리 크기: {result.processed_size}")
        print(f"\n얼굴 정보:")
        print(f"  - 위치: {result.face_info.bbox}")
        print(f"  - 신뢰도: {result.face_info.score:.2f}")
        print(f"  - 각도 (yaw/pitch/roll): "
              f"{result.face_info.angle_yaw:.1f}° / "
              f"{result.face_info.angle_pitch:.1f}° / "
              f"{result.face_info.angle_roll:.1f}°")
        print(f"  - 얼굴 영역: {result.face_info.face_area*100:.1f}%")
        print(f"  - 선명도: {result.face_info.sharpness:.2f}")
        print(f"  - 밝기: {result.face_info.brightness:.2f}")

        print(f"\n품질 점수: {result.quality_score:.1f}/100 ({result.quality_grade.value})")

        if result.warnings:
            print(f"\n경고 ({len(result.warnings)}개):")
            for warning in result.warnings:
                print(f"  ⚠ {warning}")

        print("\n처리 단계:")
        print(f"  ✓ 얼굴 감지")
        print(f"  {'✓' if result.face_aligned else '✗'} 얼굴 정렬")
        print(f"  {'✓' if result.background_removed else '✗'} 배경 제거")
        print(f"  {'✓' if result.upscaled else '✗'} 업스케일링")

        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"\n에러: {e}\n")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
