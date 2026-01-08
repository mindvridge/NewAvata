"""
MuseTalk Avatar Module
Handles avatar animation and lip-sync generation
"""

from .musetalk_wrapper import (
    MuseTalkAvatar,
    MuseTalkConfig,
    AvatarState,
    DeviceType,
    VideoFrame,
    FaceLandmarkDetector,
    AudioFeatureExtractor,
    create_musetalk_avatar,
)
from .image_processor import (
    AvatarImageProcessor,
    ProcessingResult,
    ProcessingQuality,
    FaceInfo,
    FaceAngle,
)
from .face_enhancer import (
    FaceEnhancer,
    EnhancementConfig,
    EnhancementMode,
    EnhancementModel,
    EnhancementResult,
    create_realtime_enhancer,
    create_quality_enhancer,
    create_balanced_enhancer,
)

__all__ = [
    # Avatar
    "MuseTalkAvatar",
    "MuseTalkConfig",
    "AvatarState",
    "DeviceType",
    "VideoFrame",
    # Components
    "FaceLandmarkDetector",
    "AudioFeatureExtractor",
    # Image Processing
    "AvatarImageProcessor",
    "ProcessingResult",
    "ProcessingQuality",
    "FaceInfo",
    "FaceAngle",
    # Face Enhancement
    "FaceEnhancer",
    "EnhancementConfig",
    "EnhancementMode",
    "EnhancementModel",
    "EnhancementResult",
    "create_realtime_enhancer",
    "create_quality_enhancer",
    "create_balanced_enhancer",
    # Factory
    "create_musetalk_avatar",
]
