"""
Speech-to-Text Module
Deepgram STT integration for speech recognition and text input
"""

from .deepgram_service import (
    DeepgramSTTService,
    PipecatDeepgramSTTService,
    TranscriptResult,
)
from .vad_config import (
    SileroVADAnalyzer,
    VADConfig,
    VADMode,
    VAD_PRESETS,
    create_interview_vad,
    create_conversation_vad,
)
from .text_input_service import (
    TextInputService,
    TextInputConfig,
    WebSocketTextInputService,
    UnifiedInputService,
    create_text_input_service,
)

__all__ = [
    # Deepgram STT
    "DeepgramSTTService",
    "PipecatDeepgramSTTService",
    "TranscriptResult",
    # VAD
    "SileroVADAnalyzer",
    "VADConfig",
    "VADMode",
    "VAD_PRESETS",
    "create_interview_vad",
    "create_conversation_vad",
    # Text Input (Chat mode)
    "TextInputService",
    "TextInputConfig",
    "WebSocketTextInputService",
    "UnifiedInputService",
    "create_text_input_service",
]
