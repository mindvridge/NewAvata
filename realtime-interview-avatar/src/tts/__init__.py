"""
Text-to-Speech Module
ElevenLabs TTS integration for voice synthesis and Korean TTS alternatives
"""

from .elevenlabs_service import (
    ElevenLabsTTSService,
    PipecatElevenLabsTTSService,
    TTSConfig,
    AudioChunk,
    TextChunker,
    create_elevenlabs_tts,
)
from .cache import (
    TTSCache,
    CachedTTSService,
    CacheEntry,
    LRUCache,
    INTERVIEW_PHRASES,
    create_cached_tts,
)
from .korean_tts import (
    EdgeTTSService,
    EdgeTTSConfig,
    NaverClovaTTSService,
    NaverClovaTTSConfig,
    TTSProvider,
    TTSFactory,
    FallbackTTSService,
    create_korean_tts,
)

__all__ = [
    # TTS Service
    "ElevenLabsTTSService",
    "PipecatElevenLabsTTSService",
    "TTSConfig",
    "AudioChunk",
    "TextChunker",
    "create_elevenlabs_tts",
    # Cache
    "TTSCache",
    "CachedTTSService",
    "CacheEntry",
    "LRUCache",
    "INTERVIEW_PHRASES",
    "create_cached_tts",
    # Korean TTS
    "EdgeTTSService",
    "EdgeTTSConfig",
    "NaverClovaTTSService",
    "NaverClovaTTSConfig",
    "TTSProvider",
    "TTSFactory",
    "FallbackTTSService",
    "create_korean_tts",
]
