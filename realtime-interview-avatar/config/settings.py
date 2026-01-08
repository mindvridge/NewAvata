"""
Application Settings
환경 변수 및 설정 관리를 위한 Pydantic Settings
"""

from pathlib import Path
from typing import Optional, Tuple
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    애플리케이션 전역 설정
    환경 변수 또는 .env 파일에서 자동으로 로드됩니다.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # =========================================================================
    # API Keys
    # =========================================================================
    deepgram_api_key: str = Field(
        default="",
        description="Deepgram STT API 키 (선택 - 로컬 Whisper 사용 시 불필요)"
    )

    elevenlabs_api_key: str = Field(
        default="",
        description="ElevenLabs TTS API 키 (선택 - EdgeTTS 사용 시 불필요)"
    )

    openai_api_key: str = Field(
        default="",
        description="OpenAI GPT API 키"
    )

    daily_api_key: Optional[str] = Field(
        default=None,
        description="Daily.co WebRTC API 키 (선택사항)"
    )

    # =========================================================================
    # Avatar Settings
    # =========================================================================
    avatar_image_path: Path = Field(
        default=Path("./assets/images/avatar.jpg"),
        description="아바타 소스 이미지 경로"
    )

    musetalk_model_path: Path = Field(
        default=Path("./models/musetalk"),
        description="MuseTalk 모델 경로"
    )

    # =========================================================================
    # STT (Speech-to-Text) Settings
    # =========================================================================
    stt_language: str = Field(
        default="ko",
        description="STT 언어 코드 (ko, en, ja 등)"
    )

    stt_model: str = Field(
        default="nova-2",
        description="Deepgram 모델 (nova-2, enhanced, base 등)"
    )

    stt_interim_results: bool = Field(
        default=True,
        description="중간 결과 반환 여부"
    )

    # =========================================================================
    # TTS (Text-to-Speech) Settings
    # =========================================================================
    tts_voice_id: str = Field(
        default="",
        description="ElevenLabs 음성 ID (선택 - EdgeTTS 사용 시 불필요)"
    )

    tts_model: str = Field(
        default="eleven_multilingual_v2",
        description="ElevenLabs TTS 모델"
    )

    tts_stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="음성 안정성 (0.0-1.0)"
    )

    tts_similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="음성 유사도 부스트 (0.0-1.0)"
    )

    tts_optimize_streaming_latency: int = Field(
        default=3,
        ge=0,
        le=4,
        description="스트리밍 레이턴시 최적화 레벨 (0-4)"
    )

    # =========================================================================
    # LLM Settings
    # =========================================================================
    llm_model: str = Field(
        default="gpt-4o",
        description="사용할 LLM 모델 (gpt-4o, gpt-4-turbo, gpt-3.5-turbo 등)"
    )

    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM 생성 온도 (0.0-2.0)"
    )

    llm_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="LLM 최대 토큰 수"
    )

    # =========================================================================
    # Performance & Latency Settings
    # =========================================================================
    max_latency_ms: int = Field(
        default=500,
        gt=0,
        description="목표 최대 레이턴시 (밀리초)"
    )

    pipeline_buffer_size: int = Field(
        default=1024,
        gt=0,
        description="파이프라인 버퍼 크기"
    )

    audio_sample_rate: int = Field(
        default=16000,
        description="오디오 샘플레이트 (Hz)"
    )

    # =========================================================================
    # Video Settings
    # =========================================================================
    video_fps: int = Field(
        default=30,
        gt=0,
        le=60,
        description="출력 비디오 FPS"
    )

    video_resolution: str = Field(
        default="512x512",
        description="출력 해상도 (WIDTHxHEIGHT)"
    )

    video_batch_size: int = Field(
        default=8,
        gt=0,
        description="비디오 처리 배치 크기"
    )

    use_float16: bool = Field(
        default=True,
        description="Float16 사용 여부 (GPU 메모리 절약)"
    )

    # =========================================================================
    # Server Settings
    # =========================================================================
    host: str = Field(
        default="0.0.0.0",
        description="서버 호스트"
    )

    port: int = Field(
        default=8000,
        gt=0,
        le=65535,
        description="서버 포트"
    )

    debug: bool = Field(
        default=False,
        description="디버그 모드"
    )

    log_level: str = Field(
        default="INFO",
        description="로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # =========================================================================
    # Interview Settings
    # =========================================================================
    interview_timeout: int = Field(
        default=3600,
        gt=0,
        description="면접 타임아웃 (초)"
    )

    max_interview_duration: int = Field(
        default=7200,
        gt=0,
        description="최대 면접 시간 (초)"
    )

    # =========================================================================
    # Storage Settings
    # =========================================================================
    recording_dir: Path = Field(
        default=Path("./recordings"),
        description="녹화 파일 저장 디렉토리"
    )

    cache_dir: Path = Field(
        default=Path("./cache"),
        description="캐시 디렉토리"
    )

    # =========================================================================
    # Validators
    # =========================================================================
    @field_validator("video_resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        """해상도 형식 검증 (WIDTHxHEIGHT)"""
        try:
            width, height = v.split("x")
            int(width)
            int(height)
            return v
        except (ValueError, AttributeError):
            raise ValueError(
                "video_resolution must be in format 'WIDTHxHEIGHT' (e.g., '512x512')"
            )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """로그 레벨 검증"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}"
            )
        return v_upper

    # =========================================================================
    # Helper Properties
    # =========================================================================
    @property
    def resolution_tuple(self) -> Tuple[int, int]:
        """해상도를 (width, height) 튜플로 반환"""
        width, height = self.video_resolution.split("x")
        return (int(width), int(height))

    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return not self.debug

    def ensure_directories(self) -> None:
        """필요한 디렉토리들을 생성합니다."""
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.musetalk_model_path.mkdir(parents=True, exist_ok=True)

        # assets 디렉토리도 생성
        self.avatar_image_path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Global Settings Instance
# ============================================================================
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    전역 설정 인스턴스를 반환합니다.
    싱글톤 패턴으로 구현되어 있습니다.

    Returns:
        Settings: 애플리케이션 설정 인스턴스
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


def reload_settings() -> Settings:
    """
    설정을 다시 로드합니다.
    주로 테스트나 설정 변경 후 사용합니다.

    Returns:
        Settings: 새로운 설정 인스턴스
    """
    global _settings
    _settings = Settings()
    _settings.ensure_directories()
    return _settings


# ============================================================================
# Convenience Export
# ============================================================================
# Lazy loading - settings will be created on first access
# This prevents validation errors during module import
def __getattr__(name):
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    # 설정 테스트/확인용
    import json

    settings = get_settings()

    print("=" * 80)
    print("Current Settings")
    print("=" * 80)

    # API 키는 마스킹해서 출력
    config_dict = settings.model_dump()
    for key in config_dict:
        if "api_key" in key.lower() and config_dict[key]:
            config_dict[key] = config_dict[key][:8] + "..." if len(config_dict[key]) > 8 else "***"

    print(json.dumps(config_dict, indent=2, default=str))
    print("=" * 80)
