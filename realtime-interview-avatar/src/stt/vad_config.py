"""
Silero VAD Configuration
면접 시나리오에 최적화된 Voice Activity Detection 설정
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
from loguru import logger


class VADMode(Enum):
    """VAD 동작 모드"""
    INTERVIEW_RELAXED = "interview_relaxed"  # 면접 - 편안한 모드 (긴 침묵 허용)
    INTERVIEW_NORMAL = "interview_normal"    # 면접 - 일반 모드
    CONVERSATION = "conversation"            # 대화 - 빠른 턴테이킹
    CUSTOM = "custom"                        # 커스텀 설정


@dataclass
class VADConfig:
    """
    Silero VAD 설정

    Attributes:
        min_speech_duration: 최소 발화 길이 (초) - 이보다 짧은 소리는 무시
        min_silence_duration: 최소 침묵 길이 (초) - 발화 종료 판단 기준
        speech_pad_ms: 발화 전후 패딩 (밀리초) - 여유있는 감지
        threshold: VAD 임계값 (0.0-1.0) - 낮을수록 민감
        sampling_rate: 샘플링 레이트 (Hz)
        chunk_size: 오디오 청크 크기 (샘플 수)
    """
    min_speech_duration: float = 0.1          # 최소 발화 0.1초
    min_silence_duration: float = 0.7         # 최소 침묵 0.7초
    speech_pad_ms: int = 200                  # 200ms 패딩
    threshold: float = 0.5                    # 중간 민감도
    sampling_rate: int = 16000                # 16kHz
    chunk_size: int = 512                     # 512 샘플 (32ms @ 16kHz)

    # 추가 고급 설정
    max_speech_duration: Optional[float] = None  # 최대 발화 길이 (None = 무제한)
    return_seconds: bool = True               # 초 단위로 반환
    visualize_probs: bool = False             # 확률 시각화 (디버깅용)

    def __post_init__(self):
        """초기화 후 검증"""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if self.min_speech_duration < 0:
            raise ValueError("min_speech_duration must be positive")
        if self.min_silence_duration < 0:
            raise ValueError("min_silence_duration must be positive")
        if self.speech_pad_ms < 0:
            raise ValueError("speech_pad_ms must be non-negative")


# =============================================================================
# 면접 시나리오별 프리셋
# =============================================================================

VAD_PRESETS = {
    VADMode.INTERVIEW_RELAXED: VADConfig(
        min_speech_duration=0.1,      # 짧은 응답도 감지
        min_silence_duration=1.0,     # 긴 침묵 허용 (생각할 시간)
        speech_pad_ms=300,            # 여유있는 패딩
        threshold=0.5,                # 중간 민감도
    ),

    VADMode.INTERVIEW_NORMAL: VADConfig(
        min_speech_duration=0.1,      # 짧은 응답도 감지
        min_silence_duration=0.7,     # 일반적인 침묵 허용
        speech_pad_ms=200,            # 표준 패딩
        threshold=0.5,                # 중간 민감도
    ),

    VADMode.CONVERSATION: VADConfig(
        min_speech_duration=0.05,     # 매우 짧은 반응도 감지
        min_silence_duration=0.3,     # 빠른 턴테이킹
        speech_pad_ms=100,            # 짧은 패딩
        threshold=0.6,                # 약간 낮은 민감도 (노이즈 제거)
    ),
}


class SileroVADAnalyzer:
    """
    Silero VAD 래퍼 클래스

    면접 시나리오에 최적화된 Voice Activity Detection을 제공합니다.
    동적으로 모드를 전환할 수 있습니다.
    """

    def __init__(
        self,
        mode: VADMode = VADMode.INTERVIEW_NORMAL,
        custom_config: Optional[VADConfig] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[float], None]] = None,
        on_vad_event: Optional[Callable[[dict], None]] = None,
    ):
        """
        Args:
            mode: VAD 동작 모드
            custom_config: 커스텀 설정 (mode가 CUSTOM일 때 사용)
            on_speech_start: 발화 시작 콜백
            on_speech_end: 발화 종료 콜백 (발화 길이 전달)
            on_vad_event: VAD 이벤트 콜백 (모든 이벤트)
        """
        self.mode = mode
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_vad_event = on_vad_event

        # 설정 로드
        if mode == VADMode.CUSTOM:
            if custom_config is None:
                raise ValueError("custom_config must be provided when mode is CUSTOM")
            self.config = custom_config
        else:
            self.config = VAD_PRESETS[mode]

        # Silero VAD 모델 (실제 사용 시 lazy loading)
        self._vad_model = None
        self._is_initialized = False

        # 상태 추적
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._total_speech_time = 0.0
        self._total_silence_time = 0.0

        logger.info(
            f"SileroVADAnalyzer initialized: mode={mode.value}, "
            f"min_silence={self.config.min_silence_duration}s, "
            f"threshold={self.config.threshold}"
        )

    def initialize(self):
        """
        Silero VAD 모델을 초기화합니다.

        Note: 실제 Silero VAD를 사용하려면 torch와 silero-vad 패키지가 필요합니다.
        """
        if self._is_initialized:
            return

        try:
            import torch

            # Silero VAD 모델 로드
            # torch.hub를 통해 로드하거나, 로컬 파일에서 로드
            self._vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,  # ONNX 대신 PyTorch 모델 사용
            )

            # VAD 유틸리티 함수들
            (get_speech_timestamps, _, read_audio, *_) = utils

            self._get_speech_timestamps = get_speech_timestamps
            self._vad_model.eval()  # Evaluation 모드

            self._is_initialized = True
            logger.info("Silero VAD model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            raise

    def process_audio(self, audio_chunk: bytes) -> dict:
        """
        오디오 청크를 처리하여 VAD 분석 결과를 반환합니다.

        Args:
            audio_chunk: PCM 오디오 데이터 (int16)

        Returns:
            dict: VAD 분석 결과
                - is_speech: 발화 여부
                - probability: 발화 확률 (0.0-1.0)
                - timestamp: 타임스탬프
        """
        if not self._is_initialized:
            self.initialize()

        try:
            import torch
            import numpy as np

            # bytes -> numpy array -> torch tensor
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0  # 정규화
            audio_tensor = torch.from_numpy(audio_float32)

            # VAD 확률 계산
            with torch.no_grad():
                speech_prob = self._vad_model(audio_tensor, self.config.sampling_rate).item()

            # 발화 여부 판단
            is_speech = speech_prob >= self.config.threshold

            result = {
                "is_speech": is_speech,
                "probability": speech_prob,
                "timestamp": self._total_speech_time + self._total_silence_time,
            }

            # 콜백 호출
            if self._on_vad_event:
                self._on_vad_event(result)

            return result

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                "is_speech": False,
                "probability": 0.0,
                "timestamp": 0.0,
                "error": str(e),
            }

    def switch_mode(self, mode: VADMode, custom_config: Optional[VADConfig] = None):
        """
        VAD 모드를 동적으로 전환합니다.

        Args:
            mode: 새로운 VAD 모드
            custom_config: CUSTOM 모드일 때 사용할 설정
        """
        old_mode = self.mode
        self.mode = mode

        if mode == VADMode.CUSTOM:
            if custom_config is None:
                raise ValueError("custom_config must be provided when mode is CUSTOM")
            self.config = custom_config
        else:
            self.config = VAD_PRESETS[mode]

        logger.info(
            f"VAD mode switched: {old_mode.value} -> {mode.value} "
            f"(silence_duration={self.config.min_silence_duration}s)"
        )

    def update_config(self, **kwargs):
        """
        현재 설정을 업데이트합니다.

        Args:
            **kwargs: 업데이트할 설정 키-값 쌍
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"VAD config updated: {key}={value}")
            else:
                logger.warning(f"Unknown config key: {key}")

    def reset_stats(self):
        """통계를 초기화합니다."""
        self._total_speech_time = 0.0
        self._total_silence_time = 0.0
        self._is_speaking = False
        self._speech_start_time = None
        logger.debug("VAD stats reset")

    def get_stats(self) -> dict:
        """
        현재 통계를 반환합니다.

        Returns:
            dict: VAD 통계 정보
        """
        return {
            "mode": self.mode.value,
            "is_speaking": self._is_speaking,
            "total_speech_time": self._total_speech_time,
            "total_silence_time": self._total_silence_time,
            "config": {
                "min_speech_duration": self.config.min_speech_duration,
                "min_silence_duration": self.config.min_silence_duration,
                "threshold": self.config.threshold,
            },
        }

    @property
    def is_speaking(self) -> bool:
        """현재 발화 중 여부"""
        return self._is_speaking

    @property
    def speech_ratio(self) -> float:
        """발화 비율 (0.0-1.0)"""
        total = self._total_speech_time + self._total_silence_time
        if total == 0:
            return 0.0
        return self._total_speech_time / total


# =============================================================================
# Pipecat 통합용 헬퍼
# =============================================================================

def create_interview_vad(
    relaxed: bool = False,
    on_speech_start: Optional[Callable] = None,
    on_speech_end: Optional[Callable] = None,
) -> SileroVADAnalyzer:
    """
    면접용 VAD 인스턴스를 생성하는 헬퍼 함수

    Args:
        relaxed: True면 INTERVIEW_RELAXED 모드, False면 INTERVIEW_NORMAL
        on_speech_start: 발화 시작 콜백
        on_speech_end: 발화 종료 콜백

    Returns:
        SileroVADAnalyzer: 설정된 VAD 인스턴스
    """
    mode = VADMode.INTERVIEW_RELAXED if relaxed else VADMode.INTERVIEW_NORMAL

    return SileroVADAnalyzer(
        mode=mode,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
    )


def create_conversation_vad(
    on_speech_start: Optional[Callable] = None,
    on_speech_end: Optional[Callable] = None,
) -> SileroVADAnalyzer:
    """
    일반 대화용 VAD 인스턴스를 생성하는 헬퍼 함수

    Args:
        on_speech_start: 발화 시작 콜백
        on_speech_end: 발화 종료 콜백

    Returns:
        SileroVADAnalyzer: 설정된 VAD 인스턴스
    """
    return SileroVADAnalyzer(
        mode=VADMode.CONVERSATION,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
    )


# =============================================================================
# 예시 및 테스트
# =============================================================================

if __name__ == "__main__":
    """VAD 설정 테스트"""

    print("=" * 80)
    print("Silero VAD Configuration Test")
    print("=" * 80)

    # 각 모드별 설정 출력
    for mode in [VADMode.INTERVIEW_RELAXED, VADMode.INTERVIEW_NORMAL, VADMode.CONVERSATION]:
        config = VAD_PRESETS[mode]
        print(f"\n[{mode.value.upper()}]")
        print(f"  Min Speech Duration: {config.min_speech_duration}s")
        print(f"  Min Silence Duration: {config.min_silence_duration}s")
        print(f"  Speech Padding: {config.speech_pad_ms}ms")
        print(f"  Threshold: {config.threshold}")

    # VAD 분석기 생성 예시
    print("\n" + "=" * 80)
    print("VAD Analyzer Creation")
    print("=" * 80)

    def on_speech_start():
        print("🎤 Speech started!")

    def on_speech_end(duration: float):
        print(f"🔇 Speech ended (duration: {duration:.2f}s)")

    # 면접 모드 (일반)
    vad = create_interview_vad(
        relaxed=False,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
    )
    print(f"\n✓ Interview VAD created: {vad.mode.value}")
    print(f"  Config: {vad.get_stats()['config']}")

    # 모드 전환 테스트
    print("\n" + "=" * 80)
    print("Mode Switching Test")
    print("=" * 80)

    vad.switch_mode(VADMode.CONVERSATION)
    print(f"✓ Switched to: {vad.mode.value}")
    print(f"  New silence duration: {vad.config.min_silence_duration}s")

    vad.switch_mode(VADMode.INTERVIEW_RELAXED)
    print(f"✓ Switched to: {vad.mode.value}")
    print(f"  New silence duration: {vad.config.min_silence_duration}s")

    # 커스텀 설정 테스트
    print("\n" + "=" * 80)
    print("Custom Configuration Test")
    print("=" * 80)

    custom_config = VADConfig(
        min_speech_duration=0.2,
        min_silence_duration=0.5,
        speech_pad_ms=150,
        threshold=0.7,
    )

    custom_vad = SileroVADAnalyzer(
        mode=VADMode.CUSTOM,
        custom_config=custom_config,
    )
    print(f"✓ Custom VAD created")
    print(f"  Config: {custom_vad.get_stats()['config']}")

    print("\n" + "=" * 80)
    print("Note: To actually use VAD, run: vad.initialize()")
    print("      This will download the Silero VAD model from torch.hub")
    print("=" * 80)
