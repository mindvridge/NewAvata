"""
STT 모듈 테스트

Deepgram STT와 VAD 기능을 테스트합니다.
"""

import asyncio
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.stt import (
    DeepgramSTTService,
    TranscriptResult,
    SileroVADAnalyzer,
    VADConfig,
    VADMode,
    create_interview_vad,
)


# =============================================================================
# 테스트 픽스처
# =============================================================================

@pytest.fixture
def mock_deepgram_api_key():
    """테스트용 API 키"""
    return "test_api_key_12345"


@pytest.fixture
def sample_audio_chunk():
    """테스트용 오디오 청크 (16kHz, 1초)"""
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)

    # 440Hz 사인파 생성 (A4 음)
    t = np.linspace(0, duration, samples, False)
    audio = np.sin(2 * np.pi * 440 * t)

    # int16으로 변환
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


@pytest.fixture
def transcript_callback():
    """테스트용 transcript 콜백"""
    return Mock()


# =============================================================================
# Deepgram STT 테스트
# =============================================================================

class TestDeepgramSTTService:
    """DeepgramSTTService 테스트"""

    def test_initialization(self, mock_deepgram_api_key):
        """초기화 테스트"""
        stt = DeepgramSTTService(
            api_key=mock_deepgram_api_key,
            language="ko",
            model="nova-3",
        )

        assert stt.api_key == mock_deepgram_api_key
        assert stt.language == "ko"
        assert stt.model == "nova-3"
        assert not stt.is_connected
        assert not stt.is_speaking

    def test_initialization_with_callbacks(self, mock_deepgram_api_key):
        """콜백 함수와 함께 초기화 테스트"""
        on_transcript = Mock()
        on_speech_started = Mock()
        on_speech_ended = Mock()
        on_error = Mock()

        stt = DeepgramSTTService(
            api_key=mock_deepgram_api_key,
            on_transcript=on_transcript,
            on_speech_started=on_speech_started,
            on_speech_ended=on_speech_ended,
            on_error=on_error,
        )

        assert stt._on_transcript == on_transcript
        assert stt._on_speech_started == on_speech_started
        assert stt._on_speech_ended == on_speech_ended
        assert stt._on_error == on_error

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, mock_deepgram_api_key):
        """연결/해제 테스트 (모킹)"""
        with patch('src.stt.deepgram_service.DeepgramClient') as mock_client:
            # Mock WebSocket 연결
            mock_connection = AsyncMock()
            mock_connection.start = AsyncMock(return_value=True)
            mock_connection.finish = AsyncMock()

            mock_client.return_value.listen.asyncwebsocket.v.return_value = mock_connection

            stt = DeepgramSTTService(api_key=mock_deepgram_api_key)

            # 연결
            await stt.connect()
            assert stt.is_connected

            # 해제
            await stt.disconnect()
            assert not stt.is_connected

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_deepgram_api_key):
        """컨텍스트 매니저 테스트"""
        with patch('src.stt.deepgram_service.DeepgramClient') as mock_client:
            mock_connection = AsyncMock()
            mock_connection.start = AsyncMock(return_value=True)
            mock_connection.finish = AsyncMock()
            mock_client.return_value.listen.asyncwebsocket.v.return_value = mock_connection

            async with DeepgramSTTService(api_key=mock_deepgram_api_key) as stt:
                assert stt.is_connected

            assert not stt.is_connected

    @pytest.mark.asyncio
    async def test_send_audio(self, mock_deepgram_api_key, sample_audio_chunk):
        """오디오 전송 테스트"""
        with patch('src.stt.deepgram_service.DeepgramClient') as mock_client:
            mock_connection = AsyncMock()
            mock_connection.start = AsyncMock(return_value=True)
            mock_connection.send = Mock()
            mock_client.return_value.listen.asyncwebsocket.v.return_value = mock_connection

            stt = DeepgramSTTService(api_key=mock_deepgram_api_key)
            await stt.connect()

            # 오디오 전송
            await stt.send_audio(sample_audio_chunk)

            # send 메서드가 호출되었는지 확인
            mock_connection.send.assert_called_once_with(sample_audio_chunk)

    def test_transcript_callback(self, mock_deepgram_api_key):
        """Transcript 콜백 테스트"""
        transcript_callback = Mock()

        stt = DeepgramSTTService(
            api_key=mock_deepgram_api_key,
            on_transcript=transcript_callback,
        )

        # Mock 결과 생성
        mock_result = Mock()
        mock_result.channel.alternatives = [Mock(transcript="안녕하세요", confidence=0.95)]
        mock_result.is_final = True
        mock_result.duration = 1.5

        # 콜백 직접 호출 (실제로는 Deepgram 이벤트에서 호출됨)
        stt._on_message(result=mock_result)

        # 콜백이 호출되었는지 확인
        assert transcript_callback.called
        call_args = transcript_callback.call_args[0][0]
        assert isinstance(call_args, TranscriptResult)
        assert call_args.text == "안녕하세요"
        assert call_args.is_final is True


# =============================================================================
# VAD 테스트
# =============================================================================

class TestVADConfig:
    """VADConfig 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        config = VADConfig()

        assert config.min_speech_duration == 0.1
        assert config.min_silence_duration == 0.7
        assert config.speech_pad_ms == 200
        assert config.threshold == 0.5
        assert config.sampling_rate == 16000

    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = VADConfig(
            min_speech_duration=0.2,
            min_silence_duration=1.0,
            threshold=0.6,
        )

        assert config.min_speech_duration == 0.2
        assert config.min_silence_duration == 1.0
        assert config.threshold == 0.6

    def test_invalid_threshold(self):
        """잘못된 threshold 검증"""
        with pytest.raises(ValueError):
            VADConfig(threshold=1.5)

        with pytest.raises(ValueError):
            VADConfig(threshold=-0.1)

    def test_invalid_durations(self):
        """잘못된 duration 검증"""
        with pytest.raises(ValueError):
            VADConfig(min_speech_duration=-0.1)

        with pytest.raises(ValueError):
            VADConfig(min_silence_duration=-0.5)


class TestSileroVADAnalyzer:
    """SileroVADAnalyzer 테스트"""

    def test_initialization_normal_mode(self):
        """일반 모드 초기화 테스트"""
        vad = SileroVADAnalyzer(mode=VADMode.INTERVIEW_NORMAL)

        assert vad.mode == VADMode.INTERVIEW_NORMAL
        assert vad.config.min_silence_duration == 0.7
        assert not vad.is_speaking

    def test_initialization_relaxed_mode(self):
        """편안한 모드 초기화 테스트"""
        vad = SileroVADAnalyzer(mode=VADMode.INTERVIEW_RELAXED)

        assert vad.mode == VADMode.INTERVIEW_RELAXED
        assert vad.config.min_silence_duration == 1.0

    def test_initialization_conversation_mode(self):
        """대화 모드 초기화 테스트"""
        vad = SileroVADAnalyzer(mode=VADMode.CONVERSATION)

        assert vad.mode == VADMode.CONVERSATION
        assert vad.config.min_silence_duration == 0.3

    def test_custom_mode(self):
        """커스텀 모드 테스트"""
        custom_config = VADConfig(
            min_speech_duration=0.15,
            min_silence_duration=0.8,
        )

        vad = SileroVADAnalyzer(
            mode=VADMode.CUSTOM,
            custom_config=custom_config,
        )

        assert vad.mode == VADMode.CUSTOM
        assert vad.config.min_silence_duration == 0.8

    def test_custom_mode_without_config(self):
        """커스텀 모드인데 설정이 없으면 에러"""
        with pytest.raises(ValueError):
            SileroVADAnalyzer(mode=VADMode.CUSTOM)

    def test_switch_mode(self):
        """모드 전환 테스트"""
        vad = SileroVADAnalyzer(mode=VADMode.INTERVIEW_NORMAL)

        # RELAXED 모드로 전환
        vad.switch_mode(VADMode.INTERVIEW_RELAXED)
        assert vad.mode == VADMode.INTERVIEW_RELAXED
        assert vad.config.min_silence_duration == 1.0

        # CONVERSATION 모드로 전환
        vad.switch_mode(VADMode.CONVERSATION)
        assert vad.mode == VADMode.CONVERSATION
        assert vad.config.min_silence_duration == 0.3

    def test_update_config(self):
        """설정 업데이트 테스트"""
        vad = SileroVADAnalyzer(mode=VADMode.INTERVIEW_NORMAL)

        vad.update_config(
            min_silence_duration=0.9,
            threshold=0.6,
        )

        assert vad.config.min_silence_duration == 0.9
        assert vad.config.threshold == 0.6

    def test_get_stats(self):
        """통계 조회 테스트"""
        vad = SileroVADAnalyzer(mode=VADMode.INTERVIEW_NORMAL)

        stats = vad.get_stats()

        assert stats["mode"] == "interview_normal"
        assert stats["is_speaking"] is False
        assert "config" in stats
        assert stats["config"]["min_silence_duration"] == 0.7

    def test_reset_stats(self):
        """통계 리셋 테스트"""
        vad = SileroVADAnalyzer(mode=VADMode.INTERVIEW_NORMAL)

        vad._total_speech_time = 10.0
        vad._total_silence_time = 5.0

        vad.reset_stats()

        assert vad._total_speech_time == 0.0
        assert vad._total_silence_time == 0.0

    def test_callbacks(self):
        """콜백 테스트"""
        on_speech_start = Mock()
        on_speech_end = Mock()
        on_vad_event = Mock()

        vad = SileroVADAnalyzer(
            mode=VADMode.INTERVIEW_NORMAL,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            on_vad_event=on_vad_event,
        )

        assert vad._on_speech_start == on_speech_start
        assert vad._on_speech_end == on_speech_end
        assert vad._on_vad_event == on_vad_event


# =============================================================================
# 헬퍼 함수 테스트
# =============================================================================

class TestHelperFunctions:
    """헬퍼 함수 테스트"""

    def test_create_interview_vad_normal(self):
        """일반 면접 VAD 생성 테스트"""
        vad = create_interview_vad(relaxed=False)

        assert vad.mode == VADMode.INTERVIEW_NORMAL
        assert vad.config.min_silence_duration == 0.7

    def test_create_interview_vad_relaxed(self):
        """편안한 면접 VAD 생성 테스트"""
        vad = create_interview_vad(relaxed=True)

        assert vad.mode == VADMode.INTERVIEW_RELAXED
        assert vad.config.min_silence_duration == 1.0

    def test_create_interview_vad_with_callbacks(self):
        """콜백과 함께 면접 VAD 생성"""
        on_speech_start = Mock()
        on_speech_end = Mock()

        vad = create_interview_vad(
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
        )

        assert vad._on_speech_start == on_speech_start
        assert vad._on_speech_end == on_speech_end


# =============================================================================
# 통합 테스트
# =============================================================================

class TestIntegration:
    """통합 테스트"""

    @pytest.mark.asyncio
    async def test_stt_with_vad(self, mock_deepgram_api_key):
        """STT와 VAD 함께 사용하는 통합 테스트"""
        transcript_results = []

        def on_transcript(result: TranscriptResult):
            transcript_results.append(result)

        vad_events = []

        def on_vad_event(event: dict):
            vad_events.append(event)

        # STT 생성
        with patch('src.stt.deepgram_service.DeepgramClient') as mock_client:
            mock_connection = AsyncMock()
            mock_connection.start = AsyncMock(return_value=True)
            mock_connection.send = Mock()
            mock_client.return_value.listen.asyncwebsocket.v.return_value = mock_connection

            stt = DeepgramSTTService(
                api_key=mock_deepgram_api_key,
                on_transcript=on_transcript,
            )

            # VAD 생성
            vad = create_interview_vad(on_vad_event=on_vad_event)

            # STT 연결
            await stt.connect()
            assert stt.is_connected

            await stt.disconnect()

    @pytest.mark.asyncio
    async def test_latency_measurement(self, mock_deepgram_api_key, sample_audio_chunk):
        """레이턴시 측정 테스트 (목표: 100ms 이하)"""
        import time

        with patch('src.stt.deepgram_service.DeepgramClient') as mock_client:
            mock_connection = AsyncMock()
            mock_connection.start = AsyncMock(return_value=True)
            mock_connection.send = Mock()
            mock_client.return_value.listen.asyncwebsocket.v.return_value = mock_connection

            stt = DeepgramSTTService(api_key=mock_deepgram_api_key)
            await stt.connect()

            # 오디오 전송 시간 측정
            start_time = time.perf_counter()
            await stt.send_audio(sample_audio_chunk)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000

            # 오디오 전송 자체는 매우 빠름 (네트워크 레이턴시 제외)
            assert latency_ms < 100, f"Audio send latency too high: {latency_ms:.2f}ms"

            await stt.disconnect()


# =============================================================================
# 마커 기반 테스트 (실제 API 사용 - 선택적 실행)
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_deepgram_connection():
    """
    실제 Deepgram API 연결 테스트

    실행:
    pytest tests/test_stt.py -m integration --api-key=YOUR_KEY
    """
    from config import settings

    # 실제 API 키가 있어야 실행
    if not settings.deepgram_api_key or settings.deepgram_api_key == "your_deepgram_api_key_here":
        pytest.skip("No valid Deepgram API key")

    transcript_received = asyncio.Event()
    transcript_text = []

    def on_transcript(result: TranscriptResult):
        if result.is_final:
            transcript_text.append(result.text)
            transcript_received.set()

    async with DeepgramSTTService(
        language="ko",
        model="nova-3",
        on_transcript=on_transcript,
    ) as stt:
        assert stt.is_connected

        # 테스트 오디오 전송 (실제 환경에서는 마이크 입력)
        # 여기서는 연결 테스트만 수행
        await asyncio.sleep(1)


@pytest.mark.integration
def test_real_vad_initialization():
    """
    실제 Silero VAD 초기화 테스트

    실행:
    pytest tests/test_stt.py -m integration
    """
    vad = SileroVADAnalyzer(mode=VADMode.INTERVIEW_NORMAL)

    # 실제 모델 로딩은 시간이 걸리므로 선택적으로 실행
    # vad.initialize()  # 주석 해제하면 실제 모델 다운로드

    assert vad.mode == VADMode.INTERVIEW_NORMAL


# =============================================================================
# 테스트 실행 설정
# =============================================================================

if __name__ == "__main__":
    """직접 실행 시 pytest 실행"""
    import sys

    # 기본 테스트 실행 (integration 제외)
    sys.exit(pytest.main([__file__, "-v", "-m", "not integration"]))
