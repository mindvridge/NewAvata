"""
Deepgram STT Service
실시간 스트리밍 음성 인식 서비스
"""

import asyncio
import time
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from loguru import logger

from config import settings


@dataclass
class TranscriptResult:
    """음성 인식 결과"""
    text: str
    is_final: bool
    confidence: float
    language: str
    duration: float
    timestamp: float


class DeepgramSTTService:
    """
    Deepgram 실시간 STT 서비스

    WebSocket을 통한 실시간 음성 인식을 제공합니다.
    Pipecat의 DeepgramSTTService와 호환되도록 설계되었습니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "ko",
        model: str = "nova-3",
        sample_rate: int = 16000,
        channels: int = 1,
        interim_results: bool = True,
        vad_enabled: bool = True,
        smart_format: bool = True,
        punctuate: bool = True,
        on_transcript: Optional[Callable[[TranscriptResult], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_speech_ended: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Args:
            api_key: Deepgram API 키 (None이면 settings에서 로드)
            language: 언어 코드 (ko, en, ja 등)
            model: Deepgram 모델 (nova-3, nova-2, enhanced, base)
            sample_rate: 오디오 샘플레이트 (Hz)
            channels: 오디오 채널 수
            interim_results: 중간 결과 반환 여부
            vad_enabled: Voice Activity Detection 활성화
            smart_format: 스마트 포맷팅 (숫자, 날짜 등)
            punctuate: 구두점 자동 추가
            on_transcript: 텍스트 인식 시 콜백
            on_speech_started: 발화 시작 시 콜백
            on_speech_ended: 발화 종료 시 콜백
            on_error: 에러 발생 시 콜백
        """
        self.api_key = api_key or settings.deepgram_api_key
        self.language = language
        self.model = model
        self.sample_rate = sample_rate
        self.channels = channels
        self.interim_results = interim_results
        self.vad_enabled = vad_enabled
        self.smart_format = smart_format
        self.punctuate = punctuate

        # 콜백 함수들
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._on_speech_ended = on_speech_ended
        self._on_error = on_error

        # Deepgram 클라이언트
        self._client: Optional[DeepgramClient] = None
        self._connection = None
        self._is_connected = False
        self._is_speaking = False

        # 레이턴시 측정
        self._audio_start_time: Optional[float] = None
        self._last_transcript_time: Optional[float] = None

        # 재연결 설정
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 1.0  # 초

        logger.info(
            f"DeepgramSTTService initialized: model={model}, language={language}, "
            f"sample_rate={sample_rate}, vad_enabled={vad_enabled}"
        )

    async def connect(self) -> None:
        """Deepgram WebSocket 연결을 시작합니다."""
        if self._is_connected:
            logger.warning("Already connected to Deepgram")
            return

        try:
            # Deepgram 클라이언트 생성
            config = DeepgramClientOptions(
                options={"keepalive": "true"}
            )
            self._client = DeepgramClient(self.api_key, config)

            # 연결 옵션 설정
            options = LiveOptions(
                model=self.model,
                language=self.language,
                encoding="linear16",
                sample_rate=self.sample_rate,
                channels=self.channels,
                interim_results=self.interim_results,
                smart_format=self.smart_format,
                punctuate=self.punctuate,
                vad_events=self.vad_enabled,
                endpointing=300 if self.vad_enabled else False,  # 300ms silence
            )

            # WebSocket 연결 생성
            self._connection = self._client.listen.asyncwebsocket.v("1")

            # 이벤트 핸들러 등록
            self._connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
            self._connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started_event)
            self._connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
            self._connection.on(LiveTranscriptionEvents.Error, self._on_error_event)
            self._connection.on(LiveTranscriptionEvents.Close, self._on_close)

            # 연결 시작
            if not await self._connection.start(options):
                raise Exception("Failed to start Deepgram connection")

            self._is_connected = True
            logger.info("Successfully connected to Deepgram")

        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            if self._on_error:
                self._on_error(e)
            raise

    async def disconnect(self) -> None:
        """Deepgram WebSocket 연결을 종료합니다."""
        if not self._is_connected:
            return

        try:
            if self._connection:
                await self._connection.finish()
                self._connection = None

            self._is_connected = False
            logger.info("Disconnected from Deepgram")

        except Exception as e:
            logger.error(f"Error disconnecting from Deepgram: {e}")

    async def send_audio(self, audio_data: bytes) -> None:
        """
        오디오 데이터를 Deepgram으로 전송합니다.

        Args:
            audio_data: PCM 오디오 데이터 (linear16)
        """
        if not self._is_connected or not self._connection:
            logger.warning("Not connected to Deepgram, attempting to reconnect...")
            await self._reconnect()
            return

        try:
            # 첫 오디오 청크 시간 기록
            if self._audio_start_time is None:
                self._audio_start_time = time.time()

            # 오디오 전송
            self._connection.send(audio_data)

        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")
            if self._on_error:
                self._on_error(e)
            await self._reconnect()

    async def _reconnect(self) -> None:
        """연결 재시도 로직"""
        for attempt in range(self._max_reconnect_attempts):
            try:
                logger.info(f"Reconnecting to Deepgram (attempt {attempt + 1}/{self._max_reconnect_attempts})")
                await self.disconnect()
                await asyncio.sleep(self._reconnect_delay)
                await self.connect()
                logger.info("Reconnection successful")
                return
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
                if attempt == self._max_reconnect_attempts - 1:
                    logger.error("Max reconnection attempts reached")
                    if self._on_error:
                        self._on_error(Exception("Failed to reconnect to Deepgram"))

    # =========================================================================
    # 이벤트 핸들러들
    # =========================================================================

    def _on_open(self, *args, **kwargs) -> None:
        """연결 열림 이벤트"""
        logger.debug("Deepgram connection opened")

    def _on_message(self, *args, **kwargs) -> None:
        """음성 인식 결과 수신 이벤트"""
        try:
            result = kwargs.get("result")
            if not result:
                return

            # 채널 데이터 추출
            channel = result.channel
            if not channel or not channel.alternatives:
                return

            alternative = channel.alternatives[0]
            transcript_text = alternative.transcript

            # 빈 텍스트 무시
            if not transcript_text or transcript_text.strip() == "":
                return

            # 최종/중간 결과 구분
            is_final = result.is_final if hasattr(result, 'is_final') else False
            confidence = alternative.confidence if hasattr(alternative, 'confidence') else 0.0
            duration = result.duration if hasattr(result, 'duration') else 0.0

            # 레이턴시 계산
            current_time = time.time()
            latency_ms = 0
            if self._audio_start_time:
                latency_ms = (current_time - self._audio_start_time) * 1000

            # 결과 객체 생성
            transcript_result = TranscriptResult(
                text=transcript_text,
                is_final=is_final,
                confidence=confidence,
                language=self.language,
                duration=duration,
                timestamp=current_time,
            )

            # 로깅
            result_type = "FINAL" if is_final else "INTERIM"
            logger.debug(
                f"[{result_type}] Transcript: '{transcript_text}' "
                f"(confidence: {confidence:.2f}, latency: {latency_ms:.0f}ms)"
            )

            # 콜백 실행
            if self._on_transcript:
                self._on_transcript(transcript_result)

            # 최종 결과면 타이머 리셋
            if is_final:
                self._audio_start_time = None
                self._last_transcript_time = current_time

        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
            if self._on_error:
                self._on_error(e)

    def _on_speech_started_event(self, *args, **kwargs) -> None:
        """발화 시작 이벤트"""
        if not self._is_speaking:
            self._is_speaking = True
            self._audio_start_time = time.time()
            logger.debug("Speech started")

            if self._on_speech_started:
                self._on_speech_started()

    def _on_utterance_end(self, *args, **kwargs) -> None:
        """발화 종료 이벤트"""
        if self._is_speaking:
            self._is_speaking = False
            logger.debug("Speech ended")

            if self._on_speech_ended:
                self._on_speech_ended()

    def _on_error_event(self, *args, **kwargs) -> None:
        """에러 이벤트"""
        error = kwargs.get("error", "Unknown error")
        logger.error(f"Deepgram error: {error}")

        if self._on_error:
            self._on_error(Exception(str(error)))

    def _on_close(self, *args, **kwargs) -> None:
        """연결 종료 이벤트"""
        logger.debug("Deepgram connection closed")
        self._is_connected = False

    # =========================================================================
    # 유틸리티 메서드
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """연결 상태 반환"""
        return self._is_connected

    @property
    def is_speaking(self) -> bool:
        """현재 발화 중 여부"""
        return self._is_speaking

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.disconnect()


# =============================================================================
# Pipecat 호환 래퍼 (선택사항)
# =============================================================================

class PipecatDeepgramSTTService(DeepgramSTTService):
    """
    Pipecat 프레임워크와의 완전한 호환성을 위한 래퍼 클래스

    Pipecat의 표준 인터페이스를 따릅니다.
    """

    async def run(self, audio_frames) -> None:
        """
        Pipecat 프레임워크에서 사용하는 표준 run 메서드

        Args:
            audio_frames: 오디오 프레임 스트림
        """
        await self.connect()

        try:
            async for frame in audio_frames:
                await self.send_audio(frame.audio)
        except Exception as e:
            logger.error(f"Error in STT pipeline: {e}")
            raise
        finally:
            await self.disconnect()


if __name__ == "__main__":
    # 테스트/예시 코드
    import sys

    async def test_stt():
        """STT 서비스 테스트"""

        def on_transcript(result: TranscriptResult):
            print(f"[{'FINAL' if result.is_final else 'INTERIM'}] {result.text}")

        def on_speech_started():
            print("[EVENT] Speech started")

        def on_speech_ended():
            print("[EVENT] Speech ended")

        def on_error(error: Exception):
            print(f"[ERROR] {error}")

        # STT 서비스 생성
        stt_service = DeepgramSTTService(
            language="ko",
            model="nova-3",
            on_transcript=on_transcript,
            on_speech_started=on_speech_started,
            on_speech_ended=on_speech_ended,
            on_error=on_error,
        )

        # 연결
        async with stt_service:
            print("Connected to Deepgram. Press Ctrl+C to stop.")
            print("Note: This is a test mode. Connect a real audio source for actual transcription.")

            # 실제 사용 시에는 오디오 스트림을 여기서 처리
            try:
                await asyncio.sleep(3600)  # 1시간 대기 (실제로는 오디오 처리)
            except KeyboardInterrupt:
                print("\nStopping...")

    # 테스트 실행
    asyncio.run(test_stt())
