"""
ElevenLabs TTS Service
실시간 스트리밍 음성 합성 서비스
"""

import asyncio
import time
import io
from typing import Optional, Callable, AsyncIterator, List
from dataclasses import dataclass

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from loguru import logger
import numpy as np

from config import settings


@dataclass
class TTSConfig:
    """TTS 설정"""
    voice_id: str                          # ElevenLabs 음성 ID
    model: str = "eleven_multilingual_v2"  # 모델 (한국어 지원)
    stability: float = 0.5                 # 안정성 (0.0-1.0)
    similarity_boost: float = 0.75         # 유사도 부스트 (0.0-1.0)
    style: float = 0.0                     # 스타일 (0.0-1.0)
    use_speaker_boost: bool = True         # 스피커 부스트
    optimize_streaming_latency: int = 3    # 스트리밍 레이턴시 최적화 (0-4)
    output_format: str = "pcm_16000"       # 출력 포맷 (MuseTalk 호환)
    chunk_size: int = 1024                 # 청크 크기 (바이트)


@dataclass
class AudioChunk:
    """오디오 청크"""
    data: bytes                    # PCM 오디오 데이터
    sample_rate: int = 16000       # 샘플레이트
    channels: int = 1              # 채널 수
    sample_width: int = 2          # 샘플 너비 (바이트)
    timestamp: float = 0.0         # 타임스탬프


class TextChunker:
    """
    텍스트 청킹 유틸리티

    긴 문장을 자연스러운 단위로 분할합니다.
    """

    def __init__(
        self,
        max_chunk_size: int = 500,
        sentence_endings: str = ".!?。！？",
        comma_pause: bool = True,
    ):
        """
        Args:
            max_chunk_size: 최대 청크 크기 (문자 수)
            sentence_endings: 문장 종결 문자
            comma_pause: 쉼표에서도 분할할지 여부
        """
        self.max_chunk_size = max_chunk_size
        self.sentence_endings = sentence_endings
        self.comma_pause = comma_pause

    def chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할

        Args:
            text: 입력 텍스트

        Returns:
            List[str]: 분할된 텍스트 청크들
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        # 문장 단위로 분할
        sentences = self._split_sentences(text)

        for sentence in sentences:
            # 현재 청크에 추가했을 때 크기 확인
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # 단일 문장이 너무 길면 강제 분할
                    chunks.extend(self._force_split(sentence))
            else:
                current_chunk += sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """문장 단위로 분할"""
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in self.sentence_endings:
                sentences.append(current)
                current = ""
            elif self.comma_pause and char in ",，":
                sentences.append(current)
                current = ""

        if current:
            sentences.append(current)

        return sentences

    def _force_split(self, text: str) -> List[str]:
        """강제로 텍스트 분할 (너무 긴 문장)"""
        return [
            text[i:i + self.max_chunk_size]
            for i in range(0, len(text), self.max_chunk_size)
        ]


class ElevenLabsTTSService:
    """
    ElevenLabs 실시간 TTS 서비스

    스트리밍 방식으로 텍스트를 음성으로 변환합니다.
    Pipecat의 ElevenLabsTTSService와 호환됩니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[TTSConfig] = None,
        on_audio_chunk: Optional[Callable[[AudioChunk], None]] = None,
        on_synthesis_start: Optional[Callable[[str], None]] = None,
        on_synthesis_end: Optional[Callable[[float], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Args:
            api_key: ElevenLabs API 키
            config: TTS 설정
            on_audio_chunk: 오디오 청크 수신 시 콜백
            on_synthesis_start: 합성 시작 시 콜백 (텍스트 전달)
            on_synthesis_end: 합성 종료 시 콜백 (소요 시간 전달)
            on_error: 에러 발생 시 콜백
        """
        self.api_key = api_key or settings.elevenlabs_api_key

        # 설정 로드
        if config is None:
            config = TTSConfig(
                voice_id=settings.tts_voice_id,
                stability=settings.tts_stability,
                similarity_boost=settings.tts_similarity_boost,
                optimize_streaming_latency=settings.tts_optimize_streaming_latency,
            )
        self.config = config

        # 콜백 함수들
        self._on_audio_chunk = on_audio_chunk
        self._on_synthesis_start = on_synthesis_start
        self._on_synthesis_end = on_synthesis_end
        self._on_error = on_error

        # ElevenLabs 클라이언트
        self._client = ElevenLabs(api_key=self.api_key)

        # 음성 설정
        self._voice_settings = VoiceSettings(
            stability=self.config.stability,
            similarity_boost=self.config.similarity_boost,
            style=self.config.style,
            use_speaker_boost=self.config.use_speaker_boost,
        )

        # 텍스트 청커
        self._text_chunker = TextChunker()

        # 통계
        self._total_chars_synthesized = 0
        self._total_synthesis_time = 0.0
        self._ttfb_times = []  # Time To First Byte

        logger.info(
            f"ElevenLabsTTSService initialized: voice_id={self.config.voice_id}, "
            f"model={self.config.model}, output_format={self.config.output_format}"
        )

    async def synthesize(self, text: str, streaming: bool = True) -> Optional[bytes]:
        """
        텍스트를 음성으로 합성

        Args:
            text: 합성할 텍스트
            streaming: 스트리밍 모드 (True면 청크 단위 반환, False면 전체 반환)

        Returns:
            bytes: 오디오 데이터 (streaming=False일 때만)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        text = text.strip()
        start_time = time.time()

        try:
            # 합성 시작 콜백
            if self._on_synthesis_start:
                self._on_synthesis_start(text)

            logger.debug(f"Synthesizing: {text[:50]}{'...' if len(text) > 50 else ''}")

            if streaming:
                # 스트리밍 합성
                await self._synthesize_streaming(text)
                return None
            else:
                # 전체 합성
                audio_data = await self._synthesize_full(text)

                # 합성 종료 콜백
                synthesis_time = time.time() - start_time
                if self._on_synthesis_end:
                    self._on_synthesis_end(synthesis_time)

                # 통계 업데이트
                self._total_chars_synthesized += len(text)
                self._total_synthesis_time += synthesis_time

                logger.debug(
                    f"Synthesis completed: {len(text)} chars in {synthesis_time:.2f}s "
                    f"({len(audio_data)} bytes)"
                )

                return audio_data

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            if self._on_error:
                self._on_error(e)
            raise

    async def _synthesize_streaming(self, text: str) -> None:
        """스트리밍 방식 합성"""
        ttfb_start = time.time()
        first_byte_received = False
        total_bytes = 0

        try:
            # ElevenLabs 스트리밍 API 호출
            audio_stream = self._client.generate(
                text=text,
                voice=self.config.voice_id,
                model=self.config.model,
                voice_settings=self._voice_settings,
                optimize_streaming_latency=self.config.optimize_streaming_latency,
                stream=True,
            )

            # 청크 단위로 처리
            for chunk in audio_stream:
                if not first_byte_received:
                    ttfb = (time.time() - ttfb_start) * 1000  # ms
                    self._ttfb_times.append(ttfb)
                    logger.debug(f"TTFB: {ttfb:.0f}ms")
                    first_byte_received = True

                if chunk:
                    total_bytes += len(chunk)

                    # AudioChunk 생성
                    audio_chunk = AudioChunk(
                        data=chunk,
                        sample_rate=16000,
                        channels=1,
                        sample_width=2,
                        timestamp=time.time(),
                    )

                    # 콜백 호출
                    if self._on_audio_chunk:
                        self._on_audio_chunk(audio_chunk)

                    # 비동기 처리를 위해 약간의 딜레이
                    await asyncio.sleep(0.001)

            logger.debug(f"Streaming completed: {total_bytes} bytes received")

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            raise

    async def _synthesize_full(self, text: str) -> bytes:
        """전체 합성 (비스트리밍)"""
        try:
            # ElevenLabs API 호출 (비동기 실행)
            audio_generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.generate(
                    text=text,
                    voice=self.config.voice_id,
                    model=self.config.model,
                    voice_settings=self._voice_settings,
                    stream=False,
                )
            )

            # 오디오 데이터 수집
            audio_data = b"".join(audio_generator)
            return audio_data

        except Exception as e:
            logger.error(f"Full synthesis error: {e}")
            raise

    async def synthesize_chunked(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        긴 텍스트를 청크로 나눠서 합성

        Args:
            text: 합성할 텍스트

        Yields:
            AudioChunk: 오디오 청크
        """
        # 텍스트 분할
        chunks = self._text_chunker.chunk_text(text)
        logger.debug(f"Text split into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {i}/{len(chunks)}: {chunk[:30]}...")

            # 청크 합성
            audio_data = await self._synthesize_full(chunk)

            if audio_data:
                yield AudioChunk(
                    data=audio_data,
                    sample_rate=16000,
                    channels=1,
                    sample_width=2,
                    timestamp=time.time(),
                )

    def get_available_voices(self) -> list:
        """
        사용 가능한 음성 목록 조회

        Returns:
            list: 음성 정보 리스트
        """
        try:
            voices = self._client.voices.get_all()
            return voices.voices
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []

    def get_voice_info(self, voice_id: Optional[str] = None) -> Optional[dict]:
        """
        특정 음성 정보 조회

        Args:
            voice_id: 음성 ID (None이면 현재 설정된 음성)

        Returns:
            dict: 음성 정보
        """
        voice_id = voice_id or self.config.voice_id

        try:
            voice = self._client.voices.get(voice_id)
            return {
                "voice_id": voice.voice_id,
                "name": voice.name,
                "category": voice.category,
                "description": voice.description,
                "labels": voice.labels,
            }
        except Exception as e:
            logger.error(f"Failed to get voice info: {e}")
            return None

    def get_stats(self) -> dict:
        """
        TTS 통계 조회

        Returns:
            dict: 통계 정보
        """
        avg_ttfb = sum(self._ttfb_times) / len(self._ttfb_times) if self._ttfb_times else 0

        return {
            "total_chars_synthesized": self._total_chars_synthesized,
            "total_synthesis_time": self._total_synthesis_time,
            "average_ttfb_ms": avg_ttfb,
            "ttfb_samples": len(self._ttfb_times),
        }

    def reset_stats(self) -> None:
        """통계 초기화"""
        self._total_chars_synthesized = 0
        self._total_synthesis_time = 0.0
        self._ttfb_times = []
        logger.debug("TTS stats reset")


# =============================================================================
# Pipecat 호환 래퍼
# =============================================================================

class PipecatElevenLabsTTSService(ElevenLabsTTSService):
    """
    Pipecat 프레임워크와의 완전한 호환성을 위한 래퍼 클래스
    """

    async def run(self, text_frames) -> None:
        """
        Pipecat 프레임워크에서 사용하는 표준 run 메서드

        Args:
            text_frames: 텍스트 프레임 스트림
        """
        try:
            async for frame in text_frames:
                text = frame.text
                await self.synthesize(text, streaming=True)
        except Exception as e:
            logger.error(f"Error in TTS pipeline: {e}")
            raise


# =============================================================================
# 헬퍼 함수
# =============================================================================

def create_elevenlabs_tts(
    voice_id: Optional[str] = None,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    on_audio_chunk: Optional[Callable] = None,
) -> ElevenLabsTTSService:
    """
    ElevenLabs TTS 서비스를 생성하는 헬퍼 함수

    Args:
        voice_id: 음성 ID (None이면 설정에서 로드)
        stability: 안정성
        similarity_boost: 유사도 부스트
        on_audio_chunk: 오디오 청크 콜백

    Returns:
        ElevenLabsTTSService: TTS 서비스
    """
    config = TTSConfig(
        voice_id=voice_id or settings.tts_voice_id,
        stability=stability,
        similarity_boost=similarity_boost,
    )

    return ElevenLabsTTSService(
        config=config,
        on_audio_chunk=on_audio_chunk,
    )


# =============================================================================
# 테스트/예시 코드
# =============================================================================

if __name__ == "__main__":
    """TTS 서비스 테스트"""
    import sys

    async def test_tts():
        """TTS 서비스 기본 테스트"""

        def on_audio_chunk(chunk: AudioChunk):
            print(f"Audio chunk received: {len(chunk.data)} bytes")

        def on_synthesis_start(text: str):
            print(f"\n[TTS] Synthesizing: {text[:50]}...")

        def on_synthesis_end(duration: float):
            print(f"[TTS] Completed in {duration:.2f}s")

        # TTS 서비스 생성
        tts = create_elevenlabs_tts(
            on_audio_chunk=on_audio_chunk,
        )

        # 음성 정보 확인
        voice_info = tts.get_voice_info()
        if voice_info:
            print(f"\nUsing voice: {voice_info['name']}")
            print(f"Description: {voice_info['description']}")

        # 테스트 텍스트
        test_texts = [
            "안녕하세요. ElevenLabs 음성 합성 테스트입니다.",
            "한국어 지원이 잘 되는지 확인해보겠습니다.",
            "실시간 스트리밍 모드로 동작합니다.",
        ]

        print("\n" + "=" * 80)
        print("TTS Synthesis Test")
        print("=" * 80)

        for i, text in enumerate(test_texts, 1):
            print(f"\n[Test {i}] {text}")
            await tts.synthesize(text, streaming=True)
            await asyncio.sleep(1)

        # 통계 출력
        stats = tts.get_stats()
        print("\n" + "=" * 80)
        print("Statistics")
        print("=" * 80)
        print(f"Total characters: {stats['total_chars_synthesized']}")
        print(f"Total time: {stats['total_synthesis_time']:.2f}s")
        print(f"Average TTFB: {stats['average_ttfb_ms']:.0f}ms")

    # 테스트 실행
    asyncio.run(test_tts())
