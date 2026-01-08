"""
Deepgram STT Service 사용 예시

실제 마이크 입력을 받아서 실시간으로 음성 인식하는 예제입니다.
"""

import asyncio
import pyaudio
from loguru import logger

from src.stt.deepgram_service import DeepgramSTTService, TranscriptResult


class AudioStreamer:
    """마이크 오디오 스트리밍 클래스"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        """오디오 스트림 시작"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        logger.info("Audio stream started")

    def stop(self):
        """오디오 스트림 종료"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        logger.info("Audio stream stopped")

    def read(self) -> bytes:
        """오디오 청크 읽기"""
        if self.stream:
            return self.stream.read(self.chunk_size, exception_on_overflow=False)
        return b""


async def example_basic_usage():
    """기본 사용 예제"""

    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # 콜백 함수 정의
    def on_transcript(result: TranscriptResult):
        if result.is_final:
            print(f"\n✓ [{result.confidence:.2f}] {result.text}")
        else:
            print(f"  ... {result.text}", end="\r")

    def on_speech_started():
        print("\n🎤 [Speech Started]")

    def on_speech_ended():
        print("\n🔇 [Speech Ended]")

    # STT 서비스 생성 및 연결
    stt_service = DeepgramSTTService(
        language="ko",
        model="nova-3",
        interim_results=True,
        vad_enabled=True,
        on_transcript=on_transcript,
        on_speech_started=on_speech_started,
        on_speech_ended=on_speech_ended,
    )

    await stt_service.connect()

    print("\n마이크로 말해보세요. Ctrl+C를 눌러 종료합니다.\n")

    # 오디오 스트리머 시작
    audio_streamer = AudioStreamer(sample_rate=16000)
    audio_streamer.start()

    try:
        # 오디오 데이터를 계속 전송
        while True:
            audio_data = audio_streamer.read()
            await stt_service.send_audio(audio_data)
            await asyncio.sleep(0.01)  # 작은 딜레이

    except KeyboardInterrupt:
        print("\n\n종료 중...")
    finally:
        audio_streamer.stop()
        await stt_service.disconnect()


async def example_with_context_manager():
    """컨텍스트 매니저 사용 예제"""

    print("=" * 80)
    print("Example 2: With Context Manager")
    print("=" * 80)

    transcript_buffer = []

    def on_transcript(result: TranscriptResult):
        if result.is_final:
            transcript_buffer.append(result.text)
            print(f"\n[{len(transcript_buffer)}] {result.text}")

    # 컨텍스트 매니저로 자동 연결/해제
    async with DeepgramSTTService(
        language="ko",
        model="nova-3",
        on_transcript=on_transcript,
    ) as stt_service:
        audio_streamer = AudioStreamer()
        audio_streamer.start()

        try:
            print("\n마이크로 말해보세요. (10초 후 자동 종료)\n")

            # 10초 동안만 실행
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < 10:
                audio_data = audio_streamer.read()
                await stt_service.send_audio(audio_data)
                await asyncio.sleep(0.01)

        finally:
            audio_streamer.stop()

    print(f"\n총 {len(transcript_buffer)}개의 문장이 인식되었습니다:")
    for i, text in enumerate(transcript_buffer, 1):
        print(f"{i}. {text}")


async def example_multi_language():
    """다국어 지원 예제"""

    print("=" * 80)
    print("Example 3: Multi-language Support")
    print("=" * 80)

    languages = [
        ("ko", "한국어로 말해보세요"),
        ("en", "Speak in English"),
        ("ja", "日本語で話してください"),
    ]

    for lang_code, instruction in languages:
        print(f"\n{instruction} (5초)")

        def on_transcript(result: TranscriptResult):
            if result.is_final:
                print(f"  → {result.text}")

        async with DeepgramSTTService(
            language=lang_code,
            model="nova-3",
            on_transcript=on_transcript,
        ) as stt_service:
            audio_streamer = AudioStreamer()
            audio_streamer.start()

            try:
                start_time = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - start_time < 5:
                    audio_data = audio_streamer.read()
                    await stt_service.send_audio(audio_data)
                    await asyncio.sleep(0.01)
            finally:
                audio_streamer.stop()


async def example_error_handling():
    """에러 핸들링 예제"""

    print("=" * 80)
    print("Example 4: Error Handling")
    print("=" * 80)

    error_count = 0

    def on_transcript(result: TranscriptResult):
        print(f"✓ {result.text}")

    def on_error(error: Exception):
        nonlocal error_count
        error_count += 1
        print(f"⚠️  Error #{error_count}: {error}")

    stt_service = DeepgramSTTService(
        language="ko",
        model="nova-3",
        on_transcript=on_transcript,
        on_error=on_error,
    )

    try:
        await stt_service.connect()
        print("✓ Connected successfully")

        # 상태 확인
        print(f"Is connected: {stt_service.is_connected}")
        print(f"Is speaking: {stt_service.is_speaking}")

        # 오디오 스트리밍 (5초)
        audio_streamer = AudioStreamer()
        audio_streamer.start()

        print("\n마이크로 말해보세요 (5초)\n")
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 5:
            audio_data = audio_streamer.read()
            await stt_service.send_audio(audio_data)
            await asyncio.sleep(0.01)

        audio_streamer.stop()

    except Exception as e:
        print(f"❌ Exception: {e}")
    finally:
        await stt_service.disconnect()
        print(f"\n총 {error_count}개의 에러가 발생했습니다.")


async def main():
    """메인 함수 - 예제 선택"""

    print("\n실시간 면접 아바타 - Deepgram STT 예제")
    print("=" * 80)
    print("1. 기본 사용")
    print("2. 컨텍스트 매니저 사용")
    print("3. 다국어 지원")
    print("4. 에러 핸들링")
    print("=" * 80)

    choice = input("\n실행할 예제 번호를 선택하세요 (1-4, 기본값: 1): ").strip()

    if choice == "2":
        await example_with_context_manager()
    elif choice == "3":
        await example_multi_language()
    elif choice == "4":
        await example_error_handling()
    else:
        await example_basic_usage()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n프로그램이 종료되었습니다.")
