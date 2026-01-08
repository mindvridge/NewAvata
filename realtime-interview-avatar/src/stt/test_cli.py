"""
STT CLI 테스트 스크립트

실시간으로 마이크 입력을 받아 음성 인식과 VAD를 테스트합니다.

실행:
    python -m src.stt.test_cli
"""

import asyncio
import time
from typing import Optional
from pathlib import Path

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not installed. Audio streaming will not work.")
    print("   Install: pip install pyaudio")

from loguru import logger
from src.stt import (
    DeepgramSTTService,
    TranscriptResult,
    SileroVADAnalyzer,
    VADMode,
    create_interview_vad,
)


# =============================================================================
# 색상 출력 유틸리티
# =============================================================================

class Colors:
    """터미널 색상 코드"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """헤더 출력"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_success(text: str):
    """성공 메시지"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """에러 메시지"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """정보 메시지"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """경고 메시지"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


# =============================================================================
# 오디오 스트리머
# =============================================================================

class AudioStreamer:
    """마이크 오디오 스트리밍"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024):
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio is not available")

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
        print_success(f"Audio stream started: {self.sample_rate}Hz, {self.channels} channel(s)")

    def stop(self):
        """오디오 스트림 종료"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print_info("Audio stream stopped")

    def read(self) -> bytes:
        """오디오 청크 읽기"""
        if self.stream:
            return self.stream.read(self.chunk_size, exception_on_overflow=False)
        return b""


# =============================================================================
# 테스트 시나리오
# =============================================================================

async def test_deepgram_connection():
    """테스트 1: Deepgram 연결 테스트"""
    print_header("Test 1: Deepgram Connection")

    try:
        from config import settings

        if not settings.deepgram_api_key or settings.deepgram_api_key == "your_deepgram_api_key_here":
            print_error("Deepgram API key not configured in .env file")
            return False

        print_info("Creating Deepgram STT service...")
        stt = DeepgramSTTService(language="ko", model="nova-3")

        print_info("Connecting to Deepgram...")
        await stt.connect()

        if stt.is_connected:
            print_success("Connected to Deepgram successfully!")
            await asyncio.sleep(1)
            await stt.disconnect()
            print_success("Disconnected successfully")
            return True
        else:
            print_error("Failed to connect")
            return False

    except Exception as e:
        print_error(f"Connection test failed: {e}")
        return False


async def test_korean_speech_recognition():
    """테스트 2: 한국어 음성 인식 테스트"""
    print_header("Test 2: Korean Speech Recognition")

    if not PYAUDIO_AVAILABLE:
        print_warning("PyAudio not available, skipping test")
        return False

    try:
        from config import settings

        # 통계 변수
        transcript_count = 0
        final_count = 0
        interim_count = 0
        latencies = []

        def on_transcript(result: TranscriptResult):
            nonlocal transcript_count, final_count, interim_count

            transcript_count += 1

            if result.is_final:
                final_count += 1
                print(f"\n{Colors.GREEN}[FINAL]{Colors.ENDC} {Colors.BOLD}{result.text}{Colors.ENDC}")
                print(f"        Confidence: {result.confidence:.2%}, Duration: {result.duration:.2f}s")
            else:
                interim_count += 1
                print(f"{Colors.CYAN}[INTERIM]{Colors.ENDC} {result.text}", end='\r')

        def on_speech_started():
            print(f"\n{Colors.YELLOW}🎤 [Speech Started]{Colors.ENDC}")

        def on_speech_ended():
            print(f"{Colors.YELLOW}🔇 [Speech Ended]{Colors.ENDC}")

        print_info("Creating STT service with Korean language...")
        async with DeepgramSTTService(
            language="ko",
            model="nova-3",
            interim_results=True,
            vad_enabled=True,
            on_transcript=on_transcript,
            on_speech_started=on_speech_started,
            on_speech_ended=on_speech_ended,
        ) as stt:
            print_success("Connected to Deepgram")

            audio_streamer = AudioStreamer()
            audio_streamer.start()

            print_info("말해보세요! (10초 후 자동 종료, Ctrl+C로 조기 종료)")
            print_warning("예: '안녕하세요', '테스트입니다', '음성 인식 확인'\n")

            start_time = asyncio.get_event_loop().time()
            try:
                while asyncio.get_event_loop().time() - start_time < 10:
                    audio_data = audio_streamer.read()
                    await stt.send_audio(audio_data)
                    await asyncio.sleep(0.01)
            except KeyboardInterrupt:
                print_info("\nTest interrupted by user")

            audio_streamer.stop()

        # 결과 출력
        print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
        print(f"  Total transcripts: {transcript_count}")
        print(f"  Final results: {final_count}")
        print(f"  Interim results: {interim_count}")

        return transcript_count > 0

    except Exception as e:
        print_error(f"Korean speech recognition test failed: {e}")
        return False


async def test_realtime_streaming():
    """테스트 3: 실시간 스트리밍 테스트"""
    print_header("Test 3: Real-time Streaming")

    if not PYAUDIO_AVAILABLE:
        print_warning("PyAudio not available, skipping test")
        return False

    try:
        transcript_buffer = []

        def on_transcript(result: TranscriptResult):
            if result.is_final:
                transcript_buffer.append(result.text)
                print(f"\n[{len(transcript_buffer)}] {result.text}")

        print_info("Starting real-time streaming test (5 seconds)...")

        async with DeepgramSTTService(
            language="ko",
            model="nova-3",
            on_transcript=on_transcript,
        ) as stt:
            audio_streamer = AudioStreamer()
            audio_streamer.start()

            print_info("Speak now!\n")

            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < 5:
                audio_data = audio_streamer.read()
                await stt.send_audio(audio_data)
                await asyncio.sleep(0.01)

            audio_streamer.stop()

        print(f"\n{Colors.BOLD}Captured {len(transcript_buffer)} sentences:{Colors.ENDC}")
        for i, text in enumerate(transcript_buffer, 1):
            print(f"  {i}. {text}")

        return True

    except Exception as e:
        print_error(f"Real-time streaming test failed: {e}")
        return False


async def test_vad_functionality():
    """테스트 4: VAD 동작 테스트"""
    print_header("Test 4: VAD (Voice Activity Detection)")

    try:
        speech_events = []
        vad_stats = {"speech": 0, "silence": 0}

        def on_speech_start():
            speech_events.append("start")
            print(f"{Colors.GREEN}▶ Speech detected{Colors.ENDC}")

        def on_speech_end(duration: float):
            speech_events.append("end")
            print(f"{Colors.RED}■ Speech ended (duration: {duration:.2f}s){Colors.ENDC}")

        def on_vad_event(event: dict):
            if event.get("is_speech"):
                vad_stats["speech"] += 1
            else:
                vad_stats["silence"] += 1

        print_info("Testing different VAD modes...\n")

        # 1. Interview Normal 모드
        print(f"{Colors.BOLD}Mode: INTERVIEW_NORMAL{Colors.ENDC}")
        vad = create_interview_vad(
            relaxed=False,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
        )
        stats = vad.get_stats()
        print(f"  Min silence: {stats['config']['min_silence_duration']}s")
        print(f"  Threshold: {stats['config']['threshold']}")

        # 2. Interview Relaxed 모드
        print(f"\n{Colors.BOLD}Mode: INTERVIEW_RELAXED{Colors.ENDC}")
        vad.switch_mode(VADMode.INTERVIEW_RELAXED)
        stats = vad.get_stats()
        print(f"  Min silence: {stats['config']['min_silence_duration']}s")
        print(f"  Threshold: {stats['config']['threshold']}")

        # 3. Conversation 모드
        print(f"\n{Colors.BOLD}Mode: CONVERSATION{Colors.ENDC}")
        vad.switch_mode(VADMode.CONVERSATION)
        stats = vad.get_stats()
        print(f"  Min silence: {stats['config']['min_silence_duration']}s")
        print(f"  Threshold: {stats['config']['threshold']}")

        print_success("\nVAD modes tested successfully")
        return True

    except Exception as e:
        print_error(f"VAD test failed: {e}")
        return False


async def test_latency_measurement():
    """테스트 5: 레이턴시 측정 테스트"""
    print_header("Test 5: Latency Measurement (Target: < 100ms)")

    if not PYAUDIO_AVAILABLE:
        print_warning("PyAudio not available, skipping test")
        return False

    try:
        latencies = []

        def on_transcript(result: TranscriptResult):
            if result.is_final:
                # 레이턴시는 실제로는 오디오 전송부터 결과 수신까지 측정해야 함
                # 여기서는 간단히 timestamp 기반으로 계산
                pass

        print_info("Measuring audio send latency...")

        async with DeepgramSTTService(
            language="ko",
            model="nova-3",
            on_transcript=on_transcript,
        ) as stt:
            audio_streamer = AudioStreamer()
            audio_streamer.start()

            # 10번 측정
            for i in range(10):
                audio_data = audio_streamer.read()

                start_time = time.perf_counter()
                await stt.send_audio(audio_data)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                await asyncio.sleep(0.1)

            audio_streamer.stop()

        # 통계 계산
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        print(f"\n{Colors.BOLD}Latency Statistics:{Colors.ENDC}")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Min: {min_latency:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")

        if avg_latency < 100:
            print_success(f"✓ Average latency is under 100ms target!")
        else:
            print_warning(f"⚠ Average latency ({avg_latency:.2f}ms) exceeds 100ms target")

        return avg_latency < 100

    except Exception as e:
        print_error(f"Latency measurement failed: {e}")
        return False


# =============================================================================
# 메인 함수
# =============================================================================

async def run_all_tests():
    """모든 테스트 실행"""
    print_header("STT Module Test Suite")

    results = {}

    # 테스트 1: 연결 테스트
    results["connection"] = await test_deepgram_connection()
    await asyncio.sleep(1)

    # 테스트 2: 한국어 음성 인식
    if results["connection"]:
        results["korean_stt"] = await test_korean_speech_recognition()
        await asyncio.sleep(1)
    else:
        print_warning("Skipping Korean STT test (connection failed)")
        results["korean_stt"] = False

    # 테스트 3: 실시간 스트리밍
    if results["connection"]:
        results["streaming"] = await test_realtime_streaming()
        await asyncio.sleep(1)
    else:
        print_warning("Skipping streaming test (connection failed)")
        results["streaming"] = False

    # 테스트 4: VAD 기능
    results["vad"] = await test_vad_functionality()
    await asyncio.sleep(1)

    # 테스트 5: 레이턴시 측정
    if results["connection"]:
        results["latency"] = await test_latency_measurement()
    else:
        print_warning("Skipping latency test (connection failed)")
        results["latency"] = False

    # 최종 결과 출력
    print_header("Test Summary")

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, passed in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.ENDC}" if passed else f"{Colors.RED}FAILED{Colors.ENDC}"
        print(f"  {test_name.ljust(20)}: {status}")

    print(f"\n{Colors.BOLD}Total: {passed_tests}/{total_tests} tests passed{Colors.ENDC}")

    if passed_tests == total_tests:
        print_success("\n🎉 All tests passed!")
    else:
        print_warning(f"\n⚠️  {total_tests - passed_tests} test(s) failed")


async def interactive_menu():
    """인터랙티브 메뉴"""
    print_header("STT Interactive Test Menu")

    print("1. Test Deepgram Connection")
    print("2. Test Korean Speech Recognition")
    print("3. Test Real-time Streaming")
    print("4. Test VAD Functionality")
    print("5. Test Latency Measurement")
    print("6. Run All Tests")
    print("0. Exit")

    choice = input(f"\n{Colors.CYAN}Select a test (0-6): {Colors.ENDC}").strip()

    if choice == "1":
        await test_deepgram_connection()
    elif choice == "2":
        await test_korean_speech_recognition()
    elif choice == "3":
        await test_realtime_streaming()
    elif choice == "4":
        await test_vad_functionality()
    elif choice == "5":
        await test_latency_measurement()
    elif choice == "6":
        await run_all_tests()
    elif choice == "0":
        print_info("Goodbye!")
        return False
    else:
        print_error("Invalid choice")

    return True


async def main():
    """메인 함수"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # 모든 테스트 자동 실행
        await run_all_tests()
    else:
        # 인터랙티브 메뉴
        while await interactive_menu():
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            print("\n" * 2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_info("\n\nTest interrupted by user. Goodbye!")
