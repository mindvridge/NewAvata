#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실시간 파이프라인 테스트 스크립트
"""

import os
import sys
import asyncio
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_status(name: str, status: bool, detail: str = ""):
    icon = "[OK]" if status else "[FAIL]"
    print(f"  {icon} {name}: {detail}")


async def test_tts():
    """TTS 테스트"""
    print_header("TTS (EdgeTTS) 테스트")

    try:
        from src.pipeline.realtime_pipeline import TTSService

        tts = TTSService(voice="ko-KR-SunHiNeural")
        print_status("TTS initialized", True, "ko-KR-SunHiNeural")

        # 테스트 텍스트
        test_text = "안녕하세요. 면접에 참여해 주셔서 감사합니다."
        print(f"  Synthesizing: '{test_text}'")

        audio = await tts.synthesize(test_text)

        if audio:
            print_status("TTS synthesis", True, f"{len(audio)} bytes")

            # 오디오 파일로 저장 (테스트용)
            output_path = project_root / "test_tts_output.mp3"
            with open(output_path, "wb") as f:
                f.write(audio)
            print_status("Audio saved", True, str(output_path))
            return True
        else:
            print_status("TTS synthesis", False, "No audio generated")
            return False

    except Exception as e:
        print_status("TTS test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_llm():
    """LLM 테스트"""
    print_header("LLM (OpenAI) 테스트")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_status("API key", False, "OPENAI_API_KEY not set")
        return False

    # API 키 마스킹
    masked_key = api_key[:10] + "..." + api_key[-4:]
    print_status("API key", True, masked_key)

    try:
        from src.pipeline.realtime_pipeline import LLMService

        llm = LLMService(
            api_key=api_key,
            model="gpt-3.5-turbo",
            max_tokens=100
        )

        if llm.client is None:
            print_status("LLM initialized", False, "Client not created")
            return False

        print_status("LLM initialized", True, "gpt-3.5-turbo")

        # 테스트 메시지
        test_message = "안녕하세요"
        print(f"  Sending: '{test_message}'")

        response = ""
        ttft = None
        import time
        start = time.time()

        async for token in llm.generate_response(test_message, stream=True):
            if ttft is None:
                ttft = (time.time() - start) * 1000
            response += token

        total_time = (time.time() - start) * 1000

        print_status("LLM response", True, f"TTFT: {ttft:.0f}ms, Total: {total_time:.0f}ms")
        print(f"  Response: '{response[:100]}...'")

        return True

    except Exception as e:
        print_status("LLM test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_stt():
    """STT 테스트"""
    print_header("STT (Whisper) 테스트")

    try:
        from src.pipeline.realtime_pipeline import WhisperSTTService
        import numpy as np

        # CPU 사용 (테스트용)
        stt = WhisperSTTService(model_name="tiny", language="ko", device="cpu")

        if stt.model is None:
            print_status("STT initialized", False, "Model not loaded")
            return False

        print_status("STT initialized", True, f"tiny on {stt.device}")

        # 더미 오디오 생성 (1초, 16kHz)
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01
        print("  Transcribing dummy audio (random noise)...")

        transcript = await stt.transcribe(dummy_audio)

        # 노이즈는 보통 빈 문자열이나 짧은 결과를 반환
        print_status("STT transcribe", True, f"Result: '{transcript or '(empty)'}'")

        return True

    except Exception as e:
        print_status("STT test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_pipeline_text():
    """파이프라인 텍스트 테스트"""
    print_header("파이프라인 텍스트 테스트 (LLM -> TTS)")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_status("API key", False, "OPENAI_API_KEY not set")
        return False

    try:
        from src.pipeline.realtime_pipeline import create_pipeline

        print("  Creating pipeline...")
        pipeline = await create_pipeline(
            openai_api_key=api_key,
            llm_model="gpt-3.5-turbo",
            tts_voice="ko-KR-SunHiNeural"
        )

        status = pipeline.get_status()
        print_status("Pipeline created", True, f"LLM: {status['llm_available']}, TTS: {status['tts_available']}")

        # 텍스트 처리 테스트
        test_input = "저는 3년차 백엔드 개발자입니다."
        print(f"  Input: '{test_input}'")

        result = await pipeline.process_text(test_input)

        if result.get("error"):
            print_status("Pipeline process", False, result["error"])
            return False

        print_status("Pipeline process", True, "")
        print(f"  Response: '{result['response'][:80]}...'")

        if result.get("audio"):
            print_status("Audio generated", True, f"{len(result['audio'])} bytes (base64)")
        else:
            print_status("Audio generated", False, "No audio")

        if result.get("metrics"):
            m = result["metrics"]
            print(f"  Metrics: LLM TTFT={m.get('llm_ttft_ms', 0):.0f}ms, "
                  f"LLM Total={m.get('llm_total_ms', 0):.0f}ms, "
                  f"TTS={m.get('tts_ms', 0):.0f}ms, "
                  f"Total={m.get('total_ms', 0):.0f}ms")

        return True

    except Exception as e:
        print_status("Pipeline test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_greeting():
    """인사 생성 테스트"""
    print_header("인사 생성 테스트")

    api_key = os.getenv("OPENAI_API_KEY")

    try:
        from src.pipeline.realtime_pipeline import create_pipeline

        pipeline = await create_pipeline(
            openai_api_key=api_key,
            tts_voice="ko-KR-SunHiNeural"
        )

        result = await pipeline.get_greeting()

        print_status("Greeting generated", True, "")
        print(f"  Text: '{result['response']}'")

        if result.get("audio"):
            print_status("Audio generated", True, f"{len(result['audio'])} bytes (base64)")

            # 오디오 저장
            import base64
            audio_data = base64.b64decode(result["audio"])
            output_path = project_root / "test_greeting.mp3"
            with open(output_path, "wb") as f:
                f.write(audio_data)
            print_status("Audio saved", True, str(output_path))

        return True

    except Exception as e:
        print_status("Greeting test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("  Realtime Pipeline Test Script")
    print("=" * 60)

    results = {}

    # 1. TTS 테스트
    results["tts"] = await test_tts()

    # 2. LLM 테스트
    results["llm"] = await test_llm()

    # 3. STT 테스트 (선택적)
    try:
        import whisper
        results["stt"] = await test_stt()
    except ImportError:
        print_header("STT (Whisper) 테스트")
        print_status("Whisper", False, "Not installed")
        results["stt"] = False

    # 4. 파이프라인 통합 테스트
    if results["llm"] and results["tts"]:
        results["pipeline"] = await test_pipeline_text()
    else:
        print_header("파이프라인 텍스트 테스트")
        print("  [SKIP] LLM 또는 TTS 테스트 실패")
        results["pipeline"] = False

    # 5. 인사 생성 테스트
    if results["tts"]:
        results["greeting"] = await test_greeting()
    else:
        results["greeting"] = False

    # 결과 요약
    print_header("테스트 결과 요약")

    all_passed = True
    for name, passed in results.items():
        print_status(name.upper(), passed, "PASSED" if passed else "FAILED")
        if not passed and name in ["llm", "tts", "pipeline"]:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  [SUCCESS] 핵심 테스트 통과!")
    else:
        print("  [WARNING] 일부 테스트 실패")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
