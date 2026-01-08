#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipecat 파이프라인 테스트 스크립트
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


async def test_pipecat_import():
    """Pipecat 임포트 테스트"""
    print_header("Pipecat 임포트 테스트")

    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.tts import OpenAITTSService
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        print_status("Pipeline", True, "imported")
        print_status("PipelineRunner", True, "imported")
        print_status("OpenAILLMService", True, "imported")
        print_status("OpenAITTSService", True, "imported")
        print_status("SileroVADAnalyzer", True, "imported")

        return True

    except ImportError as e:
        print_status("Pipecat import", False, str(e))
        return False


async def test_openai_services():
    """OpenAI 서비스 테스트"""
    print_header("OpenAI 서비스 테스트")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_status("API key", False, "OPENAI_API_KEY not set")
        return False

    masked_key = api_key[:10] + "..." + api_key[-4:]
    print_status("API key", True, masked_key)

    try:
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.tts import OpenAITTSService

        # LLM 서비스
        llm = OpenAILLMService(
            api_key=api_key,
            model="gpt-4o-mini"
        )
        print_status("LLM service", True, "gpt-4o-mini")

        # TTS 서비스
        tts = OpenAITTSService(
            api_key=api_key,
            model="tts-1",
            voice="nova",
            sample_rate=16000
        )
        print_status("TTS service", True, "tts-1 / nova")

        return True

    except Exception as e:
        print_status("OpenAI services", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_vad():
    """VAD 테스트"""
    print_header("VAD (Silero) 테스트")

    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        import numpy as np

        vad = SileroVADAnalyzer(
            params=VADParams(
                threshold=0.5,
                min_speech_duration=0.1,
                min_silence_duration=0.5,
            )
        )

        print_status("VAD initialized", True, "Silero")

        # 테스트 오디오 (노이즈)
        test_audio = (np.random.randn(16000) * 0.01).astype(np.float32)
        print_status("Test audio generated", True, "1 second noise")

        return True

    except Exception as e:
        print_status("VAD test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_pipecat_pipeline():
    """Pipecat 파이프라인 통합 테스트"""
    print_header("Pipecat 파이프라인 테스트")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_status("API key", False, "OPENAI_API_KEY not set")
        return False

    try:
        # 직접 pipecat_pipeline.py 모듈만 import (main_pipeline의 Daily 의존성 우회)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pipecat_pipeline",
            project_root / "src" / "pipeline" / "pipecat_pipeline.py"
        )
        pipecat_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipecat_module)

        PipecatInterviewPipeline = pipecat_module.PipecatInterviewPipeline
        PipecatConfig = pipecat_module.PipecatConfig
        create_pipecat_pipeline = pipecat_module.create_pipecat_pipeline

        # 설정
        config = PipecatConfig(
            openai_api_key=api_key,
            llm_model="gpt-4o-mini",
            tts_voice="nova",
            system_prompt="You are a helpful assistant. Respond in Korean briefly."
        )

        print_status("Config created", True, f"model: {config.llm_model}")

        # 파이프라인 생성
        pipeline = PipecatInterviewPipeline(config)
        print_status("Pipeline created", True, "")

        # 콜백 설정
        received_response = []
        received_audio = []

        def on_response(text):
            received_response.append(text)
            print_status("Response received", True, f"'{text[:30]}...'")

        def on_audio(audio_data, sample_rate):
            received_audio.append((len(audio_data), sample_rate))
            print_status("Audio received", True, f"{len(audio_data)} bytes @ {sample_rate}Hz")

        pipeline.on_response = on_response
        pipeline.on_audio = on_audio

        # 파이프라인 시작
        await pipeline.start()
        print_status("Pipeline started", True, "")

        # 텍스트 처리 테스트
        print("  Testing text processing...")
        await pipeline.process_text("안녕하세요, 반갑습니다.")

        # 결과 확인
        print_status("LLM response", len(received_response) > 0, f"{len(received_response)} responses")
        print_status("TTS audio", len(received_audio) > 0, f"{len(received_audio)} audio chunks")

        # 상태 확인
        status = pipeline.get_status()
        print_status("Status", True, f"state: {status['state']}, messages: {status['message_count']}")

        # 정리
        await pipeline.stop()
        print_status("Pipeline stopped", True, "")

        return len(received_response) > 0 and len(received_audio) > 0

    except Exception as e:
        print_status("Pipeline test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_simple_llm_tts():
    """간단한 LLM + TTS 테스트"""
    print_header("간단한 LLM + TTS 테스트")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_status("API key", False, "OPENAI_API_KEY not set")
        return False

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # LLM 테스트
        print("  Testing LLM...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in Korean, one sentence only."}
            ],
            max_tokens=50
        )

        llm_response = response.choices[0].message.content
        print_status("LLM response", True, f"'{llm_response}'")

        # TTS 테스트
        print("  Testing TTS...")
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=llm_response
        )

        audio_data = tts_response.content
        print_status("TTS audio", True, f"{len(audio_data)} bytes")

        # 오디오 저장
        output_path = project_root / "test_pipecat_tts.mp3"
        with open(output_path, "wb") as f:
            f.write(audio_data)
        print_status("Audio saved", True, str(output_path))

        return True

    except Exception as e:
        print_status("LLM + TTS test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("  Pipecat Pipeline Test Script")
    print("=" * 60)

    results = {}

    # 1. Pipecat 임포트 테스트
    results["import"] = await test_pipecat_import()

    # 2. OpenAI 서비스 테스트
    results["openai"] = await test_openai_services()

    # 3. VAD 테스트
    results["vad"] = await test_vad()

    # 4. 간단한 LLM + TTS 테스트
    results["llm_tts"] = await test_simple_llm_tts()

    # 5. Pipecat 파이프라인 테스트
    if all(results.values()):
        results["pipeline"] = await test_pipecat_pipeline()
    else:
        print_header("Pipecat 파이프라인 테스트")
        print("  [SKIP] 이전 테스트 실패")
        results["pipeline"] = False

    # 결과 요약
    print_header("테스트 결과 요약")

    all_passed = True
    for name, passed in results.items():
        print_status(name.upper(), passed, "PASSED" if passed else "FAILED")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  [SUCCESS] 모든 테스트 통과!")
    else:
        print("  [WARNING] 일부 테스트 실패")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
