"""
ElevenLabs TTS Service 사용 예제

음성 합성 기능을 다양한 방식으로 사용하는 예제입니다.
"""

import asyncio
import wave
from pathlib import Path
from typing import List

from loguru import logger

from src.tts import (
    ElevenLabsTTSService,
    TTSConfig,
    AudioChunk,
    create_elevenlabs_tts,
)


# =============================================================================
# 예제 1: 기본 TTS 합성
# =============================================================================

async def example_basic_synthesis():
    """기본 음성 합성 예제"""
    print("=" * 80)
    print("예제 1: 기본 음성 합성")
    print("=" * 80)

    def on_audio_chunk(chunk: AudioChunk):
        print(f"  오디오 청크: {len(chunk.data)} bytes, {chunk.sample_rate}Hz")

    def on_synthesis_start(text: str):
        print(f"\n[시작] {text}")

    def on_synthesis_end(duration: float):
        print(f"[완료] {duration:.2f}초 소요\n")

    # TTS 서비스 생성
    tts = create_elevenlabs_tts(
        on_audio_chunk=on_audio_chunk,
    )
    tts._on_synthesis_start = on_synthesis_start
    tts._on_synthesis_end = on_synthesis_end

    # 음성 합성
    texts = [
        "안녕하세요, 면접에 참여해 주셔서 감사합니다.",
        "자기소개 부탁드립니다.",
        "지원 동기에 대해 말씀해 주세요.",
    ]

    for text in texts:
        await tts.synthesize(text, streaming=True)
        await asyncio.sleep(0.5)

    # 통계 출력
    stats = tts.get_stats()
    print(f"\n통계:")
    print(f"  총 문자 수: {stats['total_chars_synthesized']}")
    print(f"  평균 TTFB: {stats['average_ttfb_ms']:.0f}ms")


# =============================================================================
# 예제 2: 오디오 파일로 저장
# =============================================================================

async def example_save_to_file():
    """음성을 WAV 파일로 저장"""
    print("=" * 80)
    print("예제 2: 오디오 파일로 저장")
    print("=" * 80)

    audio_chunks: List[bytes] = []

    def on_audio_chunk(chunk: AudioChunk):
        audio_chunks.append(chunk.data)

    tts = create_elevenlabs_tts(on_audio_chunk=on_audio_chunk)

    text = "이것은 오디오 파일로 저장되는 테스트 음성입니다."
    print(f"\n합성 중: {text}")

    await tts.synthesize(text, streaming=True)

    # WAV 파일로 저장
    output_path = Path("./cache/tts_output.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)  # 모노
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)  # 16kHz
        wf.writeframes(b''.join(audio_chunks))

    print(f"\n저장 완료: {output_path}")
    print(f"파일 크기: {output_path.stat().st_size} bytes")


# =============================================================================
# 예제 3: 긴 텍스트 청킹
# =============================================================================

async def example_long_text_chunking():
    """긴 텍스트를 청크로 나눠서 합성"""
    print("=" * 80)
    print("예제 3: 긴 텍스트 청킹")
    print("=" * 80)

    long_text = """
    안녕하세요. 오늘은 소프트웨어 엔지니어 포지션에 대한 면접을 진행하겠습니다.
    먼저 간단한 자기소개 부탁드립니다. 학력, 경력, 그리고 주요 기술 스택에 대해 말씀해 주세요.
    다음으로 지원 동기에 대해 설명해 주시기 바랍니다. 왜 저희 회사에 관심을 가지게 되셨나요?
    마지막으로 본인의 강점과 약점에 대해 이야기해 주세요.
    """

    chunk_count = 0

    def on_audio_chunk(chunk: AudioChunk):
        nonlocal chunk_count
        chunk_count += 1

    tts = create_elevenlabs_tts(on_audio_chunk=on_audio_chunk)

    print(f"\n원본 텍스트 길이: {len(long_text)} 문자")
    print("\n청킹하여 합성 중...")

    async for audio_chunk in tts.synthesize_chunked(long_text.strip()):
        print(f"  청크 수신: {len(audio_chunk.data)} bytes")

    print(f"\n총 {chunk_count}개의 오디오 청크 생성됨")


# =============================================================================
# 예제 4: 음성 설정 커스터마이징
# =============================================================================

async def example_voice_customization():
    """다양한 음성 설정으로 테스트"""
    print("=" * 80)
    print("예제 4: 음성 설정 커스터마이징")
    print("=" * 80)

    test_text = "이것은 음성 설정 테스트입니다."

    # 다양한 설정 조합
    configs = [
        ("안정적", 0.8, 0.5),  # (이름, stability, similarity_boost)
        ("일반", 0.5, 0.75),
        ("표현력", 0.2, 0.9),
    ]

    for name, stability, similarity in configs:
        print(f"\n[{name}] stability={stability}, similarity_boost={similarity}")

        config = TTSConfig(
            voice_id="설정에서_로드",  # 실제로는 settings에서 로드됨
            stability=stability,
            similarity_boost=similarity,
        )

        tts = ElevenLabsTTSService(config=config)
        await tts.synthesize(test_text, streaming=False)
        await asyncio.sleep(0.5)


# =============================================================================
# 예제 5: 사용 가능한 음성 조회
# =============================================================================

async def example_list_voices():
    """사용 가능한 음성 목록 조회"""
    print("=" * 80)
    print("예제 5: 사용 가능한 음성 목록")
    print("=" * 80)

    tts = create_elevenlabs_tts()

    # 모든 음성 조회
    voices = tts.get_available_voices()

    print(f"\n사용 가능한 음성: {len(voices)}개\n")

    for i, voice in enumerate(voices[:10], 1):  # 처음 10개만 출력
        print(f"{i}. {voice.name}")
        print(f"   ID: {voice.voice_id}")
        print(f"   Category: {voice.category}")
        if voice.description:
            print(f"   Description: {voice.description[:60]}...")
        print()

    # 현재 설정된 음성 정보
    current_voice = tts.get_voice_info()
    if current_voice:
        print("=" * 80)
        print("현재 설정된 음성:")
        print("=" * 80)
        print(f"Name: {current_voice['name']}")
        print(f"ID: {current_voice['voice_id']}")
        print(f"Category: {current_voice['category']}")
        if current_voice.get('description'):
            print(f"Description: {current_voice['description']}")


# =============================================================================
# 예제 6: 실시간 스트리밍 vs 전체 합성 비교
# =============================================================================

async def example_streaming_vs_full():
    """스트리밍과 전체 합성 비교"""
    print("=" * 80)
    print("예제 6: 스트리밍 vs 전체 합성")
    print("=" * 80)

    test_text = "실시간 면접 아바타 시스템에 오신 것을 환영합니다. 준비되셨나요?"

    tts = create_elevenlabs_tts()

    # 1. 스트리밍 모드
    print("\n[1] 스트리밍 모드")
    import time
    start = time.time()

    chunk_count = 0

    def count_chunks(chunk):
        nonlocal chunk_count
        chunk_count += 1

    tts._on_audio_chunk = count_chunks

    await tts.synthesize(test_text, streaming=True)
    streaming_time = time.time() - start

    print(f"  소요 시간: {streaming_time:.2f}초")
    print(f"  청크 수: {chunk_count}")

    # 2. 전체 합성 모드
    print("\n[2] 전체 합성 모드")
    start = time.time()

    audio_data = await tts.synthesize(test_text, streaming=False)
    full_time = time.time() - start

    print(f"  소요 시간: {full_time:.2f}초")
    print(f"  데이터 크기: {len(audio_data)} bytes")

    # 비교
    print("\n[비교]")
    print(f"  스트리밍이 {'더 빠름' if streaming_time < full_time else '더 느림'}")
    print(f"  차이: {abs(streaming_time - full_time):.2f}초")


# =============================================================================
# 예제 7: TTFB (Time To First Byte) 측정
# =============================================================================

async def example_ttfb_measurement():
    """TTFB 측정"""
    print("=" * 80)
    print("예제 7: TTFB (Time To First Byte) 측정")
    print("=" * 80)

    tts = create_elevenlabs_tts()

    test_texts = [
        "짧은 문장",
        "조금 더 긴 문장입니다.",
        "이것은 훨씬 더 긴 문장으로 TTFB를 측정하기 위한 테스트입니다.",
    ]

    print("\nTTFB 측정 중...\n")

    for i, text in enumerate(test_texts, 1):
        print(f"[{i}] {text}")
        await tts.synthesize(text, streaming=True)
        await asyncio.sleep(0.3)

    # 통계 조회
    stats = tts.get_stats()
    print(f"\n평균 TTFB: {stats['average_ttfb_ms']:.0f}ms")
    print(f"샘플 수: {stats['ttfb_samples']}")


# =============================================================================
# 예제 8: 면접 시나리오 시뮬레이션
# =============================================================================

async def example_interview_scenario():
    """면접 시나리오 전체 시뮬레이션"""
    print("=" * 80)
    print("예제 8: 면접 시나리오 시뮬레이션")
    print("=" * 80)

    # 면접 질문들
    interview_questions = [
        "안녕하세요. 면접을 시작하겠습니다.",
        "먼저, 간단한 자기소개 부탁드립니다.",
        "이 직무에 지원하신 동기가 무엇인가요?",
        "본인의 가장 큰 강점은 무엇이라고 생각하시나요?",
        "최근에 진행한 프로젝트에 대해 설명해 주세요.",
        "우리 회사에 대해 알고 계신 것이 있나요?",
        "마지막으로, 궁금한 점이 있으시면 질문해 주세요.",
        "면접에 참여해 주셔서 감사합니다. 결과는 일주일 내로 알려드리겠습니다.",
    ]

    total_audio_size = 0

    def on_audio_chunk(chunk: AudioChunk):
        nonlocal total_audio_size
        total_audio_size += len(chunk.data)

    tts = create_elevenlabs_tts(on_audio_chunk=on_audio_chunk)

    print("\n면접 시뮬레이션 시작...\n")

    for i, question in enumerate(interview_questions, 1):
        print(f"[면접관 {i}/{len(interview_questions)}] {question}")
        await tts.synthesize(question, streaming=True)

        # 지원자 답변 대기 시뮬레이션
        if i < len(interview_questions):
            await asyncio.sleep(1)
            print(f"[지원자] (답변 중...)\n")
            await asyncio.sleep(1)

    # 최종 통계
    stats = tts.get_stats()
    print("\n" + "=" * 80)
    print("면접 통계")
    print("=" * 80)
    print(f"총 질문 수: {len(interview_questions)}")
    print(f"총 문자 수: {stats['total_chars_synthesized']}")
    print(f"총 오디오 크기: {total_audio_size:,} bytes ({total_audio_size / 1024 / 1024:.2f} MB)")
    print(f"총 합성 시간: {stats['total_synthesis_time']:.2f}초")
    print(f"평균 TTFB: {stats['average_ttfb_ms']:.0f}ms")


# =============================================================================
# 메인 함수
# =============================================================================

async def main():
    """메인 함수 - 예제 선택"""
    print("\n실시간 면접 아바타 - ElevenLabs TTS 예제")
    print("=" * 80)
    print("1. 기본 음성 합성")
    print("2. 오디오 파일로 저장")
    print("3. 긴 텍스트 청킹")
    print("4. 음성 설정 커스터마이징")
    print("5. 사용 가능한 음성 목록")
    print("6. 스트리밍 vs 전체 합성")
    print("7. TTFB 측정")
    print("8. 면접 시나리오 시뮬레이션")
    print("0. 종료")
    print("=" * 80)

    choice = input("\n실행할 예제 번호를 선택하세요 (0-8, 기본값: 1): ").strip()

    if choice == "2":
        await example_save_to_file()
    elif choice == "3":
        await example_long_text_chunking()
    elif choice == "4":
        await example_voice_customization()
    elif choice == "5":
        await example_list_voices()
    elif choice == "6":
        await example_streaming_vs_full()
    elif choice == "7":
        await example_ttfb_measurement()
    elif choice == "8":
        await example_interview_scenario()
    elif choice == "0":
        print("종료합니다.")
        return
    else:
        await example_basic_synthesis()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n프로그램이 종료되었습니다.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
