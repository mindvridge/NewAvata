"""
텍스트 입력 서비스 사용 예제

STT 대신 채팅으로 면접을 진행하는 예제입니다.
"""

import asyncio
from loguru import logger

from src.stt import (
    TextInputService,
    create_text_input_service,
    UnifiedInputService,
    TranscriptResult,
)


# =============================================================================
# 예제 1: 기본 텍스트 입력
# =============================================================================

async def example_basic_text_input():
    """기본 텍스트 입력 예제"""
    print("=" * 80)
    print("예제 1: 기본 텍스트 입력")
    print("=" * 80)

    def on_transcript(result: TranscriptResult):
        print(f"\n[수신됨] {result.text}")
        print(f"  - 신뢰도: {result.confidence:.0%}")
        print(f"  - 최종 여부: {'예' if result.is_final else '아니오'}")

    async with create_text_input_service(
        on_transcript=on_transcript,
    ) as service:
        print("\n채팅 모드로 면접을 진행합니다.")
        print("텍스트를 입력하고 Enter를 누르세요.")
        print("'quit'를 입력하면 종료됩니다.\n")

        # 대화형 입력
        await service.send_text_interactive()

        # 히스토리 출력
        print("\n" + "=" * 80)
        print("대화 히스토리:")
        print("=" * 80)
        for i, item in enumerate(service.get_history(), 1):
            print(f"{i}. {item['text']}")


# =============================================================================
# 예제 2: 프로그래밍 방식 텍스트 전송
# =============================================================================

async def example_programmatic_input():
    """프로그래밍 방식으로 텍스트 전송"""
    print("=" * 80)
    print("예제 2: 프로그래밍 방식 텍스트 전송")
    print("=" * 80)

    responses = []

    def on_transcript(result: TranscriptResult):
        if result.is_final:
            responses.append(result.text)
            print(f"[지원자] {result.text}")

    async with create_text_input_service(
        simulate_typing=True,  # 타이핑 지연 시뮬레이션
        on_transcript=on_transcript,
    ) as service:
        # 면접 시나리오
        interview_answers = [
            "안녕하세요. 저는 김철수입니다.",
            "저는 소프트웨어 개발 경력이 5년 있습니다.",
            "Python과 JavaScript를 주로 사용합니다.",
            "팀 프로젝트에서 리더 역할을 맡은 경험이 있습니다.",
            "감사합니다.",
        ]

        print("\n[시뮬레이션] 면접 답변을 전송합니다...\n")

        for i, answer in enumerate(interview_answers, 1):
            print(f"[면접관] 질문 {i}...")
            await asyncio.sleep(0.5)

            # 답변 전송
            await service.send_text(answer, is_final=True)
            await asyncio.sleep(1)

        print(f"\n총 {len(responses)}개의 답변이 전송되었습니다.")


# =============================================================================
# 예제 3: Interim 결과 처리
# =============================================================================

async def example_interim_results():
    """Interim 결과 처리 예제"""
    print("=" * 80)
    print("예제 3: Interim 결과 처리 (타이핑 중 미리보기)")
    print("=" * 80)

    def on_transcript(result: TranscriptResult):
        if result.is_final:
            print(f"\n✓ [최종] {result.text}")
        else:
            print(f"... {result.text}", end='\r')

    async with create_text_input_service(
        on_transcript=on_transcript,
    ) as service:
        # 긴 문장을 단어별로 interim 전송
        sentence = "저는 대학교에서 컴퓨터 공학을 전공했습니다"
        words = sentence.split()

        print("\n타이핑 시뮬레이션 시작...\n")

        accumulated = ""
        for word in words:
            accumulated += word + " "
            await service.send_text(accumulated.strip(), is_final=False)
            await asyncio.sleep(0.3)

        # 최종 전송
        await service.send_text(sentence, is_final=True)
        await asyncio.sleep(0.5)


# =============================================================================
# 예제 4: WebSocket 텍스트 입력 (모의)
# =============================================================================

async def example_websocket_simulation():
    """WebSocket 텍스트 입력 시뮬레이션"""
    print("=" * 80)
    print("예제 4: WebSocket 텍스트 입력 (시뮬레이션)")
    print("=" * 80)

    from src.stt import WebSocketTextInputService

    messages_received = []

    def on_transcript(result: TranscriptResult):
        if result.is_final:
            messages_received.append(result.text)
            print(f"[WebSocket] 메시지 수신: {result.text}")

    service = WebSocketTextInputService(
        on_transcript=on_transcript,
    )

    await service.connect()

    # WebSocket 메시지 시뮬레이션
    print("\nWebSocket 메시지 시뮬레이션...\n")

    messages = [
        {"type": "text", "text": "안녕하세요", "is_final": True},
        {"type": "text", "text": "면접 잘 부탁드립니다", "is_final": True},
        {"type": "text", "text": "감사합니다", "is_final": True},
    ]

    for msg in messages:
        await service.send_text(msg["text"], is_final=msg["is_final"])
        await asyncio.sleep(0.5)

    await service.disconnect()

    print(f"\n총 {len(messages_received)}개 메시지 처리 완료")


# =============================================================================
# 예제 5: STT와 텍스트 입력 통합 (UnifiedInputService)
# =============================================================================

async def example_unified_input():
    """STT와 텍스트 입력을 통합한 예제"""
    print("=" * 80)
    print("예제 5: 통합 입력 서비스 (STT + Text)")
    print("=" * 80)

    def on_transcript(result: TranscriptResult):
        print(f"[입력] {result.text} (신뢰도: {result.confidence:.0%})")

    # 텍스트 모드로 시작
    text_service = create_text_input_service(on_transcript=on_transcript)

    unified = UnifiedInputService(
        mode="text",  # "stt", "text", "both"
        text_service=text_service,
        on_transcript=on_transcript,
    )

    await unified.connect()

    print("\n통합 입력 서비스 연결 완료")
    print(f"모드: {unified.mode}")
    print(f"연결 상태: {unified.is_connected}")

    # 텍스트 입력 테스트
    print("\n텍스트 입력 테스트...")
    await text_service.send_text("통합 서비스 테스트입니다", is_final=True)

    await asyncio.sleep(1)
    await unified.disconnect()


# =============================================================================
# 예제 6: 콜백 이벤트 처리
# =============================================================================

async def example_callbacks():
    """콜백 이벤트 처리 예제"""
    print("=" * 80)
    print("예제 6: 콜백 이벤트 처리")
    print("=" * 80)

    event_log = []

    def on_transcript(result: TranscriptResult):
        event_log.append(("transcript", result.text))
        print(f"📝 Transcript: {result.text}")

    def on_speech_started():
        event_log.append(("started", None))
        print("▶️  입력 시작")

    def on_speech_ended():
        event_log.append(("ended", None))
        print("⏹️  입력 종료")

    async with create_text_input_service(
        on_transcript=on_transcript,
        on_speech_started=on_speech_started,
        on_speech_ended=on_speech_ended,
    ) as service:
        # 여러 메시지 전송
        messages = ["첫 번째 메시지", "두 번째 메시지", "세 번째 메시지"]

        print("\n이벤트 로그 테스트...\n")

        for msg in messages:
            await service.send_text(msg, is_final=True)
            await asyncio.sleep(0.5)

        # 이벤트 로그 출력
        print("\n" + "=" * 80)
        print("이벤트 로그:")
        print("=" * 80)
        for event_type, data in event_log:
            if event_type == "transcript":
                print(f"  - 📝 Transcript: {data}")
            elif event_type == "started":
                print(f"  - ▶️  Started")
            elif event_type == "ended":
                print(f"  - ⏹️  Ended")


# =============================================================================
# 메인 함수
# =============================================================================

async def main():
    """메인 함수 - 예제 선택"""
    print("\n실시간 면접 아바타 - 텍스트 입력 예제")
    print("=" * 80)
    print("1. 기본 텍스트 입력 (대화형)")
    print("2. 프로그래밍 방식 텍스트 전송")
    print("3. Interim 결과 처리")
    print("4. WebSocket 시뮬레이션")
    print("5. 통합 입력 서비스")
    print("6. 콜백 이벤트 처리")
    print("0. 종료")
    print("=" * 80)

    choice = input("\n실행할 예제 번호를 선택하세요 (0-6, 기본값: 1): ").strip()

    if choice == "2":
        await example_programmatic_input()
    elif choice == "3":
        await example_interim_results()
    elif choice == "4":
        await example_websocket_simulation()
    elif choice == "5":
        await example_unified_input()
    elif choice == "6":
        await example_callbacks()
    elif choice == "0":
        print("종료합니다.")
        return
    else:
        await example_basic_text_input()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n프로그램이 종료되었습니다.")
