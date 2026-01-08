"""
Text Input Service
STT 대신 채팅으로 텍스트 입력을 받는 서비스

음성 인식 대신 키보드로 텍스트를 입력받아 면접을 진행할 수 있습니다.
"""

import asyncio
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from .deepgram_service import TranscriptResult


@dataclass
class TextInputConfig:
    """텍스트 입력 설정"""
    simulate_typing_delay: bool = False      # 타이핑 지연 시뮬레이션
    typing_delay_per_char: float = 0.05     # 문자당 지연 (초)
    auto_finalize: bool = True               # 엔터 입력 시 자동으로 final 처리
    language: str = "ko"                     # 언어 (호환성을 위해)
    enable_history: bool = True              # 대화 히스토리 저장


class TextInputService:
    """
    텍스트 입력 기반 면접 서비스

    STT 대신 키보드로 텍스트를 입력받아 DeepgramSTTService와 동일한 인터페이스 제공
    """

    def __init__(
        self,
        config: Optional[TextInputConfig] = None,
        on_transcript: Optional[Callable[[TranscriptResult], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_speech_ended: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Args:
            config: 텍스트 입력 설정
            on_transcript: 텍스트 입력 시 콜백
            on_speech_started: 입력 시작 시 콜백
            on_speech_ended: 입력 종료 시 콜백
            on_error: 에러 발생 시 콜백
        """
        self.config = config or TextInputConfig()
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._on_speech_ended = on_speech_ended
        self._on_error = on_error

        # 상태 추적
        self._is_connected = False
        self._is_typing = False
        self._input_queue: asyncio.Queue = asyncio.Queue()
        self._history = [] if self.config.enable_history else None

        logger.info("TextInputService initialized (Chat mode)")

    async def connect(self) -> None:
        """연결 시작 (호환성을 위한 메서드)"""
        self._is_connected = True
        logger.info("TextInputService connected (ready to receive text)")

    async def disconnect(self) -> None:
        """연결 종료"""
        self._is_connected = False
        logger.info("TextInputService disconnected")

    async def send_text(self, text: str, is_final: bool = True) -> None:
        """
        텍스트를 전송합니다.

        Args:
            text: 입력 텍스트
            is_final: 최종 입력 여부 (False면 interim으로 처리)
        """
        if not self._is_connected:
            logger.warning("TextInputService not connected")
            return

        try:
            # 타이핑 시작
            if not self._is_typing and self._on_speech_started:
                self._is_typing = True
                self._on_speech_started()

            # 타이핑 지연 시뮬레이션
            if self.config.simulate_typing_delay and is_final:
                typing_time = len(text) * self.config.typing_delay_per_char
                await asyncio.sleep(typing_time)

            # TranscriptResult 생성
            result = TranscriptResult(
                text=text,
                is_final=is_final,
                confidence=1.0,  # 텍스트 입력은 항상 100% 정확
                language=self.config.language,
                duration=0.0,
                timestamp=datetime.now().timestamp(),
            )

            # 히스토리 저장
            if is_final and self._history is not None:
                self._history.append({
                    "text": text,
                    "timestamp": result.timestamp,
                })

            # 콜백 실행
            if self._on_transcript:
                self._on_transcript(result)

            # 타이핑 종료
            if is_final and self._is_typing:
                self._is_typing = False
                if self._on_speech_ended:
                    self._on_speech_ended()

            logger.debug(f"Text {'final' if is_final else 'interim'}: {text}")

        except Exception as e:
            logger.error(f"Error sending text: {e}")
            if self._on_error:
                self._on_error(e)

    async def send_text_interactive(self) -> None:
        """
        대화형 텍스트 입력 (stdin에서 입력 받음)

        사용자가 직접 터미널에서 텍스트를 입력합니다.
        빈 줄이나 'quit'를 입력하면 종료됩니다.
        """
        print("\n채팅 모드 시작 (Enter로 전송, 'quit'로 종료)")
        print("-" * 60)

        while self._is_connected:
            try:
                # 비동기로 입력 받기
                text = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )

                text = text.strip()

                # 종료 명령
                if text.lower() in ['quit', 'exit', 'q']:
                    logger.info("Interactive mode terminated by user")
                    break

                # 빈 입력 무시
                if not text:
                    continue

                # 텍스트 전송
                await self.send_text(text, is_final=True)

            except KeyboardInterrupt:
                logger.info("Interactive mode interrupted")
                break
            except Exception as e:
                logger.error(f"Error in interactive input: {e}")
                if self._on_error:
                    self._on_error(e)

    def get_history(self) -> list:
        """
        대화 히스토리 반환

        Returns:
            list: 입력 히스토리 [{text, timestamp}, ...]
        """
        return self._history.copy() if self._history else []

    def clear_history(self) -> None:
        """히스토리 초기화"""
        if self._history is not None:
            self._history.clear()
            logger.debug("History cleared")

    @property
    def is_connected(self) -> bool:
        """연결 상태"""
        return self._is_connected

    @property
    def is_typing(self) -> bool:
        """현재 타이핑 중 여부 (is_speaking과 호환)"""
        return self._is_typing

    # is_speaking 별칭 (STT 호환성)
    @property
    def is_speaking(self) -> bool:
        """is_typing의 별칭 (STT 호환성)"""
        return self._is_typing

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.disconnect()


# =============================================================================
# WebSocket 기반 텍스트 입력 (웹 인터페이스용)
# =============================================================================

class WebSocketTextInputService(TextInputService):
    """
    WebSocket을 통한 텍스트 입력 서비스

    웹 브라우저에서 채팅 메시지를 받아 처리합니다.
    """

    def __init__(
        self,
        config: Optional[TextInputConfig] = None,
        on_transcript: Optional[Callable[[TranscriptResult], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_speech_ended: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        super().__init__(
            config=config,
            on_transcript=on_transcript,
            on_speech_started=on_speech_started,
            on_speech_ended=on_speech_ended,
            on_error=on_error,
        )
        self._websocket = None
        logger.info("WebSocketTextInputService initialized")

    async def handle_websocket(self, websocket):
        """
        WebSocket 연결 핸들러

        Args:
            websocket: WebSocket 연결 객체
        """
        self._websocket = websocket
        await self.connect()

        try:
            async for message in websocket:
                # JSON 파싱
                import json
                data = json.loads(message) if isinstance(message, str) else message

                # 메시지 타입에 따라 처리
                msg_type = data.get("type", "text")

                if msg_type == "text":
                    text = data.get("text", "").strip()
                    is_final = data.get("is_final", True)

                    if text:
                        await self.send_text(text, is_final=is_final)

                elif msg_type == "typing":
                    # 타이핑 중 표시
                    if not self._is_typing and self._on_speech_started:
                        self._is_typing = True
                        self._on_speech_started()

                elif msg_type == "stop_typing":
                    # 타이핑 종료
                    if self._is_typing and self._on_speech_ended:
                        self._is_typing = False
                        self._on_speech_ended()

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if self._on_error:
                self._on_error(e)
        finally:
            await self.disconnect()
            self._websocket = None

    async def send_response(self, text: str):
        """
        WebSocket으로 응답 전송

        Args:
            text: 전송할 텍스트
        """
        if self._websocket:
            import json
            await self._websocket.send(json.dumps({
                "type": "response",
                "text": text,
                "timestamp": datetime.now().isoformat(),
            }))


# =============================================================================
# 유니파이드 인터페이스 (STT + Text 통합)
# =============================================================================

class UnifiedInputService:
    """
    STT와 Text Input을 통합한 인터페이스

    음성 또는 텍스트 중 하나를 선택하여 사용하거나,
    둘 다 동시에 사용할 수 있습니다.
    """

    def __init__(
        self,
        mode: str = "text",  # "stt", "text", "both"
        stt_service: Optional["DeepgramSTTService"] = None,
        text_service: Optional[TextInputService] = None,
        on_transcript: Optional[Callable[[TranscriptResult], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_speech_ended: Optional[Callable[[], None]] = None,
    ):
        """
        Args:
            mode: 입력 모드 ("stt", "text", "both")
            stt_service: STT 서비스 인스턴스
            text_service: 텍스트 입력 서비스 인스턴스
            on_transcript: 텍스트 인식 시 콜백
            on_speech_started: 발화/입력 시작 시 콜백
            on_speech_ended: 발화/입력 종료 시 콜백
        """
        self.mode = mode
        self._stt_service = stt_service
        self._text_service = text_service

        # 콜백 함수들
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._on_speech_ended = on_speech_ended

        logger.info(f"UnifiedInputService initialized: mode={mode}")

    async def connect(self):
        """모든 활성화된 서비스 연결"""
        if self.mode in ["stt", "both"] and self._stt_service:
            await self._stt_service.connect()

        if self.mode in ["text", "both"] and self._text_service:
            await self._text_service.connect()

    async def disconnect(self):
        """모든 활성화된 서비스 연결 해제"""
        if self._stt_service:
            await self._stt_service.disconnect()

        if self._text_service:
            await self._text_service.disconnect()

    @property
    def is_connected(self) -> bool:
        """연결 상태"""
        if self.mode == "stt":
            return self._stt_service.is_connected if self._stt_service else False
        elif self.mode == "text":
            return self._text_service.is_connected if self._text_service else False
        else:  # both
            stt_ok = self._stt_service.is_connected if self._stt_service else True
            text_ok = self._text_service.is_connected if self._text_service else True
            return stt_ok and text_ok

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


# =============================================================================
# 헬퍼 함수
# =============================================================================

def create_text_input_service(
    simulate_typing: bool = False,
    on_transcript: Optional[Callable] = None,
    on_speech_started: Optional[Callable] = None,
    on_speech_ended: Optional[Callable] = None,
) -> TextInputService:
    """
    텍스트 입력 서비스를 생성하는 헬퍼 함수

    Args:
        simulate_typing: 타이핑 지연 시뮬레이션 여부
        on_transcript: 텍스트 입력 시 콜백
        on_speech_started: 입력 시작 시 콜백
        on_speech_ended: 입력 종료 시 콜백

    Returns:
        TextInputService: 텍스트 입력 서비스
    """
    config = TextInputConfig(simulate_typing_delay=simulate_typing)

    return TextInputService(
        config=config,
        on_transcript=on_transcript,
        on_speech_started=on_speech_started,
        on_speech_ended=on_speech_ended,
    )


# =============================================================================
# 테스트/예시 코드
# =============================================================================

if __name__ == "__main__":
    """텍스트 입력 서비스 테스트"""
    import sys

    async def test_text_input():
        """기본 텍스트 입력 테스트"""

        def on_transcript(result: TranscriptResult):
            print(f"\n[면접관] 응답 처리 중...")
            print(f"입력: {result.text}")
            print(f"신뢰도: {result.confidence:.0%}")
            print()

        def on_speech_started():
            print("[시스템] 입력 시작...")

        def on_speech_ended():
            print("[시스템] 입력 완료")

        async with create_text_input_service(
            simulate_typing=True,
            on_transcript=on_transcript,
            on_speech_started=on_speech_started,
            on_speech_ended=on_speech_ended,
        ) as service:
            # 대화형 입력
            await service.send_text_interactive()

            # 히스토리 출력
            print("\n" + "=" * 60)
            print("대화 히스토리:")
            for i, item in enumerate(service.get_history(), 1):
                print(f"{i}. {item['text']}")

    # 테스트 실행
    asyncio.run(test_text_input())
