"""
Pipecat 기반 실시간 면접 아바타 파이프라인

Pipecat 프레임워크를 사용하여 LLM → TTS 파이프라인 구현
STT는 외부(Whisper)에서 처리하고 텍스트를 받아 처리
"""

import asyncio
import os
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService

# LLM Context - 새로운 경로 사용
try:
    from pipecat.services.llm_service import OpenAILLMContext
except ImportError:
    from pipecat.services.ai_services import OpenAILLMContext


class PipelineState(Enum):
    """파이프라인 상태"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass
class PipecatConfig:
    """Pipecat 파이프라인 설정"""
    # OpenAI
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    tts_model: str = "tts-1"
    tts_voice: str = "nova"

    # Audio settings
    sample_rate: int = 24000  # OpenAI TTS는 24kHz만 지원

    # Interview settings
    system_prompt: str = """당신은 전문적인 면접관입니다. 한국어로 대화합니다.
지원자를 존중하며 전문적이고 따뜻한 태도로 면접을 진행합니다.
답변은 간결하게 2-3문장으로 해주세요."""


class StateProcessor(FrameProcessor):
    """파이프라인 상태를 추적하는 프로세서"""

    def __init__(self, on_state_change: Optional[Callable[[PipelineState], None]] = None):
        super().__init__()
        self.on_state_change = on_state_change
        self._current_state = PipelineState.IDLE

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        new_state = None

        if isinstance(frame, LLMFullResponseStartFrame):
            new_state = PipelineState.PROCESSING
        elif isinstance(frame, TTSStartedFrame):
            new_state = PipelineState.SPEAKING
        elif isinstance(frame, TTSStoppedFrame):
            new_state = PipelineState.IDLE

        if new_state and new_state != self._current_state:
            self._current_state = new_state
            if self.on_state_change:
                self.on_state_change(new_state)

        await self.push_frame(frame, direction)


class AudioOutputProcessor(FrameProcessor):
    """TTS 오디오를 외부로 전달하는 프로세서"""

    def __init__(self, on_audio: Optional[Callable[[bytes, int], None]] = None):
        super().__init__()
        self.on_audio = on_audio

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            if self.on_audio:
                self.on_audio(frame.audio, frame.sample_rate)

        await self.push_frame(frame, direction)


class ResponseProcessor(FrameProcessor):
    """LLM 응답 텍스트를 추출하는 프로세서"""

    def __init__(self, on_response: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.on_response = on_response
        self._current_response = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            self._current_response += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self.on_response and self._current_response:
                self.on_response(self._current_response)
            self._current_response = ""

        await self.push_frame(frame, direction)


class TextInputProcessor(FrameProcessor):
    """외부에서 텍스트 입력을 받아 처리하는 프로세서"""

    def __init__(self):
        super().__init__()
        self._input_queue: asyncio.Queue = asyncio.Queue()

    async def queue_text(self, text: str):
        """텍스트 입력 큐에 추가"""
        await self._input_queue.put(text)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class PipecatInterviewPipeline:
    """Pipecat 기반 면접 파이프라인

    STT는 외부(Whisper)에서 처리하고, 텍스트 입력을 받아
    LLM → TTS 파이프라인을 실행합니다.
    """

    def __init__(self, config: PipecatConfig):
        self.config = config

        # Pipeline components
        self.pipeline: Optional[Pipeline] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None

        # Services
        self.llm_service: Optional[OpenAILLMService] = None
        self.tts_service: Optional[OpenAITTSService] = None

        # Processors
        self.text_input_processor: Optional[TextInputProcessor] = None
        self.state_processor: Optional[StateProcessor] = None
        self.audio_output_processor: Optional[AudioOutputProcessor] = None
        self.response_processor: Optional[ResponseProcessor] = None

        # Context
        self.context = None
        self.context_aggregator = None

        # State
        self.state = PipelineState.IDLE
        self.is_running = False
        self._task_handle = None

        # Callbacks
        self.on_response: Optional[Callable[[str], None]] = None
        self.on_audio: Optional[Callable[[bytes, int], None]] = None
        self.on_state_change: Optional[Callable[[PipelineState], None]] = None

        # Conversation history
        self._messages: List[Dict[str, str]] = []

        logger.info("PipecatInterviewPipeline created")

    async def initialize(self):
        """파이프라인 초기화"""
        logger.info("Initializing Pipecat pipeline...")

        # OpenAI LLM 서비스
        self.llm_service = OpenAILLMService(
            api_key=self.config.openai_api_key,
            model=self.config.llm_model,
        )

        # OpenAI TTS 서비스
        self.tts_service = OpenAITTSService(
            api_key=self.config.openai_api_key,
            model=self.config.tts_model,
            voice=self.config.tts_voice,
            sample_rate=self.config.sample_rate,
        )

        # 커스텀 프로세서 초기화
        self.text_input_processor = TextInputProcessor()

        self.state_processor = StateProcessor(
            on_state_change=self._handle_state_change
        )

        self.audio_output_processor = AudioOutputProcessor(
            on_audio=self._handle_audio
        )

        self.response_processor = ResponseProcessor(
            on_response=self._handle_response
        )

        # LLM 컨텍스트 설정
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        self._messages = messages.copy()
        self.context = OpenAILLMContext(messages=messages)
        self.context_aggregator = self.llm_service.create_context_aggregator(self.context)

        # 파이프라인 구성 (VAD 없이 간단한 LLM → TTS 체인)
        self.pipeline = Pipeline([
            self.text_input_processor,
            self.context_aggregator.user(),
            self.llm_service,
            self.state_processor,
            self.response_processor,
            self.tts_service,
            self.audio_output_processor,
            self.context_aggregator.assistant(),
        ])

        logger.info("Pipecat pipeline initialized")

    def _handle_response(self, text: str):
        """LLM 응답 처리"""
        logger.debug(f"LLM response: {text[:50]}...")
        if self.on_response:
            self.on_response(text)

    def _handle_audio(self, audio_data: bytes, sample_rate: int):
        """TTS 오디오 처리"""
        logger.debug(f"TTS audio: {len(audio_data)} bytes, {sample_rate}Hz")
        if self.on_audio:
            self.on_audio(audio_data, sample_rate)

    def _handle_state_change(self, state: PipelineState):
        """상태 변경 처리"""
        self.state = state
        logger.debug(f"State changed: {state.value}")
        if self.on_state_change:
            self.on_state_change(state)

    async def start(self):
        """파이프라인 시작"""
        if self.is_running:
            return

        if not self.pipeline:
            await self.initialize()

        logger.info("Starting Pipecat pipeline...")

        # Task 생성
        self.task = PipelineTask(
            self.pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        # Runner 생성 및 실행
        self.runner = PipelineRunner()
        self.is_running = True

        logger.info("Pipecat pipeline started")

    async def stop(self):
        """파이프라인 중지"""
        if not self.is_running:
            return

        logger.info("Stopping Pipecat pipeline...")

        if self.task:
            await self.task.queue_frame(EndFrame())

        self.is_running = False
        logger.info("Pipecat pipeline stopped")

    async def process_text(self, text: str):
        """텍스트 입력 처리 (STT 결과를 받아 LLM으로 전달)"""
        if not self.is_running:
            await self.start()

        logger.info(f"Processing text: {text}")

        # 사용자 메시지 추가
        self._messages.append({"role": "user", "content": text})

        # 직접 LLM 호출하여 응답 생성
        await self._generate_response(text)

    async def _generate_response(self, user_text: str):
        """LLM 응답 생성 및 TTS 변환"""
        try:
            # 상태 변경: 처리 중
            self._handle_state_change(PipelineState.PROCESSING)

            # OpenAI LLM 직접 호출
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.config.openai_api_key)

            response = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=self._messages,
                max_tokens=200,
            )

            assistant_response = response.choices[0].message.content
            self._messages.append({"role": "assistant", "content": assistant_response})

            # LLM 응답 콜백
            if self.on_response:
                self.on_response(assistant_response)

            # 상태 변경: 말하는 중
            self._handle_state_change(PipelineState.SPEAKING)

            # TTS 생성
            tts_response = await client.audio.speech.create(
                model=self.config.tts_model,
                voice=self.config.tts_voice,
                input=assistant_response,
                response_format="pcm",
            )

            # 오디오 데이터 처리
            audio_data = tts_response.content
            if self.on_audio:
                self.on_audio(audio_data, self.config.sample_rate)

            # 상태 변경: 대기
            self._handle_state_change(PipelineState.IDLE)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self._handle_state_change(PipelineState.IDLE)
            raise

    def get_status(self) -> Dict[str, Any]:
        """파이프라인 상태 반환"""
        return {
            "state": self.state.value,
            "is_running": self.is_running,
            "message_count": len(self._messages),
        }

    def reset_conversation(self):
        """대화 초기화"""
        self._messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        logger.info("Conversation reset")


class PipecatWebSocketAdapter:
    """WebSocket과 Pipecat 파이프라인을 연결하는 어댑터"""

    def __init__(self, pipeline: PipecatInterviewPipeline):
        self.pipeline = pipeline
        self._audio_buffer = bytearray()

    def set_callbacks(
        self,
        on_response: Optional[Callable[[str], None]] = None,
        on_audio: Optional[Callable[[bytes, int], None]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
    ):
        """콜백 설정"""
        if on_response:
            self.pipeline.on_response = on_response
        if on_audio:
            self.pipeline.on_audio = on_audio
        if on_state_change:
            self.pipeline.on_state_change = lambda s: on_state_change(s.value)

    async def handle_transcript(self, text: str, is_final: bool = True):
        """STT 결과 처리"""
        if is_final and text.strip():
            await self.pipeline.process_text(text)

    async def start(self):
        """어댑터 시작"""
        await self.pipeline.start()

    async def stop(self):
        """어댑터 중지"""
        await self.pipeline.stop()


def create_pipecat_pipeline(
    openai_api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> PipecatInterviewPipeline:
    """Pipecat 파이프라인 팩토리 함수"""
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")

    config = PipecatConfig(
        openai_api_key=api_key,
        system_prompt=system_prompt or PipecatConfig.system_prompt,
    )

    return PipecatInterviewPipeline(config)


# Export
__all__ = [
    "PipelineState",
    "PipecatConfig",
    "PipecatInterviewPipeline",
    "PipecatWebSocketAdapter",
    "create_pipecat_pipeline",
]
