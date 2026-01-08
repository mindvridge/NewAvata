"""
Main Interview Avatar Pipeline
Pipecat 기반 실시간 면접 아바타 파이프라인
"""

import asyncio
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

from loguru import logger

# Pipecat imports
PIPECAT_AVAILABLE = False
DAILY_AVAILABLE = False

try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask, PipelineParams

    from pipecat.frames.frames import (
        Frame,
        AudioRawFrame,
        TextFrame,
        ImageRawFrame,
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        ErrorFrame,
        StartFrame,
        EndFrame,
        CancelFrame,
    )

    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    # VAD - 새 경로 사용
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    PIPECAT_AVAILABLE = True
    logger.info("Pipecat core available")

except ImportError as e:
    logger.warning(f"Pipecat core not available: {e}. Install: pip install pipecat-ai")
    # Stub classes when Pipecat is not available
    FrameProcessor = object
    Frame = object
    FrameDirection = object
    AudioRawFrame = object
    TextFrame = object
    ImageRawFrame = object
    TranscriptionFrame = object
    UserStartedSpeakingFrame = object
    UserStoppedSpeakingFrame = object
    ErrorFrame = object
    StartFrame = object
    EndFrame = object
    CancelFrame = object

# Daily Transport - disabled (not available on Windows)
# daily-python package causes import errors on Windows even with try-except
DAILY_AVAILABLE = False
DailyTransport = None
DailyParams = None
logger.info("Daily transport disabled (Windows compatibility)")

# Optional services - disabled to avoid import errors
# pipecat services raise exceptions internally when dependencies are missing
DeepgramSTTService = None
OpenAILLMService = None
ElevenLabsTTSService = None
logger.info("External pipecat services disabled (using built-in pipeline)")

from config.settings import get_settings


class PipelineState(Enum):
    """파이프라인 상태"""
    IDLE = "idle"               # 대기 중
    LISTENING = "listening"     # 사용자 말하는 중
    PROCESSING = "processing"   # LLM 처리 중
    SPEAKING = "speaking"       # 아바타 말하는 중
    ERROR = "error"             # 에러 상태


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # Daily.co (WebRTC)
    daily_room_url: str
    daily_token: Optional[str] = None

    # Deepgram (STT)
    deepgram_api_key: Optional[str] = None
    deepgram_model: str = "nova-2"
    deepgram_language: str = "ko"

    # OpenAI (LLM)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"

    # ElevenLabs (TTS)
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    # MuseTalk (Avatar)
    avatar_source_image: str = "./assets/avatar_source.jpg"
    avatar_resolution: int = 512

    # VAD
    vad_silence_duration: float = 0.7  # 초

    # 기타
    enable_face_enhancement: bool = False
    device: str = "cpu"


class AudioInputProcessor(FrameProcessor):
    """오디오 입력 전처리"""

    def __init__(self):
        super().__init__()
        self._audio_buffer = bytearray()
        logger.info("AudioInputProcessor initialized")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """오디오 프레임 전처리"""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # 오디오 데이터 버퍼링 (필요시)
            # self._audio_buffer.extend(frame.audio)

            # 전처리 (노이즈 제거 등) - 선택적
            # processed_audio = self._preprocess_audio(frame.audio)
            # frame.audio = processed_audio

            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class TranscriptionProcessor(FrameProcessor):
    """STT 결과 처리"""

    def __init__(self, on_transcription: Optional[Callable] = None):
        super().__init__()
        self._on_transcription = on_transcription
        self._current_transcription = ""
        logger.info("TranscriptionProcessor initialized")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """전사 결과 처리"""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # 전사 결과 저장
            text = frame.text
            is_final = getattr(frame, 'is_final', True)

            if is_final:
                self._current_transcription = text
                logger.info(f"Transcription (final): {text}")

                # 콜백 호출
                if self._on_transcription:
                    await self._on_transcription(text, is_final=True)
            else:
                logger.debug(f"Transcription (interim): {text}")

                if self._on_transcription:
                    await self._on_transcription(text, is_final=False)

            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class LLMProcessor(FrameProcessor):
    """LLM 응답 생성 (면접관 에이전트)"""

    def __init__(self, interviewer_agent):
        super().__init__()
        self.interviewer_agent = interviewer_agent
        self._conversation_history = []
        logger.info("LLMProcessor initialized")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """LLM 처리"""
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            # 사용자 입력
            user_text = frame.text

            logger.info(f"User input: {user_text}")

            # 면접관 에이전트로 응답 생성
            try:
                response = await self.interviewer_agent.process_answer(user_text)

                logger.info(f"Interviewer response: {response}")

                # 응답 프레임 생성
                response_frame = TextFrame(text=response)
                await self.push_frame(response_frame, direction)

            except Exception as e:
                logger.error(f"LLM processing failed: {e}")

                # 에러 프레임
                error_frame = ErrorFrame(error=str(e))
                await self.push_frame(error_frame, direction)
        else:
            await self.push_frame(frame, direction)


class TTSProcessor(FrameProcessor):
    """TTS 음성 합성"""

    def __init__(self, tts_service):
        super().__init__()
        self.tts_service = tts_service
        logger.info("TTSProcessor initialized")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """TTS 처리"""
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            # 텍스트를 음성으로 변환
            text = frame.text

            logger.info(f"Synthesizing: {text}")

            try:
                # TTS 서비스로 음성 합성
                # (실제 구현은 tts_service에 따라 다름)
                # audio_data = await self.tts_service.synthesize(text)

                # 오디오 프레임 생성
                # audio_frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
                # await self.push_frame(audio_frame, direction)

                # Pipecat의 TTS 서비스가 자동으로 처리하므로
                # 여기서는 프레임만 전달
                await self.push_frame(frame, direction)

            except Exception as e:
                logger.error(f"TTS processing failed: {e}")
                error_frame = ErrorFrame(error=str(e))
                await self.push_frame(error_frame, direction)
        else:
            await self.push_frame(frame, direction)


class AvatarProcessor(FrameProcessor):
    """아바타 비디오 생성 (MuseTalk)"""

    def __init__(self, avatar):
        super().__init__()
        self.avatar = avatar
        self._frame_count = 0
        logger.info("AvatarProcessor initialized")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """아바타 비디오 생성"""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # 오디오로부터 아바타 비디오 프레임 생성
            try:
                import numpy as np

                # 오디오 데이터
                audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

                # MuseTalk로 비디오 프레임 생성
                video_frame = self.avatar.process_audio_chunk(audio_data)

                # 비디오 프레임 전송
                image_frame = ImageRawFrame(
                    image=video_frame.frame.tobytes(),
                    size=(video_frame.frame.shape[1], video_frame.frame.shape[0]),
                    format="RGB"
                )

                await self.push_frame(image_frame, direction)

                self._frame_count += 1

                if self._frame_count % 25 == 0:
                    logger.debug(f"Avatar frames generated: {self._frame_count}")

            except Exception as e:
                logger.error(f"Avatar processing failed: {e}")
                # 에러 발생 시 대기 프레임 전송
                idle_frame = self.avatar.get_idle_frame()
                image_frame = ImageRawFrame(
                    image=idle_frame.tobytes(),
                    size=(idle_frame.shape[1], idle_frame.shape[0]),
                    format="RGB"
                )
                await self.push_frame(image_frame, direction)
        else:
            await self.push_frame(frame, direction)


class VideoOutputProcessor(FrameProcessor):
    """비디오 출력 인코딩"""

    def __init__(self):
        super().__init__()
        self._frame_count = 0
        logger.info("VideoOutputProcessor initialized")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """비디오 인코딩"""
        await super().process_frame(frame, direction)

        if isinstance(frame, ImageRawFrame):
            # 비디오 프레임 후처리 (필요시)
            # - 얼굴 향상
            # - 워터마크 추가
            # - 리사이즈

            self._frame_count += 1

            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class InterviewAvatarPipeline:
    """실시간 면접 아바타 파이프라인"""

    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: 파이프라인 설정
        """
        if not PIPECAT_AVAILABLE:
            raise ImportError("Pipecat is required. Install: pip install pipecat-ai")

        self.config = config
        self._state = PipelineState.IDLE

        # 컴포넌트
        self._transport = None
        self._pipeline = None
        self._task = None
        self._runner = None

        # 프로세서
        self._audio_input_processor = None
        self._transcription_processor = None
        self._llm_processor = None
        self._tts_processor = None
        self._avatar_processor = None
        self._video_output_processor = None

        # 서비스
        self._stt_service = None
        self._llm_service = None
        self._tts_service = None
        self._vad_analyzer = None

        # 에이전트
        self._interviewer_agent = None
        self._avatar = None

        # 이벤트 핸들러
        self._event_handlers = {
            'on_participant_joined': [],
            'on_participant_left': [],
            'on_error': [],
            'on_pipeline_started': [],
            'on_pipeline_stopped': [],
            'on_state_changed': [],
        }

        logger.info("InterviewAvatarPipeline created")

    async def initialize(self):
        """파이프라인 초기화"""
        logger.info("Initializing pipeline...")

        # 1. 설정 로드
        settings = get_settings()

        # 2. VAD 초기화
        self._vad_analyzer = SileroVADAnalyzer(
            min_silence_duration=self.config.vad_silence_duration
        )
        logger.info("VAD initialized")

        # 3. STT 서비스 초기화 (Deepgram)
        self._stt_service = DeepgramSTTService(
            api_key=self.config.deepgram_api_key or settings.deepgram_api_key,
            model=self.config.deepgram_model,
            language=self.config.deepgram_language
        )
        logger.info("STT service initialized")

        # 4. LLM 서비스 초기화 (OpenAI)
        self._llm_service = OpenAILLMService(
            api_key=self.config.openai_api_key or settings.openai_api_key,
            model=self.config.openai_model
        )
        logger.info("LLM service initialized")

        # 5. 면접관 에이전트 초기화
        from src.llm import create_interviewer
        self._interviewer_agent = create_interviewer(
            api_key=self.config.openai_api_key or settings.openai_api_key,
            model=self.config.openai_model
        )
        logger.info("Interviewer agent initialized")

        # 6. TTS 서비스 초기화 (ElevenLabs)
        self._tts_service = ElevenLabsTTSService(
            api_key=self.config.elevenlabs_api_key or settings.elevenlabs_api_key,
            voice_id=self.config.elevenlabs_voice_id
        )
        logger.info("TTS service initialized")

        # 7. 아바타 초기화 (MuseTalk)
        from src.avatar import create_musetalk_avatar
        self._avatar = create_musetalk_avatar(
            source_image_path=self.config.avatar_source_image,
            resolution=self.config.avatar_resolution,
            use_face_enhance=self.config.enable_face_enhancement,
            device=self.config.device
        )
        logger.info("Avatar initialized")

        # 8. 프로세서 초기화
        self._audio_input_processor = AudioInputProcessor()
        self._transcription_processor = TranscriptionProcessor(
            on_transcription=self._on_transcription
        )
        self._llm_processor = LLMProcessor(self._interviewer_agent)
        self._tts_processor = TTSProcessor(self._tts_service)
        self._avatar_processor = AvatarProcessor(self._avatar)
        self._video_output_processor = VideoOutputProcessor()

        logger.info("Processors initialized")

        # 9. Transport 초기화 (Daily.co)
        self._transport = DailyTransport(
            room_url=self.config.daily_room_url,
            token=self.config.daily_token,
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=self._vad_analyzer,
                transcription_enabled=True
            )
        )
        logger.info("Transport initialized")

        # 10. 파이프라인 구성
        self._build_pipeline()

        # 11. 이벤트 핸들러 등록
        self._register_event_handlers()

        logger.info("Pipeline initialization completed")

    def _build_pipeline(self):
        """파이프라인 구축"""
        logger.info("Building pipeline...")

        # 파이프라인 구조:
        # [WebRTC Input]
        #   ↓
        # [Audio Input Processor]
        #   ↓
        # [VAD - Silero]
        #   ↓
        # [STT - Deepgram]
        #   ↓
        # [Transcription Processor]
        #   ↓
        # [LLM Processor - Interviewer Agent]
        #   ↓
        # [TTS - ElevenLabs]
        #   ↓
        # [TTS Processor]
        #   ↓
        # [Avatar Processor - MuseTalk]
        #   ↓
        # [Video Output Processor]
        #   ↓
        # [WebRTC Output]

        self._pipeline = Pipeline([
            self._transport.input(),           # WebRTC 입력
            self._audio_input_processor,       # 오디오 전처리
            self._stt_service,                 # STT (Deepgram)
            self._transcription_processor,     # 전사 결과 처리
            self._llm_processor,               # LLM (면접관)
            self._tts_service,                 # TTS (ElevenLabs)
            self._tts_processor,               # TTS 후처리
            self._avatar_processor,            # 아바타 생성
            self._video_output_processor,      # 비디오 후처리
            self._transport.output(),          # WebRTC 출력
        ])

        logger.info("Pipeline built successfully")

    def _register_event_handlers(self):
        """이벤트 핸들러 등록"""
        # Transport 이벤트
        @self._transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            logger.info(f"Participant joined: {participant}")
            await self._emit_event('on_participant_joined', participant)

            # 환영 인사
            greeting = await self._interviewer_agent.start_interview()
            await self._send_message(greeting)

        @self._transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant}, reason: {reason}")
            await self._emit_event('on_participant_left', participant, reason)

        @self._transport.event_handler("on_error")
        async def on_error(transport, error):
            logger.error(f"Transport error: {error}")
            self._set_state(PipelineState.ERROR)
            await self._emit_event('on_error', error)

    async def _on_transcription(self, text: str, is_final: bool):
        """전사 결과 콜백"""
        if is_final:
            # 사용자 말하기 완료
            self._set_state(PipelineState.PROCESSING)

            logger.info(f"User said: {text}")

    async def _send_message(self, text: str):
        """메시지 전송 (TTS + 아바타)"""
        logger.info(f"Sending message: {text}")

        self._set_state(PipelineState.SPEAKING)

        # TextFrame을 파이프라인에 주입
        text_frame = TextFrame(text=text)
        await self._pipeline.queue_frame(text_frame)

    def _set_state(self, state: PipelineState):
        """상태 변경"""
        if self._state != state:
            old_state = self._state
            self._state = state

            logger.info(f"State changed: {old_state.value} → {state.value}")

            # 이벤트 발생
            asyncio.create_task(
                self._emit_event('on_state_changed', old_state, state)
            )

    def get_state(self) -> PipelineState:
        """현재 상태 반환"""
        return self._state

    async def start(self):
        """파이프라인 시작"""
        logger.info("Starting pipeline...")

        if self._pipeline is None:
            await self.initialize()

        # Runner 생성
        self._task = PipelineTask(
            self._pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True
            )
        )

        self._runner = PipelineRunner()

        # 시작 이벤트
        await self._emit_event('on_pipeline_started')

        # 파이프라인 실행
        await self._runner.run(self._task)

        logger.info("Pipeline started")

    async def stop(self):
        """파이프라인 중지"""
        logger.info("Stopping pipeline...")

        if self._runner:
            await self._runner.stop()

        self._set_state(PipelineState.IDLE)

        # 중지 이벤트
        await self._emit_event('on_pipeline_stopped')

        logger.info("Pipeline stopped")

    def on(self, event_name: str, handler: Callable):
        """이벤트 핸들러 등록"""
        if event_name in self._event_handlers:
            self._event_handlers[event_name].append(handler)
        else:
            logger.warning(f"Unknown event: {event_name}")

    async def _emit_event(self, event_name: str, *args, **kwargs):
        """이벤트 발생"""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(*args, **kwargs)
                    else:
                        handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Event handler error ({event_name}): {e}")

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        stats = {
            "state": self._state.value,
            "interviewer_stage": self._interviewer_agent.current_stage.value if self._interviewer_agent else None,
            "avatar_fps": 0.0,
        }

        if self._avatar:
            avatar_stats = self._avatar.get_stats()
            stats["avatar_fps"] = avatar_stats.get("fps", 0.0)

        return stats


# 헬퍼 함수
async def create_interview_pipeline(
    daily_room_url: str,
    daily_token: Optional[str] = None,
    avatar_source_image: str = "./assets/avatar_source.jpg",
    device: str = "cpu"
) -> InterviewAvatarPipeline:
    """
    면접 파이프라인 생성 및 초기화

    Args:
        daily_room_url: Daily.co 룸 URL
        daily_token: Daily.co 토큰
        avatar_source_image: 아바타 소스 이미지
        device: 디바이스 (cpu/cuda)

    Returns:
        InterviewAvatarPipeline: 초기화된 파이프라인
    """
    config = PipelineConfig(
        daily_room_url=daily_room_url,
        daily_token=daily_token,
        avatar_source_image=avatar_source_image,
        device=device
    )

    pipeline = InterviewAvatarPipeline(config)
    await pipeline.initialize()

    return pipeline


# 사용 예시
if __name__ == "__main__":
    async def main():
        # 설정
        config = PipelineConfig(
            daily_room_url="https://your-domain.daily.co/your-room",
            daily_token="your-token",
            avatar_source_image="./assets/avatar_source.jpg",
            device="cuda"
        )

        # 파이프라인 생성
        pipeline = InterviewAvatarPipeline(config)

        # 이벤트 핸들러 등록
        @pipeline.on('on_participant_joined')
        async def on_joined(participant):
            print(f"참가자 입장: {participant}")

        @pipeline.on('on_state_changed')
        async def on_state_changed(old_state, new_state):
            print(f"상태 변경: {old_state.value} → {new_state.value}")

        # 초기화 및 시작
        await pipeline.initialize()
        await pipeline.start()

        # 통계 출력 (주기적)
        while True:
            await asyncio.sleep(5)
            stats = pipeline.get_stats()
            print(f"Stats: {stats}")

    asyncio.run(main())
