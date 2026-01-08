"""
Realtime Interview Avatar Pipeline
STT -> LLM -> TTS -> MuseTalk Avatar
"""

import asyncio
import time
import tempfile
import os
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from loguru import logger

# Import services
from src.stt.deepgram_service import DeepgramSTTService, TranscriptResult
from src.llm.interviewer_agent import InterviewerAgent, create_interviewer, InterviewStage
from src.tts.elevenlabs_service import ElevenLabsTTSService, AudioChunk, create_elevenlabs_tts

# Import Avatar
try:
    from src.avatar.musetalk_realtime import (
        MuseTalkRealtimeAvatar,
        create_realtime_avatar,
        VideoFrame,
        AvatarConfig
    )
    AVATAR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Avatar not available: {e}")
    AVATAR_AVAILABLE = False


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # STT
    stt_language: str = "ko"
    stt_model: str = "nova-3"

    # LLM
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7

    # TTS
    tts_voice_id: str = ""  # Set from env
    tts_stability: float = 0.5
    tts_similarity_boost: float = 0.75

    # Avatar
    avatar_source_image: str = "./assets/images/avatar.jpg"
    avatar_fps: int = 25
    use_float16: bool = True
    silence_blend: bool = True

    # General
    device: str = "cuda"


class RealtimeAvatarPipeline:
    """
    Real-time Interview Avatar Pipeline

    Flow:
    1. User speaks -> STT (Deepgram) -> Text
    2. Text -> LLM (GPT-4o Interviewer) -> Response
    3. Response -> TTS (ElevenLabs) -> Audio
    4. Audio -> Avatar (MuseTalk) -> Video frames
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_transcript: Optional[Callable[[str, bool], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        on_video_frame: Optional[Callable[[np.ndarray], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Initialize pipeline

        Args:
            config: Pipeline configuration
            on_transcript: Callback for STT results (text, is_final)
            on_response: Callback for LLM responses
            on_audio_chunk: Callback for TTS audio chunks
            on_video_frame: Callback for avatar video frames
            on_error: Callback for errors
        """
        self.config = config or PipelineConfig()

        # Callbacks
        self._on_transcript = on_transcript
        self._on_response = on_response
        self._on_audio_chunk = on_audio_chunk
        self._on_video_frame = on_video_frame
        self._on_error = on_error

        # Services
        self._stt: Optional[DeepgramSTTService] = None
        self._llm: Optional[InterviewerAgent] = None
        self._tts: Optional[ElevenLabsTTSService] = None
        self._avatar: Optional[MuseTalkRealtimeAvatar] = None

        # State
        self._is_initialized = False
        self._is_processing = False
        self._current_transcript = ""

        # Statistics
        self._stats = {
            "stt_calls": 0,
            "llm_calls": 0,
            "tts_calls": 0,
            "avatar_frames": 0,
            "total_latency_ms": [],
        }

        logger.info("RealtimeAvatarPipeline created")

    async def initialize(self) -> None:
        """Initialize all services"""
        if self._is_initialized:
            return

        logger.info("Initializing pipeline...")
        start_time = time.time()

        # 1. Initialize STT
        self._stt = DeepgramSTTService(
            language=self.config.stt_language,
            model=self.config.stt_model,
            on_transcript=self._handle_transcript,
            on_error=self._handle_error,
        )
        logger.info("STT service initialized")

        # 2. Initialize LLM (Interviewer Agent)
        self._llm = create_interviewer(
            model=self.config.llm_model,
            on_question_generated=self._handle_question,
        )
        logger.info("LLM service initialized")

        # 3. Initialize TTS
        self._tts = create_elevenlabs_tts(
            voice_id=self.config.tts_voice_id or os.getenv("TTS_VOICE_ID"),
            stability=self.config.tts_stability,
            similarity_boost=self.config.tts_similarity_boost,
            on_audio_chunk=self._handle_audio_chunk,
        )
        logger.info("TTS service initialized")

        # 4. Initialize Avatar (if available)
        if AVATAR_AVAILABLE:
            try:
                self._avatar = create_realtime_avatar(
                    source_image_path=self.config.avatar_source_image,
                    device=self.config.device,
                    use_float16=self.config.use_float16,
                    silence_blend=self.config.silence_blend,
                    on_frame_ready=self._handle_video_frame,
                )
                self._avatar.initialize()
                logger.info("Avatar service initialized")
            except Exception as e:
                logger.warning(f"Avatar initialization failed: {e}")
                self._avatar = None
        else:
            logger.warning("Avatar not available in this environment")

        self._is_initialized = True
        init_time = time.time() - start_time
        logger.info(f"Pipeline initialized in {init_time:.2f}s")

    async def start_interview(self) -> str:
        """
        Start the interview session

        Returns:
            Initial greeting message
        """
        if not self._is_initialized:
            await self.initialize()

        # Get greeting from interviewer
        greeting = await self._llm.start_interview()

        # Generate TTS and avatar for greeting
        await self._process_response(greeting)

        return greeting

    async def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process audio input through the full pipeline

        Args:
            audio_data: Raw PCM audio data

        Returns:
            Dict with transcript, response, audio, and video frames
        """
        if not self._is_initialized:
            await self.initialize()

        self._is_processing = True
        start_time = time.time()

        result = {
            "transcript": "",
            "response": "",
            "audio": None,
            "video_frames": [],
            "metrics": {}
        }

        try:
            # 1. STT: Audio -> Text
            stt_start = time.time()
            await self._stt.connect()
            await self._stt.send_audio(audio_data)
            # Wait for transcript (handled via callback)
            await asyncio.sleep(0.5)  # Wait for final transcript
            await self._stt.disconnect()

            transcript = self._current_transcript
            result["transcript"] = transcript
            result["metrics"]["stt_ms"] = (time.time() - stt_start) * 1000
            self._stats["stt_calls"] += 1

            if not transcript.strip():
                logger.warning("Empty transcript, skipping")
                return result

            # 2. LLM: Text -> Response
            llm_start = time.time()
            response = await self._llm.process_answer(transcript)
            result["response"] = response
            result["metrics"]["llm_ms"] = (time.time() - llm_start) * 1000
            self._stats["llm_calls"] += 1

            # 3. TTS + Avatar: Response -> Audio + Video
            tts_start = time.time()
            audio_video = await self._process_response(response)
            result["audio"] = audio_video.get("audio")
            result["video_frames"] = audio_video.get("video_frames", [])
            result["metrics"]["tts_avatar_ms"] = (time.time() - tts_start) * 1000

            # Total latency
            total_ms = (time.time() - start_time) * 1000
            result["metrics"]["total_ms"] = total_ms
            self._stats["total_latency_ms"].append(total_ms)

            logger.info(f"Pipeline complete: {total_ms:.0f}ms total")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result["error"] = str(e)
            if self._on_error:
                self._on_error(e)

        finally:
            self._is_processing = False

        return result

    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input (skip STT)

        Args:
            text: User text input

        Returns:
            Dict with response, audio, and video frames
        """
        if not self._is_initialized:
            await self.initialize()

        start_time = time.time()

        result = {
            "response": "",
            "audio": None,
            "video_frames": [],
            "metrics": {}
        }

        try:
            # 1. LLM: Text -> Response
            llm_start = time.time()
            response = await self._llm.process_answer(text)
            result["response"] = response
            result["metrics"]["llm_ms"] = (time.time() - llm_start) * 1000
            self._stats["llm_calls"] += 1

            # 2. TTS + Avatar
            tts_start = time.time()
            audio_video = await self._process_response(response)
            result["audio"] = audio_video.get("audio")
            result["video_frames"] = audio_video.get("video_frames", [])
            result["metrics"]["tts_avatar_ms"] = (time.time() - tts_start) * 1000

            total_ms = (time.time() - start_time) * 1000
            result["metrics"]["total_ms"] = total_ms

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result["error"] = str(e)
            if self._on_error:
                self._on_error(e)

        return result

    async def _process_response(self, response: str) -> Dict[str, Any]:
        """
        Process LLM response through TTS and Avatar

        Args:
            response: LLM response text

        Returns:
            Dict with audio and video frames
        """
        result = {
            "audio": None,
            "video_frames": []
        }

        # 1. TTS: Text -> Audio
        audio_data = await self._tts.synthesize(response, streaming=False)
        result["audio"] = audio_data
        self._stats["tts_calls"] += 1

        if self._on_audio_chunk and audio_data:
            self._on_audio_chunk(audio_data)

        # 2. Avatar: Audio -> Video (if available)
        if self._avatar and audio_data:
            try:
                # Save audio to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    # Convert to WAV format if needed
                    f.write(audio_data)
                    temp_audio_path = f.name

                # Process through avatar
                video_frames = self._avatar.process_audio(temp_audio_path)
                result["video_frames"] = video_frames
                self._stats["avatar_frames"] += len(video_frames)

                # Cleanup
                os.unlink(temp_audio_path)

            except Exception as e:
                logger.error(f"Avatar processing failed: {e}")

        return result

    def _handle_transcript(self, result: TranscriptResult) -> None:
        """Handle STT transcript callback"""
        if result.is_final:
            self._current_transcript = result.text
            logger.info(f"[STT Final] {result.text}")
        else:
            logger.debug(f"[STT Interim] {result.text}")

        if self._on_transcript:
            self._on_transcript(result.text, result.is_final)

    def _handle_question(self, question: str) -> None:
        """Handle LLM question callback"""
        logger.info(f"[LLM] {question}")
        if self._on_response:
            self._on_response(question)

    def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        """Handle TTS audio chunk callback"""
        if self._on_audio_chunk:
            self._on_audio_chunk(chunk.data)

    def _handle_video_frame(self, frame: VideoFrame) -> None:
        """Handle avatar video frame callback"""
        if self._on_video_frame:
            self._on_video_frame(frame.frame)

    def _handle_error(self, error: Exception) -> None:
        """Handle error callback"""
        logger.error(f"Pipeline error: {error}")
        if self._on_error:
            self._on_error(error)

    async def get_greeting(self) -> Dict[str, Any]:
        """
        Get initial greeting with TTS and avatar

        Returns:
            Dict with greeting text, audio, and video frames
        """
        greeting = await self.start_interview()

        result = {
            "response": greeting,
            "audio": None,
            "video_frames": []
        }

        # Generate TTS and avatar for greeting
        audio_video = await self._process_response(greeting)
        result["audio"] = audio_video.get("audio")
        result["video_frames"] = audio_video.get("video_frames", [])

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        avg_latency = np.mean(self._stats["total_latency_ms"]) if self._stats["total_latency_ms"] else 0

        return {
            "stt_calls": self._stats["stt_calls"],
            "llm_calls": self._stats["llm_calls"],
            "tts_calls": self._stats["tts_calls"],
            "avatar_frames": self._stats["avatar_frames"],
            "avg_latency_ms": avg_latency,
            "interview_stage": self._llm._current_stage.value if self._llm else None,
        }

    def reset(self) -> None:
        """Reset pipeline state"""
        self._current_transcript = ""
        self._stats = {
            "stt_calls": 0,
            "llm_calls": 0,
            "tts_calls": 0,
            "avatar_frames": 0,
            "total_latency_ms": [],
        }
        if self._llm:
            self._llm.reset()
        if self._avatar:
            self._avatar.reset()
        logger.info("Pipeline reset")


# Helper function
async def create_pipeline(
    openai_api_key: Optional[str] = None,
    deepgram_api_key: Optional[str] = None,
    elevenlabs_api_key: Optional[str] = None,
    avatar_source_image: str = "./assets/images/avatar.jpg",
    stt_model: str = "nova-3",
    llm_model: str = "gpt-4o",
    tts_voice: str = "",
) -> RealtimeAvatarPipeline:
    """
    Create and initialize a pipeline

    Args:
        openai_api_key: OpenAI API key
        deepgram_api_key: Deepgram API key
        elevenlabs_api_key: ElevenLabs API key
        avatar_source_image: Path to avatar source image
        stt_model: STT model name
        llm_model: LLM model name
        tts_voice: TTS voice ID

    Returns:
        Initialized RealtimeAvatarPipeline
    """
    # Set API keys to environment if provided
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if deepgram_api_key:
        os.environ["DEEPGRAM_API_KEY"] = deepgram_api_key
    if elevenlabs_api_key:
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key

    config = PipelineConfig(
        stt_model=stt_model,
        llm_model=llm_model,
        tts_voice_id=tts_voice,
        avatar_source_image=avatar_source_image,
    )

    pipeline = RealtimeAvatarPipeline(config)
    await pipeline.initialize()

    return pipeline


# Test
if __name__ == "__main__":
    async def test_pipeline():
        """Test pipeline"""
        print("=" * 60)
        print("Realtime Avatar Pipeline Test")
        print("=" * 60)

        # Create pipeline
        pipeline = RealtimeAvatarPipeline()
        await pipeline.initialize()

        # Test greeting
        print("\n--- Getting greeting ---")
        greeting = await pipeline.get_greeting()
        print(f"Greeting: {greeting['response'][:100]}...")
        print(f"Audio bytes: {len(greeting.get('audio', b''))}")
        print(f"Video frames: {len(greeting.get('video_frames', []))}")

        # Test text input
        print("\n--- Processing text ---")
        result = await pipeline.process_text("안녕하세요, 저는 5년 경력의 개발자입니다.")
        print(f"Response: {result['response'][:100]}...")
        print(f"Metrics: {result.get('metrics', {})}")

        # Stats
        print("\n--- Stats ---")
        print(pipeline.get_stats())

    asyncio.run(test_pipeline())
