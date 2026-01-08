"""
실시간 면접 파이프라인
STT → LLM → TTS → Avatar 통합 파이프라인
"""

import asyncio
import time
import base64
import numpy as np
from typing import Optional, Callable, AsyncGenerator, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from loguru import logger

# Whisper for STT
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("openai-whisper not installed. Local STT will not work.")

# OpenAI for LLM
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. LLM will not work.")

# EdgeTTS
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts not installed. TTS will not work.")


class PipelineState(Enum):
    """파이프라인 상태"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING_STT = "processing_stt"
    PROCESSING_LLM = "processing_llm"
    PROCESSING_TTS = "processing_tts"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # STT
    stt_model: str = "base"  # whisper model: tiny, base, small, medium
    stt_language: str = "ko"

    # LLM
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150
    openai_api_key: Optional[str] = None

    # TTS
    tts_voice: str = "ko-KR-SunHiNeural"
    tts_rate: str = "+0%"

    # Audio
    sample_rate: int = 16000

    # Callbacks
    on_state_change: Optional[Callable[[PipelineState], None]] = None
    on_stt_result: Optional[Callable[[str, bool], None]] = None
    on_llm_token: Optional[Callable[[str], None]] = None
    on_tts_audio: Optional[Callable[[bytes], None]] = None


@dataclass
class PipelineMetrics:
    """파이프라인 성능 메트릭"""
    stt_latency_ms: float = 0.0
    llm_ttft_ms: float = 0.0  # Time to first token
    llm_total_ms: float = 0.0
    tts_ttfb_ms: float = 0.0  # Time to first byte
    tts_total_ms: float = 0.0
    total_latency_ms: float = 0.0


class WhisperSTTService:
    """로컬 Whisper STT 서비스"""

    def __init__(
        self,
        model_name: str = "base",
        language: str = "ko",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.model = None

        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available")
            return

        logger.info(f"Loading Whisper model: {model_name}")
        try:
            self.model = whisper.load_model(model_name, device=device)
            logger.info(f"Whisper model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            # CPU fallback
            try:
                self.model = whisper.load_model(model_name, device="cpu")
                self.device = "cpu"
                logger.info("Whisper loaded on CPU (fallback)")
            except Exception as e2:
                logger.error(f"Failed to load Whisper on CPU: {e2}")

    async def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """
        오디오를 텍스트로 변환

        Args:
            audio_data: numpy array (float32, 16kHz)

        Returns:
            transcribed text or None
        """
        if self.model is None:
            logger.warning("Whisper model not loaded")
            return None

        try:
            # Whisper expects float32 audio normalized to [-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / 32768.0

            # Run transcription in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    audio_data,
                    language=self.language,
                    fp16=(self.device == "cuda"),
                    task="transcribe"
                )
            )

            text = result.get("text", "").strip()
            return text if text else None

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None


class LLMService:
    """OpenAI LLM 서비스"""

    SYSTEM_PROMPT = """당신은 전문적이고 친근한 면접관입니다.

# 역할
- 지원자의 역량, 경험, 기술을 공정하게 평가합니다
- 존중하고 격려하는 태도로 면접을 진행합니다

# 응답 스타일
- 한국어로 존댓말을 사용합니다
- 짧고 명확한 문장을 사용합니다 (2-3문장)
- 한 번에 하나의 질문만 합니다
- 따뜻하고 격려하는 톤을 유지합니다

# 면접 진행
1. 먼저 간단한 자기소개를 요청합니다
2. 경험과 프로젝트에 대해 질문합니다
3. 기술적인 질문을 합니다
4. 상황 대처 능력을 확인합니다
5. 마무리 질문을 합니다"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 150
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self.conversation_history = []

        if not OPENAI_AVAILABLE:
            logger.error("OpenAI not available")
            return

        if not api_key:
            logger.error("OpenAI API key not provided")
            return

        self.client = AsyncOpenAI(api_key=api_key)
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        logger.info(f"LLM service initialized: {model}")

    def reset_conversation(self):
        """대화 기록 초기화"""
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

    async def generate_response(
        self,
        user_message: str,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        사용자 메시지에 대한 응답 생성

        Args:
            user_message: 사용자 입력
            stream: 스트리밍 여부

        Yields:
            response tokens or full response
        """
        if self.client is None:
            logger.error("LLM client not initialized")
            yield "[LLM 서비스 오류] 잠시 후 다시 시도해주세요."
            return

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            if stream:
                # Streaming response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )

                full_response = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        yield token

                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
            else:
                # Non-streaming response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False
                )

                full_response = response.choices[0].message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                yield full_response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            error_msg = "죄송합니다. 일시적인 오류가 발생했습니다. 다시 말씀해 주시겠어요?"
            self.conversation_history.append({
                "role": "assistant",
                "content": error_msg
            })
            yield error_msg


class TTSService:
    """EdgeTTS 서비스"""

    def __init__(
        self,
        voice: str = "ko-KR-SunHiNeural",
        rate: str = "+0%",
        volume: str = "+0%"
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume

        if not EDGE_TTS_AVAILABLE:
            logger.error("EdgeTTS not available")
            return

        logger.info(f"TTS service initialized: {voice}")

    async def synthesize(self, text: str) -> Optional[bytes]:
        """
        텍스트를 음성으로 합성

        Args:
            text: 합성할 텍스트

        Returns:
            audio bytes (MP3) or None
        """
        if not EDGE_TTS_AVAILABLE:
            return None

        if not text or not text.strip():
            return None

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )

            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            return b"".join(audio_chunks)

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        텍스트를 음성으로 스트리밍 합성

        Args:
            text: 합성할 텍스트

        Yields:
            audio chunks (MP3)
        """
        if not EDGE_TTS_AVAILABLE:
            return

        if not text or not text.strip():
            return

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")


class RealtimePipeline:
    """
    실시간 면접 파이프라인

    STT → LLM → TTS 통합 처리
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.state = PipelineState.IDLE

        # Services
        self.stt: Optional[WhisperSTTService] = None
        self.llm: Optional[LLMService] = None
        self.tts: Optional[TTSService] = None

        # Metrics
        self.last_metrics = PipelineMetrics()

        # Audio buffer
        self.audio_buffer = []

        logger.info("RealtimePipeline created")

    def _set_state(self, state: PipelineState):
        """상태 변경 및 콜백 호출"""
        self.state = state
        if self.config.on_state_change:
            self.config.on_state_change(state)

    async def initialize(self):
        """파이프라인 초기화"""
        logger.info("Initializing pipeline services...")

        # Initialize STT
        if WHISPER_AVAILABLE:
            self.stt = WhisperSTTService(
                model_name=self.config.stt_model,
                language=self.config.stt_language
            )

        # Initialize LLM
        if OPENAI_AVAILABLE and self.config.openai_api_key:
            self.llm = LLMService(
                api_key=self.config.openai_api_key,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )

        # Initialize TTS
        if EDGE_TTS_AVAILABLE:
            self.tts = TTSService(
                voice=self.config.tts_voice,
                rate=self.config.tts_rate
            )

        logger.info(f"Pipeline initialized - STT: {self.stt is not None}, "
                   f"LLM: {self.llm is not None}, TTS: {self.tts is not None}")

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        오디오 데이터 처리 (전체 파이프라인)

        Args:
            audio_data: raw audio bytes (PCM int16 or base64)
            sample_rate: sample rate

        Returns:
            result dict with transcript, response, audio
        """
        result = {
            "transcript": None,
            "response": None,
            "audio": None,
            "metrics": None,
            "error": None
        }

        total_start = time.time()

        try:
            # 1. Decode audio
            if isinstance(audio_data, str):
                # Base64 encoded
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data

            # Convert to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0  # Normalize to [-1, 1]

            # 2. STT
            self._set_state(PipelineState.PROCESSING_STT)
            stt_start = time.time()

            if self.stt:
                transcript = await self.stt.transcribe(audio_np)
            else:
                transcript = None
                logger.warning("STT not available")

            stt_latency = (time.time() - stt_start) * 1000
            self.last_metrics.stt_latency_ms = stt_latency

            if not transcript:
                result["error"] = "음성을 인식하지 못했습니다"
                self._set_state(PipelineState.IDLE)
                return result

            result["transcript"] = transcript

            # Callback
            if self.config.on_stt_result:
                self.config.on_stt_result(transcript, True)

            logger.info(f"STT: '{transcript}' ({stt_latency:.0f}ms)")

            # 3. LLM
            self._set_state(PipelineState.PROCESSING_LLM)
            llm_start = time.time()
            ttft = None

            if self.llm:
                full_response = ""
                async for token in self.llm.generate_response(transcript, stream=True):
                    if ttft is None:
                        ttft = (time.time() - llm_start) * 1000
                        self.last_metrics.llm_ttft_ms = ttft

                    full_response += token

                    if self.config.on_llm_token:
                        self.config.on_llm_token(token)

                result["response"] = full_response
            else:
                result["response"] = f"[AI] '{transcript}'에 대한 답변입니다."
                logger.warning("LLM not available, using placeholder")

            llm_total = (time.time() - llm_start) * 1000
            self.last_metrics.llm_total_ms = llm_total

            logger.info(f"LLM: '{result['response'][:50]}...' "
                       f"(TTFT: {ttft:.0f}ms, Total: {llm_total:.0f}ms)")

            # 4. TTS
            self._set_state(PipelineState.PROCESSING_TTS)
            tts_start = time.time()

            if self.tts:
                audio_output = await self.tts.synthesize(result["response"])
                if audio_output:
                    result["audio"] = base64.b64encode(audio_output).decode()
            else:
                logger.warning("TTS not available")

            tts_total = (time.time() - tts_start) * 1000
            self.last_metrics.tts_total_ms = tts_total

            logger.info(f"TTS: {len(result.get('audio', '') or '')} bytes ({tts_total:.0f}ms)")

            # Total metrics
            total_latency = (time.time() - total_start) * 1000
            self.last_metrics.total_latency_ms = total_latency

            result["metrics"] = {
                "stt_ms": stt_latency,
                "llm_ttft_ms": ttft or 0,
                "llm_total_ms": llm_total,
                "tts_ms": tts_total,
                "total_ms": total_latency
            }

            self._set_state(PipelineState.SPEAKING)

            logger.info(f"Pipeline complete: {total_latency:.0f}ms total")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
            self._set_state(PipelineState.ERROR)

        return result

    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        텍스트 입력 처리 (LLM → TTS)

        Args:
            text: user input text

        Returns:
            result dict
        """
        result = {
            "transcript": text,
            "response": None,
            "audio": None,
            "metrics": None,
            "error": None
        }

        total_start = time.time()

        try:
            # LLM
            self._set_state(PipelineState.PROCESSING_LLM)
            llm_start = time.time()
            ttft = None

            if self.llm:
                full_response = ""
                async for token in self.llm.generate_response(text, stream=True):
                    if ttft is None:
                        ttft = (time.time() - llm_start) * 1000
                    full_response += token

                    if self.config.on_llm_token:
                        self.config.on_llm_token(token)

                result["response"] = full_response
            else:
                result["response"] = f"[AI] '{text}'에 대한 답변입니다."

            llm_total = (time.time() - llm_start) * 1000

            # TTS
            self._set_state(PipelineState.PROCESSING_TTS)
            tts_start = time.time()

            if self.tts:
                audio_output = await self.tts.synthesize(result["response"])
                if audio_output:
                    result["audio"] = base64.b64encode(audio_output).decode()

            tts_total = (time.time() - tts_start) * 1000
            total_latency = (time.time() - total_start) * 1000

            result["metrics"] = {
                "llm_ttft_ms": ttft or 0,
                "llm_total_ms": llm_total,
                "tts_ms": tts_total,
                "total_ms": total_latency
            }

            self._set_state(PipelineState.SPEAKING)

        except Exception as e:
            logger.error(f"Text pipeline error: {e}")
            result["error"] = str(e)
            self._set_state(PipelineState.ERROR)

        return result

    async def get_greeting(self) -> Dict[str, Any]:
        """첫 인사 생성"""
        greeting = "안녕하세요! 면접에 참여해 주셔서 감사합니다. 편안하게 답변해 주시면 됩니다. 먼저 간단한 자기소개 부탁드립니다."

        result = {
            "response": greeting,
            "audio": None
        }

        if self.tts:
            audio = await self.tts.synthesize(greeting)
            if audio:
                result["audio"] = base64.b64encode(audio).decode()

        # Add to LLM history
        if self.llm:
            self.llm.conversation_history.append({
                "role": "assistant",
                "content": greeting
            })

        return result

    def reset(self):
        """파이프라인 상태 초기화"""
        self.state = PipelineState.IDLE
        self.audio_buffer.clear()
        self.last_metrics = PipelineMetrics()

        if self.llm:
            self.llm.reset_conversation()

        logger.info("Pipeline reset")

    def get_status(self) -> Dict[str, Any]:
        """파이프라인 상태 조회"""
        return {
            "state": self.state.value,
            "stt_available": self.stt is not None,
            "llm_available": self.llm is not None and self.llm.client is not None,
            "tts_available": self.tts is not None,
            "last_metrics": {
                "stt_ms": self.last_metrics.stt_latency_ms,
                "llm_ttft_ms": self.last_metrics.llm_ttft_ms,
                "llm_total_ms": self.last_metrics.llm_total_ms,
                "tts_ms": self.last_metrics.tts_total_ms,
                "total_ms": self.last_metrics.total_latency_ms
            }
        }


# Factory function
async def create_pipeline(
    openai_api_key: Optional[str] = None,
    stt_model: str = "base",
    llm_model: str = "gpt-3.5-turbo",
    tts_voice: str = "ko-KR-SunHiNeural"
) -> RealtimePipeline:
    """
    파이프라인 생성 및 초기화

    Args:
        openai_api_key: OpenAI API key
        stt_model: Whisper model name
        llm_model: OpenAI model name
        tts_voice: EdgeTTS voice name

    Returns:
        initialized pipeline
    """
    config = PipelineConfig(
        stt_model=stt_model,
        llm_model=llm_model,
        tts_voice=tts_voice,
        openai_api_key=openai_api_key
    )

    pipeline = RealtimePipeline(config)
    await pipeline.initialize()

    return pipeline
