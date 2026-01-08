"""
Custom Pipecat Processors
실시간 면접 아바타를 위한 커스텀 프레임 프로세서
"""

import asyncio
import time
import numpy as np
from typing import Optional, Dict, List, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field

from loguru import logger

try:
    from pipecat.frames.frames import (
        Frame,
        AudioRawFrame,
        ImageRawFrame,
        TextFrame,
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        CancelFrame,
        ErrorFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    PIPECAT_AVAILABLE = True
except ImportError:
    logger.warning("Pipecat not available")
    PIPECAT_AVAILABLE = False
    FrameProcessor = object
    Frame = object
    FrameDirection = object
    AudioRawFrame = object
    ImageRawFrame = object
    TextFrame = object
    TranscriptionFrame = object
    UserStartedSpeakingFrame = object
    UserStoppedSpeakingFrame = object
    CancelFrame = object
    ErrorFrame = object


# ============================================================================
# 1. AvatarFrameProcessor
# ============================================================================

class AvatarFrameProcessor(FrameProcessor):
    """
    MuseTalk 아바타 프레임 생성 프로세서

    TTS 오디오를 받아서 립싱크된 비디오 프레임으로 변환
    """

    def __init__(
        self,
        avatar,
        target_fps: int = 25,
        idle_animation: bool = True
    ):
        """
        Args:
            avatar: MuseTalkAvatar 인스턴스
            target_fps: 목표 FPS
            idle_animation: 대기 애니메이션 사용 여부
        """
        super().__init__()

        self.avatar = avatar
        self.target_fps = target_fps
        self.idle_animation = idle_animation

        # 프레임 타이밍
        self.frame_duration = 1.0 / target_fps  # 초
        self.last_frame_time = 0

        # 오디오 버퍼
        self.audio_buffer = bytearray()
        self.chunk_size = int(16000 * self.frame_duration)  # 16kHz 기준

        # 상태
        self.is_speaking = False
        self._idle_task = None

        # 통계
        self.frames_generated = 0

        logger.info(
            f"AvatarFrameProcessor initialized: "
            f"fps={target_fps}, idle={idle_animation}"
        )

    async def start(self):
        """프로세서 시작"""
        await super().start()

        # Idle 애니메이션 시작
        if self.idle_animation:
            self._idle_task = asyncio.create_task(self._idle_animation_loop())

    async def stop(self):
        """프로세서 중지"""
        # Idle 애니메이션 중지
        if self._idle_task:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass

        await super().stop()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """프레임 처리"""
        await super().process_frame(frame, direction)

        # TTS 오디오 → 아바타 비디오
        if isinstance(frame, AudioRawFrame):
            self.is_speaking = True

            # 오디오 버퍼에 추가
            self.audio_buffer.extend(frame.audio)

            # 충분한 오디오가 모이면 비디오 프레임 생성
            while len(self.audio_buffer) >= self.chunk_size:
                audio_chunk = bytes(self.audio_buffer[:self.chunk_size])
                self.audio_buffer = self.audio_buffer[self.chunk_size:]

                # MuseTalk로 비디오 프레임 생성
                await self._generate_avatar_frame(audio_chunk)

        # 발화 종료
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.is_speaking = False

            # 남은 오디오 처리
            if len(self.audio_buffer) > 0:
                audio_chunk = bytes(self.audio_buffer)
                self.audio_buffer.clear()

                await self._generate_avatar_frame(audio_chunk)

        # 다른 프레임은 그대로 전달
        else:
            await self.push_frame(frame, direction)

    async def _generate_avatar_frame(self, audio_chunk: bytes):
        """아바타 비디오 프레임 생성"""
        try:
            # 오디오 → numpy
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

            # MuseTalk 처리
            video_frame = self.avatar.process_audio_chunk(audio_data)

            # ImageRawFrame 생성
            image_frame = ImageRawFrame(
                image=video_frame.frame.tobytes(),
                size=(video_frame.frame.shape[1], video_frame.frame.shape[0]),
                format="RGB"
            )

            # 프레임 전송
            await self.push_frame(image_frame, FrameDirection.DOWNSTREAM)

            self.frames_generated += 1

            # FPS 조절
            current_time = time.time()
            elapsed = current_time - self.last_frame_time

            if elapsed < self.frame_duration:
                await asyncio.sleep(self.frame_duration - elapsed)

            self.last_frame_time = time.time()

        except Exception as e:
            logger.error(f"Avatar frame generation failed: {e}")

    async def _idle_animation_loop(self):
        """대기 애니메이션 루프"""
        logger.info("Idle animation started")

        try:
            while True:
                # 말하는 중이 아닐 때만
                if not self.is_speaking:
                    # Idle 프레임 생성
                    idle_frame = self.avatar.get_idle_frame()

                    image_frame = ImageRawFrame(
                        image=idle_frame.tobytes(),
                        size=(idle_frame.shape[1], idle_frame.shape[0]),
                        format="RGB"
                    )

                    await self.push_frame(image_frame, FrameDirection.DOWNSTREAM)

                # FPS에 맞춰 대기
                await asyncio.sleep(self.frame_duration)

        except asyncio.CancelledError:
            logger.info("Idle animation stopped")
            raise


# ============================================================================
# 2. InterviewContextProcessor
# ============================================================================

@dataclass
class InterviewContext:
    """면접 컨텍스트"""
    # 히스토리
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)

    # 현재 상태
    current_question: Optional[str] = None
    current_stage: str = "greeting"

    # 타임스탬프
    interview_start_time: Optional[float] = None
    last_interaction_time: Optional[float] = None

    # 통계
    total_questions: int = 0
    total_answers: int = 0


class InterviewContextProcessor(FrameProcessor):
    """
    면접 컨텍스트 관리 프로세서

    질문/응답 히스토리 및 면접 단계 추적
    """

    def __init__(self, interviewer_agent):
        """
        Args:
            interviewer_agent: InterviewerAgent 인스턴스
        """
        super().__init__()

        self.interviewer_agent = interviewer_agent
        self.context = InterviewContext()

        logger.info("InterviewContextProcessor initialized")

    async def start(self):
        """프로세서 시작"""
        await super().start()

        # 면접 시작
        self.context.interview_start_time = time.time()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """프레임 처리"""
        await super().process_frame(frame, direction)

        # 사용자 응답 (전사 결과)
        if isinstance(frame, TranscriptionFrame) and frame.is_final:
            answer = frame.text

            # 컨텍스트 업데이트
            self.context.answers.append(answer)
            self.context.total_answers += 1
            self.context.last_interaction_time = time.time()

            logger.info(f"Answer recorded: {answer[:50]}...")

            await self.push_frame(frame, direction)

        # 면접관 질문 (LLM 응답)
        elif isinstance(frame, TextFrame):
            question = frame.text

            # 컨텍스트 업데이트
            self.context.current_question = question
            self.context.questions.append(question)
            self.context.total_questions += 1
            self.context.last_interaction_time = time.time()

            # 현재 단계 업데이트
            if hasattr(self.interviewer_agent, 'current_stage'):
                self.context.current_stage = self.interviewer_agent.current_stage.value

            logger.info(
                f"Question recorded: {question[:50]}... "
                f"(Stage: {self.context.current_stage})"
            )

            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    def get_context(self) -> InterviewContext:
        """컨텍스트 반환"""
        return self.context

    def get_statistics(self) -> Dict[str, Any]:
        """통계 반환"""
        duration = 0
        if self.context.interview_start_time:
            duration = time.time() - self.context.interview_start_time

        return {
            "duration_seconds": duration,
            "total_questions": self.context.total_questions,
            "total_answers": self.context.total_answers,
            "current_stage": self.context.current_stage,
            "questions_per_minute": (
                self.context.total_questions / (duration / 60) if duration > 0 else 0
            )
        }


# ============================================================================
# 3. LatencyMonitorProcessor
# ============================================================================

@dataclass
class LatencyMetrics:
    """레이턴시 메트릭"""
    # 단계별 레이턴시 (ms)
    stt_latency: List[float] = field(default_factory=list)
    llm_latency: List[float] = field(default_factory=list)
    tts_latency: List[float] = field(default_factory=list)
    avatar_latency: List[float] = field(default_factory=list)

    # 전체 레이턴시
    end_to_end_latency: List[float] = field(default_factory=list)


class LatencyMonitorProcessor(FrameProcessor):
    """
    레이턴시 모니터링 프로세서

    각 단계별 레이턴시 측정 및 통계 수집
    """

    def __init__(self, warning_threshold_ms: float = 500.0):
        """
        Args:
            warning_threshold_ms: 경고 임계값 (ms)
        """
        super().__init__()

        self.warning_threshold = warning_threshold_ms
        self.metrics = LatencyMetrics()

        # 타이밍 추적
        self._timestamps = {}

        # 통계 (최대 1000개 유지)
        self._max_samples = 1000

        logger.info(f"LatencyMonitorProcessor initialized: threshold={warning_threshold_ms}ms")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """프레임 처리"""
        await super().process_frame(frame, direction)

        current_time = time.time()

        # 사용자 말하기 시작
        if isinstance(frame, UserStartedSpeakingFrame):
            self._timestamps['user_started'] = current_time

        # 사용자 말하기 종료
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._timestamps['user_stopped'] = current_time

        # 전사 완료 (STT)
        elif isinstance(frame, TranscriptionFrame) and frame.is_final:
            if 'user_stopped' in self._timestamps:
                stt_latency = (current_time - self._timestamps['user_stopped']) * 1000
                self._record_latency('stt', stt_latency)

                self._timestamps['stt_completed'] = current_time

        # LLM 응답 생성 완료
        elif isinstance(frame, TextFrame):
            if 'stt_completed' in self._timestamps:
                llm_latency = (current_time - self._timestamps['stt_completed']) * 1000
                self._record_latency('llm', llm_latency)

                self._timestamps['llm_completed'] = current_time

        # TTS 오디오 생성
        elif isinstance(frame, AudioRawFrame):
            if 'llm_completed' in self._timestamps:
                # 첫 오디오 청크
                if 'tts_first_chunk' not in self._timestamps:
                    tts_latency = (current_time - self._timestamps['llm_completed']) * 1000
                    self._record_latency('tts', tts_latency)

                    self._timestamps['tts_first_chunk'] = current_time

        # 아바타 비디오 프레임
        elif isinstance(frame, ImageRawFrame):
            if 'tts_first_chunk' in self._timestamps:
                # 첫 비디오 프레임
                if 'avatar_first_frame' not in self._timestamps:
                    avatar_latency = (current_time - self._timestamps['tts_first_chunk']) * 1000
                    self._record_latency('avatar', avatar_latency)

                    self._timestamps['avatar_first_frame'] = current_time

                    # End-to-End 레이턴시 계산
                    if 'user_stopped' in self._timestamps:
                        e2e_latency = (current_time - self._timestamps['user_stopped']) * 1000
                        self._record_latency('end_to_end', e2e_latency)

                    # 타임스탬프 초기화
                    self._timestamps.clear()

        await self.push_frame(frame, direction)

    def _record_latency(self, stage: str, latency_ms: float):
        """레이턴시 기록"""
        # 메트릭에 추가
        if stage == 'stt':
            self.metrics.stt_latency.append(latency_ms)
        elif stage == 'llm':
            self.metrics.llm_latency.append(latency_ms)
        elif stage == 'tts':
            self.metrics.tts_latency.append(latency_ms)
        elif stage == 'avatar':
            self.metrics.avatar_latency.append(latency_ms)
        elif stage == 'end_to_end':
            self.metrics.end_to_end_latency.append(latency_ms)

        # 최대 샘플 수 유지
        self._trim_metrics()

        # 경고 로깅
        if latency_ms > self.warning_threshold:
            logger.warning(f"High latency detected: {stage} = {latency_ms:.1f}ms")
        else:
            logger.debug(f"Latency: {stage} = {latency_ms:.1f}ms")

    def _trim_metrics(self):
        """메트릭 크기 제한"""
        for attr in ['stt_latency', 'llm_latency', 'tts_latency', 'avatar_latency', 'end_to_end_latency']:
            latency_list = getattr(self.metrics, attr)
            if len(latency_list) > self._max_samples:
                setattr(self.metrics, attr, latency_list[-self._max_samples:])

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """통계 반환"""
        def calc_stats(latencies: List[float]) -> Dict[str, float]:
            if not latencies:
                return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}

            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)

            return {
                "avg": sum(sorted_latencies) / n,
                "p50": sorted_latencies[int(n * 0.5)],
                "p95": sorted_latencies[int(n * 0.95)],
                "p99": sorted_latencies[int(n * 0.99)],
                "min": sorted_latencies[0],
                "max": sorted_latencies[-1]
            }

        return {
            "stt": calc_stats(self.metrics.stt_latency),
            "llm": calc_stats(self.metrics.llm_latency),
            "tts": calc_stats(self.metrics.tts_latency),
            "avatar": calc_stats(self.metrics.avatar_latency),
            "end_to_end": calc_stats(self.metrics.end_to_end_latency)
        }


# ============================================================================
# 4. FillerProcessor
# ============================================================================

class FillerProcessor(FrameProcessor):
    """
    필러 사운드 프로세서

    LLM 응답 대기 중 "음...", "잠시만요" 등의 필러 삽입
    """

    # 필러 구문
    FILLERS = [
        "음...",
        "잠시만요...",
        "아...",
        "네, 생각해보니...",
        "그러니까...",
    ]

    def __init__(
        self,
        tts_cache,
        delay_threshold: float = 2.0,
        filler_interval: float = 3.0
    ):
        """
        Args:
            tts_cache: TTSCache 인스턴스
            delay_threshold: 필러 시작 임계값 (초)
            filler_interval: 필러 반복 간격 (초)
        """
        super().__init__()

        self.tts_cache = tts_cache
        self.delay_threshold = delay_threshold
        self.filler_interval = filler_interval

        # 타이밍
        self._llm_start_time = None
        self._filler_task = None

        # 필러 인덱스
        self._filler_index = 0

        logger.info(
            f"FillerProcessor initialized: "
            f"threshold={delay_threshold}s, interval={filler_interval}s"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """프레임 처리"""
        await super().process_frame(frame, direction)

        # 전사 완료 (LLM 처리 시작)
        if isinstance(frame, TranscriptionFrame) and frame.is_final:
            self._llm_start_time = time.time()

            # 필러 태스크 시작
            if self._filler_task is None or self._filler_task.done():
                self._filler_task = asyncio.create_task(self._filler_loop())

        # LLM 응답 생성 완료 (필러 중지)
        elif isinstance(frame, TextFrame):
            if self._filler_task and not self._filler_task.done():
                self._filler_task.cancel()
                try:
                    await self._filler_task
                except asyncio.CancelledError:
                    pass

            self._llm_start_time = None

        await self.push_frame(frame, direction)

    async def _filler_loop(self):
        """필러 루프"""
        try:
            # 임계값까지 대기
            await asyncio.sleep(self.delay_threshold)

            # 필러 반복
            while True:
                filler_text = self.FILLERS[self._filler_index % len(self.FILLERS)]
                self._filler_index += 1

                logger.info(f"Playing filler: {filler_text}")

                # 캐시에서 오디오 가져오기
                audio_data = self.tts_cache.get(filler_text)

                if audio_data:
                    # 오디오 프레임 생성
                    audio_frame = AudioRawFrame(
                        audio=audio_data,
                        sample_rate=16000,
                        num_channels=1
                    )

                    await self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)
                else:
                    logger.warning(f"Filler not in cache: {filler_text}")

                # 다음 필러까지 대기
                await asyncio.sleep(self.filler_interval)

        except asyncio.CancelledError:
            logger.debug("Filler loop cancelled")
            raise


# ============================================================================
# 5. InterruptionHandler
# ============================================================================

class InterruptionHandler(FrameProcessor):
    """
    인터럽션 핸들러 프로세서

    사용자가 아바타 말하는 중에 끼어들 때 처리
    """

    def __init__(self, interruption_threshold: float = 0.5):
        """
        Args:
            interruption_threshold: 인터럽션 감지 임계값 (초)
        """
        super().__init__()

        self.interruption_threshold = interruption_threshold

        # 상태
        self._is_avatar_speaking = False
        self._last_avatar_frame_time = None

        # 통계
        self.interruption_count = 0

        logger.info(
            f"InterruptionHandler initialized: "
            f"threshold={interruption_threshold}s"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """프레임 처리"""
        await super().process_frame(frame, direction)

        current_time = time.time()

        # 아바타 말하기 시작 (TTS 오디오)
        if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            self._is_avatar_speaking = True
            self._last_avatar_frame_time = current_time

            await self.push_frame(frame, direction)

        # 사용자 말하기 시작 (인터럽션 가능)
        elif isinstance(frame, UserStartedSpeakingFrame):
            # 아바타가 말하는 중이면 인터럽션
            if self._is_avatar_speaking:
                # 임계값 체크
                if self._last_avatar_frame_time:
                    elapsed = current_time - self._last_avatar_frame_time

                    if elapsed < self.interruption_threshold:
                        # 인터럽션 감지
                        logger.warning("User interruption detected!")

                        self.interruption_count += 1

                        # Cancel 프레임 전송 (아바타 발화 중단)
                        cancel_frame = CancelFrame()
                        await self.push_frame(cancel_frame, FrameDirection.DOWNSTREAM)

                        self._is_avatar_speaking = False

            await self.push_frame(frame, direction)

        # 아바타 말하기 종료
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_avatar_speaking = False
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    def get_statistics(self) -> Dict[str, int]:
        """통계 반환"""
        return {
            "total_interruptions": self.interruption_count
        }


# 헬퍼 함수
def create_default_processors(
    avatar,
    interviewer_agent,
    tts_cache,
    enable_latency_monitoring: bool = True,
    enable_filler: bool = True,
    enable_interruption_handling: bool = True
):
    """
    기본 프로세서 세트 생성

    Returns:
        Dict[str, FrameProcessor]: 프로세서 딕셔너리
    """
    processors = {}

    # 1. 아바타 프로세서
    processors['avatar'] = AvatarFrameProcessor(
        avatar=avatar,
        target_fps=25,
        idle_animation=True
    )

    # 2. 컨텍스트 프로세서
    processors['context'] = InterviewContextProcessor(
        interviewer_agent=interviewer_agent
    )

    # 3. 레이턴시 모니터 (옵션)
    if enable_latency_monitoring:
        processors['latency'] = LatencyMonitorProcessor(
            warning_threshold_ms=500.0
        )

    # 4. 필러 프로세서 (옵션)
    if enable_filler and tts_cache:
        processors['filler'] = FillerProcessor(
            tts_cache=tts_cache,
            delay_threshold=2.0,
            filler_interval=3.0
        )

    # 5. 인터럽션 핸들러 (옵션)
    if enable_interruption_handling:
        processors['interruption'] = InterruptionHandler(
            interruption_threshold=0.5
        )

    return processors
