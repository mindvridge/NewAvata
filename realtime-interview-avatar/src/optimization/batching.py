"""
배치 처리 최적화 모듈

오디오 청크 배치 처리 및 프레임 배치 렌더링
"""

import asyncio
import time
from typing import List, Optional, Callable, TypeVar, Generic, Any
from dataclasses import dataclass, field
from collections import deque
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# Batch Configuration
# ============================================================================

@dataclass
class BatchConfig:
    """배치 처리 설정"""
    batch_size: int = 4
    max_wait_time_ms: int = 50  # 최대 대기 시간 (ms)
    enable_dynamic_batching: bool = True  # 동적 배치 크기 조정


# ============================================================================
# Generic Batch Processor
# ============================================================================

class BatchProcessor(Generic[T, R]):
    """
    범용 배치 프로세서

    여러 요청을 모아서 한 번에 처리
    """

    def __init__(
        self,
        process_fn: Callable[[List[T]], List[R]],
        config: BatchConfig = None,
    ):
        """
        Args:
            process_fn: 배치 처리 함수 (List[T] → List[R])
            config: 배치 설정
        """
        if config is None:
            config = BatchConfig()

        self.process_fn = process_fn
        self.config = config

        self.queue: deque[tuple[T, asyncio.Future]] = deque()
        self.processing = False
        self.total_processed = 0
        self.total_batches = 0

    async def process(self, item: T) -> R:
        """
        단일 아이템 처리

        내부적으로 배치에 추가하고 결과를 대기
        """
        # Future 생성
        future = asyncio.get_event_loop().create_future()

        # 큐에 추가
        self.queue.append((item, future))

        # 배치 처리 트리거
        if not self.processing:
            asyncio.create_task(self._process_batch())

        # 결과 대기
        result = await future
        return result

    async def _process_batch(self):
        """배치 처리 (내부)"""
        if self.processing:
            return

        self.processing = True

        try:
            while self.queue:
                # 배치 수집
                batch_items = []
                batch_futures = []

                # 배치 크기만큼 또는 max_wait_time 동안 수집
                start_time = time.time()

                while len(batch_items) < self.config.batch_size:
                    # 큐에서 아이템 가져오기
                    if not self.queue:
                        # 대기 시간 초과 확인
                        elapsed_ms = (time.time() - start_time) * 1000
                        if elapsed_ms > self.config.max_wait_time_ms:
                            break

                        # 짧은 대기
                        await asyncio.sleep(0.001)
                        continue

                    item, future = self.queue.popleft()
                    batch_items.append(item)
                    batch_futures.append(future)

                if not batch_items:
                    break

                # 배치 처리
                try:
                    results = self.process_fn(batch_items)

                    # 결과 분배
                    for future, result in zip(batch_futures, results):
                        if not future.done():
                            future.set_result(result)

                    self.total_processed += len(batch_items)
                    self.total_batches += 1

                    logger.debug(
                        f"배치 처리: {len(batch_items)}개 아이템 "
                        f"(총 {self.total_processed}개, {self.total_batches}배치)"
                    )

                except Exception as e:
                    logger.error(f"배치 처리 실패: {e}")

                    # 모든 Future에 예외 전달
                    for future in batch_futures:
                        if not future.done():
                            future.set_exception(e)

        finally:
            self.processing = False

    def stats(self) -> dict:
        """통계"""
        avg_batch_size = (
            self.total_processed / self.total_batches
            if self.total_batches > 0 else 0
        )

        return {
            "total_processed": self.total_processed,
            "total_batches": self.total_batches,
            "avg_batch_size": avg_batch_size,
            "queue_size": len(self.queue),
        }


# ============================================================================
# Audio Chunk Batch Processor
# ============================================================================

class AudioChunkBatcher:
    """
    오디오 청크 배치 처리

    여러 오디오 청크를 모아서 한 번에 처리 (STT, 특징 추출 등)
    """

    def __init__(
        self,
        process_fn: Callable[[np.ndarray], Any],
        config: BatchConfig = None,
    ):
        """
        Args:
            process_fn: 배치 오디오 처리 함수
                입력: (batch_size, audio_length) np.ndarray
                출력: List[결과]
        """
        if config is None:
            config = BatchConfig(batch_size=4, max_wait_time_ms=50)

        self.config = config
        self.process_fn = process_fn

        self.queue: deque[tuple[np.ndarray, asyncio.Future]] = deque()
        self.processing = False

    async def process_chunk(self, audio_chunk: np.ndarray) -> Any:
        """
        단일 오디오 청크 처리

        Args:
            audio_chunk: (audio_length,) 오디오 데이터

        Returns:
            처리 결과
        """
        future = asyncio.get_event_loop().create_future()
        self.queue.append((audio_chunk, future))

        if not self.processing:
            asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """배치 처리"""
        if self.processing:
            return

        self.processing = True

        try:
            while self.queue:
                # 배치 수집
                batch_chunks = []
                batch_futures = []

                start_time = time.time()

                while len(batch_chunks) < self.config.batch_size:
                    if not self.queue:
                        elapsed_ms = (time.time() - start_time) * 1000
                        if elapsed_ms > self.config.max_wait_time_ms:
                            break
                        await asyncio.sleep(0.001)
                        continue

                    chunk, future = self.queue.popleft()
                    batch_chunks.append(chunk)
                    batch_futures.append(future)

                if not batch_chunks:
                    break

                # 배치 텐서 생성 (패딩 필요 시)
                try:
                    # 모든 청크를 같은 길이로 패딩
                    max_length = max(chunk.shape[0] for chunk in batch_chunks)
                    padded_chunks = []

                    for chunk in batch_chunks:
                        if chunk.shape[0] < max_length:
                            # Zero padding
                            padded = np.pad(
                                chunk,
                                (0, max_length - chunk.shape[0]),
                                mode='constant'
                            )
                            padded_chunks.append(padded)
                        else:
                            padded_chunks.append(chunk)

                    # (batch_size, audio_length)
                    batch_array = np.stack(padded_chunks, axis=0)

                    # 배치 처리
                    results = self.process_fn(batch_array)

                    # 결과 분배
                    for future, result in zip(batch_futures, results):
                        if not future.done():
                            future.set_result(result)

                    logger.debug(f"오디오 배치 처리: {len(batch_chunks)}개 청크")

                except Exception as e:
                    logger.error(f"오디오 배치 처리 실패: {e}")
                    for future in batch_futures:
                        if not future.done():
                            future.set_exception(e)

        finally:
            self.processing = False


# ============================================================================
# Frame Batch Renderer
# ============================================================================

class FrameBatchRenderer:
    """
    프레임 배치 렌더링

    여러 프레임을 GPU에서 한 번에 렌더링 (MuseTalk 등)
    """

    def __init__(
        self,
        render_fn: Callable[[torch.Tensor], torch.Tensor],
        config: BatchConfig = None,
    ):
        """
        Args:
            render_fn: 배치 렌더링 함수
                입력: (batch_size, channels, height, width) Tensor
                출력: (batch_size, channels, height, width) Tensor
        """
        if config is None:
            config = BatchConfig(batch_size=8, max_wait_time_ms=40)

        self.config = config
        self.render_fn = render_fn

        self.queue: deque[tuple[torch.Tensor, asyncio.Future]] = deque()
        self.processing = False

    async def render_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        단일 프레임 렌더링

        Args:
            frame: (channels, height, width) 프레임

        Returns:
            렌더링된 프레임
        """
        future = asyncio.get_event_loop().create_future()
        self.queue.append((frame, future))

        if not self.processing:
            asyncio.create_task(self._render_batch())

        return await future

    async def _render_batch(self):
        """배치 렌더링"""
        if self.processing:
            return

        self.processing = True

        try:
            while self.queue:
                # 배치 수집
                batch_frames = []
                batch_futures = []

                start_time = time.time()

                while len(batch_frames) < self.config.batch_size:
                    if not self.queue:
                        elapsed_ms = (time.time() - start_time) * 1000
                        if elapsed_ms > self.config.max_wait_time_ms:
                            break
                        await asyncio.sleep(0.001)
                        continue

                    frame, future = self.queue.popleft()
                    batch_frames.append(frame)
                    batch_futures.append(future)

                if not batch_frames:
                    break

                try:
                    # 배치 텐서 생성
                    batch_tensor = torch.stack(batch_frames, dim=0)

                    # GPU로 이동
                    if torch.cuda.is_available():
                        batch_tensor = batch_tensor.cuda()

                    # 배치 렌더링
                    with torch.no_grad():
                        rendered_batch = self.render_fn(batch_tensor)

                    # CPU로 이동 및 분할
                    if torch.cuda.is_available():
                        rendered_batch = rendered_batch.cpu()

                    rendered_frames = torch.unbind(rendered_batch, dim=0)

                    # 결과 분배
                    for future, frame in zip(batch_futures, rendered_frames):
                        if not future.done():
                            future.set_result(frame)

                    logger.debug(f"프레임 배치 렌더링: {len(batch_frames)}개 프레임")

                except Exception as e:
                    logger.error(f"프레임 배치 렌더링 실패: {e}")
                    for future in batch_futures:
                        if not future.done():
                            future.set_exception(e)

        finally:
            self.processing = False


# ============================================================================
# Dynamic Batch Sizer
# ============================================================================

class DynamicBatchSizer:
    """
    동적 배치 크기 조정

    시스템 부하에 따라 배치 크기를 자동으로 조정
    """

    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 16,
        target_latency_ms: float = 50.0,
    ):
        """
        Args:
            initial_batch_size: 초기 배치 크기
            min_batch_size: 최소 배치 크기
            max_batch_size: 최대 배치 크기
            target_latency_ms: 목표 레이턴시 (ms)
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms

        self.recent_latencies: deque[float] = deque(maxlen=10)

    def update(self, latency_ms: float):
        """
        레이턴시 기반 배치 크기 조정

        Args:
            latency_ms: 최근 배치 처리 레이턴시
        """
        self.recent_latencies.append(latency_ms)

        if len(self.recent_latencies) < 5:
            return  # 충분한 샘플이 모일 때까지 대기

        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)

        # 레이턴시가 목표보다 높으면 배치 크기 감소
        if avg_latency > self.target_latency_ms * 1.2:
            new_size = max(self.min_batch_size, self.current_batch_size - 1)
            if new_size != self.current_batch_size:
                logger.info(
                    f"배치 크기 감소: {self.current_batch_size} → {new_size} "
                    f"(레이턴시: {avg_latency:.2f}ms)"
                )
                self.current_batch_size = new_size

        # 레이턴시가 목표보다 낮으면 배치 크기 증가
        elif avg_latency < self.target_latency_ms * 0.8:
            new_size = min(self.max_batch_size, self.current_batch_size + 1)
            if new_size != self.current_batch_size:
                logger.info(
                    f"배치 크기 증가: {self.current_batch_size} → {new_size} "
                    f"(레이턴시: {avg_latency:.2f}ms)"
                )
                self.current_batch_size = new_size

    def get_batch_size(self) -> int:
        """현재 배치 크기 반환"""
        return self.current_batch_size


# ============================================================================
# Adaptive Batch Processor
# ============================================================================

class AdaptiveBatchProcessor(Generic[T, R]):
    """
    적응형 배치 프로세서

    동적 배치 크기 조정을 포함한 배치 프로세서
    """

    def __init__(
        self,
        process_fn: Callable[[List[T]], List[R]],
        config: BatchConfig = None,
    ):
        if config is None:
            config = BatchConfig()

        self.config = config
        self.process_fn = process_fn

        # 동적 배치 크기 조정
        self.batch_sizer = DynamicBatchSizer(
            initial_batch_size=config.batch_size,
            target_latency_ms=config.max_wait_time_ms,
        )

        self.batch_processor = BatchProcessor(
            process_fn=self._process_with_timing,
            config=config,
        )

    def _process_with_timing(self, items: List[T]) -> List[R]:
        """타이밍 측정을 포함한 배치 처리"""
        start = time.time()

        results = self.process_fn(items)

        latency_ms = (time.time() - start) * 1000

        # 배치 크기 조정
        if self.config.enable_dynamic_batching:
            self.batch_sizer.update(latency_ms)
            # 배치 설정 업데이트
            self.config.batch_size = self.batch_sizer.get_batch_size()

        return results

    async def process(self, item: T) -> R:
        """단일 아이템 처리"""
        return await self.batch_processor.process(item)

    def stats(self) -> dict:
        """통계"""
        base_stats = self.batch_processor.stats()
        base_stats["current_batch_size"] = self.batch_sizer.get_batch_size()
        return base_stats


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    # 예제: 오디오 청크 배치 처리
    def process_audio_batch(batch: np.ndarray) -> List[str]:
        """더미 오디오 처리 함수"""
        time.sleep(0.1)  # 처리 시뮬레이션
        return [f"result_{i}" for i in range(batch.shape[0])]

    async def test_audio_batching():
        batcher = AudioChunkBatcher(
            process_fn=process_audio_batch,
            config=BatchConfig(batch_size=4, max_wait_time_ms=50),
        )

        # 여러 청크 동시 처리
        chunks = [np.random.randn(1600) for _ in range(10)]

        tasks = [batcher.process_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        print(f"처리 결과: {results}")

    asyncio.run(test_audio_batching())
