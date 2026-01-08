"""
비동기 처리 최적화 모듈

비동기 처리 최적화 및 파이프라인 병렬화
"""

import asyncio
from typing import List, Optional, Callable, TypeVar, Coroutine, Any
from dataclasses import dataclass
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# Async Pipeline
# ============================================================================

@dataclass
class PipelineStage:
    """파이프라인 단계"""
    name: str
    process_fn: Callable[[Any], Coroutine[Any, Any, Any]]
    max_concurrency: int = 1


class AsyncPipeline:
    """
    비동기 파이프라인

    여러 단계를 병렬로 처리하는 파이프라인
    """

    def __init__(self, stages: List[PipelineStage]):
        """
        Args:
            stages: 파이프라인 단계 리스트 (순서대로)
        """
        self.stages = stages
        self.stage_queues: List[asyncio.Queue] = [
            asyncio.Queue() for _ in range(len(stages) + 1)
        ]
        self.workers: List[asyncio.Task] = []
        self.running = False

    async def start(self):
        """파이프라인 시작"""
        if self.running:
            return

        self.running = True

        # 각 단계별 워커 생성
        for i, stage in enumerate(self.stages):
            for _ in range(stage.max_concurrency):
                worker = asyncio.create_task(
                    self._stage_worker(i, stage)
                )
                self.workers.append(worker)

        logger.info(f"파이프라인 시작: {len(self.stages)}개 단계, {len(self.workers)}개 워커")

    async def stop(self):
        """파이프라인 중지"""
        if not self.running:
            return

        self.running = False

        # 모든 큐에 종료 신호
        for queue in self.stage_queues:
            await queue.put(None)

        # 워커 종료 대기
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("파이프라인 중지")

    async def _stage_worker(self, stage_idx: int, stage: PipelineStage):
        """단계별 워커"""
        input_queue = self.stage_queues[stage_idx]
        output_queue = self.stage_queues[stage_idx + 1]

        logger.debug(f"워커 시작: {stage.name}")

        while self.running:
            try:
                # 입력 큐에서 아이템 가져오기
                item = await input_queue.get()

                if item is None:
                    # 종료 신호
                    break

                input_data, result_future = item

                # 처리
                try:
                    output_data = await stage.process_fn(input_data)

                    # 마지막 단계면 결과 반환
                    if stage_idx == len(self.stages) - 1:
                        if not result_future.done():
                            result_future.set_result(output_data)
                    else:
                        # 다음 단계로 전달
                        await output_queue.put((output_data, result_future))

                except Exception as e:
                    logger.error(f"{stage.name} 처리 실패: {e}")
                    if not result_future.done():
                        result_future.set_exception(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"워커 오류 ({stage.name}): {e}")

        logger.debug(f"워커 종료: {stage.name}")

    async def process(self, input_data: Any) -> Any:
        """
        파이프라인 처리

        Args:
            input_data: 입력 데이터

        Returns:
            최종 출력 데이터
        """
        if not self.running:
            raise RuntimeError("파이프라인이 시작되지 않았습니다.")

        # 결과 Future 생성
        result_future = asyncio.get_event_loop().create_future()

        # 첫 번째 큐에 추가
        await self.stage_queues[0].put((input_data, result_future))

        # 결과 대기
        return await result_future


# ============================================================================
# Concurrent Executor
# ============================================================================

class ConcurrentExecutor:
    """
    동시 실행 관리자

    여러 비동기 작업을 동시에 실행하고 결과를 수집
    """

    def __init__(self, max_concurrency: int = 10):
        """
        Args:
            max_concurrency: 최대 동시 실행 수
        """
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(
        self,
        tasks: List[Callable[[], Coroutine]],
    ) -> List[Any]:
        """
        여러 작업을 동시에 실행

        Args:
            tasks: 비동기 함수 리스트

        Returns:
            결과 리스트
        """
        async def run_with_semaphore(task):
            async with self.semaphore:
                return await task()

        results = await asyncio.gather(
            *[run_with_semaphore(task) for task in tasks],
            return_exceptions=False,
        )

        return results


# ============================================================================
# Async Pool
# ============================================================================

class AsyncPool:
    """
    비동기 워커 풀

    고정된 수의 워커로 작업을 처리
    """

    def __init__(
        self,
        worker_fn: Callable[[Any], Coroutine],
        num_workers: int = 4,
    ):
        """
        Args:
            worker_fn: 워커 함수
            num_workers: 워커 수
        """
        self.worker_fn = worker_fn
        self.num_workers = num_workers

        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.running = False

    async def start(self):
        """풀 시작"""
        if self.running:
            return

        self.running = True

        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

        logger.info(f"워커 풀 시작: {self.num_workers}개 워커")

    async def stop(self):
        """풀 중지"""
        if not self.running:
            return

        self.running = False

        # 종료 신호
        for _ in range(self.num_workers):
            await self.task_queue.put(None)

        # 워커 종료 대기
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("워커 풀 중지")

    async def _worker(self, worker_id: int):
        """워커"""
        logger.debug(f"워커 {worker_id} 시작")

        while self.running:
            try:
                task = await self.task_queue.get()

                if task is None:
                    break

                input_data, result_future = task

                try:
                    result = await self.worker_fn(input_data)
                    if not result_future.done():
                        result_future.set_result(result)
                except Exception as e:
                    logger.error(f"워커 {worker_id} 처리 실패: {e}")
                    if not result_future.done():
                        result_future.set_exception(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"워커 {worker_id} 오류: {e}")

        logger.debug(f"워커 {worker_id} 종료")

    async def submit(self, input_data: Any) -> Any:
        """
        작업 제출

        Args:
            input_data: 입력 데이터

        Returns:
            결과
        """
        if not self.running:
            raise RuntimeError("워커 풀이 시작되지 않았습니다.")

        result_future = asyncio.get_event_loop().create_future()
        await self.task_queue.put((input_data, result_future))

        return await result_future


# ============================================================================
# Stream Processor
# ============================================================================

class StreamProcessor:
    """
    스트림 프로세서

    비동기 스트림을 효율적으로 처리
    """

    @staticmethod
    async def map(
        stream: asyncio.Queue,
        fn: Callable[[Any], Coroutine],
        max_concurrency: int = 10,
    ) -> asyncio.Queue:
        """
        스트림 맵

        각 아이템에 함수를 적용
        """
        output_queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_item(item):
            async with semaphore:
                result = await fn(item)
                await output_queue.put(result)

        async def process_stream():
            tasks = []
            while True:
                item = await stream.get()
                if item is None:
                    break

                task = asyncio.create_task(process_item(item))
                tasks.append(task)

            await asyncio.gather(*tasks)
            await output_queue.put(None)  # 종료 신호

        asyncio.create_task(process_stream())
        return output_queue

    @staticmethod
    async def filter(
        stream: asyncio.Queue,
        predicate: Callable[[Any], Coroutine],
    ) -> asyncio.Queue:
        """
        스트림 필터

        조건을 만족하는 아이템만 통과
        """
        output_queue = asyncio.Queue()

        async def process_stream():
            while True:
                item = await stream.get()
                if item is None:
                    await output_queue.put(None)
                    break

                if await predicate(item):
                    await output_queue.put(item)

        asyncio.create_task(process_stream())
        return output_queue

    @staticmethod
    async def buffer(
        stream: asyncio.Queue,
        size: int,
    ) -> asyncio.Queue:
        """
        스트림 버퍼링

        여러 아이템을 모아서 배치로 전달
        """
        output_queue = asyncio.Queue()

        async def process_stream():
            buffer = []

            while True:
                try:
                    item = await asyncio.wait_for(stream.get(), timeout=0.1)

                    if item is None:
                        if buffer:
                            await output_queue.put(buffer)
                        await output_queue.put(None)
                        break

                    buffer.append(item)

                    if len(buffer) >= size:
                        await output_queue.put(buffer)
                        buffer = []

                except asyncio.TimeoutError:
                    if buffer:
                        await output_queue.put(buffer)
                        buffer = []

        asyncio.create_task(process_stream())
        return output_queue


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """
    속도 제한기

    초당 요청 수를 제한
    """

    def __init__(self, max_rate: float):
        """
        Args:
            max_rate: 초당 최대 요청 수
        """
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate
        self.last_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """속도 제한 획득"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_time

            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                await asyncio.sleep(wait_time)

            self.last_time = time.time()


# ============================================================================
# Interview Avatar Pipeline (Example)
# ============================================================================

class InterviewAvatarAsyncPipeline:
    """
    면접 아바타 비동기 파이프라인

    STT → LLM → TTS → Avatar 단계를 병렬로 처리
    """

    def __init__(self):
        # 각 단계 정의
        self.stt_stage = PipelineStage(
            name="STT",
            process_fn=self._stt_process,
            max_concurrency=2,
        )

        self.llm_stage = PipelineStage(
            name="LLM",
            process_fn=self._llm_process,
            max_concurrency=1,
        )

        self.tts_stage = PipelineStage(
            name="TTS",
            process_fn=self._tts_process,
            max_concurrency=2,
        )

        self.avatar_stage = PipelineStage(
            name="Avatar",
            process_fn=self._avatar_process,
            max_concurrency=4,
        )

        # 파이프라인 생성
        self.pipeline = AsyncPipeline([
            self.stt_stage,
            self.llm_stage,
            self.tts_stage,
            self.avatar_stage,
        ])

    async def start(self):
        """파이프라인 시작"""
        await self.pipeline.start()

    async def stop(self):
        """파이프라인 중지"""
        await self.pipeline.stop()

    async def _stt_process(self, audio_data):
        """STT 처리 (더미)"""
        await asyncio.sleep(0.1)  # STT 시뮬레이션
        return {"transcript": "Hello"}

    async def _llm_process(self, stt_result):
        """LLM 처리 (더미)"""
        await asyncio.sleep(0.2)  # LLM 시뮬레이션
        return {"response": "Hi, how can I help you?"}

    async def _tts_process(self, llm_result):
        """TTS 처리 (더미)"""
        await asyncio.sleep(0.15)  # TTS 시뮬레이션
        return {"audio": b"audio_data"}

    async def _avatar_process(self, tts_result):
        """Avatar 처리 (더미)"""
        await asyncio.sleep(0.05)  # Avatar 시뮬레이션
        return {"video_frame": "frame_data"}

    async def process_audio(self, audio_data):
        """오디오 처리"""
        return await self.pipeline.process(audio_data)


# ============================================================================
# Parallel Task Runner
# ============================================================================

async def run_parallel(
    tasks: List[Coroutine],
    max_concurrency: int = 10,
) -> List[Any]:
    """
    여러 작업을 병렬로 실행

    Args:
        tasks: 코루틴 리스트
        max_concurrency: 최대 동시 실행 수

    Returns:
        결과 리스트
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_semaphore(task):
        async with semaphore:
            return await task

    return await asyncio.gather(
        *[run_with_semaphore(task) for task in tasks],
        return_exceptions=False,
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        # 파이프라인 테스트
        pipeline = InterviewAvatarAsyncPipeline()
        await pipeline.start()

        # 여러 오디오 동시 처리
        audio_data_list = [f"audio_{i}" for i in range(5)]

        tasks = [
            pipeline.process_audio(audio_data)
            for audio_data in audio_data_list
        ]

        results = await asyncio.gather(*tasks)

        print(f"처리 결과: {results}")

        await pipeline.stop()

    asyncio.run(main())
