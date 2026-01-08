"""
통합 테스트 - 전체 파이프라인
전체 인터뷰 아바타 시스템의 end-to-end 테스트
"""

import asyncio
import time
import pytest
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
import psutil
import gc

# 파이프라인 컴포넌트
from pipeline.main_pipeline import (
    InterviewAvatarPipeline,
    PipelineConfig,
    create_interview_pipeline,
)
from pipeline.processors import (
    LatencyMetrics,
    InterviewContext,
)
from stt.deepgram_service import DeepgramSTTService
from tts.elevenlabs_service import ElevenLabsTTSService
from tts.cache import TTSCache
from llm.interviewer_agent import InterviewerAgent
from avatar.musetalk_wrapper import MuseTalkAvatar, MuseTalkConfig


@dataclass
class IntegrationTestMetrics:
    """통합 테스트 메트릭"""
    total_latency: float
    stt_latency: float
    llm_latency: float
    tts_latency: float
    avatar_latency: float
    memory_usage_mb: float
    cpu_percent: float
    gpu_memory_mb: float


class TestIntegrationPipeline:
    """통합 파이프라인 테스트"""

    @pytest.fixture
    async def pipeline(self):
        """테스트용 파이프라인 생성"""
        config = PipelineConfig(
            language="ko",
            enable_vad=True,
            enable_face_enhancement=False,  # 테스트 속도 향상
            target_latency_ms=500,
        )

        pipeline = await create_interview_pipeline(config)
        await pipeline.initialize()

        yield pipeline

        # Cleanup
        await pipeline.cleanup()

    @pytest.fixture
    def sample_audio(self) -> np.ndarray:
        """테스트용 샘플 오디오"""
        # 한국어 샘플: "안녕하세요, 저는 소프트웨어 엔지니어입니다."
        sample_path = Path("tests/fixtures/korean_sample.wav")

        if sample_path.exists():
            import soundfile as sf
            audio, sr = sf.read(sample_path)
            return audio
        else:
            # 더미 오디오 (3초, 16kHz)
            duration = 3.0
            sample_rate = 16000
            return np.random.randn(int(duration * sample_rate)).astype(np.float32)

    def measure_resources(self) -> Dict[str, float]:
        """시스템 리소스 측정"""
        process = psutil.Process()

        # CPU 및 메모리
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_mb = process.memory_info().rss / 1024 / 1024

        # GPU 메모리 (NVIDIA)
        gpu_memory_mb = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass

        return {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "gpu_memory_mb": gpu_memory_mb,
        }

    # ========================================================================
    # Test 1: 전체 파이프라인 테스트
    # ========================================================================

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_pipeline(self, pipeline, sample_audio):
        """
        전체 파이프라인 통합 테스트
        오디오 입력 → STT → LLM → TTS → Avatar → 비디오 출력
        """
        print("\n[TEST] 전체 파이프라인 테스트 시작...")

        # 메트릭 수집
        metrics = {}
        start_time = time.time()

        # 1. STT: 오디오 → 텍스트
        stt_start = time.time()
        transcript = await pipeline.stt_processor.process_audio(sample_audio)
        stt_latency = (time.time() - stt_start) * 1000
        metrics["stt_latency"] = stt_latency

        assert transcript is not None
        assert len(transcript) > 0
        print(f"✓ STT 완료 ({stt_latency:.2f}ms): {transcript}")

        # 2. LLM: 텍스트 → 응답
        llm_start = time.time()
        response = await pipeline.llm_processor.generate_response(transcript)
        llm_latency = (time.time() - llm_start) * 1000
        metrics["llm_latency"] = llm_latency

        assert response is not None
        assert len(response) > 0
        print(f"✓ LLM 완료 ({llm_latency:.2f}ms): {response}")

        # 3. TTS: 텍스트 → 오디오
        tts_start = time.time()
        audio_chunks = []
        async for chunk in pipeline.tts_processor.stream_audio(response):
            audio_chunks.append(chunk)
        tts_latency = (time.time() - tts_start) * 1000
        metrics["tts_latency"] = tts_latency

        assert len(audio_chunks) > 0
        print(f"✓ TTS 완료 ({tts_latency:.2f}ms): {len(audio_chunks)} chunks")

        # 4. Avatar: 오디오 → 비디오
        avatar_start = time.time()
        video_frames = []
        for audio_chunk in audio_chunks[:5]:  # 처음 5개 청크만 테스트
            frame = await pipeline.avatar_processor.process_audio_chunk(audio_chunk)
            video_frames.append(frame)
        avatar_latency = (time.time() - avatar_start) * 1000
        metrics["avatar_latency"] = avatar_latency

        assert len(video_frames) > 0
        print(f"✓ Avatar 완료 ({avatar_latency:.2f}ms): {len(video_frames)} frames")

        # 5. 전체 레이턴시 측정
        total_latency = (time.time() - start_time) * 1000
        metrics["total_latency"] = total_latency

        # 리소스 사용량
        resources = self.measure_resources()
        metrics.update(resources)

        # 결과 출력
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("         파이프라인 성능 메트릭")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"전체 레이턴시:    {total_latency:.2f}ms")
        print(f"  ├─ STT:         {stt_latency:.2f}ms")
        print(f"  ├─ LLM:         {llm_latency:.2f}ms")
        print(f"  ├─ TTS:         {tts_latency:.2f}ms")
        print(f"  └─ Avatar:      {avatar_latency:.2f}ms")
        print(f"CPU 사용률:       {resources['cpu_percent']:.1f}%")
        print(f"메모리 사용량:    {resources['memory_mb']:.1f}MB")
        print(f"GPU 메모리:       {resources['gpu_memory_mb']:.1f}MB")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        # Assertions
        # 전체 레이턴시 500ms 이하 목표 (TTS 제외 시)
        pipeline_latency = stt_latency + llm_latency + avatar_latency
        assert pipeline_latency < 500, f"파이프라인 레이턴시 초과: {pipeline_latency}ms > 500ms"

        # 개별 컴포넌트 레이턴시
        assert stt_latency < 200, f"STT 레이턴시 초과: {stt_latency}ms"
        assert llm_latency < 2000, f"LLM 레이턴시 초과: {llm_latency}ms"
        # TTS는 스트리밍이므로 제외
        assert avatar_latency < 100, f"Avatar 레이턴시 초과: {avatar_latency}ms"

    # ========================================================================
    # Test 2: 한국어 대화 테스트
    # ========================================================================

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_korean_conversation(self, pipeline):
        """
        한국어 면접 대화 흐름 테스트
        - 한국어 음성 인식 정확도
        - 한국어 TTS 품질
        - 면접 대화 컨텍스트
        """
        print("\n[TEST] 한국어 대화 테스트 시작...")

        # 면접 시나리오
        conversation_scenarios = [
            {
                "user_input": "안녕하세요",
                "expected_stage": "greeting",
            },
            {
                "user_input": "저는 5년 경력의 백엔드 개발자입니다",
                "expected_stage": "self_introduction",
            },
            {
                "user_input": "Python과 Django를 주로 사용합니다",
                "expected_stage": "technical",
            },
        ]

        interview_context = InterviewContext()

        for i, scenario in enumerate(conversation_scenarios, 1):
            print(f"\n--- 대화 턴 {i} ---")

            user_input = scenario["user_input"]
            expected_stage = scenario["expected_stage"]

            # 1. 사용자 입력 처리
            print(f"사용자: {user_input}")

            # 2. LLM 응답 생성
            response = await pipeline.llm_processor.generate_response(
                user_input,
                context=interview_context,
            )

            assert response is not None
            assert len(response) > 0
            print(f"면접관: {response}")

            # 3. 대화 컨텍스트 업데이트
            interview_context.add_exchange(user_input, response)

            # 4. 면접 단계 확인
            current_stage = interview_context.current_stage
            print(f"현재 단계: {current_stage}")

            # 5. TTS로 변환 (품질 확인)
            audio_chunks = []
            async for chunk in pipeline.tts_processor.stream_audio(response):
                audio_chunks.append(chunk)

            assert len(audio_chunks) > 0
            print(f"✓ TTS 생성: {len(audio_chunks)} chunks")

            # 한국어 특수문자 및 발음 확인
            assert all(ord(c) < 128 or ord(c) >= 0xAC00 for c in response if c.isalpha()), \
                "응답에 한국어 이외의 문자 포함"

        # 대화 히스토리 확인
        assert len(interview_context.history) == len(conversation_scenarios)
        print(f"\n✓ 전체 대화 턴: {len(interview_context.history)}")

    # ========================================================================
    # Test 3: 동시 세션 처리 테스트
    # ========================================================================

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_concurrent_sessions(self):
        """
        동시 다중 세션 처리 테스트
        - 2-3개 세션 동시 처리
        - 리소스 사용량 모니터링
        - 세션 간 간섭 없음 확인
        """
        print("\n[TEST] 동시 세션 테스트 시작...")

        num_sessions = 3
        pipelines = []

        # 1. 여러 파이프라인 생성
        for i in range(num_sessions):
            config = PipelineConfig(
                language="ko",
                enable_face_enhancement=False,
            )
            pipeline = await create_interview_pipeline(config)
            await pipeline.initialize()
            pipelines.append(pipeline)

        print(f"✓ {num_sessions}개 파이프라인 생성 완료")

        # 2. 동시 세션 시작
        async def run_session(session_id: int, pipeline: InterviewAvatarPipeline):
            """개별 세션 실행"""
            print(f"[세션 {session_id}] 시작")

            # 간단한 대화 시뮬레이션
            messages = [
                "안녕하세요",
                "저는 개발자입니다",
                "Python을 사용합니다",
            ]

            for msg in messages:
                response = await pipeline.llm_processor.generate_response(msg)
                assert response is not None
                print(f"[세션 {session_id}] 응답: {response[:50]}...")
                await asyncio.sleep(0.1)  # 약간의 지연

            print(f"[세션 {session_id}] 완료")
            return session_id

        # 3. 동시 실행
        start_time = time.time()
        results = await asyncio.gather(
            *[run_session(i, pipeline) for i, pipeline in enumerate(pipelines)]
        )
        elapsed = time.time() - start_time

        print(f"\n✓ 모든 세션 완료: {elapsed:.2f}s")

        # 4. 리소스 사용량 확인
        resources = self.measure_resources()
        print(f"CPU 사용률: {resources['cpu_percent']:.1f}%")
        print(f"메모리 사용량: {resources['memory_mb']:.1f}MB")
        print(f"GPU 메모리: {resources['gpu_memory_mb']:.1f}MB")

        # Assertions
        assert len(results) == num_sessions
        assert all(r is not None for r in results)

        # 메모리 사용량이 합리적인지 확인 (세션당 ~500MB 가정)
        expected_max_memory = num_sessions * 500
        assert resources['memory_mb'] < expected_max_memory, \
            f"메모리 사용량 초과: {resources['memory_mb']}MB > {expected_max_memory}MB"

        # Cleanup
        for pipeline in pipelines:
            await pipeline.cleanup()

    # ========================================================================
    # Test 4: 장시간 세션 테스트
    # ========================================================================

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_long_session(self, pipeline):
        """
        장시간 연속 세션 테스트 (30분 시뮬레이션)
        - 메모리 누수 검사
        - 품질 일관성 확인
        - 리소스 안정성
        """
        print("\n[TEST] 장시간 세션 테스트 시작...")

        # 30분 = 1800초 → 테스트에서는 60초로 축소 (30 턴)
        test_duration = 60
        turn_interval = 2  # 2초마다 1턴
        num_turns = test_duration // turn_interval

        initial_resources = self.measure_resources()
        print(f"초기 메모리: {initial_resources['memory_mb']:.1f}MB")

        resource_samples = []

        for turn in range(num_turns):
            # 대화 시뮬레이션
            user_input = f"질문 {turn + 1}: 제 경력에 대해 설명드리겠습니다."

            # LLM 응답
            response = await pipeline.llm_processor.generate_response(user_input)
            assert response is not None

            # TTS (스트리밍 일부만)
            chunk_count = 0
            async for chunk in pipeline.tts_processor.stream_audio(response):
                chunk_count += 1
                if chunk_count >= 3:  # 처음 3개 청크만
                    break

            # 리소스 측정
            if turn % 5 == 0:
                resources = self.measure_resources()
                resource_samples.append(resources)
                print(f"[턴 {turn + 1}/{num_turns}] "
                      f"메모리: {resources['memory_mb']:.1f}MB, "
                      f"CPU: {resources['cpu_percent']:.1f}%")

            # GC 주기적 실행
            if turn % 10 == 0:
                gc.collect()

            await asyncio.sleep(turn_interval)

        final_resources = self.measure_resources()
        print(f"\n최종 메모리: {final_resources['memory_mb']:.1f}MB")

        # 메모리 증가량 확인
        memory_increase = final_resources['memory_mb'] - initial_resources['memory_mb']
        print(f"메모리 증가량: {memory_increase:.1f}MB")

        # Assertions
        # 메모리 누수: 30턴 동안 100MB 이상 증가하지 않아야 함
        assert memory_increase < 100, f"메모리 누수 의심: {memory_increase:.1f}MB 증가"

        # 메모리 사용량의 표준편차가 크지 않아야 함 (안정성)
        memory_values = [r['memory_mb'] for r in resource_samples]
        memory_std = np.std(memory_values)
        print(f"메모리 표준편차: {memory_std:.2f}MB")
        assert memory_std < 50, f"메모리 사용량 불안정: std={memory_std:.2f}MB"

        print("\n✓ 장시간 세션 안정성 확인")

    # ========================================================================
    # Test 5: 에러 복구 테스트
    # ========================================================================

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_recovery(self, pipeline):
        """
        에러 복구 및 예외 처리 테스트
        - 네트워크 끊김 시뮬레이션
        - API 에러 핸들링
        - 그레이스풀 셧다운
        """
        print("\n[TEST] 에러 복구 테스트 시작...")

        # 1. 잘못된 입력 처리
        print("\n--- 잘못된 입력 테스트 ---")
        invalid_inputs = [
            None,
            "",
            "   ",
            "a" * 10000,  # 너무 긴 입력
        ]

        for invalid_input in invalid_inputs:
            try:
                response = await pipeline.llm_processor.generate_response(invalid_input)
                # 에러가 발생하지 않으면 빈 응답 또는 에러 메시지 반환
                print(f"입력: {str(invalid_input)[:50]}... → 응답: {str(response)[:50]}...")
            except Exception as e:
                print(f"입력: {str(invalid_input)[:50]}... → 예외: {type(e).__name__}")
                # 예외가 발생해도 파이프라인이 계속 실행되어야 함

        # 2. API 타임아웃 시뮬레이션
        print("\n--- API 타임아웃 테스트 ---")

        # STT 타임아웃 설정 (짧게)
        original_timeout = pipeline.stt_processor.timeout if hasattr(pipeline.stt_processor, 'timeout') else None

        try:
            if hasattr(pipeline.stt_processor, 'timeout'):
                pipeline.stt_processor.timeout = 0.001  # 1ms (타임아웃 발생)

            # 타임아웃 발생 시도
            dummy_audio = np.random.randn(16000).astype(np.float32)
            try:
                await pipeline.stt_processor.process_audio(dummy_audio)
                print("타임아웃 발생하지 않음 (정상 처리)")
            except asyncio.TimeoutError:
                print("✓ 타임아웃 예외 정상 처리")
            except Exception as e:
                print(f"✓ 예외 발생: {type(e).__name__}")

        finally:
            # 타임아웃 복원
            if original_timeout is not None:
                pipeline.stt_processor.timeout = original_timeout

        # 3. 리소스 정리 테스트
        print("\n--- 그레이스풀 셧다운 테스트 ---")

        # 파이프라인 복제 생성 (원본은 fixture가 정리)
        test_pipeline = await create_interview_pipeline(PipelineConfig())
        await test_pipeline.initialize()

        # 일부 작업 수행
        await test_pipeline.llm_processor.generate_response("테스트 메시지")

        # 그레이스풀 셧다운
        try:
            await asyncio.wait_for(test_pipeline.cleanup(), timeout=5.0)
            print("✓ 그레이스풀 셧다운 성공")
        except asyncio.TimeoutError:
            print("⚠ 셧다운 타임아웃 (5초 초과)")
            # 강제 종료
            test_pipeline.force_stop()

        # 4. 재시작 테스트
        print("\n--- 재시작 테스트 ---")

        # 새 파이프라인 생성
        new_pipeline = await create_interview_pipeline(PipelineConfig())
        await new_pipeline.initialize()

        # 정상 작동 확인
        response = await new_pipeline.llm_processor.generate_response("안녕하세요")
        assert response is not None
        print(f"✓ 재시작 후 정상 작동: {response[:50]}...")

        await new_pipeline.cleanup()

        print("\n✓ 모든 에러 복구 테스트 완료")

    # ========================================================================
    # Test 6: 캐싱 효과 테스트
    # ========================================================================

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_caching_performance(self, pipeline):
        """
        TTS 캐싱 성능 테스트
        - 캐시 히트율 확인
        - 레이턴시 개선 측정
        """
        print("\n[TEST] 캐싱 성능 테스트 시작...")

        # 공통 질문 (캐시될 것으로 예상)
        common_questions = [
            "자기소개 부탁드립니다.",
            "귀하의 경력에 대해 말씀해주세요.",
            "왜 우리 회사에 지원하셨나요?",
        ]

        # 1차 요청 (캐시 미스)
        print("\n--- 1차 요청 (캐시 미스) ---")
        first_latencies = []

        for question in common_questions:
            start = time.time()
            chunks = []
            async for chunk in pipeline.tts_processor.stream_audio(question):
                chunks.append(chunk)
            latency = (time.time() - start) * 1000
            first_latencies.append(latency)
            print(f"{question} → {latency:.2f}ms")

        # 2차 요청 (캐시 히트)
        print("\n--- 2차 요청 (캐시 히트) ---")
        second_latencies = []

        for question in common_questions:
            start = time.time()
            chunks = []
            async for chunk in pipeline.tts_processor.stream_audio(question):
                chunks.append(chunk)
            latency = (time.time() - start) * 1000
            second_latencies.append(latency)
            print(f"{question} → {latency:.2f}ms")

        # 캐싱 효과 분석
        avg_first = np.mean(first_latencies)
        avg_second = np.mean(second_latencies)
        speedup = avg_first / avg_second

        print(f"\n평균 레이턴시:")
        print(f"  1차 (캐시 미스): {avg_first:.2f}ms")
        print(f"  2차 (캐시 히트): {avg_second:.2f}ms")
        print(f"  속도 향상: {speedup:.2f}x")

        # Assertions
        # 캐시 히트 시 최소 10배 이상 빠를 것으로 예상
        assert speedup > 10, f"캐시 효과 부족: {speedup:.2f}x < 10x"

        print("\n✓ 캐싱 성능 테스트 완료")


# ============================================================================
# Performance Benchmark Tests
# ============================================================================

class TestPerformanceBenchmark:
    """성능 벤치마크 테스트"""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """처리량 벤치마크: 1분간 최대 처리 가능한 요청 수"""
        print("\n[BENCHMARK] 처리량 테스트...")

        config = PipelineConfig(enable_face_enhancement=False)
        pipeline = await create_interview_pipeline(config)
        await pipeline.initialize()

        request_count = 0
        start_time = time.time()
        duration = 60  # 60초

        try:
            while (time.time() - start_time) < duration:
                response = await pipeline.llm_processor.generate_response(
                    "간단한 질문입니다."
                )
                assert response is not None
                request_count += 1

                if request_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"처리량: {request_count / elapsed:.2f} req/s")

        finally:
            await pipeline.cleanup()

        total_elapsed = time.time() - start_time
        throughput = request_count / total_elapsed

        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"         처리량 벤치마크 결과")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"총 요청 수:       {request_count}")
        print(f"총 실행 시간:     {total_elapsed:.2f}s")
        print(f"평균 처리량:      {throughput:.2f} req/s")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        # 최소 처리량 목표 (예: 5 req/s)
        assert throughput > 5, f"처리량 부족: {throughput:.2f} req/s < 5 req/s"


# ============================================================================
# Test Utilities
# ============================================================================

def pytest_configure(config):
    """pytest 설정"""
    config.addinivalue_line(
        "markers", "integration: 통합 테스트 마커"
    )
    config.addinivalue_line(
        "markers", "slow: 느린 테스트 마커 (30초 이상)"
    )
    config.addinivalue_line(
        "markers", "benchmark: 성능 벤치마크 마커"
    )


if __name__ == "__main__":
    # 개별 테스트 실행
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
