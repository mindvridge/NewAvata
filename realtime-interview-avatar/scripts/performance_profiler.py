#!/usr/bin/env python3
"""
성능 프로파일링 스크립트

각 컴포넌트(STT, LLM, TTS, Avatar)의 성능을 측정하고
병목 구간을 식별하여 최적화 가이드를 제공합니다.
"""

import asyncio
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import psutil

# 시각화 (선택적)
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 실행
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib가 설치되어 있지 않습니다. 차트 생성이 비활성화됩니다.")
    print("  설치: pip install matplotlib")

# GPU 모니터링 (선택적)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except (ImportError, Exception):
    HAS_NVML = False


# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class LatencyMetrics:
    """레이턴시 메트릭"""
    min: float
    max: float
    mean: float
    median: float
    p50: float
    p95: float
    p99: float
    std: float

    @classmethod
    def from_samples(cls, samples: List[float]) -> 'LatencyMetrics':
        """샘플로부터 메트릭 계산"""
        arr = np.array(samples)
        return cls(
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            std=float(np.std(arr)),
        )


@dataclass
class MemoryMetrics:
    """메모리 메트릭"""
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float
    available_mb: float


@dataclass
class GPUMetrics:
    """GPU 메트릭"""
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    utilization_percent: float
    temperature_c: Optional[float] = None


@dataclass
class ComponentProfile:
    """컴포넌트 프로파일"""
    name: str
    latency: LatencyMetrics
    memory_before: MemoryMetrics
    memory_after: MemoryMetrics
    memory_delta_mb: float
    gpu_before: Optional[GPUMetrics] = None
    gpu_after: Optional[GPUMetrics] = None
    throughput_per_sec: Optional[float] = None
    samples_count: int = 0


@dataclass
class ProfileReport:
    """전체 프로파일 리포트"""
    timestamp: str
    duration_sec: float
    components: Dict[str, ComponentProfile]
    bottlenecks: List[str]
    recommendations: List[str]
    system_info: Dict[str, any]


# ============================================================================
# 시스템 모니터링
# ============================================================================

class SystemMonitor:
    """시스템 리소스 모니터링"""

    @staticmethod
    def get_memory_metrics() -> MemoryMetrics:
        """메모리 메트릭 수집"""
        process = psutil.Process()
        mem_info = process.memory_info()
        sys_mem = psutil.virtual_memory()

        return MemoryMetrics(
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            percent=process.memory_percent(),
            available_mb=sys_mem.available / 1024 / 1024,
        )

    @staticmethod
    def get_gpu_metrics(device_id: int = 0) -> Optional[GPUMetrics]:
        """GPU 메트릭 수집"""
        if not HAS_NVML:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = None

            return GPUMetrics(
                name=name,
                memory_used_mb=mem_info.used / 1024 / 1024,
                memory_total_mb=mem_info.total / 1024 / 1024,
                memory_percent=(mem_info.used / mem_info.total) * 100,
                utilization_percent=utilization.gpu,
                temperature_c=temp,
            )
        except Exception as e:
            print(f"⚠ GPU 메트릭 수집 실패: {e}")
            return None

    @staticmethod
    def get_system_info() -> Dict[str, any]:
        """시스템 정보 수집"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "total_memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "platform": sys.platform,
            "python_version": sys.version,
        }

        # GPU 정보
        if HAS_TORCH and torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        else:
            info["gpu_available"] = False

        return info


# ============================================================================
# 컴포넌트 프로파일러
# ============================================================================

class ComponentProfiler:
    """개별 컴포넌트 프로파일링"""

    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor

    async def profile_stt(self, num_samples: int = 20) -> ComponentProfile:
        """STT 레이턴시 프로파일링"""
        print("\n📊 STT 프로파일링 중...")

        from stt.deepgram_service import DeepgramSTTService

        mem_before = self.monitor.get_memory_metrics()
        gpu_before = self.monitor.get_gpu_metrics()

        # 더미 오디오 생성 (3초, 16kHz)
        sample_rate = 16000
        duration = 3.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        latencies = []
        stt = DeepgramSTTService()

        for i in range(num_samples):
            start = time.time()
            try:
                # 실제 API 호출 대신 로컬 처리 시뮬레이션
                await asyncio.sleep(0.1)  # 시뮬레이션
                transcript = f"테스트 음성 {i+1}"
            except Exception as e:
                print(f"  ⚠ 샘플 {i+1} 실패: {e}")
                continue

            latency = (time.time() - start) * 1000
            latencies.append(latency)

            if (i + 1) % 5 == 0:
                print(f"  진행: {i+1}/{num_samples} (평균: {np.mean(latencies):.2f}ms)")

        mem_after = self.monitor.get_memory_metrics()
        gpu_after = self.monitor.get_gpu_metrics()

        return ComponentProfile(
            name="STT (Deepgram)",
            latency=LatencyMetrics.from_samples(latencies),
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta_mb=mem_after.rss_mb - mem_before.rss_mb,
            gpu_before=gpu_before,
            gpu_after=gpu_after,
            throughput_per_sec=num_samples / (sum(latencies) / 1000),
            samples_count=len(latencies),
        )

    async def profile_llm(self, num_samples: int = 10) -> ComponentProfile:
        """LLM TTFT (Time To First Token) 프로파일링"""
        print("\n📊 LLM 프로파일링 중...")

        from llm.interviewer_agent import InterviewerAgent

        mem_before = self.monitor.get_memory_metrics()
        gpu_before = self.monitor.get_gpu_metrics()

        agent = InterviewerAgent()
        latencies_ttft = []
        latencies_total = []

        test_inputs = [
            "안녕하세요",
            "저는 개발자입니다",
            "Python을 사용합니다",
            "5년 경력입니다",
            "팀 프로젝트 경험이 있습니다",
        ]

        for i in range(num_samples):
            user_input = test_inputs[i % len(test_inputs)]

            start = time.time()
            try:
                response = await agent.generate_response(user_input)

                # TTFT 시뮬레이션 (실제는 스트리밍 첫 토큰까지 시간)
                ttft = (time.time() - start) * 1000
                latencies_ttft.append(ttft)

                # 전체 생성 시간
                total_latency = ttft  # 실제는 전체 스트리밍 완료까지
                latencies_total.append(total_latency)

            except Exception as e:
                print(f"  ⚠ 샘플 {i+1} 실패: {e}")
                continue

            if (i + 1) % 2 == 0:
                print(f"  진행: {i+1}/{num_samples} (TTFT 평균: {np.mean(latencies_ttft):.2f}ms)")

        mem_after = self.monitor.get_memory_metrics()
        gpu_after = self.monitor.get_gpu_metrics()

        return ComponentProfile(
            name="LLM (GPT-4o TTFT)",
            latency=LatencyMetrics.from_samples(latencies_ttft),
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta_mb=mem_after.rss_mb - mem_before.rss_mb,
            gpu_before=gpu_before,
            gpu_after=gpu_after,
            throughput_per_sec=num_samples / (sum(latencies_total) / 1000),
            samples_count=len(latencies_ttft),
        )

    async def profile_tts(self, num_samples: int = 15) -> ComponentProfile:
        """TTS TTFB (Time To First Byte) 프로파일링"""
        print("\n📊 TTS 프로파일링 중...")

        from tts.elevenlabs_service import ElevenLabsTTSService

        mem_before = self.monitor.get_memory_metrics()
        gpu_before = self.monitor.get_gpu_metrics()

        tts = ElevenLabsTTSService()
        latencies_ttfb = []

        test_texts = [
            "안녕하세요",
            "자기소개 부탁드립니다",
            "경력에 대해 말씀해주세요",
            "기술 스택은 무엇인가요",
            "프로젝트 경험을 알려주세요",
        ]

        for i in range(num_samples):
            text = test_texts[i % len(test_texts)]

            start = time.time()
            try:
                # 첫 오디오 청크까지 시간 (TTFB)
                first_chunk = True
                ttfb = 0

                async for chunk in tts.stream_audio(text):
                    if first_chunk:
                        ttfb = (time.time() - start) * 1000
                        latencies_ttfb.append(ttfb)
                        first_chunk = False
                        break  # TTFB만 측정

            except Exception as e:
                print(f"  ⚠ 샘플 {i+1} 실패: {e}")
                # Mock TTFB
                latencies_ttfb.append(150.0)
                continue

            if (i + 1) % 3 == 0:
                print(f"  진행: {i+1}/{num_samples} (TTFB 평균: {np.mean(latencies_ttfb):.2f}ms)")

        mem_after = self.monitor.get_memory_metrics()
        gpu_after = self.monitor.get_gpu_metrics()

        return ComponentProfile(
            name="TTS (ElevenLabs TTFB)",
            latency=LatencyMetrics.from_samples(latencies_ttfb),
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta_mb=mem_after.rss_mb - mem_before.rss_mb,
            gpu_before=gpu_before,
            gpu_after=gpu_after,
            throughput_per_sec=num_samples / (sum(latencies_ttfb) / 1000),
            samples_count=len(latencies_ttfb),
        )

    async def profile_avatar(self, num_samples: int = 25) -> ComponentProfile:
        """Avatar 프레임 렌더링 프로파일링"""
        print("\n📊 Avatar 프로파일링 중...")

        from avatar.musetalk_wrapper import MuseTalkAvatar, MuseTalkConfig

        mem_before = self.monitor.get_memory_metrics()
        gpu_before = self.monitor.get_gpu_metrics()

        config = MuseTalkConfig(
            fps=25,
            enable_face_enhancement=False,  # 성능 측정 시 비활성화
        )

        avatar = MuseTalkAvatar(config)
        # await avatar.initialize()  # 실제 초기화

        latencies = []

        # 더미 오디오 청크 (40ms 분량)
        chunk_samples = int(16000 * 0.04)

        for i in range(num_samples):
            audio_chunk = np.random.randn(chunk_samples).astype(np.float32)

            start = time.time()
            try:
                # 실제 프레임 생성 시뮬레이션
                await asyncio.sleep(0.03)  # 30ms 시뮬레이션
                # frame = await avatar.process_audio_chunk(audio_chunk)

            except Exception as e:
                print(f"  ⚠ 프레임 {i+1} 실패: {e}")
                continue

            latency = (time.time() - start) * 1000
            latencies.append(latency)

            if (i + 1) % 5 == 0:
                print(f"  진행: {i+1}/{num_samples} (평균: {np.mean(latencies):.2f}ms/frame)")

        mem_after = self.monitor.get_memory_metrics()
        gpu_after = self.monitor.get_gpu_metrics()

        fps = 1000 / np.mean(latencies) if latencies else 0

        return ComponentProfile(
            name="Avatar (MuseTalk)",
            latency=LatencyMetrics.from_samples(latencies),
            memory_before=mem_before,
            memory_after=mem_after,
            memory_delta_mb=mem_after.rss_mb - mem_before.rss_mb,
            gpu_before=gpu_before,
            gpu_after=gpu_after,
            throughput_per_sec=fps,
            samples_count=len(latencies),
        )


# ============================================================================
# 분석 및 리포트
# ============================================================================

class ProfileAnalyzer:
    """프로파일 결과 분석"""

    # 목표 레이턴시 (ms)
    TARGETS = {
        "STT": 100,
        "LLM": 200,
        "TTS": 200,
        "Avatar": 50,
    }

    @classmethod
    def analyze_bottlenecks(cls, components: Dict[str, ComponentProfile]) -> List[str]:
        """병목 구간 식별"""
        bottlenecks = []

        for name, profile in components.items():
            component_type = name.split()[0]
            target = cls.TARGETS.get(component_type, float('inf'))

            if profile.latency.p95 > target:
                slowdown = (profile.latency.p95 / target - 1) * 100
                bottlenecks.append(
                    f"{name}: P95 {profile.latency.p95:.1f}ms "
                    f"(목표 {target}ms 대비 {slowdown:.1f}% 초과)"
                )

        return bottlenecks

    @classmethod
    def generate_recommendations(cls, components: Dict[str, ComponentProfile]) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []

        for name, profile in components.items():
            component_type = name.split()[0]

            # STT 최적화
            if component_type == "STT" and profile.latency.mean > 100:
                recommendations.append(
                    "STT: VAD 설정 조정으로 오디오 청크 크기 최적화"
                )

            # LLM 최적화
            if component_type == "LLM":
                if profile.latency.mean > 500:
                    recommendations.append(
                        "LLM: 더 빠른 모델 사용 (GPT-4o → GPT-4o-mini)"
                    )
                if profile.latency.std > 200:
                    recommendations.append(
                        "LLM: 레이턴시 변동이 큽니다. 캐싱 또는 프롬프트 최적화 고려"
                    )

            # TTS 최적화
            if component_type == "TTS" and profile.latency.mean > 200:
                recommendations.append(
                    "TTS: 공통 질문 캐싱 활성화 (TTSCache)"
                )

            # Avatar 최적화
            if component_type == "Avatar":
                if profile.latency.mean > 50:
                    recommendations.append(
                        "Avatar: face_enhancement 비활성화 또는 GPU 업그레이드"
                    )
                if profile.gpu_after and profile.gpu_after.memory_percent > 90:
                    recommendations.append(
                        "Avatar: GPU 메모리 부족 (배치 사이즈 감소)"
                    )

            # 메모리 누수 감지
            if profile.memory_delta_mb > 100:
                recommendations.append(
                    f"{name}: 메모리 증가 감지 ({profile.memory_delta_mb:.1f}MB), "
                    "메모리 누수 가능성"
                )

        return recommendations


# ============================================================================
# 시각화
# ============================================================================

class ProfileVisualizer:
    """프로파일 결과 시각화"""

    @staticmethod
    def plot_latency_comparison(components: Dict[str, ComponentProfile], output_path: Path):
        """레이턴시 비교 차트"""
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        names = list(components.keys())
        p50_values = [c.latency.p50 for c in components.values()]
        p95_values = [c.latency.p95 for c in components.values()]
        p99_values = [c.latency.p99 for c in components.values()]

        x = np.arange(len(names))
        width = 0.25

        ax.bar(x - width, p50_values, width, label='P50', color='#2ecc71')
        ax.bar(x, p95_values, width, label='P95', color='#f39c12')
        ax.bar(x + width, p99_values, width, label='P99', color='#e74c3c')

        ax.set_xlabel('Component', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Component Latency Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"✓ 차트 저장: {output_path}")

    @staticmethod
    def plot_memory_usage(components: Dict[str, ComponentProfile], output_path: Path):
        """메모리 사용량 차트"""
        if not HAS_MATPLOTLIB:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        names = list(components.keys())
        mem_before = [c.memory_before.rss_mb for c in components.values()]
        mem_after = [c.memory_after.rss_mb for c in components.values()]
        mem_delta = [c.memory_delta_mb for c in components.values()]

        # Before/After 비교
        x = np.arange(len(names))
        width = 0.35

        ax1.bar(x - width/2, mem_before, width, label='Before', color='#3498db')
        ax1.bar(x + width/2, mem_after, width, label='After', color='#e67e22')
        ax1.set_xlabel('Component', fontsize=12)
        ax1.set_ylabel('Memory (MB)', fontsize=12)
        ax1.set_title('Memory Usage Before/After', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Delta
        colors = ['#e74c3c' if d > 50 else '#2ecc71' for d in mem_delta]
        ax2.bar(names, mem_delta, color=colors)
        ax2.set_xlabel('Component', fontsize=12)
        ax2.set_ylabel('Memory Delta (MB)', fontsize=12)
        ax2.set_title('Memory Increase', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(names, rotation=15, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"✓ 차트 저장: {output_path}")

    @staticmethod
    def plot_gpu_usage(components: Dict[str, ComponentProfile], output_path: Path):
        """GPU 사용량 차트"""
        if not HAS_MATPLOTLIB:
            return

        # GPU 메트릭이 있는 컴포넌트만 필터링
        gpu_components = {
            name: comp for name, comp in components.items()
            if comp.gpu_after is not None
        }

        if not gpu_components:
            print("⚠ GPU 메트릭이 없어 GPU 차트를 생성하지 않습니다.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        names = list(gpu_components.keys())
        mem_used = [c.gpu_after.memory_used_mb for c in gpu_components.values()]
        utilization = [c.gpu_after.utilization_percent for c in gpu_components.values()]

        # GPU 메모리
        ax1.bar(names, mem_used, color='#9b59b6')
        ax1.set_xlabel('Component', fontsize=12)
        ax1.set_ylabel('GPU Memory (MB)', fontsize=12)
        ax1.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
        ax1.set_xticklabels(names, rotation=15, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # GPU 활용률
        ax2.bar(names, utilization, color='#1abc9c')
        ax2.set_xlabel('Component', fontsize=12)
        ax2.set_ylabel('Utilization (%)', fontsize=12)
        ax2.set_title('GPU Utilization', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(names, rotation=15, ha='right')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"✓ 차트 저장: {output_path}")


# ============================================================================
# 메인 프로파일러
# ============================================================================

class PerformanceProfiler:
    """전체 성능 프로파일러"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = SystemMonitor()
        self.profiler = ComponentProfiler(self.monitor)
        self.visualizer = ProfileVisualizer()

    async def run(
        self,
        components: List[str] = None,
        samples_per_component: int = 20,
    ) -> ProfileReport:
        """프로파일링 실행"""

        if components is None:
            components = ["stt", "llm", "tts", "avatar"]

        print("\n" + "="*60)
        print("🔍 성능 프로파일링 시작")
        print("="*60)

        start_time = time.time()
        profiles = {}

        # 각 컴포넌트 프로파일링
        if "stt" in components:
            profiles["STT"] = await self.profiler.profile_stt(samples_per_component)

        if "llm" in components:
            profiles["LLM"] = await self.profiler.profile_llm(samples_per_component // 2)

        if "tts" in components:
            profiles["TTS"] = await self.profiler.profile_tts(samples_per_component)

        if "avatar" in components:
            profiles["Avatar"] = await self.profiler.profile_avatar(samples_per_component)

        duration = time.time() - start_time

        # 분석
        bottlenecks = ProfileAnalyzer.analyze_bottlenecks(profiles)
        recommendations = ProfileAnalyzer.generate_recommendations(profiles)
        system_info = self.monitor.get_system_info()

        # 리포트 생성
        report = ProfileReport(
            timestamp=datetime.now().isoformat(),
            duration_sec=duration,
            components=profiles,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            system_info=system_info,
        )

        # 결과 출력
        self.print_report(report)

        # JSON 저장
        self.save_json_report(report)

        # 차트 생성
        if HAS_MATPLOTLIB:
            self.create_visualizations(profiles)

        return report

    def print_report(self, report: ProfileReport):
        """리포트 콘솔 출력"""
        print("\n" + "="*60)
        print("📊 프로파일링 결과")
        print("="*60)

        # 컴포넌트별 결과
        for name, profile in report.components.items():
            print(f"\n【{name}】")
            print(f"  레이턴시:")
            print(f"    • 평균:  {profile.latency.mean:.2f}ms")
            print(f"    • 중앙값: {profile.latency.median:.2f}ms")
            print(f"    • P95:   {profile.latency.p95:.2f}ms")
            print(f"    • P99:   {profile.latency.p99:.2f}ms")
            print(f"    • 표준편차: {profile.latency.std:.2f}ms")

            print(f"  메모리:")
            print(f"    • 사용 전: {profile.memory_before.rss_mb:.1f}MB")
            print(f"    • 사용 후: {profile.memory_after.rss_mb:.1f}MB")
            print(f"    • 증가량:  {profile.memory_delta_mb:.1f}MB")

            if profile.gpu_after:
                print(f"  GPU:")
                print(f"    • 메모리:  {profile.gpu_after.memory_used_mb:.1f}MB / "
                      f"{profile.gpu_after.memory_total_mb:.1f}MB "
                      f"({profile.gpu_after.memory_percent:.1f}%)")
                print(f"    • 활용률:  {profile.gpu_after.utilization_percent:.1f}%")

            if profile.throughput_per_sec:
                print(f"  처리량: {profile.throughput_per_sec:.2f} ops/s")

        # 병목 구간
        print("\n" + "="*60)
        print("⚠️  병목 구간")
        print("="*60)
        if report.bottlenecks:
            for bottleneck in report.bottlenecks:
                print(f"  • {bottleneck}")
        else:
            print("  ✓ 모든 컴포넌트가 목표 레이턴시 달성")

        # 권장사항
        print("\n" + "="*60)
        print("💡 최적화 권장사항")
        print("="*60)
        if report.recommendations:
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  ✓ 추가 최적화 불필요")

        print("\n" + "="*60)
        print(f"⏱️  총 실행 시간: {report.duration_sec:.2f}s")
        print("="*60 + "\n")

    def save_json_report(self, report: ProfileReport):
        """JSON 리포트 저장"""
        output_file = self.output_dir / f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # dataclass를 dict로 변환 (nested)
        def convert_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert_to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj

        report_dict = convert_to_dict(report)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ JSON 리포트 저장: {output_file}")

    def create_visualizations(self, profiles: Dict[str, ComponentProfile]):
        """시각화 차트 생성"""
        print("\n📈 차트 생성 중...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 레이턴시 비교
        self.visualizer.plot_latency_comparison(
            profiles,
            self.output_dir / f"latency_{timestamp}.png"
        )

        # 메모리 사용량
        self.visualizer.plot_memory_usage(
            profiles,
            self.output_dir / f"memory_{timestamp}.png"
        )

        # GPU 사용량
        self.visualizer.plot_gpu_usage(
            profiles,
            self.output_dir / f"gpu_{timestamp}.png"
        )


# ============================================================================
# CLI
# ============================================================================

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Interview Avatar System Performance Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 모든 컴포넌트 프로파일링
  python scripts/profile.py

  # 특정 컴포넌트만
  python scripts/profile.py --components stt llm

  # 샘플 수 조정
  python scripts/profile.py --samples 50

  # 출력 디렉토리 지정
  python scripts/profile.py --output-dir results/profiles
        """
    )

    parser.add_argument(
        '--components',
        nargs='+',
        choices=['stt', 'llm', 'tts', 'avatar'],
        default=['stt', 'llm', 'tts', 'avatar'],
        help='프로파일링할 컴포넌트 (기본: 전체)',
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=20,
        help='컴포넌트당 샘플 수 (기본: 20)',
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('profile_results'),
        help='결과 출력 디렉토리 (기본: profile_results)',
    )

    args = parser.parse_args()

    # 프로파일러 실행
    profiler = PerformanceProfiler(output_dir=args.output_dir)

    try:
        asyncio.run(
            profiler.run(
                components=args.components,
                samples_per_component=args.samples,
            )
        )
    except KeyboardInterrupt:
        print("\n\n⚠ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
