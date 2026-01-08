"""
Optimization Module
최적화 모듈 - 모델, 캐싱, 배치 처리, 비동기 최적화
"""

from .model_optimization import (
    PrecisionMode,
    OptimizationConfig,
    TensorRTOptimizer,
    ONNXOptimizer,
    ModelQuantizer,
    ModelOptimizer,
)

from .caching import (
    LRUCache,
    TTLCache,
    TTSAudioCache,
    FaceFeatureCache,
    cached,
    async_cached,
    CacheManager,
)

from .batching import (
    BatchConfig,
    BatchProcessor,
    AudioChunkBatcher,
    FrameBatchRenderer,
    DynamicBatchSizer,
    AdaptiveBatchProcessor,
)

from .async_optimization import (
    PipelineStage,
    AsyncPipeline,
    ConcurrentExecutor,
    AsyncPool,
    StreamProcessor,
    RateLimiter,
    run_parallel,
)

__all__ = [
    # Model Optimization
    "PrecisionMode",
    "OptimizationConfig",
    "TensorRTOptimizer",
    "ONNXOptimizer",
    "ModelQuantizer",
    "ModelOptimizer",
    # Caching
    "LRUCache",
    "TTLCache",
    "TTSAudioCache",
    "FaceFeatureCache",
    "cached",
    "async_cached",
    "CacheManager",
    # Batching
    "BatchConfig",
    "BatchProcessor",
    "AudioChunkBatcher",
    "FrameBatchRenderer",
    "DynamicBatchSizer",
    "AdaptiveBatchProcessor",
    # Async Optimization
    "PipelineStage",
    "AsyncPipeline",
    "ConcurrentExecutor",
    "AsyncPool",
    "StreamProcessor",
    "RateLimiter",
    "run_parallel",
]
