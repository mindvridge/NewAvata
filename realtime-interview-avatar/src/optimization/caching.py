"""
캐싱 최적화 모듈

LRU 캐시, TTS 오디오 캐싱, 얼굴 특징 캐싱
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, Callable, TypeVar, Generic
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# LRU Cache
# ============================================================================

class LRUCache(Generic[T]):
    """
    LRU (Least Recently Used) 캐시

    메모리 기반 캐시로, 가장 오래 사용되지 않은 항목을 자동으로 제거
    """

    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: 최대 캐시 크기
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, T] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[T]:
        """캐시에서 값 조회"""
        if key in self.cache:
            # 최근 사용으로 이동
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, key: str, value: T) -> None:
        """캐시에 값 저장"""
        if key in self.cache:
            # 기존 항목 업데이트
            self.cache.move_to_end(key)
        else:
            # 새 항목 추가
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거 (LRU)
                self.cache.popitem(last=False)

        self.cache[key] = value

    def clear(self) -> None:
        """캐시 전체 삭제"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def size(self) -> int:
        """현재 캐시 크기"""
        return len(self.cache)

    def hit_rate(self) -> float:
        """캐시 히트율"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate(),
        }


# ============================================================================
# TTL Cache
# ============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """캐시 엔트리 (TTL 포함)"""
    value: T
    created_at: datetime
    expires_at: Optional[datetime] = None


class TTLCache(Generic[T]):
    """
    TTL (Time To Live) 캐시

    일정 시간 후 자동으로 만료되는 캐시
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Args:
            max_size: 최대 캐시 크기
            ttl_seconds: 캐시 만료 시간 (초)
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[T]:
        """캐시에서 값 조회 (만료 확인)"""
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # 만료 확인
        if entry.expires_at and datetime.now() > entry.expires_at:
            del self.cache[key]
            self.misses += 1
            return None

        # 최근 사용으로 이동
        self.cache.move_to_end(key)
        self.hits += 1
        return entry.value

    def put(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> None:
        """캐시에 값 저장"""
        now = datetime.now()

        # TTL 계산
        if ttl_seconds is not None:
            expires_at = now + timedelta(seconds=ttl_seconds)
        else:
            expires_at = now + self.ttl

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=expires_at,
        )

        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

        self.cache[key] = entry

    def cleanup_expired(self) -> int:
        """만료된 항목 정리"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.expires_at and now > entry.expires_at
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)


# ============================================================================
# TTS Audio Cache
# ============================================================================

class TTSAudioCache:
    """
    TTS 오디오 캐싱

    텍스트 → 오디오 변환 결과를 캐싱하여 레이턴시 대폭 감소
    """

    def __init__(
        self,
        max_memory_items: int = 100,
        enable_disk_cache: bool = True,
        cache_dir: Path = Path("cache/tts"),
        ttl_seconds: int = 86400,  # 24시간
    ):
        """
        Args:
            max_memory_items: 메모리 캐시 최대 크기
            enable_disk_cache: 디스크 캐시 활성화
            cache_dir: 디스크 캐시 디렉토리
            ttl_seconds: 캐시 만료 시간
        """
        self.memory_cache = TTLCache[bytes](
            max_size=max_memory_items,
            ttl_seconds=ttl_seconds,
        )
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = cache_dir

        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_text(self, text: str, voice_id: str = "default") -> str:
        """텍스트와 음성 ID를 해시화"""
        combined = f"{text}:{voice_id}"
        return hashlib.sha256(combined.encode()).hexdigest()

    async def get(self, text: str, voice_id: str = "default") -> Optional[bytes]:
        """캐시에서 오디오 조회"""
        cache_key = self._hash_text(text, voice_id)

        # 1. 메모리 캐시 확인
        audio = self.memory_cache.get(cache_key)
        if audio is not None:
            logger.debug(f"TTS 메모리 캐시 히트: {text[:30]}...")
            return audio

        # 2. 디스크 캐시 확인
        if self.enable_disk_cache:
            disk_path = self.cache_dir / f"{cache_key}.bin"
            if disk_path.exists():
                try:
                    audio = disk_path.read_bytes()
                    # 메모리 캐시에도 저장
                    self.memory_cache.put(cache_key, audio)
                    logger.debug(f"TTS 디스크 캐시 히트: {text[:30]}...")
                    return audio
                except Exception as e:
                    logger.warning(f"디스크 캐시 읽기 실패: {e}")

        logger.debug(f"TTS 캐시 미스: {text[:30]}...")
        return None

    async def put(
        self,
        text: str,
        audio: bytes,
        voice_id: str = "default",
    ) -> None:
        """캐시에 오디오 저장"""
        cache_key = self._hash_text(text, voice_id)

        # 1. 메모리 캐시 저장
        self.memory_cache.put(cache_key, audio)

        # 2. 디스크 캐시 저장 (비동기)
        if self.enable_disk_cache:
            disk_path = self.cache_dir / f"{cache_key}.bin"
            try:
                await asyncio.to_thread(disk_path.write_bytes, audio)
                logger.debug(f"TTS 캐시 저장: {text[:30]}...")
            except Exception as e:
                logger.warning(f"디스크 캐시 저장 실패: {e}")

    async def prewarm(self, texts: list[str], voice_id: str = "default") -> None:
        """
        캐시 미리 워밍

        자주 사용되는 텍스트를 미리 캐싱
        """
        logger.info(f"TTS 캐시 prewarming 시작: {len(texts)}개 항목")

        # 실제 TTS 서비스 필요 (여기서는 시뮬레이션)
        for text in texts:
            cache_key = self._hash_text(text, voice_id)

            # 이미 캐시된 경우 건너뛰기
            if self.memory_cache.get(cache_key) is not None:
                continue

            # 디스크 캐시 확인
            if self.enable_disk_cache:
                disk_path = self.cache_dir / f"{cache_key}.bin"
                if disk_path.exists():
                    audio = disk_path.read_bytes()
                    self.memory_cache.put(cache_key, audio)
                    continue

            logger.info(f"  - {text[:50]}... (캐시 없음)")

        logger.info("TTS 캐시 prewarming 완료")

    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        disk_size_mb = 0
        if self.enable_disk_cache and self.cache_dir.exists():
            disk_size_mb = sum(
                f.stat().st_size for f in self.cache_dir.glob("*.bin")
            ) / 1024 / 1024

        return {
            "memory": {
                "size": self.memory_cache.cache.size(),
                "hits": self.memory_cache.hits,
                "misses": self.memory_cache.misses,
                "hit_rate": self.memory_cache.hit_rate(),
            },
            "disk": {
                "enabled": self.enable_disk_cache,
                "size_mb": disk_size_mb if self.enable_disk_cache else 0,
            },
        }


# ============================================================================
# Face Feature Cache
# ============================================================================

class FaceFeatureCache:
    """
    얼굴 특징 캐싱

    얼굴 랜드마크, 임베딩 등을 캐싱하여 반복 처리 방지
    """

    def __init__(
        self,
        max_size: int = 50,
        cache_dir: Path = Path("cache/face_features"),
    ):
        """
        Args:
            max_size: 최대 캐시 크기
            cache_dir: 디스크 캐시 디렉토리
        """
        self.cache = LRUCache[np.ndarray](max_size=max_size)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_image(self, image: np.ndarray) -> str:
        """이미지를 해시화"""
        # 이미지 데이터를 해시
        image_bytes = image.tobytes()
        return hashlib.sha256(image_bytes).hexdigest()

    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 랜드마크 조회"""
        cache_key = self._hash_image(image)
        return self.cache.get(cache_key)

    def put_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> None:
        """얼굴 랜드마크 저장"""
        cache_key = self._hash_image(image)
        self.cache.put(cache_key, landmarks)

        # 디스크에도 저장 (선택적)
        try:
            disk_path = self.cache_dir / f"{cache_key}.npy"
            np.save(disk_path, landmarks)
        except Exception as e:
            logger.warning(f"랜드마크 디스크 저장 실패: {e}")

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 임베딩 조회"""
        cache_key = f"embed_{self._hash_image(image)}"
        return self.cache.get(cache_key)

    def put_embedding(self, image: np.ndarray, embedding: np.ndarray) -> None:
        """얼굴 임베딩 저장"""
        cache_key = f"embed_{self._hash_image(image)}"
        self.cache.put(cache_key, embedding)


# ============================================================================
# General Cache Decorator
# ============================================================================

def cached(cache: LRUCache, key_func: Optional[Callable] = None):
    """
    함수 결과를 캐싱하는 데코레이터

    Usage:
        cache = LRUCache(max_size=100)

        @cached(cache)
        def expensive_function(x, y):
            # ... 비용이 큰 연산
            return result
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # 캐시 조회
            result = cache.get(cache_key)
            if result is not None:
                return result

            # 함수 실행 및 캐싱
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            return result

        return wrapper
    return decorator


def async_cached(cache: LRUCache, key_func: Optional[Callable] = None):
    """
    비동기 함수용 캐싱 데코레이터

    Usage:
        cache = LRUCache(max_size=100)

        @async_cached(cache)
        async def expensive_async_function(x, y):
            # ... 비동기 연산
            return result
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # 캐시 키 생성
            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # 캐시 조회
            result = cache.get(cache_key)
            if result is not None:
                return result

            # 함수 실행 및 캐싱
            result = await func(*args, **kwargs)
            cache.put(cache_key, result)
            return result

        return wrapper
    return decorator


# ============================================================================
# Cache Manager
# ============================================================================

class CacheManager:
    """
    캐시 통합 관리

    여러 캐시를 중앙에서 관리하고 통계 제공
    """

    def __init__(self):
        self.caches: Dict[str, Any] = {}

    def register(self, name: str, cache: Any) -> None:
        """캐시 등록"""
        self.caches[name] = cache
        logger.info(f"캐시 등록: {name}")

    def get_cache(self, name: str) -> Optional[Any]:
        """캐시 조회"""
        return self.caches.get(name)

    def clear_all(self) -> None:
        """모든 캐시 삭제"""
        for name, cache in self.caches.items():
            if hasattr(cache, 'clear'):
                cache.clear()
                logger.info(f"캐시 삭제: {name}")

    def stats(self) -> Dict[str, Any]:
        """전체 캐시 통계"""
        stats = {}
        for name, cache in self.caches.items():
            if hasattr(cache, 'stats'):
                stats[name] = cache.stats()
        return stats

    def print_stats(self) -> None:
        """통계 출력"""
        stats = self.stats()

        print("\n" + "="*60)
        print("캐시 통계")
        print("="*60)

        for name, stat in stats.items():
            print(f"\n【{name}】")
            if isinstance(stat, dict):
                for key, value in stat.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    - {k}: {v}")
                    else:
                        print(f"  - {key}: {value}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        # TTS 캐시 테스트
        tts_cache = TTSAudioCache(max_memory_items=10, enable_disk_cache=True)

        # 샘플 텍스트
        texts = [
            "안녕하세요",
            "자기소개 부탁드립니다",
            "경력에 대해 말씀해주세요",
        ]

        # 캐시 저장
        for text in texts:
            audio = f"audio_data_{text}".encode()  # 더미 오디오
            await tts_cache.put(text, audio)

        # 캐시 조회
        for text in texts:
            audio = await tts_cache.get(text)
            print(f"{text}: {'HIT' if audio else 'MISS'}")

        # 통계
        print("\nTTS 캐시 통계:")
        print(tts_cache.stats())

        # Cache Manager
        manager = CacheManager()
        manager.register("tts", tts_cache)
        manager.register("face", FaceFeatureCache())

        manager.print_stats()

    asyncio.run(main())
