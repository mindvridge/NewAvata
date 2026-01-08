"""
TTS Cache
자주 사용되는 문구를 캐싱하여 레이턴시를 줄입니다.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict, List
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from config import settings


# =============================================================================
# 면접용 사전 정의 문구
# =============================================================================

INTERVIEW_PHRASES = {
    # 인사말
    "greeting": [
        "안녕하세요, 면접을 시작하겠습니다.",
        "만나서 반갑습니다.",
        "오늘 면접에 참여해 주셔서 감사합니다.",
    ],

    # 필러 (자연스러운 대화를 위한 추임새)
    "filler": [
        "네, 알겠습니다.",
        "좋은 답변이네요.",
        "흥미롭습니다.",
        "잘 들었습니다.",
        "잠시만요.",
        "아, 그렇군요.",
        "네, 네.",
        "음...",
    ],

    # 전환 (질문 간 전환)
    "transition": [
        "다음 질문입니다.",
        "그럼 다음으로 넘어가겠습니다.",
        "이제 다른 질문을 드리겠습니다.",
        "마지막 질문입니다.",
    ],

    # 프롬프트 (답변 유도)
    "prompt": [
        "조금 더 자세히 말씀해 주시겠어요?",
        "구체적인 예시를 들어주실 수 있나요?",
        "그 부분에 대해 설명해 주세요.",
        "어떻게 생각하시나요?",
    ],

    # 긍정 피드백
    "positive": [
        "좋습니다.",
        "훌륭한 답변입니다.",
        "잘 이해했습니다.",
        "완벽합니다.",
    ],

    # 종료 인사
    "closing": [
        "면접이 종료되었습니다. 수고하셨습니다.",
        "오늘 면접 감사했습니다.",
        "좋은 하루 되세요.",
        "결과는 빠른 시일 내에 알려드리겠습니다.",
    ],

    # 대기/생각
    "thinking": [
        "잠시 생각해 보겠습니다.",
        "음, 그러니까...",
        "좋은 질문이네요.",
    ],
}


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    audio_data: bytes              # 오디오 데이터
    text: str                      # 원본 텍스트
    created_at: datetime           # 생성 시간
    last_accessed: datetime        # 마지막 접근 시간
    access_count: int = 0          # 접근 횟수
    sample_rate: int = 16000       # 샘플레이트
    channels: int = 1              # 채널 수


class LRUCache:
    """
    LRU (Least Recently Used) 캐시

    메모리 기반 캐시로, 최대 크기를 초과하면 가장 오래 사용되지 않은 항목을 제거합니다.
    """

    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: 최대 캐시 항목 수
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        logger.info(f"LRUCache initialized: max_size={max_size}")

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        캐시에서 항목 조회

        Args:
            key: 캐시 키

        Returns:
            CacheEntry: 캐시 엔트리 (없으면 None)
        """
        if key not in self.cache:
            return None

        # LRU 업데이트 (최근 사용된 항목으로 이동)
        self.cache.move_to_end(key)

        # 통계 업데이트
        entry = self.cache[key]
        entry.last_accessed = datetime.now()
        entry.access_count += 1

        return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        """
        캐시에 항목 추가

        Args:
            key: 캐시 키
            entry: 캐시 엔트리
        """
        # 이미 존재하면 업데이트
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = entry
            return

        # 최대 크기 초과 시 가장 오래된 항목 제거
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            removed_entry = self.cache.pop(oldest_key)
            logger.debug(f"Cache evicted: {removed_entry.text[:30]}...")

        self.cache[key] = entry

    def clear(self) -> None:
        """캐시 전체 삭제"""
        self.cache.clear()
        logger.debug("Cache cleared")

    def size(self) -> int:
        """현재 캐시 크기"""
        return len(self.cache)

    def get_all_keys(self) -> List[str]:
        """모든 캐시 키 조회"""
        return list(self.cache.keys())


class TTSCache:
    """
    TTS 캐시 관리자

    메모리와 디스크에 TTS 결과를 캐싱하여 레이턴시를 줄입니다.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_size: int = 100,
        enable_disk_cache: bool = True,
    ):
        """
        Args:
            cache_dir: 디스크 캐시 디렉토리
            max_memory_size: 메모리 캐시 최대 크기
            enable_disk_cache: 디스크 캐시 활성화 여부
        """
        self.cache_dir = cache_dir or settings.cache_dir / "tts"
        self.enable_disk_cache = enable_disk_cache

        # 메모리 캐시 (LRU)
        self._memory_cache = LRUCache(max_size=max_memory_size)

        # 통계
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0

        # 캐시 디렉토리 생성
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"TTSCache initialized: cache_dir={self.cache_dir}, "
            f"max_memory_size={max_memory_size}, disk_cache={enable_disk_cache}"
        )

    def _get_cache_key(self, text: str, voice_id: str = "default") -> str:
        """
        텍스트와 음성 ID로 캐시 키 생성

        Args:
            text: 텍스트
            voice_id: 음성 ID

        Returns:
            str: 캐시 키 (해시)
        """
        key_string = f"{text}:{voice_id}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_disk_path(self, cache_key: str) -> Path:
        """
        디스크 캐시 파일 경로

        Args:
            cache_key: 캐시 키

        Returns:
            Path: 캐시 파일 경로
        """
        return self.cache_dir / f"{cache_key}.cache"

    def get(self, text: str, voice_id: str = "default") -> Optional[bytes]:
        """
        캐시에서 오디오 조회

        Args:
            text: 텍스트
            voice_id: 음성 ID

        Returns:
            bytes: 오디오 데이터 (캐시 미스 시 None)
        """
        self._total_requests += 1
        cache_key = self._get_cache_key(text, voice_id)

        # 1. 메모리 캐시 확인
        entry = self._memory_cache.get(cache_key)
        if entry:
            self._cache_hits += 1
            logger.debug(f"Memory cache hit: {text[:30]}...")
            return entry.audio_data

        # 2. 디스크 캐시 확인
        if self.enable_disk_cache:
            disk_path = self._get_disk_path(cache_key)
            if disk_path.exists():
                try:
                    with open(disk_path, 'rb') as f:
                        entry = pickle.load(f)

                    # 메모리 캐시에 추가 (warm-up)
                    self._memory_cache.put(cache_key, entry)

                    self._cache_hits += 1
                    logger.debug(f"Disk cache hit: {text[:30]}...")
                    return entry.audio_data

                except Exception as e:
                    logger.error(f"Failed to load disk cache: {e}")

        # 캐시 미스
        self._cache_misses += 1
        logger.debug(f"Cache miss: {text[:30]}...")
        return None

    def put(
        self,
        text: str,
        audio_data: bytes,
        voice_id: str = "default",
        save_to_disk: bool = True,
    ) -> None:
        """
        캐시에 오디오 저장

        Args:
            text: 텍스트
            audio_data: 오디오 데이터
            voice_id: 음성 ID
            save_to_disk: 디스크에도 저장할지 여부
        """
        cache_key = self._get_cache_key(text, voice_id)

        # CacheEntry 생성
        entry = CacheEntry(
            audio_data=audio_data,
            text=text,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
        )

        # 메모리 캐시에 저장
        self._memory_cache.put(cache_key, entry)

        # 디스크 캐시에 저장
        if save_to_disk and self.enable_disk_cache:
            try:
                disk_path = self._get_disk_path(cache_key)
                with open(disk_path, 'wb') as f:
                    pickle.dump(entry, f)

                logger.debug(f"Saved to disk cache: {text[:30]}...")

            except Exception as e:
                logger.error(f"Failed to save disk cache: {e}")

        logger.debug(f"Cached: {text[:30]}... ({len(audio_data)} bytes)")

    async def prewarm(
        self,
        tts_service,
        phrases: Optional[Dict[str, List[str]]] = None,
    ) -> int:
        """
        자주 사용되는 문구를 미리 합성하여 캐싱 (프리워밍)

        Args:
            tts_service: TTS 서비스 인스턴스
            phrases: 캐싱할 문구 딕셔너리 (None이면 기본 INTERVIEW_PHRASES 사용)

        Returns:
            int: 캐싱된 문구 수
        """
        if phrases is None:
            phrases = INTERVIEW_PHRASES

        logger.info("Starting TTS cache prewarming...")
        cached_count = 0

        for category, phrase_list in phrases.items():
            logger.debug(f"Prewarming category: {category}")

            for phrase in phrase_list:
                # 이미 캐시된 경우 스킵
                if self.get(phrase) is not None:
                    logger.debug(f"Already cached: {phrase[:30]}...")
                    continue

                try:
                    # TTS 합성
                    logger.debug(f"Synthesizing: {phrase}")
                    audio_data = await tts_service.synthesize(phrase, streaming=False)

                    if audio_data:
                        # 캐시에 저장
                        self.put(phrase, audio_data, save_to_disk=True)
                        cached_count += 1

                except Exception as e:
                    logger.error(f"Failed to prewarm phrase '{phrase}': {e}")

        logger.info(f"Prewarming completed: {cached_count} phrases cached")
        return cached_count

    def get_stats(self) -> dict:
        """
        캐시 통계 조회

        Returns:
            dict: 통계 정보
        """
        hit_rate = (self._cache_hits / self._total_requests * 100) if self._total_requests > 0 else 0

        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": hit_rate,
            "memory_cache_size": self._memory_cache.size(),
            "disk_cache_enabled": self.enable_disk_cache,
        }

    def reset_stats(self) -> None:
        """통계 초기화"""
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Cache stats reset")

    def clear_memory_cache(self) -> None:
        """메모리 캐시 삭제"""
        self._memory_cache.clear()
        logger.info("Memory cache cleared")

    def clear_disk_cache(self) -> None:
        """디스크 캐시 삭제"""
        if not self.enable_disk_cache:
            return

        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()

            logger.info("Disk cache cleared")

        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")

    def clear_all(self) -> None:
        """모든 캐시 삭제 (메모리 + 디스크)"""
        self.clear_memory_cache()
        self.clear_disk_cache()
        logger.info("All caches cleared")


# =============================================================================
# 캐시 통합 TTS 서비스
# =============================================================================

class CachedTTSService:
    """
    캐시를 통합한 TTS 서비스

    자동으로 캐시를 확인하고, 없으면 합성하여 캐싱합니다.
    """

    def __init__(
        self,
        tts_service,
        cache: Optional[TTSCache] = None,
        auto_cache: bool = True,
    ):
        """
        Args:
            tts_service: 기본 TTS 서비스
            cache: TTS 캐시 (None이면 자동 생성)
            auto_cache: 자동 캐싱 여부
        """
        self.tts_service = tts_service
        self.cache = cache or TTSCache()
        self.auto_cache = auto_cache

        logger.info("CachedTTSService initialized")

    async def synthesize(
        self,
        text: str,
        streaming: bool = True,
        force_synthesis: bool = False,
    ) -> Optional[bytes]:
        """
        텍스트를 음성으로 합성 (캐시 우선)

        Args:
            text: 합성할 텍스트
            streaming: 스트리밍 모드
            force_synthesis: 캐시 무시하고 강제 합성

        Returns:
            bytes: 오디오 데이터 (streaming=False일 때만)
        """
        # 캐시 확인 (force_synthesis=False이고 비스트리밍일 때만)
        if not force_synthesis and not streaming:
            cached_audio = self.cache.get(text)
            if cached_audio:
                logger.debug(f"Using cached audio: {text[:30]}...")
                return cached_audio

        # TTS 합성
        audio_data = await self.tts_service.synthesize(text, streaming=streaming)

        # 자동 캐싱 (비스트리밍이고 데이터가 있을 때)
        if self.auto_cache and not streaming and audio_data:
            self.cache.put(text, audio_data)

        return audio_data

    async def prewarm(self, phrases: Optional[Dict[str, List[str]]] = None) -> int:
        """
        프리워밍 (자주 사용되는 문구 미리 캐싱)

        Args:
            phrases: 캐싱할 문구 딕셔너리

        Returns:
            int: 캐싱된 문구 수
        """
        return await self.cache.prewarm(self.tts_service, phrases)

    def get_cache_stats(self) -> dict:
        """캐시 통계 조회"""
        return self.cache.get_stats()


# =============================================================================
# 헬퍼 함수
# =============================================================================

def create_cached_tts(
    tts_service,
    max_memory_size: int = 100,
    enable_disk_cache: bool = True,
) -> CachedTTSService:
    """
    캐시 통합 TTS 서비스를 생성하는 헬퍼 함수

    Args:
        tts_service: 기본 TTS 서비스
        max_memory_size: 메모리 캐시 최대 크기
        enable_disk_cache: 디스크 캐시 활성화

    Returns:
        CachedTTSService: 캐시 통합 TTS 서비스
    """
    cache = TTSCache(
        max_memory_size=max_memory_size,
        enable_disk_cache=enable_disk_cache,
    )

    return CachedTTSService(
        tts_service=tts_service,
        cache=cache,
    )


# =============================================================================
# 테스트/예시 코드
# =============================================================================

if __name__ == "__main__":
    """캐시 테스트"""
    import asyncio

    async def test_cache():
        """캐시 기능 테스트"""
        from src.tts import create_elevenlabs_tts

        print("=" * 80)
        print("TTS Cache Test")
        print("=" * 80)

        # TTS 서비스 생성
        tts = create_elevenlabs_tts()

        # 캐시 통합 TTS 생성
        cached_tts = create_cached_tts(tts)

        # 프리워밍
        print("\n[1] Prewarming cache...")
        cached_count = await cached_tts.prewarm()
        print(f"Cached {cached_count} phrases")

        # 캐시 히트 테스트
        print("\n[2] Testing cache hits...")
        test_phrases = [
            "안녕하세요, 면접을 시작하겠습니다.",
            "네, 알겠습니다.",
            "다음 질문입니다.",
        ]

        for phrase in test_phrases:
            audio = await cached_tts.synthesize(phrase, streaming=False)
            print(f"  {phrase}: {len(audio) if audio else 0} bytes")

        # 통계 출력
        print("\n[3] Cache statistics:")
        stats = cached_tts.get_cache_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(f"\nCache hit rate: {stats['hit_rate_percent']:.1f}%")

    # 테스트 실행
    asyncio.run(test_cache())
