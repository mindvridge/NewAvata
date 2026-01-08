"""
한국어 TTS 서비스
비용 효율적인 한국어 TTS 대안 구현
"""

import asyncio
import io
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

from loguru import logger

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts not installed. EdgeTTSService will not work.")

from .elevenlabs_service import AudioChunk


# =============================================================================
# TTS 제공자 열거형
# =============================================================================

class TTSProvider(Enum):
    """TTS 제공자"""
    ELEVENLABS = "elevenlabs"      # 가장 자연스러운 음성, 유료
    EDGE_TTS = "edge_tts"          # Microsoft Edge TTS, 무료
    NAVER_CLOVA = "naver_clova"    # Naver Clova, 가장 자연스러운 한국어, 유료
    FALLBACK = "fallback"          # 폴백 체인


# =============================================================================
# EdgeTTS 서비스 (Microsoft Edge TTS - 무료)
# =============================================================================

@dataclass
class EdgeTTSConfig:
    """EdgeTTS 설정"""
    voice: str = "ko-KR-SunHiNeural"  # 한국어 음성
    rate: str = "+0%"                  # 속도 (-50% ~ +100%)
    volume: str = "+0%"                # 볼륨 (-50% ~ +50%)
    pitch: str = "+0Hz"                # 피치


class EdgeTTSService:
    """
    Microsoft Edge TTS 서비스

    장점:
    - 완전 무료 (API 키 불필요)
    - 낮은 레이턴시
    - 괜찮은 한국어 품질
    - 다양한 목소리 옵션

    단점:
    - ElevenLabs/Naver보다 자연스러움이 떨어짐
    - 감정 표현이 제한적
    - 상업적 사용 제약 가능성

    추천 사용 케이스:
    - 개발/테스트 환경
    - 저비용 프로덕션
    - 간단한 음성 안내
    """

    # 한국어 음성 옵션
    KOREAN_VOICES = {
        "female": "ko-KR-SunHiNeural",     # 여성 (밝고 친근한)
        "male": "ko-KR-InJoonNeural",       # 남성 (차분하고 전문적)
    }

    def __init__(
        self,
        config: Optional[EdgeTTSConfig] = None,
        on_audio_chunk: Optional[Callable[[AudioChunk], None]] = None,
        on_synthesis_start: Optional[Callable[[str], None]] = None,
        on_synthesis_end: Optional[Callable[[float], None]] = None,
    ):
        """
        Args:
            config: EdgeTTS 설정
            on_audio_chunk: 오디오 청크 콜백
            on_synthesis_start: 합성 시작 콜백
            on_synthesis_end: 합성 종료 콜백
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError(
                "edge-tts is not installed. Install it with: pip install edge-tts"
            )

        self.config = config or EdgeTTSConfig()
        self._on_audio_chunk = on_audio_chunk
        self._on_synthesis_start = on_synthesis_start
        self._on_synthesis_end = on_synthesis_end

        logger.info(f"EdgeTTSService initialized: voice={self.config.voice}")

    async def synthesize(
        self,
        text: str,
        streaming: bool = True,
    ) -> Optional[bytes]:
        """
        텍스트를 음성으로 합성

        Args:
            text: 합성할 텍스트
            streaming: 스트리밍 모드 (현재는 전체 합성만 지원)

        Returns:
            bytes: 오디오 데이터 (streaming=False일 때)
        """
        if not text or not text.strip():
            return None

        import time
        start_time = time.time()

        try:
            # 합성 시작 콜백
            if self._on_synthesis_start:
                self._on_synthesis_start(text)

            logger.debug(f"EdgeTTS synthesizing: {text[:50]}...")

            # EdgeTTS 통신 객체 생성
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.config.voice,
                rate=self.config.rate,
                volume=self.config.volume,
                pitch=self.config.pitch,
            )

            # 오디오 데이터 수집
            audio_chunks = []

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data = chunk["data"]
                    audio_chunks.append(audio_data)

                    # 스트리밍 모드면 청크 콜백 호출
                    if streaming and self._on_audio_chunk:
                        audio_chunk = AudioChunk(
                            data=audio_data,
                            sample_rate=24000,  # EdgeTTS는 24kHz
                            channels=1,
                            sample_width=2,
                            timestamp=time.time(),
                        )
                        self._on_audio_chunk(audio_chunk)

            # 전체 오디오 데이터
            full_audio = b"".join(audio_chunks)

            # 합성 종료 콜백
            synthesis_time = time.time() - start_time
            if self._on_synthesis_end:
                self._on_synthesis_end(synthesis_time)

            logger.debug(
                f"EdgeTTS synthesis completed: {len(full_audio)} bytes in {synthesis_time:.2f}s"
            )

            if streaming:
                return None
            else:
                return full_audio

        except Exception as e:
            logger.error(f"EdgeTTS synthesis error: {e}")
            raise

    @staticmethod
    async def list_voices(language: str = "ko") -> List[dict]:
        """
        사용 가능한 음성 목록 조회

        Args:
            language: 언어 코드 (ko, en 등)

        Returns:
            List[dict]: 음성 정보 리스트
        """
        if not EDGE_TTS_AVAILABLE:
            return []

        try:
            voices = await edge_tts.list_voices()

            # 해당 언어 필터링
            filtered_voices = [
                {
                    "name": v["Name"],
                    "short_name": v["ShortName"],
                    "gender": v["Gender"],
                    "locale": v["Locale"],
                }
                for v in voices
                if v["Locale"].startswith(language)
            ]

            return filtered_voices

        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []


# =============================================================================
# Naver Clova TTS 서비스 (선택적)
# =============================================================================

@dataclass
class NaverClovaTTSConfig:
    """Naver Clova TTS 설정"""
    client_id: str          # Naver Cloud Platform Client ID
    client_secret: str      # Naver Cloud Platform Client Secret
    speaker: str = "nara"   # 화자 (nara, nminyoung, nhajun 등)
    speed: int = 0          # 속도 (-5 ~ 5)
    pitch: int = 0          # 피치 (-5 ~ 5)
    format: str = "mp3"     # 포맷 (mp3, wav, pcm)


class NaverClovaTTSService:
    """
    Naver Clova TTS 서비스

    장점:
    - 가장 자연스러운 한국어 발음
    - 다양한 감정/스타일 표현
    - 한국어 특화
    - 빠른 응답 속도

    단점:
    - API 키 필요 (Naver Cloud Platform)
    - 유료 (무료 크레딧 제공)
    - 한국어 외 언어 제한적

    추천 사용 케이스:
    - 프로덕션 환경 (한국어)
    - 자연스러운 대화가 중요한 경우
    - 면접/고객 응대 시나리오
    """

    # 한국어 화자 옵션
    KOREAN_SPEAKERS = {
        "nara": "여성 (차분하고 명확한)",
        "nminyoung": "여성 (밝고 친근한)",
        "nhajun": "남성 (진중하고 차분한)",
        "ndain": "여성 (생동감 있는)",
        "njinho": "남성 (젊고 활기찬)",
    }

    def __init__(
        self,
        config: NaverClovaTTSConfig,
        on_audio_chunk: Optional[Callable[[AudioChunk], None]] = None,
        on_synthesis_start: Optional[Callable[[str], None]] = None,
        on_synthesis_end: Optional[Callable[[float], None]] = None,
    ):
        """
        Args:
            config: Naver Clova TTS 설정
            on_audio_chunk: 오디오 청크 콜백
            on_synthesis_start: 합성 시작 콜백
            on_synthesis_end: 합성 종료 콜백
        """
        self.config = config
        self._on_audio_chunk = on_audio_chunk
        self._on_synthesis_start = on_synthesis_start
        self._on_synthesis_end = on_synthesis_end

        logger.info(f"NaverClovaTTSService initialized: speaker={config.speaker}")

    async def synthesize(
        self,
        text: str,
        streaming: bool = False,
    ) -> Optional[bytes]:
        """
        텍스트를 음성으로 합성

        Args:
            text: 합성할 텍스트
            streaming: 스트리밍 모드 (현재 미지원)

        Returns:
            bytes: 오디오 데이터
        """
        if not text or not text.strip():
            return None

        import time
        import urllib.request
        import urllib.parse

        start_time = time.time()

        try:
            # 합성 시작 콜백
            if self._on_synthesis_start:
                self._on_synthesis_start(text)

            logger.debug(f"Naver Clova TTS synthesizing: {text[:50]}...")

            # API 엔드포인트
            url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"

            # 요청 데이터
            data = {
                "speaker": self.config.speaker,
                "speed": self.config.speed,
                "pitch": self.config.pitch,
                "format": self.config.format,
                "text": text,
            }

            # HTTP 요청
            request = urllib.request.Request(url)
            request.add_header("X-NCP-APIGW-API-KEY-ID", self.config.client_id)
            request.add_header("X-NCP-APIGW-API-KEY", self.config.client_secret)
            request.add_header("Content-Type", "application/x-www-form-urlencoded")

            # 비동기 실행
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: urllib.request.urlopen(
                    request, data=urllib.parse.urlencode(data).encode("utf-8")
                )
            )

            # 오디오 데이터 읽기
            audio_data = response.read()

            # 합성 종료 콜백
            synthesis_time = time.time() - start_time
            if self._on_synthesis_end:
                self._on_synthesis_end(synthesis_time)

            logger.debug(
                f"Naver Clova TTS completed: {len(audio_data)} bytes in {synthesis_time:.2f}s"
            )

            # 콜백 호출
            if self._on_audio_chunk:
                audio_chunk = AudioChunk(
                    data=audio_data,
                    sample_rate=24000,  # Clova TTS 기본값
                    channels=1,
                    sample_width=2,
                    timestamp=time.time(),
                )
                self._on_audio_chunk(audio_chunk)

            return audio_data

        except Exception as e:
            logger.error(f"Naver Clova TTS error: {e}")
            raise


# =============================================================================
# TTS Factory (통합 인터페이스)
# =============================================================================

class TTSFactory:
    """
    TTS 서비스 팩토리

    설정에 따라 적절한 TTS 서비스를 반환하고,
    폴백 체인을 구성합니다.
    """

    @staticmethod
    def create(
        provider: TTSProvider = TTSProvider.EDGE_TTS,
        fallback_chain: bool = True,
        **kwargs
    ):
        """
        TTS 서비스 생성

        Args:
            provider: TTS 제공자
            fallback_chain: 폴백 체인 활성화
            **kwargs: 제공자별 설정

        Returns:
            TTS 서비스 인스턴스
        """
        if provider == TTSProvider.ELEVENLABS:
            from .elevenlabs_service import create_elevenlabs_tts
            return create_elevenlabs_tts(**kwargs)

        elif provider == TTSProvider.EDGE_TTS:
            if not EDGE_TTS_AVAILABLE:
                logger.error("EdgeTTS not available")
                if fallback_chain:
                    logger.info("Falling back to ElevenLabs")
                    return TTSFactory.create(TTSProvider.ELEVENLABS, fallback_chain=False, **kwargs)
                raise RuntimeError("EdgeTTS not available and fallback disabled")

            config = kwargs.get("config", EdgeTTSConfig())
            return EdgeTTSService(
                config=config,
                on_audio_chunk=kwargs.get("on_audio_chunk"),
                on_synthesis_start=kwargs.get("on_synthesis_start"),
                on_synthesis_end=kwargs.get("on_synthesis_end"),
            )

        elif provider == TTSProvider.NAVER_CLOVA:
            config = kwargs.get("config")
            if not config:
                raise ValueError("NaverClovaTTSConfig is required for Naver Clova TTS")

            return NaverClovaTTSService(
                config=config,
                on_audio_chunk=kwargs.get("on_audio_chunk"),
                on_synthesis_start=kwargs.get("on_synthesis_start"),
                on_synthesis_end=kwargs.get("on_synthesis_end"),
            )

        elif provider == TTSProvider.FALLBACK:
            # 폴백 체인: ElevenLabs → EdgeTTS
            return FallbackTTSService(fallback_chain=["elevenlabs", "edge_tts"])

        else:
            raise ValueError(f"Unknown TTS provider: {provider}")

    @staticmethod
    def create_best_korean_tts(**kwargs):
        """
        한국어에 최적화된 TTS 서비스 생성

        우선순위:
        1. Naver Clova (설정된 경우)
        2. ElevenLabs (API 키 있는 경우)
        3. EdgeTTS (무료 폴백)

        Returns:
            TTS 서비스 인스턴스
        """
        # Naver Clova 설정이 있으면 우선 사용
        if "naver_config" in kwargs and kwargs["naver_config"]:
            logger.info("Using Naver Clova TTS (best for Korean)")
            return TTSFactory.create(TTSProvider.NAVER_CLOVA, config=kwargs["naver_config"])

        # ElevenLabs API 키가 있으면 사용
        try:
            from config import settings
            if settings.elevenlabs_api_key and settings.elevenlabs_api_key != "your_elevenlabs_api_key_here":
                logger.info("Using ElevenLabs TTS")
                return TTSFactory.create(TTSProvider.ELEVENLABS, **kwargs)
        except:
            pass

        # 폴백: EdgeTTS (무료)
        logger.info("Using EdgeTTS (free fallback)")
        return TTSFactory.create(TTSProvider.EDGE_TTS, **kwargs)


class FallbackTTSService:
    """
    폴백 체인 TTS 서비스

    여러 TTS 서비스를 순차적으로 시도하여
    안정성을 높입니다.
    """

    def __init__(self, fallback_chain: List[str] = None):
        """
        Args:
            fallback_chain: 폴백 체인 (순서대로 시도)
        """
        self.fallback_chain = fallback_chain or ["elevenlabs", "edge_tts"]
        self._services = {}
        self._current_provider = None

        logger.info(f"FallbackTTSService initialized: chain={self.fallback_chain}")

    def _get_service(self, provider_name: str):
        """TTS 서비스 가져오기 (캐싱)"""
        if provider_name not in self._services:
            try:
                provider = TTSProvider(provider_name)
                self._services[provider_name] = TTSFactory.create(provider, fallback_chain=False)
            except Exception as e:
                logger.error(f"Failed to create {provider_name} service: {e}")
                return None

        return self._services.get(provider_name)

    async def synthesize(self, text: str, streaming: bool = True) -> Optional[bytes]:
        """
        폴백 체인으로 텍스트를 음성으로 합성

        Args:
            text: 합성할 텍스트
            streaming: 스트리밍 모드

        Returns:
            bytes: 오디오 데이터
        """
        last_error = None

        for provider_name in self.fallback_chain:
            service = self._get_service(provider_name)

            if not service:
                continue

            try:
                logger.debug(f"Trying {provider_name}...")
                result = await service.synthesize(text, streaming=streaming)
                self._current_provider = provider_name
                logger.info(f"TTS succeeded with {provider_name}")
                return result

            except Exception as e:
                logger.warning(f"{provider_name} failed: {e}")
                last_error = e
                continue

        # 모든 서비스 실패
        logger.error("All TTS services in fallback chain failed")
        if last_error:
            raise last_error
        else:
            raise RuntimeError("No TTS service available")


# =============================================================================
# 헬퍼 함수
# =============================================================================

def create_korean_tts(
    provider: str = "auto",
    voice_gender: str = "female",
    **kwargs
):
    """
    한국어 TTS 서비스 생성 헬퍼

    Args:
        provider: "auto", "elevenlabs", "edge", "naver"
        voice_gender: "female" 또는 "male"
        **kwargs: 추가 설정

    Returns:
        TTS 서비스 인스턴스
    """
    if provider == "auto":
        return TTSFactory.create_best_korean_tts(**kwargs)

    elif provider == "edge":
        config = EdgeTTSConfig(
            voice=EdgeTTSService.KOREAN_VOICES.get(voice_gender, "ko-KR-SunHiNeural")
        )
        return EdgeTTSService(config=config, **kwargs)

    elif provider == "elevenlabs":
        from .elevenlabs_service import create_elevenlabs_tts
        return create_elevenlabs_tts(**kwargs)

    elif provider == "naver":
        # Naver 설정 필요
        naver_config = kwargs.get("config")
        if not naver_config:
            raise ValueError("NaverClovaTTSConfig required for Naver TTS")
        return NaverClovaTTSService(config=naver_config, **kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# 테스트/예시 코드
# =============================================================================

if __name__ == "__main__":
    """한국어 TTS 서비스 테스트"""
    import asyncio

    async def test_korean_tts():
        """한국어 TTS 테스트"""

        print("=" * 80)
        print("Korean TTS Services Test")
        print("=" * 80)

        test_text = "안녕하세요. 한국어 음성 합성 테스트입니다."

        # 1. EdgeTTS 테스트
        if EDGE_TTS_AVAILABLE:
            print("\n[1] Testing EdgeTTS (Free)...")
            edge_tts = EdgeTTSService()

            audio = await edge_tts.synthesize(test_text, streaming=False)
            print(f"EdgeTTS: {len(audio) if audio else 0} bytes")

            # 사용 가능한 한국어 음성 조회
            voices = await EdgeTTSService.list_voices("ko")
            print(f"Available Korean voices: {len(voices)}")
            for v in voices[:3]:
                print(f"  - {v['name']} ({v['gender']})")

        # 2. Auto 모드 (최적 선택)
        print("\n[2] Testing Auto Mode (Best for Korean)...")
        auto_tts = create_korean_tts(provider="auto")
        audio = await auto_tts.synthesize(test_text, streaming=False)
        print(f"Auto TTS: {len(audio) if audio else 0} bytes")

        # 3. 폴백 체인 테스트
        print("\n[3] Testing Fallback Chain...")
        fallback_tts = FallbackTTSService()
        audio = await fallback_tts.synthesize(test_text, streaming=False)
        print(f"Fallback TTS: {len(audio) if audio else 0} bytes")
        print(f"Used provider: {fallback_tts._current_provider}")

    # 테스트 실행
    asyncio.run(test_korean_tts())
