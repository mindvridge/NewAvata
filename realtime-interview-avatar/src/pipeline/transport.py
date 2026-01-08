"""
WebRTC Transport Layer
Daily.co 및 로컬 WebRTC 전송 계층
"""

import asyncio
import json
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

from loguru import logger

# Pipecat Daily.co transport - disabled on Windows
# daily-python package causes import errors even with try-except
DAILY_AVAILABLE = False
DailyTransport = None
DailyParams = None
DailyDialinSettings = None
logger.info("Daily transport disabled (Windows compatibility)")

# aiortc for local WebRTC
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
    from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaRelay
    from aiortc import VideoStreamTrack, AudioStreamTrack
    import websockets
    AIORTC_AVAILABLE = True
except ImportError:
    logger.warning("aiortc not available. Install: pip install aiortc")
    AIORTC_AVAILABLE = False

from config.settings import get_settings


class TransportType(Enum):
    """전송 타입"""
    DAILY = "daily"           # Daily.co (프로덕션 권장)
    LOCAL = "local"           # 로컬 WebRTC (테스트용)
    MOCK = "mock"             # Mock (개발용)


class NetworkQuality(Enum):
    """네트워크 품질"""
    EXCELLENT = "excellent"   # 손실 <1%, 지연 <50ms
    GOOD = "good"            # 손실 <3%, 지연 <100ms
    FAIR = "fair"            # 손실 <5%, 지연 <200ms
    POOR = "poor"            # 손실 >5% 또는 지연 >200ms


@dataclass
class TransportConfig:
    """전송 설정"""
    # 전송 타입
    transport_type: TransportType = TransportType.DAILY

    # Daily.co 설정
    daily_room_url: Optional[str] = None
    daily_token: Optional[str] = None
    daily_bot_name: str = "AI Interviewer"

    # 로컬 WebRTC 설정
    signaling_server_url: str = "ws://localhost:8765"
    stun_servers: list = None
    turn_servers: list = None

    # 오디오 설정
    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    # 비디오 설정
    video_out_enabled: bool = True
    video_out_width: int = 512
    video_out_height: int = 512
    video_out_fps: int = 25
    video_out_codec: str = "H264"
    video_out_bitrate: int = 1000000  # 1 Mbps

    # 품질 적응
    enable_quality_adaptation: bool = True
    min_video_width: int = 256
    max_video_width: int = 1024

    def __post_init__(self):
        # 기본 STUN 서버
        if self.stun_servers is None:
            self.stun_servers = [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
            ]


# ============================================================================
# Daily.co Transport Wrapper
# ============================================================================

class DailyTransportWrapper:
    """
    Daily.co WebRTC 전송 래퍼

    Pipecat DailyTransport를 래핑하여 추가 기능 제공
    """

    def __init__(self, config: TransportConfig):
        """
        Args:
            config: 전송 설정
        """
        if not DAILY_AVAILABLE:
            raise ImportError("Pipecat Daily transport not available")

        self.config = config

        # Daily 파라미터
        self.daily_params = DailyParams(
            audio_in_enabled=config.audio_in_enabled,
            audio_out_enabled=config.audio_out_enabled,
            audio_out_sample_rate=config.audio_sample_rate,
            video_out_enabled=config.video_out_enabled,
            video_out_width=config.video_out_width,
            video_out_height=config.video_out_height,
            video_out_bitrate=config.video_out_bitrate,
            vad_enabled=True,
            transcription_enabled=True,
        )

        # Daily Transport
        self.transport = DailyTransport(
            room_url=config.daily_room_url,
            token=config.daily_token,
            bot_name=config.daily_bot_name,
            params=self.daily_params
        )

        # 품질 모니터링
        self._network_quality = NetworkQuality.GOOD
        self._quality_stats = {
            'packet_loss': 0.0,
            'latency_ms': 0.0,
            'jitter_ms': 0.0,
        }

        logger.info(
            f"DailyTransportWrapper initialized: "
            f"room={config.daily_room_url}"
        )

    async def start(self):
        """전송 시작"""
        await self.transport.start()
        logger.info("Daily transport started")

    async def stop(self):
        """전송 중지"""
        await self.transport.stop()
        logger.info("Daily transport stopped")

    def on_participant_joined(self, handler: Callable):
        """참가자 입장 이벤트 핸들러"""
        self.transport.on("participant_joined", handler)

    def on_participant_left(self, handler: Callable):
        """참가자 퇴장 이벤트 핸들러"""
        self.transport.on("participant_left", handler)

    def on_error(self, handler: Callable):
        """에러 이벤트 핸들러"""
        self.transport.on("error", handler)

    async def update_video_quality(self, width: int, height: int, bitrate: int):
        """
        비디오 품질 동적 조정

        Args:
            width: 너비
            height: 높이
            bitrate: 비트레이트
        """
        logger.info(f"Updating video quality: {width}x{height} @ {bitrate}bps")

        # Daily API를 통해 설정 업데이트
        try:
            await self.transport.update_subscription_profiles({
                "base": {
                    "camera": {
                        "width": width,
                        "height": height,
                        "bitrate": bitrate
                    }
                }
            })
        except Exception as e:
            logger.error(f"Failed to update video quality: {e}")

    async def monitor_network_quality(self):
        """네트워크 품질 모니터링"""
        while True:
            try:
                # Daily API에서 통계 가져오기
                stats = await self.transport.get_network_stats()

                if stats:
                    # 패킷 손실률
                    packet_loss = stats.get('video', {}).get('packetLossPercent', 0)

                    # 지연
                    latency = stats.get('video', {}).get('roundTripTime', 0) * 1000  # ms

                    # Jitter
                    jitter = stats.get('video', {}).get('jitter', 0) * 1000  # ms

                    # 통계 업데이트
                    self._quality_stats = {
                        'packet_loss': packet_loss,
                        'latency_ms': latency,
                        'jitter_ms': jitter,
                    }

                    # 품질 등급 결정
                    old_quality = self._network_quality

                    if packet_loss < 1 and latency < 50:
                        self._network_quality = NetworkQuality.EXCELLENT
                    elif packet_loss < 3 and latency < 100:
                        self._network_quality = NetworkQuality.GOOD
                    elif packet_loss < 5 and latency < 200:
                        self._network_quality = NetworkQuality.FAIR
                    else:
                        self._network_quality = NetworkQuality.POOR

                    # 품질 변경 시 로깅
                    if old_quality != self._network_quality:
                        logger.info(
                            f"Network quality changed: {old_quality.value} → {self._network_quality.value}"
                        )

                    # 적응형 품질 조정
                    if self.config.enable_quality_adaptation:
                        await self._adapt_quality()

            except Exception as e:
                logger.error(f"Network quality monitoring failed: {e}")

            # 5초마다 체크
            await asyncio.sleep(5)

    async def _adapt_quality(self):
        """품질 적응"""
        # 현재 설정
        current_width = self.config.video_out_width
        current_bitrate = self.config.video_out_bitrate

        # 품질에 따른 조정
        if self._network_quality == NetworkQuality.POOR:
            # 해상도 낮춤
            new_width = max(self.config.min_video_width, current_width // 2)
            new_height = new_width  # 정사각형 유지
            new_bitrate = current_bitrate // 2

            await self.update_video_quality(new_width, new_height, new_bitrate)

        elif self._network_quality == NetworkQuality.EXCELLENT:
            # 해상도 높임
            new_width = min(self.config.max_video_width, current_width * 2)
            new_height = new_width
            new_bitrate = min(2000000, current_bitrate * 2)  # 최대 2Mbps

            await self.update_video_quality(new_width, new_height, new_bitrate)

    def get_network_quality(self) -> NetworkQuality:
        """네트워크 품질 반환"""
        return self._network_quality

    def get_quality_stats(self) -> Dict[str, float]:
        """품질 통계 반환"""
        return self._quality_stats.copy()


# ============================================================================
# Local WebRTC Transport (aiortc)
# ============================================================================

class LocalWebRTCTransport:
    """
    로컬 WebRTC 전송 (aiortc 기반)

    로컬 테스트 및 개발용
    """

    def __init__(self, config: TransportConfig):
        """
        Args:
            config: 전송 설정
        """
        if not AIORTC_AVAILABLE:
            raise ImportError("aiortc not available")

        self.config = config

        # RTCPeerConnection
        self.pc = None

        # ICE 설정
        ice_servers = []

        # STUN 서버
        for stun_url in config.stun_servers:
            ice_servers.append(RTCIceServer(urls=stun_url))

        # TURN 서버
        if config.turn_servers:
            for turn_config in config.turn_servers:
                ice_servers.append(RTCIceServer(
                    urls=turn_config['url'],
                    username=turn_config.get('username'),
                    credential=turn_config.get('credential')
                ))

        self.rtc_config = RTCConfiguration(iceServers=ice_servers)

        # 시그널링
        self.signaling_ws = None

        # 트랙
        self.audio_track = None
        self.video_track = None

        # 이벤트 핸들러
        self._event_handlers = {
            'on_participant_joined': [],
            'on_participant_left': [],
            'on_error': [],
        }

        logger.info("LocalWebRTCTransport initialized")

    async def start(self):
        """전송 시작"""
        # PeerConnection 생성
        self.pc = RTCPeerConnection(configuration=self.rtc_config)

        # 이벤트 핸들러
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state: {self.pc.connectionState}")

            if self.pc.connectionState == "connected":
                await self._emit_event('on_participant_joined', {'id': 'remote'})

            elif self.pc.connectionState == "failed":
                await self._emit_event('on_error', "Connection failed")

            elif self.pc.connectionState == "closed":
                await self._emit_event('on_participant_left', {'id': 'remote'})

        @self.pc.on("track")
        def on_track(track):
            logger.info(f"Track received: {track.kind}")

            if track.kind == "audio":
                # 오디오 트랙 처리
                pass

        # 시그널링 서버 연결
        await self._connect_signaling()

        logger.info("Local WebRTC transport started")

    async def stop(self):
        """전송 중지"""
        if self.pc:
            await self.pc.close()

        if self.signaling_ws:
            await self.signaling_ws.close()

        logger.info("Local WebRTC transport stopped")

    async def _connect_signaling(self):
        """시그널링 서버 연결"""
        try:
            self.signaling_ws = await websockets.connect(
                self.config.signaling_server_url
            )

            logger.info(f"Connected to signaling server: {self.config.signaling_server_url}")

            # 시그널링 메시지 처리
            asyncio.create_task(self._handle_signaling())

        except Exception as e:
            logger.error(f"Signaling connection failed: {e}")

    async def _handle_signaling(self):
        """시그널링 메시지 처리"""
        try:
            async for message in self.signaling_ws:
                data = json.loads(message)

                msg_type = data.get('type')

                if msg_type == 'offer':
                    # Offer 수신
                    await self._handle_offer(data['sdp'])

                elif msg_type == 'answer':
                    # Answer 수신
                    await self._handle_answer(data['sdp'])

                elif msg_type == 'ice':
                    # ICE candidate 수신
                    await self._handle_ice(data['candidate'])

        except Exception as e:
            logger.error(f"Signaling error: {e}")

    async def _handle_offer(self, sdp: str):
        """Offer 처리"""
        # Remote description 설정
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type="offer")
        )

        # Answer 생성
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        # Answer 전송
        await self.signaling_ws.send(json.dumps({
            'type': 'answer',
            'sdp': self.pc.localDescription.sdp
        }))

        logger.info("Answer sent")

    async def _handle_answer(self, sdp: str):
        """Answer 처리"""
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type="answer")
        )

        logger.info("Answer received")

    async def _handle_ice(self, candidate: dict):
        """ICE candidate 처리"""
        # ICE candidate 추가
        # await self.pc.addIceCandidate(candidate)
        pass

    def on_participant_joined(self, handler: Callable):
        """참가자 입장 핸들러"""
        self._event_handlers['on_participant_joined'].append(handler)

    def on_participant_left(self, handler: Callable):
        """참가자 퇴장 핸들러"""
        self._event_handlers['on_participant_left'].append(handler)

    def on_error(self, handler: Callable):
        """에러 핸들러"""
        self._event_handlers['on_error'].append(handler)

    async def _emit_event(self, event_name: str, *args, **kwargs):
        """이벤트 발생"""
        for handler in self._event_handlers.get(event_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Event handler error ({event_name}): {e}")


# ============================================================================
# Mock Transport (개발용)
# ============================================================================

class MockTransport:
    """
    Mock WebRTC 전송

    WebRTC 없이 로컬 테스트용
    """

    def __init__(self, config: TransportConfig):
        """
        Args:
            config: 전송 설정
        """
        self.config = config

        # 이벤트 핸들러
        self._event_handlers = {
            'on_participant_joined': [],
            'on_participant_left': [],
            'on_error': [],
        }

        logger.info("MockTransport initialized")

    async def start(self):
        """전송 시작"""
        logger.info("Mock transport started")

        # 가짜 참가자 입장
        await asyncio.sleep(1)
        await self._emit_event('on_participant_joined', {'id': 'mock_user', 'name': 'Test User'})

    async def stop(self):
        """전송 중지"""
        logger.info("Mock transport stopped")

    def on_participant_joined(self, handler: Callable):
        """참가자 입장 핸들러"""
        self._event_handlers['on_participant_joined'].append(handler)

    def on_participant_left(self, handler: Callable):
        """참가자 퇴장 핸들러"""
        self._event_handlers['on_participant_left'].append(handler)

    def on_error(self, handler: Callable):
        """에러 핸들러"""
        self._event_handlers['on_error'].append(handler)

    async def _emit_event(self, event_name: str, *args, **kwargs):
        """이벤트 발생"""
        for handler in self._event_handlers.get(event_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Event handler error ({event_name}): {e}")


# ============================================================================
# Transport Factory
# ============================================================================

def create_transport(config: TransportConfig):
    """
    전송 생성

    Args:
        config: 전송 설정

    Returns:
        Transport 인스턴스
    """
    if config.transport_type == TransportType.DAILY:
        if not DAILY_AVAILABLE:
            logger.warning("Daily transport not available. Using mock transport.")
            return MockTransport(config)
        return DailyTransportWrapper(config)

    elif config.transport_type == TransportType.LOCAL:
        if not AIORTC_AVAILABLE:
            logger.warning("aiortc not available. Using mock transport.")
            return MockTransport(config)
        return LocalWebRTCTransport(config)

    elif config.transport_type == TransportType.MOCK:
        return MockTransport(config)

    else:
        raise ValueError(f"Unknown transport type: {config.transport_type}")


# 헬퍼 함수
def create_daily_transport(
    room_url: str,
    token: Optional[str] = None,
    bot_name: str = "AI Interviewer"
) -> DailyTransportWrapper:
    """
    Daily.co 전송 생성 (간편 함수)

    Args:
        room_url: Daily 룸 URL
        token: Daily 토큰
        bot_name: 봇 이름

    Returns:
        DailyTransportWrapper
    """
    config = TransportConfig(
        transport_type=TransportType.DAILY,
        daily_room_url=room_url,
        daily_token=token,
        daily_bot_name=bot_name
    )

    return DailyTransportWrapper(config)


def create_local_transport(
    signaling_server_url: str = "ws://localhost:8765"
) -> LocalWebRTCTransport:
    """
    로컬 WebRTC 전송 생성 (간편 함수)

    Args:
        signaling_server_url: 시그널링 서버 URL

    Returns:
        LocalWebRTCTransport
    """
    config = TransportConfig(
        transport_type=TransportType.LOCAL,
        signaling_server_url=signaling_server_url
    )

    return LocalWebRTCTransport(config)


# 사용 예시
if __name__ == "__main__":
    async def main():
        # Daily.co 전송
        config = TransportConfig(
            transport_type=TransportType.DAILY,
            daily_room_url="https://your-domain.daily.co/room",
            daily_token="your-token",
            video_out_width=512,
            video_out_height=512,
            video_out_fps=25,
            enable_quality_adaptation=True
        )

        transport = create_transport(config)

        # 이벤트 핸들러
        @transport.on_participant_joined
        async def on_joined(participant):
            logger.info(f"Participant joined: {participant}")

        @transport.on_error
        async def on_error(error):
            logger.error(f"Transport error: {error}")

        # 시작
        await transport.start()

        # 네트워크 품질 모니터링
        if isinstance(transport, DailyTransportWrapper):
            asyncio.create_task(transport.monitor_network_quality())

        # 통계 출력
        while True:
            await asyncio.sleep(10)

            if isinstance(transport, DailyTransportWrapper):
                quality = transport.get_network_quality()
                stats = transport.get_quality_stats()

                logger.info(
                    f"Network: {quality.value}, "
                    f"Loss: {stats['packet_loss']:.1f}%, "
                    f"Latency: {stats['latency_ms']:.0f}ms"
                )

    asyncio.run(main())
