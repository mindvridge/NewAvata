"""
립싱크 아바타 API 클라이언트

사용법:
    from lipsync_client import LipsyncClient

    client = LipsyncClient("http://localhost:5000")

    # 서버 상태 확인
    status = client.get_status()

    # 립싱크 비디오 생성 (동기)
    result = client.generate("안녕하세요")
    print(result['video_url'])

    # 립싱크 비디오 생성 (스트리밍)
    for event in client.generate_stream("안녕하세요"):
        print(event)
"""

import requests
import json
from typing import Generator, Dict, Any, Optional


class LipsyncClient:
    """립싱크 아바타 API 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Args:
            base_url: API 서버 주소 (기본: http://localhost:5000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def get_status(self) -> Dict[str, Any]:
        """서버 상태 확인"""
        response = self.session.get(f"{self.base_url}/api/v2/status")
        response.raise_for_status()
        return response.json()

    def get_tts_engines(self) -> list:
        """TTS 엔진 목록 조회"""
        response = self.session.get(f"{self.base_url}/api/tts_engines")
        response.raise_for_status()
        return response.json()

    def get_avatars(self) -> Dict[str, Any]:
        """아바타 목록 조회"""
        response = self.session.get(f"{self.base_url}/api/avatars")
        response.raise_for_status()
        return response.json()

    def get_queue_status(self) -> Dict[str, Any]:
        """대기열/GPU 상태 조회"""
        response = self.session.get(f"{self.base_url}/api/queue_status")
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        text: str,
        resolution: str = "720p",
        tts_engine: Optional[str] = None,
        tts_voice: Optional[str] = None,
        frame_skip: int = 1,
        output_format: str = "mp4",
        persist: bool = False
    ) -> Dict[str, Any]:
        """
        립싱크 비디오 생성 (동기)

        Args:
            text: 음성으로 변환할 텍스트
            resolution: 해상도 (720p, 480p, 360p)
            tts_engine: TTS 엔진 (cosyvoice, elevenlabs)
            tts_voice: 음성 종류
            frame_skip: 프레임 스킵 (1=없음, 2=절반)
            output_format: 출력 포맷 (mp4, webm)
            persist: 영구 저장 여부 (True=녹화모드, False=대화모드)

        Returns:
            {
                "success": True,
                "video_url": "/video/output.mp4",
                "video_path": "results/realtime/output.mp4",
                "elapsed": 9.7,
                "persistent": False
            }
        """
        payload = {
            "text": text,
            "resolution": resolution,
            "frame_skip": frame_skip,
            "output_format": output_format,
            "persist": persist
        }
        if tts_engine:
            payload["tts_engine"] = tts_engine
        if tts_voice:
            payload["tts_voice"] = tts_voice

        response = self.session.post(
            f"{self.base_url}/api/v2/lipsync",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def generate_stream(
        self,
        text: str,
        resolution: str = "720p",
        tts_engine: Optional[str] = None,
        tts_voice: Optional[str] = None,
        frame_skip: int = 1,
        output_format: str = "mp4",
        persist: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        립싱크 비디오 생성 (스트리밍)

        Args:
            text: 음성으로 변환할 텍스트
            resolution: 해상도 (720p, 480p, 360p)
            tts_engine: TTS 엔진 (cosyvoice, elevenlabs)
            tts_voice: 음성 종류
            frame_skip: 프레임 스킵 (1=없음, 2=절반)
            output_format: 출력 포맷 (mp4, webm)
            persist: 영구 저장 여부 (True=녹화모드, False=대화모드)

        Yields:
            SSE 이벤트:
            - {"type": "status", "stage": "tts", "message": "..."}
            - {"type": "status", "stage": "tts_done", "audio_url": "...", "duration": 3.5}
            - {"type": "status", "stage": "lipsync", "message": "..."}
            - {"type": "done", "video_url": "...", "persistent": false, "elapsed": {...}}
            - {"type": "error", "message": "..."}
        """
        payload = {
            "text": text,
            "resolution": resolution,
            "frame_skip": frame_skip,
            "output_format": output_format,
            "persist": persist
        }
        if tts_engine:
            payload["tts_engine"] = tts_engine
        if tts_voice:
            payload["tts_voice"] = tts_voice

        response = self.session.post(
            f"{self.base_url}/api/v2/lipsync/stream",
            json=payload,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])
                        yield data
                    except json.JSONDecodeError:
                        continue

    def download_video(self, video_url: str, output_path: str) -> str:
        """
        비디오 다운로드

        Args:
            video_url: 비디오 URL (/video/xxx.mp4)
            output_path: 저장 경로

        Returns:
            저장된 파일 경로
        """
        if video_url.startswith('/'):
            url = f"{self.base_url}{video_url}"
        else:
            url = video_url

        response = self.session.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path

    def get_video_url(self, video_path: str) -> str:
        """상대 URL을 절대 URL로 변환"""
        if video_path.startswith('/'):
            return f"{self.base_url}{video_path}"
        return video_path


def main():
    """예시: 립싱크 비디오 생성"""
    import argparse

    parser = argparse.ArgumentParser(description="립싱크 아바타 API 클라이언트")
    parser.add_argument("text", help="음성으로 변환할 텍스트")
    parser.add_argument("--url", default="http://localhost:5000", help="API 서버 주소")
    parser.add_argument("--resolution", default="720p", choices=["720p", "480p", "360p"])
    parser.add_argument("--output", "-o", help="출력 파일 경로")
    parser.add_argument("--sync", action="store_true", help="동기 모드 사용")
    parser.add_argument("--persist", action="store_true", help="영구 저장 (녹화 모드)")

    args = parser.parse_args()

    client = LipsyncClient(args.url)

    # 서버 상태 확인
    try:
        status = client.get_status()
        print(f"[서버] 상태: {status['status']}, 모델 로드: {status['models_loaded']}")
    except Exception as e:
        print(f"[오류] 서버 연결 실패: {e}")
        return

    mode = "녹화 모드 (영구 저장)" if args.persist else "대화 모드 (임시)"
    print(f"[모드] {mode}")

    # 립싱크 생성
    if args.sync:
        print(f"[생성] 동기 모드로 생성 중...")
        result = client.generate(args.text, resolution=args.resolution, persist=args.persist)
        print(f"[완료] 비디오: {result['video_url']} ({result['elapsed']:.1f}초)")
        print(f"  영구저장: {result.get('persistent', False)}")
        video_url = result['video_url']
    else:
        print(f"[생성] 스트리밍 모드로 생성 중...")
        video_url = None
        for event in client.generate_stream(args.text, resolution=args.resolution, persist=args.persist):
            if event['type'] == 'status':
                print(f"  [{event.get('stage', '')}] {event['message']}")
            elif event['type'] == 'done':
                video_url = event['video_url']
                elapsed = event.get('elapsed', {})
                total = elapsed.get('total', 0) if isinstance(elapsed, dict) else elapsed
                print(f"[완료] 비디오: {video_url} ({total:.1f}초)")
                print(f"  영구저장: {event.get('persistent', False)}")
            elif event['type'] == 'error':
                print(f"[오류] {event['message']}")
                return

    # 다운로드
    if args.output and video_url:
        print(f"[다운로드] {args.output}")
        client.download_video(video_url, args.output)
        print(f"[저장] {args.output}")


if __name__ == "__main__":
    main()
