"""
Pytest 설정 및 공통 픽스처

전역 설정과 모든 테스트에서 사용할 수 있는 공통 픽스처를 정의합니다.
"""

import pytest
import numpy as np
from pathlib import Path


def pytest_configure(config):
    """Pytest 설정"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def sample_audio_file(tmp_path_factory):
    """
    테스트용 샘플 오디오 파일 생성

    16kHz, 1초, 440Hz 사인파 (A4 음)
    """
    import wave

    # 임시 디렉토리에 파일 생성
    tmp_dir = tmp_path_factory.mktemp("audio")
    audio_file = tmp_dir / "sample.wav"

    # 오디오 파라미터
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 음

    # 사인파 생성
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    audio = np.sin(2 * np.pi * frequency * t)

    # int16으로 변환
    audio_int16 = (audio * 32767).astype(np.int16)

    # WAV 파일 저장
    with wave.open(str(audio_file), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return audio_file


@pytest.fixture
def korean_sample_text():
    """테스트용 한국어 샘플 텍스트"""
    return [
        "안녕하세요",
        "저는 면접에 참여하게 되어 기쁩니다",
        "제 강점은 빠른 학습 능력입니다",
        "팀워크를 중요하게 생각합니다",
        "감사합니다",
    ]


@pytest.fixture
def mock_env_vars(monkeypatch):
    """테스트용 환경 변수 모킹"""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test_deepgram_key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_elevenlabs_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("DAILY_API_KEY", "test_daily_key")
