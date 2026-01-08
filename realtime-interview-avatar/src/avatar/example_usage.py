"""
MuseTalk Avatar 사용 예시
실시간 면접 아바타를 위한 MuseTalk 통합 예제
"""

import asyncio
import numpy as np
import cv2
from pathlib import Path

from musetalk_wrapper import (
    MuseTalkAvatar,
    MuseTalkConfig,
    AvatarState,
    create_musetalk_avatar,
)


# ============================================================================
# 예시 1: 기본 사용법
# ============================================================================
def example1_basic_usage():
    """기본적인 MuseTalk 아바타 사용"""
    print("=" * 70)
    print("예시 1: 기본 사용법")
    print("=" * 70)

    # 설정
    config = MuseTalkConfig(
        source_image_path="./assets/avatar_source.jpg",
        model_dir="./models/musetalk",
        resolution=256,
        use_face_enhance=False,
        use_fp16=True
    )

    # 아바타 생성
    avatar = MuseTalkAvatar(config)

    # 모델 로드
    avatar.load_models()

    # 소스 이미지 로드
    avatar.load_source_image()

    # 더미 오디오 생성 (16kHz, 40ms)
    audio_chunk = np.random.randn(640).astype(np.float32) * 0.1

    # 프레임 생성
    video_frame = avatar.process_audio_chunk(audio_chunk)

    print(f"생성된 프레임 크기: {video_frame.frame.shape}")
    print(f"현재 상태: {video_frame.state.value}")
    print(f"FPS: {video_frame.metadata['fps']:.1f}")
    print(f"처리 시간: {video_frame.metadata['processing_time_ms']:.1f}ms")

    # 프레임 저장 (옵션)
    output_path = "./output/frame_example1.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, video_frame.frame)
    print(f"프레임 저장됨: {output_path}\n")


# ============================================================================
# 예시 2: 헬퍼 함수 사용
# ============================================================================
def example2_helper_function():
    """헬퍼 함수로 간단하게 생성"""
    print("=" * 70)
    print("예시 2: 헬퍼 함수 사용")
    print("=" * 70)

    # 헬퍼 함수로 아바타 생성 (자동 초기화)
    avatar = create_musetalk_avatar(
        source_image_path="./assets/avatar_source.jpg",
        model_dir="./models/musetalk",
        resolution=256,
        use_face_enhance=False
    )

    # 대기 프레임 생성
    idle_frame = avatar.get_idle_frame()
    print(f"대기 프레임 크기: {idle_frame.shape}")

    # 생각 중 프레임 생성
    thinking_frame = avatar.get_thinking_frame()
    print(f"생각 중 프레임 크기: {thinking_frame.shape}\n")


# ============================================================================
# 예시 3: 실시간 스트리밍 시뮬레이션
# ============================================================================
def example3_streaming_simulation():
    """실시간 스트리밍 시뮬레이션 (25 FPS)"""
    print("=" * 70)
    print("예시 3: 실시간 스트리밍 시뮬레이션 (25 FPS)")
    print("=" * 70)

    import time

    # 아바타 생성
    avatar = create_musetalk_avatar(
        source_image_path="./assets/avatar_source.jpg",
        resolution=256
    )

    # 25 FPS로 100 프레임 생성
    num_frames = 100
    fps = 25
    frame_duration = 1.0 / fps

    frames = []
    start_time = time.time()

    for i in range(num_frames):
        frame_start = time.time()

        # 오디오 청크 생성 (40ms)
        audio_chunk = np.random.randn(640).astype(np.float32) * 0.05

        # 프레임 생성
        video_frame = avatar.process_audio_chunk(audio_chunk, timestamp=time.time())
        frames.append(video_frame.frame)

        # FPS 유지를 위한 대기
        elapsed = time.time() - frame_start
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)

        if (i + 1) % 25 == 0:
            print(f"진행: {i+1}/{num_frames} 프레임 ({(i+1)/num_frames*100:.1f}%)")

    total_time = time.time() - start_time
    actual_fps = num_frames / total_time

    print(f"\n총 처리 시간: {total_time:.2f}초")
    print(f"실제 FPS: {actual_fps:.1f}")
    print(f"생성된 프레임 수: {len(frames)}\n")

    # 비디오로 저장 (옵션)
    # save_video(frames, "./output/streaming_example.mp4", fps=25)


# ============================================================================
# 예시 4: 상태 전환
# ============================================================================
def example4_state_transitions():
    """아바타 상태 전환 예시"""
    print("=" * 70)
    print("예시 4: 아바타 상태 전환")
    print("=" * 70)

    avatar = create_musetalk_avatar(
        source_image_path="./assets/avatar_source.jpg",
        resolution=256
    )

    # 각 상태별 프레임 생성
    states = [
        (AvatarState.IDLE, "대기 중"),
        (AvatarState.SPEAKING, "말하는 중"),
        (AvatarState.THINKING, "생각 중"),
        (AvatarState.LISTENING, "듣는 중")
    ]

    for state, description in states:
        avatar.set_state(state)

        # 상태에 따른 오디오 생성
        if state == AvatarState.SPEAKING:
            # 큰 소리 (말하는 중)
            audio_chunk = np.random.randn(640).astype(np.float32) * 0.2
        else:
            # 무음
            audio_chunk = np.zeros(640, dtype=np.float32)

        video_frame = avatar.process_audio_chunk(audio_chunk)

        print(f"{description}: {video_frame.state.value}")
        print(f"  - 오디오 에너지: {video_frame.metadata['audio_energy']:.4f}")
        print(f"  - FPS: {video_frame.metadata['fps']:.1f}\n")


# ============================================================================
# 예시 5: 고해상도 + 얼굴 향상
# ============================================================================
def example5_high_quality():
    """고해상도 + 얼굴 향상"""
    print("=" * 70)
    print("예시 5: 고해상도 + 얼굴 향상")
    print("=" * 70)

    # 고해상도 설정
    config = MuseTalkConfig(
        source_image_path="./assets/avatar_source.jpg",
        resolution=512,  # 512x512
        use_face_enhance=True,  # GFPGAN 얼굴 향상
        smooth_factor=0.7,  # 강한 스무딩
        use_fp16=True
    )

    avatar = MuseTalkAvatar(config)
    avatar.load_models()
    avatar.load_source_image()

    # 오디오 청크
    audio_chunk = np.random.randn(640).astype(np.float32) * 0.1

    # 프레임 생성
    video_frame = avatar.process_audio_chunk(audio_chunk)

    print(f"해상도: {video_frame.frame.shape}")
    print(f"얼굴 향상: {config.use_face_enhance}")
    print(f"스무딩 계수: {config.smooth_factor}")
    print(f"처리 시간: {video_frame.metadata['processing_time_ms']:.1f}ms\n")


# ============================================================================
# 예시 6: 배치 처리
# ============================================================================
def example6_batch_processing():
    """배치 처리로 성능 향상"""
    print("=" * 70)
    print("예시 6: 배치 처리")
    print("=" * 70)

    import time

    # 배치 처리 설정
    config = MuseTalkConfig(
        source_image_path="./assets/avatar_source.jpg",
        resolution=256,
        batch_size=4,  # 4개 프레임 동시 처리
        use_fp16=True
    )

    avatar = MuseTalkAvatar(config)
    avatar.load_models()
    avatar.load_source_image()

    # 100개 오디오 청크 준비
    audio_chunks = [
        np.random.randn(640).astype(np.float32) * 0.1
        for _ in range(100)
    ]

    # 순차 처리
    start_time = time.time()
    for audio_chunk in audio_chunks[:50]:
        avatar.process_audio_chunk(audio_chunk)
    sequential_time = time.time() - start_time

    print(f"순차 처리 (50 프레임): {sequential_time:.2f}초")
    print(f"평균 FPS: {50/sequential_time:.1f}\n")

    # 배치 처리는 실제 MuseTalk 구현에 따라 다름
    # (여기서는 순차 처리만 시뮬레이션)


# ============================================================================
# 예시 7: 성능 통계 모니터링
# ============================================================================
def example7_performance_monitoring():
    """성능 통계 모니터링"""
    print("=" * 70)
    print("예시 7: 성능 통계 모니터링")
    print("=" * 70)

    avatar = create_musetalk_avatar(
        source_image_path="./assets/avatar_source.jpg",
        resolution=256
    )

    # 50 프레임 처리
    for i in range(50):
        audio_chunk = np.random.randn(640).astype(np.float32) * 0.1
        avatar.process_audio_chunk(audio_chunk)

    # 통계 출력
    stats = avatar.get_stats()

    print("=== 성능 통계 ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print()


# ============================================================================
# 예시 8: 실제 오디오 파일로 비디오 생성
# ============================================================================
async def example8_audio_to_video():
    """실제 오디오 파일로 비디오 생성"""
    print("=" * 70)
    print("예시 8: 오디오 파일로 비디오 생성")
    print("=" * 70)

    import soundfile as sf

    # 오디오 파일 로드 (예시)
    audio_path = "./assets/interview_audio.wav"

    # 파일이 존재하지 않으면 더미 생성
    if not Path(audio_path).exists():
        print(f"오디오 파일이 없습니다: {audio_path}")
        print("더미 오디오 생성 중...")

        # 더미 오디오 (16kHz, 5초)
        sample_rate = 16000
        duration = 5.0
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, audio_data, sample_rate)
        print(f"더미 오디오 생성됨: {audio_path}")
    else:
        # 오디오 로드
        audio_data, sample_rate = sf.read(audio_path)
        print(f"오디오 로드됨: {audio_path}")
        print(f"  - 샘플레이트: {sample_rate}Hz")
        print(f"  - 길이: {len(audio_data)/sample_rate:.2f}초")

    # 아바타 생성
    avatar = create_musetalk_avatar(
        source_image_path="./assets/avatar_source.jpg",
        resolution=256
    )

    # 청크 크기 (40ms)
    chunk_size = int(sample_rate * 0.04)

    # 프레임 생성
    frames = []
    num_chunks = len(audio_data) // chunk_size

    print(f"\n비디오 생성 중... ({num_chunks} 프레임)")

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        audio_chunk = audio_data[start_idx:end_idx]

        video_frame = avatar.process_audio_chunk(audio_chunk)
        frames.append(video_frame.frame)

        if (i + 1) % 25 == 0:
            print(f"진행: {i+1}/{num_chunks} 프레임")

    print(f"\n총 {len(frames)} 프레임 생성됨")

    # 비디오 저장 (옵션)
    output_video_path = "./output/interview_avatar.mp4"
    save_video(frames, output_video_path, fps=25)
    print(f"비디오 저장됨: {output_video_path}\n")


# ============================================================================
# 유틸리티 함수
# ============================================================================
def save_video(frames: list, output_path: str, fps: int = 25):
    """프레임을 비디오로 저장"""
    import cv2

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        print("저장할 프레임이 없습니다.")
        return

    height, width = frames[0].shape[:2]

    # VideoWriter 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"비디오 저장 완료: {output_path} ({len(frames)} 프레임, {fps} FPS)")


# ============================================================================
# 메인 함수
# ============================================================================
def main():
    """모든 예시 실행"""
    print("\n" + "=" * 70)
    print("MuseTalk Avatar 사용 예시")
    print("=" * 70 + "\n")

    # 예시 1: 기본 사용법
    # example1_basic_usage()

    # 예시 2: 헬퍼 함수
    # example2_helper_function()

    # 예시 3: 실시간 스트리밍
    # example3_streaming_simulation()

    # 예시 4: 상태 전환
    # example4_state_transitions()

    # 예시 5: 고해상도
    # example5_high_quality()

    # 예시 6: 배치 처리
    # example6_batch_processing()

    # 예시 7: 성능 모니터링
    example7_performance_monitoring()

    # 예시 8: 오디오 파일로 비디오 생성 (비동기)
    # asyncio.run(example8_audio_to_video())

    print("\n" + "=" * 70)
    print("모든 예시 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
