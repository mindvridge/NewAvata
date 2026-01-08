"""
MuseTalk 립싱크 테스트 스크립트

ElevenLabs TTS 오디오 파일과 영상을 사용하여 립싱크 테스트
MuseTalk 공식 추론 방식 사용: 전체 오디오 미리 처리 + 슬라이싱
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger


def test_lipsync():
    """립싱크 테스트 실행 (MuseTalk 공식 추론 방식)"""

    # 파일 경로
    video_path = project_root / "assets" / "images" / "A_professional_woman_in_a_dark_gray_business_suit.mp4"
    audio_path = project_root / "assets" / "audio" / "ElevenLabs_2025-05-09T07_25_56_Psychological Consultant Woman_gen_sp100_s94_sb75_se0_b_m2.mp3"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  MuseTalk 립싱크 테스트 (공식 추론 방식)")
    print("=" * 60)
    print(f"\n  영상: {video_path.name}")
    print(f"  오디오: {audio_path.name}")
    print()

    # 파일 존재 확인
    if not video_path.exists():
        print(f"[ERROR] 영상 파일을 찾을 수 없습니다: {video_path}")
        return
    if not audio_path.exists():
        print(f"[ERROR] 오디오 파일을 찾을 수 없습니다: {audio_path}")
        return

    # 오디오 로드
    print("[1/6] 오디오 로드 중...")
    try:
        import librosa
        audio_data, sr = librosa.load(str(audio_path), sr=16000)  # 16kHz로 리샘플링
        audio_duration = len(audio_data) / sr
        print(f"       오디오 길이: {audio_duration:.2f}초, 샘플레이트: {sr}Hz")
    except Exception as e:
        print(f"[ERROR] 오디오 로드 실패: {e}")
        return

    # MuseTalk 초기화
    print("\n[2/6] MuseTalk 초기화 중...")
    try:
        from avatar.musetalk_wrapper import MuseTalkAvatar, MuseTalkConfig, SourceType

        config = MuseTalkConfig(
            source_type=SourceType.VIDEO,
            source_video_path=str(video_path),
            resolution=256,
            use_face_crop=True,
            fixed_crop_region=True,  # 루프 영상이므로 고정 크롭 사용
            video_fps=25,
            use_fp16=False,  # Whisper는 FP16을 지원하지 않음
        )

        avatar = MuseTalkAvatar(config)

    except Exception as e:
        print(f"[ERROR] MuseTalk 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 모델 로드
    print("\n[3/6] 모델 로드 중...")
    try:
        avatar.load_models()
        if avatar._models_loaded:
            print("       모델 로드 완료")
        else:
            print("[ERROR] 모델 로드 실패")
            return
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 영상 소스 로드
    print("\n[4/6] 영상 소스 로드 중...")
    try:
        if not avatar._load_video_source(str(video_path)):
            print("[ERROR] 영상 소스 로드 실패")
            return

        if avatar.video_loader:
            print(f"       프레임 수: {len(avatar.video_loader.frames)}")
            print(f"       얼굴 크롭: {'활성화' if avatar.video_loader._face_crop_enabled else '비활성화'}")
            if avatar.video_loader._face_crop_enabled and avatar.video_loader.crop_regions:
                print(f"       크롭 영역: {avatar.video_loader.crop_regions[0]}")
    except Exception as e:
        print(f"[ERROR] 영상 소스 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 전체 오디오 특징 추출 (MuseTalk 공식 방식)
    print("\n[5/6] 오디오 특징 추출 중 (전체 오디오)...")
    try:
        # 테스트용으로 처음 10초만 사용
        test_duration = 10.0  # 초
        test_samples = int(test_duration * sr)
        test_audio = audio_data[:test_samples]

        # 전체 오디오 특징 추출 (50FPS)
        audio_features_full = avatar.audio_encoder.extract_full_audio(test_audio, sample_rate=sr)
        print(f"       오디오 특징 shape: {audio_features_full.shape}")
        print(f"       (50FPS × {test_duration}초 = {int(50 * test_duration)} frames expected)")

    except Exception as e:
        print(f"[ERROR] 오디오 특징 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 립싱크 생성
    print("\n[6/6] 립싱크 생성 중...")

    output_frames = []  # 256x256 크롭 프레임
    full_frames = []    # 원본 해상도 합성 프레임
    fps = 25
    total_frames = int(test_duration * fps)

    # 원본 영상 크기 확인
    if avatar.video_loader and avatar.video_loader.original_frames:
        orig_h, orig_w = avatar.video_loader.original_frames[0].shape[:2]
        print(f"       원본 영상 크기: {orig_w}x{orig_h}")
        print(f"       크롭 영역 수: {len(avatar.video_loader.crop_regions)}")
        if avatar.video_loader.crop_regions and avatar.video_loader.crop_regions[0]:
            crop = avatar.video_loader.crop_regions[0]
            print(f"       크롭 영역: {crop}")

    print(f"       테스트 길이: {test_duration}초")
    print(f"       예상 프레임 수: {total_frames}")
    print()

    start_time = time.time()

    try:
        for i in range(total_frames):
            # 해당 프레임에 대한 오디오 특징 슬라이싱 (MuseTalk 공식 방식)
            audio_features = avatar.audio_encoder.get_sliced_feature(
                audio_features_full,
                vid_idx=i,
                audio_feat_length=(2, 2),  # 좌우 컨텍스트
                fps=fps
            )

            # 립싱크 프레임 생성
            frame = avatar._generate_frame(audio_features)

            if frame is not None:
                output_frames.append(frame)

                # 원본 해상도에 합성
                if avatar.video_loader is not None:
                    # 현재 비디오 프레임 인덱스 (루프)
                    vid_idx = i % len(avatar.video_loader.frames)
                    full_frame = avatar.composite_to_original(frame, vid_idx)
                    if full_frame is not None:
                        full_frames.append(full_frame)

            # 비디오 프레임 인덱스 업데이트
            avatar.current_frame_idx = (avatar.current_frame_idx + 1) % len(avatar.video_loader.frames) if avatar.video_loader else 0

            # 진행률 표시
            if (i + 1) % 25 == 0 or i == total_frames - 1:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_frames * 100
                fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"       진행: {progress:.1f}% ({i+1}/{total_frames} frames, {fps_actual:.1f} fps)")

    except Exception as e:
        print(f"[ERROR] 립싱크 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed_total = time.time() - start_time
    print(f"\n       완료! {len(output_frames)} 프레임 생성 ({elapsed_total:.2f}초)")
    print(f"       원본 해상도 합성: {len(full_frames)} 프레임")

    # 결과 저장
    import subprocess

    # 1. 크롭된 256x256 결과 저장
    if output_frames:
        output_path = output_dir / "lipsync_test_cropped.mp4"
        print(f"\n[크롭 결과 저장] {output_path}")

        h, w = output_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame in output_frames:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                writer.write(frame)
            else:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"       저장 완료: {w}x{h}")

    # 2. 원본 해상도 합성 결과 저장 (주요 결과물)
    if full_frames:
        full_output_path = output_dir / "lipsync_test_full.mp4"
        print(f"\n[원본 해상도 합성 결과 저장] {full_output_path}")

        h, w = full_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(full_output_path), fourcc, fps, (w, h))

        for frame in full_frames:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                writer.write(frame)
            else:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"       저장 완료: {w}x{h}")

        # 오디오와 합성 (원본 해상도 영상 사용)
        try:
            output_with_audio = output_dir / "lipsync_test_with_audio.mp4"

            cmd = [
                'ffmpeg', '-y',
                '-i', str(full_output_path),
                '-i', str(audio_path),
                '-t', str(test_duration),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'aac',
                '-shortest',
                str(output_with_audio)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"\n[오디오 합성 완료] {output_with_audio}")
            else:
                print(f"[ERROR] 오디오 합성 실패: {result.stderr}")

        except Exception as e:
            print(f"[ERROR] 오디오 합성 실패: {e}")

    elif output_frames:
        # full_frames가 없으면 크롭 결과에 오디오 합성
        try:
            output_with_audio = output_dir / "lipsync_test_with_audio.mp4"

            cmd = [
                'ffmpeg', '-y',
                '-i', str(output_path),
                '-i', str(audio_path),
                '-t', str(test_duration),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                str(output_with_audio)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"\n[오디오 합성 완료] {output_with_audio}")
            else:
                print(f"[ERROR] 오디오 합성 실패: {result.stderr}")

        except Exception as e:
            print(f"[ERROR] 오디오 합성 실패: {e}")

    print("\n" + "=" * 60)
    print("  테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    test_lipsync()
