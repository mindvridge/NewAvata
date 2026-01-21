#!/usr/bin/env python3
"""
해상도별 영상 및 프리컴퓨트 데이터 생성 스크립트

원본 720p 영상을 480p, 360p로 변환하고 각각의 프리컴퓨트 데이터를 생성합니다.

사용법:
    python create_resolution_variants.py
"""

import os
import subprocess
import sys
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
PRECOMPUTED_DIR = BASE_DIR / "precomputed"

# 해상도 설정 (원본 비율 1268:724 ≈ 1.75:1 유지)
RESOLUTIONS = {
    "720p": None,  # 원본 유지
    "480p": (848, 484),
    "360p": (632, 360)
}

# 변환할 영상 목록
SOURCE_VIDEOS = [
    "질문_long.mp4",
    "질문_short.mp4",
    "기본.mp4",
    "끄덕.mp4",
    "끄덕2.mp4",
    "서류보기.mp4",
    "서류보기2.mp4",
    "미소1.mp4",
    "미소2.mp4"
]

# 립싱크용 영상 (프리컴퓨트 필요)
LIPSYNC_VIDEOS = [
    "질문_long.mp4",
    "질문_short.mp4"
]


def create_resolution_dirs():
    """해상도별 디렉토리 생성"""
    for res in RESOLUTIONS:
        res_dir = ASSETS_DIR / res
        res_dir.mkdir(exist_ok=True)
        print(f"[디렉토리] {res_dir}")


def convert_video(src_path: Path, dst_path: Path, width: int, height: int):
    """영상을 지정 해상도로 변환"""
    if dst_path.exists():
        print(f"  [스킵] 이미 존재: {dst_path.name}")
        return True

    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        str(dst_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [완료] {dst_path.name} ({width}x{height})")
            return True
        else:
            print(f"  [오류] {dst_path.name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"  [오류] {dst_path.name}: {e}")
        return False


def copy_720p_videos():
    """720p는 원본을 720p 폴더로 복사 (심볼릭 링크 또는 복사)"""
    res_dir = ASSETS_DIR / "720p"
    for video in SOURCE_VIDEOS:
        src = ASSETS_DIR / video
        dst = res_dir / video
        if not src.exists():
            print(f"  [경고] 원본 없음: {src}")
            continue
        if dst.exists():
            print(f"  [스킵] 이미 존재: {dst.name}")
            continue

        # Windows에서는 복사, Linux에서는 심볼릭 링크
        try:
            import shutil
            shutil.copy2(src, dst)
            print(f"  [복사] {dst.name}")
        except Exception as e:
            print(f"  [오류] {dst.name}: {e}")


def convert_all_videos():
    """모든 영상을 각 해상도로 변환"""
    print("\n=== 영상 변환 시작 ===\n")

    # 720p (원본 복사)
    print("[720p] 원본 복사...")
    copy_720p_videos()

    # 480p, 360p 변환
    for res, size in RESOLUTIONS.items():
        if size is None:
            continue  # 720p는 이미 처리

        width, height = size
        res_dir = ASSETS_DIR / res
        print(f"\n[{res}] 변환 중... ({width}x{height})")

        for video in SOURCE_VIDEOS:
            src = ASSETS_DIR / video
            dst = res_dir / video
            if src.exists():
                convert_video(src, dst, width, height)


def generate_precomputed():
    """립싱크용 영상의 프리컴퓨트 데이터 생성"""
    print("\n=== 프리컴퓨트 데이터 생성 ===\n")

    # 프리컴퓨트 해상도별 폴더 생성
    for res in RESOLUTIONS:
        res_precompute_dir = PRECOMPUTED_DIR / res
        res_precompute_dir.mkdir(exist_ok=True)

    # 프리컴퓨트 스크립트 경로
    precompute_script = BASE_DIR / "precompute_face.py"
    if not precompute_script.exists():
        print("[오류] precompute_face.py를 찾을 수 없습니다")
        return

    for res in RESOLUTIONS:
        print(f"\n[{res}] 프리컴퓨트 생성...")
        res_dir = ASSETS_DIR / res

        for video in LIPSYNC_VIDEOS:
            video_path = res_dir / video
            if not video_path.exists():
                print(f"  [스킵] 영상 없음: {video_path}")
                continue

            # 출력 파일명
            base_name = video.replace(".mp4", "")
            output_name = f"{base_name}_{res}_precomputed.pkl"
            output_path = PRECOMPUTED_DIR / res / output_name

            if output_path.exists():
                print(f"  [스킵] 이미 존재: {output_name}")
                continue

            print(f"  [생성] {output_name}...")

            # precompute_face.py 실행
            cmd = [
                sys.executable, str(precompute_script),
                "--video", str(video_path),
                "--output", str(output_path)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  [완료] {output_name}")
                else:
                    print(f"  [오류] {output_name}")
                    print(f"         {result.stderr[:200]}")
            except Exception as e:
                print(f"  [오류] {output_name}: {e}")


def print_summary():
    """생성된 파일 요약 출력"""
    print("\n=== 생성 완료 요약 ===\n")

    for res in RESOLUTIONS:
        res_dir = ASSETS_DIR / res
        precompute_dir = PRECOMPUTED_DIR / res

        video_count = len(list(res_dir.glob("*.mp4"))) if res_dir.exists() else 0
        pkl_count = len(list(precompute_dir.glob("*.pkl"))) if precompute_dir.exists() else 0

        print(f"[{res}]")
        print(f"  영상: {video_count}개 ({res_dir})")
        print(f"  프리컴퓨트: {pkl_count}개 ({precompute_dir})")


def main():
    print("=" * 50)
    print("  해상도별 영상 및 프리컴퓨트 데이터 생성")
    print("=" * 50)

    # 1. 디렉토리 생성
    create_resolution_dirs()

    # 2. 영상 변환
    convert_all_videos()

    # 3. 프리컴퓨트 생성
    generate_precomputed()

    # 4. 요약
    print_summary()

    print("\n[완료] 모든 작업이 완료되었습니다.")
    print("\n다음 단계:")
    print("  1. 서버 재시작")
    print("  2. 해상도 선택 시 해당 영상/프리컴퓨트 자동 사용")


if __name__ == "__main__":
    main()
