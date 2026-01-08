"""
Face Enhancer 사용 예시
실시간 및 후처리 얼굴 품질 향상 예제
"""

import cv2
import numpy as np
from pathlib import Path
import time

from face_enhancer import (
    FaceEnhancer,
    EnhancementConfig,
    EnhancementMode,
    EnhancementModel,
    create_realtime_enhancer,
    create_quality_enhancer,
    create_balanced_enhancer,
)


# ============================================================================
# 예시 1: 실시간 향상 (빠름)
# ============================================================================
def example1_realtime_enhancement():
    """예시 1: 실시간 얼굴 향상 (~30ms/프레임)"""
    print("=" * 70)
    print("예시 1: 실시간 얼굴 향상")
    print("=" * 70)

    # 실시간 향상기 생성
    enhancer = create_realtime_enhancer(device='cpu')

    # 이미지 로드
    image_path = "./assets/sample_face.jpg"

    if not Path(image_path).exists():
        print(f"\n⚠ 이미지가 없습니다: {image_path}\n")
        return

    image = cv2.imread(image_path)

    # 향상
    result = enhancer.enhance(image)

    print(f"\n✓ 처리 완료!")
    print(f"  - 처리 시간: {result.processing_time_ms:.1f}ms")
    print(f"  - 모델: {result.model_used.value}")
    print(f"  - 업스케일: {result.upscale_factor}x")

    # 저장
    output_path = "./output/enhanced_realtime.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result.enhanced_image)

    print(f"  - 저장: {output_path}\n")


# ============================================================================
# 예시 2: 고품질 향상 (느림)
# ============================================================================
def example2_quality_enhancement():
    """예시 2: 고품질 얼굴 향상 (~200ms/프레임, 2x 업스케일)"""
    print("=" * 70)
    print("예시 2: 고품질 얼굴 향상")
    print("=" * 70)

    # 고품질 향상기 생성 (2x 업스케일)
    enhancer = create_quality_enhancer(upscale_factor=2, device='cuda')

    image_path = "./assets/sample_face.jpg"

    if not Path(image_path).exists():
        print(f"\n⚠ 이미지가 없습니다: {image_path}\n")
        return

    image = cv2.imread(image_path)

    print(f"\n원본 크기: {image.shape[1]}x{image.shape[0]}")

    # 향상
    result = enhancer.enhance(image)

    print(f"\n✓ 처리 완료!")
    print(f"  - 처리 시간: {result.processing_time_ms:.1f}ms")
    print(f"  - 향상 크기: {result.enhanced_image.shape[1]}x{result.enhanced_image.shape[0]}")
    print(f"  - 업스케일: {result.upscale_factor}x")

    # 저장
    output_path = "./output/enhanced_quality.jpg"
    cv2.imwrite(output_path, result.enhanced_image)

    print(f"  - 저장: {output_path}\n")


# ============================================================================
# 예시 3: 균형 모드
# ============================================================================
def example3_balanced_enhancement():
    """예시 3: 균형 모드 (~100ms/프레임)"""
    print("=" * 70)
    print("예시 3: 균형 모드")
    print("=" * 70)

    # 균형 향상기 생성
    enhancer = create_balanced_enhancer(device='cpu')

    image_path = "./assets/sample_face.jpg"

    if not Path(image_path).exists():
        print(f"\n⚠ 이미지가 없습니다: {image_path}\n")
        return

    image = cv2.imread(image_path)

    # 향상
    result = enhancer.enhance(image)

    print(f"\n✓ 처리 완료!")
    print(f"  - 처리 시간: {result.processing_time_ms:.1f}ms")

    # 저장
    output_path = "./output/enhanced_balanced.jpg"
    cv2.imwrite(output_path, result.enhanced_image)

    print(f"  - 저장: {output_path}\n")


# ============================================================================
# 예시 4: 블렌딩 가중치 조정
# ============================================================================
def example4_weight_blending():
    """예시 4: 원본과 향상 이미지 블렌딩"""
    print("=" * 70)
    print("예시 4: 블렌딩 가중치 조정")
    print("=" * 70)

    enhancer = create_balanced_enhancer(device='cpu')

    image_path = "./assets/sample_face.jpg"

    if not Path(image_path).exists():
        print(f"\n⚠ 이미지가 없습니다: {image_path}\n")
        return

    image = cv2.imread(image_path)

    # 다양한 가중치로 향상
    weights = [0.0, 0.3, 0.5, 0.7, 1.0]

    print(f"\n가중치별 향상 (0.0=원본, 1.0=완전향상):")

    for weight in weights:
        result = enhancer.enhance(image, face_weight=weight)

        output_path = f"./output/enhanced_weight_{weight:.1f}.jpg"
        cv2.imwrite(output_path, result.enhanced_image)

        print(f"  - 가중치 {weight:.1f}: {output_path}")

    print()


# ============================================================================
# 예시 5: 배치 처리
# ============================================================================
def example5_batch_processing():
    """예시 5: 여러 이미지 일괄 처리"""
    print("=" * 70)
    print("예시 5: 배치 처리")
    print("=" * 70)

    enhancer = create_realtime_enhancer(device='cpu')

    # 입력 디렉토리
    input_dir = Path("./assets/faces")
    output_dir = Path("./output/enhanced_batch")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"\n⚠ 입력 디렉토리가 없습니다: {input_dir}\n")
        return

    # 이미지 파일 목록
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    if not image_files:
        print(f"\n⚠ 이미지가 없습니다: {input_dir}\n")
        return

    print(f"\n총 {len(image_files)}개 이미지 처리 중...\n")

    # 이미지 로드
    images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            images.append(img)

    # 진행률 콜백
    def progress_callback(current, total):
        percent = current / total * 100
        print(f"진행: {current}/{total} ({percent:.1f}%)")

    # 배치 처리
    start_time = time.time()

    results = enhancer.enhance_batch(images, progress_callback=progress_callback)

    elapsed = time.time() - start_time

    # 결과 저장
    for i, (img_file, result) in enumerate(zip(image_files, results)):
        output_path = output_dir / f"enhanced_{img_file.name}"
        cv2.imwrite(str(output_path), result.enhanced_image)

    print(f"\n✓ 완료!")
    print(f"  - 총 처리 시간: {elapsed:.2f}초")
    print(f"  - 평균 시간/이미지: {elapsed/len(images):.2f}초")
    print(f"  - 저장 디렉토리: {output_dir}\n")


# ============================================================================
# 예시 6: 비디오 처리
# ============================================================================
def example6_video_enhancement():
    """예시 6: 비디오 파일 향상"""
    print("=" * 70)
    print("예시 6: 비디오 향상")
    print("=" * 70)

    # 실시간 향상기 (비디오는 빠른 처리 필요)
    enhancer = create_realtime_enhancer(device='cuda')

    video_path = "./assets/interview_video.mp4"
    output_path = "./output/enhanced_video.mp4"

    if not Path(video_path).exists():
        print(f"\n⚠ 비디오가 없습니다: {video_path}\n")
        return

    print(f"\n비디오 향상 중: {video_path}")

    # 진행률 콜백
    def progress_callback(current, total):
        percent = current / total * 100
        print(f"\r진행: {current}/{total} ({percent:.1f}%)", end='', flush=True)

    # 비디오 처리
    start_time = time.time()

    success = enhancer.enhance_video(
        video_path=video_path,
        output_path=output_path,
        progress_callback=progress_callback,
        save_comparison=False
    )

    elapsed = time.time() - start_time

    print()  # 줄바꿈

    if success:
        # 통계
        stats = enhancer.get_stats()

        print(f"\n✓ 완료!")
        print(f"  - 총 처리 시간: {elapsed:.1f}초")
        print(f"  - 평균 시간/프레임: {stats['average_time_ms']:.1f}ms")
        print(f"  - 실시간 가능 (25fps): {stats['realtime_25fps']}")
        print(f"  - 저장: {output_path}\n")
    else:
        print(f"\n✗ 비디오 처리 실패\n")


# ============================================================================
# 예시 7: 비교 비디오 생성
# ============================================================================
def example7_comparison_video():
    """예시 7: 원본/향상 비교 비디오 생성"""
    print("=" * 70)
    print("예시 7: 비교 비디오 생성")
    print("=" * 70)

    enhancer = create_balanced_enhancer(device='cuda')

    video_path = "./assets/interview_video.mp4"
    output_path = "./output/comparison_video.mp4"

    if not Path(video_path).exists():
        print(f"\n⚠ 비디오가 없습니다: {video_path}\n")
        return

    print(f"\n비교 비디오 생성 중...")

    def progress_callback(current, total):
        percent = current / total * 100
        print(f"\r진행: {current}/{total} ({percent:.1f}%)", end='', flush=True)

    # 비교 비디오 생성 (원본 | 향상)
    success = enhancer.enhance_video(
        video_path=video_path,
        output_path=output_path,
        progress_callback=progress_callback,
        save_comparison=True  # 비교 모드
    )

    print()

    if success:
        print(f"\n✓ 완료!")
        print(f"  - 저장: {output_path}")
        print(f"  - 형식: 원본 | 향상 (좌우 비교)\n")
    else:
        print(f"\n✗ 비디오 처리 실패\n")


# ============================================================================
# 예시 8: 실시간 처리 성능 테스트
# ============================================================================
def example8_performance_test():
    """예시 8: 실시간 처리 성능 테스트"""
    print("=" * 70)
    print("예시 8: 실시간 처리 성능 테스트")
    print("=" * 70)

    # 다양한 모드 테스트
    configs = [
        ("실시간 (CPU)", create_realtime_enhancer(device='cpu')),
        ("실시간 (CUDA)", create_realtime_enhancer(device='cuda')),
        ("균형 (CPU)", create_balanced_enhancer(device='cpu')),
        ("고품질 2x (CUDA)", create_quality_enhancer(upscale_factor=2, device='cuda')),
    ]

    # 더미 이미지 생성
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    print(f"\n테스트 이미지: 512x512")
    print(f"테스트 횟수: 10회\n")

    results = []

    for name, enhancer in configs:
        # 워밍업
        for _ in range(3):
            enhancer.enhance(test_image)

        # 실제 측정
        times = []
        for _ in range(10):
            start = time.time()
            result = enhancer.enhance(test_image)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)

        # 실시간 가능 여부
        realtime_25fps = avg_time < (1000 / 25)
        realtime_30fps = avg_time < (1000 / 30)

        results.append((name, avg_time, realtime_25fps, realtime_30fps))

        print(f"{name}:")
        print(f"  - 평균 시간: {avg_time:.1f}ms")
        print(f"  - 실시간 25fps: {'✓' if realtime_25fps else '✗'}")
        print(f"  - 실시간 30fps: {'✓' if realtime_30fps else '✗'}")
        print()

    # 요약
    print("=" * 70)
    print("요약:")
    print("-" * 70)

    for name, avg_time, rt25, rt30 in results:
        status = "✓ 실시간" if rt25 else "✗ 후처리용"
        print(f"{name:20s}: {avg_time:6.1f}ms  {status}")

    print()


# ============================================================================
# 예시 9: 커스텀 설정
# ============================================================================
def example9_custom_config():
    """예시 9: 커스텀 설정"""
    print("=" * 70)
    print("예시 9: 커스텀 설정")
    print("=" * 70)

    # 커스텀 설정
    config = EnhancementConfig(
        mode=EnhancementMode.QUALITY,
        model=EnhancementModel.GFPGAN_V1_4,
        upscale_factor=4,  # 4배 업스케일
        face_weight=0.9,   # 강한 향상
        background_enhance=True,  # 배경도 향상
        device='cuda',
        tile_size=512,     # 타일 크기 (메모리 절약)
        tile_pad=10
    )

    enhancer = FaceEnhancer(config)

    image_path = "./assets/sample_face.jpg"

    if not Path(image_path).exists():
        print(f"\n⚠ 이미지가 없습니다: {image_path}\n")
        return

    image = cv2.imread(image_path)

    print(f"\n원본 크기: {image.shape[1]}x{image.shape[0]}")

    # 향상
    result = enhancer.enhance(image)

    print(f"\n✓ 처리 완료!")
    print(f"  - 향상 크기: {result.enhanced_image.shape[1]}x{result.enhanced_image.shape[0]}")
    print(f"  - 업스케일: {result.upscale_factor}x")
    print(f"  - 처리 시간: {result.processing_time_ms:.1f}ms")

    # 저장
    output_path = "./output/enhanced_custom_4x.jpg"
    cv2.imwrite(output_path, result.enhanced_image)

    print(f"  - 저장: {output_path}\n")


# ============================================================================
# 메인 함수
# ============================================================================
def main():
    """모든 예시 실행"""
    print("\n" + "=" * 70)
    print("Face Enhancer 사용 예시")
    print("=" * 70 + "\n")

    # 예시 1: 실시간 향상
    # example1_realtime_enhancement()

    # 예시 2: 고품질 향상
    # example2_quality_enhancement()

    # 예시 3: 균형 모드
    # example3_balanced_enhancement()

    # 예시 4: 블렌딩 가중치
    # example4_weight_blending()

    # 예시 5: 배치 처리
    # example5_batch_processing()

    # 예시 6: 비디오 처리
    # example6_video_enhancement()

    # 예시 7: 비교 비디오
    # example7_comparison_video()

    # 예시 8: 성능 테스트
    example8_performance_test()

    # 예시 9: 커스텀 설정
    # example9_custom_config()


if __name__ == "__main__":
    main()
