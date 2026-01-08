"""
Avatar Image Processor 사용 예시
소스 이미지 전처리 예제
"""

from pathlib import Path
from image_processor import AvatarImageProcessor, ProcessingQuality


def example1_basic_processing():
    """예시 1: 기본 이미지 처리"""
    print("=" * 70)
    print("예시 1: 기본 이미지 처리")
    print("=" * 70)

    # 프로세서 생성
    processor = AvatarImageProcessor(
        target_size=512,
        use_face_align=True,
        use_background_removal=False,
        use_upscaling=False
    )

    # 이미지 처리
    input_path = "./assets/sample_photo.jpg"
    output_path = "./output/avatar_basic.png"

    try:
        result = processor.process(
            image_path=input_path,
            output_path=output_path
        )

        print(f"\n✓ 처리 완료!")
        print(f"  - 품질 점수: {result.quality_score:.1f}/100")
        print(f"  - 품질 등급: {result.quality_grade.value}")
        print(f"  - 얼굴 각도: {result.face_info.angle_yaw:.1f}°")
        print(f"  - 경고: {len(result.warnings)}개\n")

    except Exception as e:
        print(f"\n✗ 처리 실패: {e}\n")


def example2_high_quality():
    """예시 2: 고품질 처리 (배경 제거 + 업스케일)"""
    print("=" * 70)
    print("예시 2: 고품질 처리 (배경 제거 + 업스케일)")
    print("=" * 70)

    # 프로세서 생성
    processor = AvatarImageProcessor(
        target_size=512,
        use_face_align=True,
        use_background_removal=True,  # 배경 제거
        use_upscaling=True,            # 업스케일링
        device='cuda'                  # GPU 사용
    )

    input_path = "./assets/sample_photo.jpg"
    output_path = "./output/avatar_hq.png"

    try:
        result = processor.process(
            image_path=input_path,
            output_path=output_path,
            save_intermediate=True  # 중간 결과 저장
        )

        print(f"\n✓ 처리 완료!")
        print(f"  - 배경 제거: {result.background_removed}")
        print(f"  - 업스케일: {result.upscaled}")
        print(f"  - 품질: {result.quality_score:.1f}/100\n")

        # 중간 결과 파일들
        # - sample_photo_aligned.jpg
        # - sample_photo_cropped.jpg
        # - sample_photo_no_bg.jpg

    except Exception as e:
        print(f"\n✗ 처리 실패: {e}\n")


def example3_batch_processing():
    """예시 3: 배치 처리"""
    print("=" * 70)
    print("예시 3: 배치 처리")
    print("=" * 70)

    # 프로세서 생성
    processor = AvatarImageProcessor(target_size=512)

    # 입력 디렉토리의 모든 이미지 처리
    input_dir = Path("./assets/photos")
    output_dir = Path("./output/avatars")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"\n⚠ 입력 디렉토리가 없습니다: {input_dir}\n")
        return

    # 이미지 파일 목록
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    print(f"\n총 {len(image_files)}개 이미지 처리 중...\n")

    results = []
    for i, image_file in enumerate(image_files, 1):
        output_path = output_dir / f"avatar_{image_file.stem}.png"

        try:
            result = processor.process(
                image_path=str(image_file),
                output_path=str(output_path)
            )

            results.append((image_file.name, result.quality_score, result.quality_grade))

            print(f"[{i}/{len(image_files)}] {image_file.name}: "
                  f"{result.quality_score:.1f} ({result.quality_grade.value})")

        except Exception as e:
            print(f"[{i}/{len(image_files)}] {image_file.name}: ✗ 실패 ({e})")

    # 통계 출력
    if results:
        avg_score = sum(r[1] for r in results) / len(results)
        excellent_count = sum(1 for r in results if r[2] == ProcessingQuality.EXCELLENT)
        good_count = sum(1 for r in results if r[2] == ProcessingQuality.GOOD)

        print(f"\n=== 통계 ===")
        print(f"평균 품질: {avg_score:.1f}/100")
        print(f"Excellent: {excellent_count}개")
        print(f"Good: {good_count}개\n")


def example4_quality_check():
    """예시 4: 품질 검사"""
    print("=" * 70)
    print("예시 4: 품질 검사")
    print("=" * 70)

    processor = AvatarImageProcessor(target_size=512)

    input_path = "./assets/sample_photo.jpg"

    try:
        result = processor.process(image_path=input_path)

        print(f"\n=== 품질 리포트 ===")
        print(f"\n얼굴 정보:")
        print(f"  - 위치: {result.face_info.bbox}")
        print(f"  - 크기: {result.face_info.face_size}")
        print(f"  - 영역 비율: {result.face_info.face_area*100:.1f}%")
        print(f"  - 감지 신뢰도: {result.face_info.score:.2f}")

        print(f"\n각도:")
        print(f"  - Yaw (좌우): {result.face_info.angle_yaw:.1f}°")
        print(f"  - Pitch (상하): {result.face_info.angle_pitch:.1f}°")
        print(f"  - Roll (기울기): {result.face_info.angle_roll:.1f}°")

        print(f"\n품질 메트릭:")
        print(f"  - 선명도: {result.face_info.sharpness:.2f}")
        print(f"  - 밝기: {result.face_info.brightness:.2f}")

        print(f"\n종합 평가:")
        print(f"  - 점수: {result.quality_score:.1f}/100")
        print(f"  - 등급: {result.quality_grade.value}")

        if result.warnings:
            print(f"\n경고 ({len(result.warnings)}개):")
            for warning in result.warnings:
                print(f"  ⚠ {warning}")
        else:
            print(f"\n✓ 경고 없음 (완벽한 이미지)")

        # 품질 기준
        if result.quality_score >= 80:
            print(f"\n✓ 이 이미지는 아바타 생성에 적합합니다!")
        elif result.quality_score >= 60:
            print(f"\n⚠ 이 이미지는 사용 가능하지만 개선이 필요합니다.")
        else:
            print(f"\n✗ 다른 이미지를 사용하는 것을 권장합니다.")

        print()

    except Exception as e:
        print(f"\n✗ 처리 실패: {e}\n")


def example5_custom_settings():
    """예시 5: 커스텀 설정"""
    print("=" * 70)
    print("예시 5: 커스텀 설정")
    print("=" * 70)

    # 256x256 저해상도 (빠른 처리)
    processor_fast = AvatarImageProcessor(
        target_size=256,
        use_face_align=False,
        use_background_removal=False,
        use_upscaling=False
    )

    # 512x512 고품질
    processor_hq = AvatarImageProcessor(
        target_size=512,
        use_face_align=True,
        use_background_removal=True,
        use_upscaling=True,
        device='cuda'
    )

    input_path = "./assets/sample_photo.jpg"

    print("\n1. 빠른 처리 (256x256, 기본)")
    try:
        import time
        start = time.time()

        result_fast = processor_fast.process(
            image_path=input_path,
            output_path="./output/avatar_fast.png"
        )

        elapsed = time.time() - start

        print(f"   - 처리 시간: {elapsed:.2f}초")
        print(f"   - 품질: {result_fast.quality_score:.1f}/100")

    except Exception as e:
        print(f"   ✗ 실패: {e}")

    print("\n2. 고품질 처리 (512x512, 모든 옵션)")
    try:
        start = time.time()

        result_hq = processor_hq.process(
            image_path=input_path,
            output_path="./output/avatar_hq.png"
        )

        elapsed = time.time() - start

        print(f"   - 처리 시간: {elapsed:.2f}초")
        print(f"   - 품질: {result_hq.quality_score:.1f}/100")
        print(f"   - 배경 제거: {result_hq.background_removed}")
        print(f"   - 업스케일: {result_hq.upscaled}\n")

    except Exception as e:
        print(f"   ✗ 실패: {e}\n")


def example6_error_handling():
    """예시 6: 에러 처리"""
    print("=" * 70)
    print("예시 6: 에러 처리")
    print("=" * 70)

    processor = AvatarImageProcessor(target_size=512)

    # 1. 존재하지 않는 파일
    print("\n1. 존재하지 않는 파일:")
    try:
        result = processor.process("./nonexistent.jpg")
        print("   ✓ 처리 완료")
    except ValueError as e:
        print(f"   ✗ 예상된 에러: {e}")

    # 2. 얼굴이 없는 이미지
    print("\n2. 얼굴이 없는 이미지:")
    try:
        # 풍경 사진 등
        result = processor.process("./assets/landscape.jpg")
        print("   ✓ 처리 완료")
    except ValueError as e:
        print(f"   ✗ 예상된 에러: {e}")

    # 3. 품질이 낮은 이미지 (경고만)
    print("\n3. 품질이 낮은 이미지:")
    try:
        result = processor.process(
            image_path="./assets/low_quality.jpg",
            output_path="./output/avatar_low.png"
        )
        print(f"   ✓ 처리 완료 (품질: {result.quality_score:.1f}/100)")

        if result.warnings:
            print(f"   경고:")
            for warning in result.warnings:
                print(f"     - {warning}")

    except Exception as e:
        print(f"   ✗ 에러: {e}")

    print()


def main():
    """모든 예시 실행"""
    print("\n" + "=" * 70)
    print("Avatar Image Processor 사용 예시")
    print("=" * 70 + "\n")

    # 예시 1: 기본 처리
    # example1_basic_processing()

    # 예시 2: 고품질 처리
    # example2_high_quality()

    # 예시 3: 배치 처리
    # example3_batch_processing()

    # 예시 4: 품질 검사
    example4_quality_check()

    # 예시 5: 커스텀 설정
    # example5_custom_settings()

    # 예시 6: 에러 처리
    # example6_error_handling()


if __name__ == "__main__":
    main()
