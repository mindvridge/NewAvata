"""
해상도별 립싱크 생성 속도 측정 테스트
720p, 480p, 360p 해상도에서 생성 시간 비교
"""

import os
import sys
import time
import asyncio

# 프로젝트 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_resolution_speed():
    """해상도별 립싱크 생성 속도 측정"""

    print("=" * 60)
    print("해상도별 립싱크 생성 속도 측정 테스트")
    print("=" * 60)

    # 1. 먼저 Edge TTS로 테스트 오디오 생성
    print("\n[1단계] Edge TTS로 테스트 오디오 생성...")

    import edge_tts

    test_text = "안녕하세요. 저는 면접관입니다. 오늘 면접에 참여해 주셔서 감사합니다."
    voice_name = "ko-KR-SunHiNeural"

    async def generate_test_audio():
        communicate = edge_tts.Communicate(test_text, voice_name)
        audio_data = b''
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

    tts_start = time.time()
    mp3_data = asyncio.run(generate_test_audio())
    tts_time = time.time() - tts_start

    # MP3 저장
    test_audio_path = "results/realtime/test_resolution_audio.mp3"
    os.makedirs(os.path.dirname(test_audio_path), exist_ok=True)
    with open(test_audio_path, 'wb') as f:
        f.write(mp3_data)

    print(f"  TTS 생성 시간: {tts_time:.2f}초")
    print(f"  오디오 파일: {test_audio_path}")

    # 2. MP3를 WAV로 변환
    print("\n[2단계] MP3 -> WAV 변환...")
    from pydub import AudioSegment

    convert_start = time.time()
    audio = AudioSegment.from_mp3(test_audio_path)
    test_wav_path = test_audio_path.replace('.mp3', '.wav')
    audio.export(test_wav_path, format='wav')
    convert_time = time.time() - convert_start

    audio_duration = len(audio) / 1000.0
    print(f"  변환 시간: {convert_time:.2f}초")
    print(f"  오디오 길이: {audio_duration:.2f}초")

    # 3. 립싱크 엔진 초기화
    print("\n[3단계] 립싱크 엔진 초기화...")

    from app import LipsyncEngine

    init_start = time.time()
    engine = LipsyncEngine()

    # 모델 로드
    print("\n[4단계] 모델 로드...")
    load_start = time.time()
    engine.load_models(use_float16=True, use_tensorrt=True)
    load_time = time.time() - load_start
    print(f"  모델 로드 시간: {load_time:.2f}초")

    # 5. 해상도별 립싱크 생성 테스트
    precomputed_path = "precomputed/a24ca21b-0d02-49f0-99a7-d737c5ca6058_precomputed.pkl"
    resolutions = ["720p", "480p", "360p"]
    results = {}

    print("\n" + "=" * 60)
    print("해상도별 립싱크 생성 테스트")
    print("=" * 60)

    for resolution in resolutions:
        print(f"\n[테스트] {resolution} 해상도...")

        output_dir = f"results/realtime/test_{resolution}"
        os.makedirs(output_dir, exist_ok=True)

        # 립싱크 생성
        gen_start = time.time()
        try:
            result = engine.generate_lipsync(
                precomputed_path=precomputed_path,
                audio_input=test_wav_path,
                output_dir=output_dir,
                fps=25,
                sid=None,
                preloaded_data=None,
                frame_skip=2,  # 프레임 스킵 2 (속도 최적화)
                resolution=resolution
            )
            gen_time = time.time() - gen_start

            if result and os.path.exists(result):
                # 파일 크기 확인
                file_size = os.path.getsize(result) / (1024 * 1024)  # MB

                results[resolution] = {
                    'time': gen_time,
                    'file_size': file_size,
                    'path': result,
                    'success': True
                }

                print(f"  [OK] 생성 완료!")
                print(f"    - 소요 시간: {gen_time:.2f}초")
                print(f"    - 파일 크기: {file_size:.2f}MB")
                print(f"    - 실시간 비율: {audio_duration / gen_time:.2f}x")
            else:
                results[resolution] = {
                    'time': gen_time,
                    'success': False,
                    'error': 'No output file'
                }
                print(f"  [FAIL] 출력 파일 없음")

        except Exception as e:
            gen_time = time.time() - gen_start
            results[resolution] = {
                'time': gen_time,
                'success': False,
                'error': str(e)
            }
            print(f"  [FAIL] 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    # 6. 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"\n오디오 길이: {audio_duration:.2f}초")
    print(f"TTS 생성 시간: {tts_time:.2f}초")
    print(f"\n{'해상도':<10} {'생성시간':<12} {'파일크기':<12} {'실시간비율':<12} {'상태'}")
    print("-" * 60)

    for resolution in resolutions:
        r = results.get(resolution, {})
        if r.get('success'):
            realtime_ratio = audio_duration / r['time']
            print(f"{resolution:<10} {r['time']:.2f}초{'':<6} {r['file_size']:.2f}MB{'':<6} {realtime_ratio:.2f}x{'':<8} 성공")
        else:
            print(f"{resolution:<10} {r.get('time', 0):.2f}초{'':<6} {'-':<12} {'-':<12} 실패: {r.get('error', 'Unknown')}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

    return results

if __name__ == "__main__":
    test_resolution_speed()
