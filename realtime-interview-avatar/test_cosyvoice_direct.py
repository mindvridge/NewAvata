"""
CosyVoice 2.0 직접 테스트 스크립트
원본 zero-shot 방식으로 TTS 품질 확인
"""
import sys
sys.path.insert(0, "c:/NewAvata/NewAvata/CosyVoice")

import os
import time
import torch
import torchaudio
import subprocess

os.chdir("c:/NewAvata/NewAvata/realtime-interview-avatar")

print("=" * 60)
print("CosyVoice 2.0 직접 테스트 (원본 zero-shot 방식)")
print("=" * 60)

print("\nCosyVoice 모듈 임포트 중...")
from cosyvoice.cli.cosyvoice import CosyVoice2

MODEL_PATH = "c:/NewAvata/NewAvata/CosyVoice/pretrained_models/CosyVoice2-0.5B"
PROMPT_AUDIO = "assets/audio/ElevenLabs_2025-05-09T07_25_56_Psychological Consultant Woman_gen_sp100_s94_sb75_se0_b_m2.mp3"

print(f"\n모델 경로: {MODEL_PATH}")
print(f"프롬프트 오디오: {PROMPT_AUDIO}")

# 모델 로드
print("\n[1] CosyVoice 모델 로드 중...")
start = time.time()
cosyvoice = CosyVoice2(MODEL_PATH, fp16=True)
print(f"    모델 로드 완료: {time.time()-start:.1f}초")
print(f"    샘플레이트: {cosyvoice.sample_rate}")

# 프롬프트 오디오 준비 - CosyVoice가 자동으로 리샘플링하므로 원본 사용 가능
# 하지만 안전을 위해 16kHz로 변환 (speech_token 추출용)
prompt_wav_path = "assets/audio/test_prompt.wav"
print(f"\n[2] 프롬프트 오디오 준비 중...")
result = subprocess.run(
    f'ffmpeg -y -i "{PROMPT_AUDIO}" -ar 16000 -t 10 "{prompt_wav_path}"',
    shell=True, capture_output=True, text=True
)
if result.returncode != 0:
    print(f"    ffmpeg 오류: {result.stderr}")

if not os.path.exists(prompt_wav_path):
    print("    프롬프트 오디오 변환 실패!")
    sys.exit(1)

print(f"    프롬프트 오디오 준비 완료: {prompt_wav_path}")

# TTS 생성 테스트
# 테스트 1: 간단한 한국어 문장
test_cases = [
    {
        "name": "테스트 1: 짧은 인사",
        "text": "안녕하세요. 반갑습니다.",
        "prompt_text": "안녕하세요. 저는 심리 상담사입니다."
    },
    {
        "name": "테스트 2: 면접 질문",
        "text": "오늘 면접에 참석해 주셔서 감사합니다. 먼저 간단한 자기소개를 부탁드릴게요.",
        "prompt_text": "안녕하세요. 저는 심리 상담사입니다."
    },
    {
        "name": "테스트 3: 긴 문장",
        "text": "네, 좋은 질문이네요. 그 경험에서 배운 점은 무엇인가요? 그리고 그것을 어떻게 실제 업무에 적용하셨나요?",
        "prompt_text": "안녕하세요. 저는 심리 상담사입니다."
    }
]

for i, test in enumerate(test_cases):
    print(f"\n[{i+3}] {test['name']}")
    print(f"    합성 텍스트: {test['text']}")
    print(f"    프롬프트 텍스트: {test['prompt_text']}")
    print("    TTS 생성 중...")

    start = time.time()
    output_audio = None

    try:
        for j, result in enumerate(cosyvoice.inference_zero_shot(
            tts_text=test['text'],
            prompt_text=test['prompt_text'],
            prompt_wav=prompt_wav_path,
            stream=False,
            speed=1.0
        )):
            if output_audio is None:
                output_audio = result['tts_speech']
            else:
                output_audio = torch.cat([output_audio, result['tts_speech']], dim=1)

        elapsed = time.time() - start

        if output_audio is not None:
            duration = output_audio.shape[1] / cosyvoice.sample_rate
            rtf = elapsed / duration
            print(f"    생성 완료!")
            print(f"    - 오디오 길이: {duration:.2f}초")
            print(f"    - 생성 시간: {elapsed:.2f}초")
            print(f"    - RTF: {rtf:.3f}")

            # 저장
            output_path = f"test_output_{i+1}.wav"
            torchaudio.save(output_path, output_audio, cosyvoice.sample_rate)
            print(f"    - 저장됨: {output_path}")
        else:
            print("    출력 오디오 없음!")

    except Exception as e:
        print(f"    오류 발생: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
print("\n생성된 파일들을 재생하여 음성 품질을 확인하세요:")
for i in range(len(test_cases)):
    print(f"  - test_output_{i+1}.wav")
print("\n음성이 프롬프트 텍스트가 아닌 합성 텍스트를 말하는지 확인하세요.")
