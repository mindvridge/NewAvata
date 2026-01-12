"""
실시간 립싱크 추론 스크립트

사전 계산된 아바타 데이터를 로드하고, 오디오만 실시간으로 처리하여 립싱크 수행
UNet 추론만 실시간 루프에서 수행하므로 25 FPS 이상 가능

사용법:
    python realtime_lipsync.py --precomputed <사전계산파일.pkl> --audio_path <오디오.wav>
"""

import os
import sys
import cv2
import copy
import torch
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import time

# MuseTalk 경로 추가
MUSETALK_PATH = Path("c:/NewAvata/NewAvata/MuseTalk")
sys.path.insert(0, str(MUSETALK_PATH))

from musetalk.utils.blending import get_image_blending
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import datagen
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding
from transformers import WhisperModel


def realtime_lipsync(args):
    """실시간 립싱크 추론"""
    print("=" * 70)
    print("실시간 립싱크 추론")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ===== 1. 사전 계산 데이터 로드 =====
    print("\n[1/4] 사전 계산 데이터 로드...")
    start_time = time.time()

    with open(args.precomputed, 'rb') as f:
        precomputed = pickle.load(f)

    load_time = time.time() - start_time
    print(f"   로드 완료: {load_time:.2f}초")
    print(f"   영상: {precomputed['video_path']}")
    print(f"   프레임 수: {precomputed['total_frames']}")
    print(f"   FPS: {precomputed['fps']}")
    print(f"   얼굴 선택: {precomputed['face_index']}")

    coords_list = precomputed['coords_list']
    input_latent_list = precomputed['input_latent_list']
    frames = precomputed['frames']
    cached_mask = precomputed['cached_mask']
    cached_crop_box = precomputed['cached_crop_box']
    extra_margin = precomputed['extra_margin']
    fps = precomputed['fps']
    use_float16 = precomputed['use_float16']

    # ===== 2. 모델 로드 =====
    print("\n[2/4] 모델 로드...")
    start_time = time.time()

    # VAE
    vae = VAE(model_path="./models/sd-vae", use_float16=use_float16)
    if use_float16:
        vae.vae = vae.vae.half()
    vae.vae = vae.vae.to(device)

    # UNet
    unet = UNet(
        unet_config="./models/musetalkV15/musetalk.json",
        model_path="./models/musetalkV15/unet.pth",
        device=device
    )
    if use_float16:
        unet.model = unet.model.half()
    unet.model = unet.model.to(device)

    # Positional Encoding
    pe = PositionalEncoding(d_model=384)
    if use_float16:
        pe = pe.half()
    pe = pe.to(device)

    timesteps = torch.tensor([0], device=device)

    # Whisper
    weight_dtype = torch.float16 if use_float16 else torch.float32
    audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")
    whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    model_load_time = time.time() - start_time
    print(f"   모델 로드 완료: {model_load_time:.2f}초")

    # CUDA 워밍업 - 첫 번째 추론은 느리므로 미리 실행
    print("   CUDA 워밍업 중...")
    with torch.no_grad():
        dummy_latent = torch.randn(1, 8, 32, 32, device=device, dtype=weight_dtype)
        dummy_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
        dummy_audio_feat = pe(dummy_audio)
        _ = unet.model(dummy_latent, timesteps, encoder_hidden_states=dummy_audio_feat)
        torch.cuda.synchronize()
    print("   워밍업 완료")

    # ===== 3. 오디오 특징 추출 =====
    print("\n[3/4] 오디오 특징 추출...")
    start_time = time.time()

    whisper_input_features, librosa_length = audio_processor.get_audio_feature(args.audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=fps,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    )

    audio_time = time.time() - start_time
    print(f"   오디오 처리 완료: {audio_time:.2f}초")
    print(f"   Whisper chunks: {whisper_chunks.shape}")

    # ===== 4. 실시간 추론 (UNet만) =====
    print("\n[4/4] 실시간 추론...")

    # cycle 리스트 생성
    frame_list_cycle = frames + frames[::-1]
    coord_list_cycle = coords_list + coords_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # 배치 생성기
    batch_size = args.batch_size
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )

    res_frame_list = []
    video_num = len(whisper_chunks)
    total = int(np.ceil(float(video_num) / batch_size))

    # UNet 추론 타이밍
    unet_times = []

    print(f"   총 {video_num} 프레임 처리 (배치 크기: {batch_size})")

    with torch.no_grad():
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total, desc="   UNet 추론")):
            torch.cuda.synchronize()
            batch_start = time.time()

            audio_feature_batch = pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)

            pred_latents = unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample

            recon = vae.decode_latents(pred_latents)

            torch.cuda.synchronize()
            batch_time = time.time() - batch_start
            unet_times.append(batch_time)

            for res_frame in recon:
                res_frame_list.append(res_frame)

    # 타이밍 분석
    avg_batch_time = np.mean(unet_times) * 1000
    avg_frame_time = avg_batch_time / batch_size
    estimated_fps = 1000 / avg_frame_time

    print(f"\n   === 성능 분석 ===")
    print(f"   평균 배치 시간: {avg_batch_time:.1f}ms ({batch_size}프레임)")
    print(f"   평균 프레임 시간: {avg_frame_time:.1f}ms")
    print(f"   예상 FPS: {estimated_fps:.1f}")

    if estimated_fps >= 25:
        print(f"   [OK] 실시간 가능! (25 FPS 충족)")
    else:
        print(f"   [X] 실시간 불가 (25 FPS 미달)")

    # ===== 5. 블렌딩 & 저장 (최적화: 메모리 기반) =====
    print("\n[5/5] 블렌딩 & 비디오 저장...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    # 메모리에 프레임 저장 (파일 I/O 제거)
    blended_frames = []
    blend_times = []

    # 첫 프레임에서 비디오 크기 결정
    first_frame = frame_list_cycle[0]
    frame_height, frame_width = first_frame.shape[:2]

    for i, res_frame in enumerate(tqdm(res_frame_list, desc="   블렌딩")):
        blend_start = time.time()

        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = frame_list_cycle[i % len(frame_list_cycle)].copy()

        if bbox == coord_placeholder:
            blended_frames.append(ori_frame)
            blend_times.append(time.time() - blend_start)
            continue

        x1, y1, x2, y2 = bbox
        y2 = y2 + extra_margin
        y2 = min(y2, ori_frame.shape[0])

        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except:
            blended_frames.append(ori_frame)
            blend_times.append(time.time() - blend_start)
            continue

        # 캐시된 마스크로 빠른 블렌딩
        if cached_mask is not None and cached_crop_box is not None:
            combine_frame = get_image_blending(
                ori_frame, res_frame, [x1, y1, x2, y2],
                cached_mask, cached_crop_box
            )
        else:
            # fallback
            combine_frame = ori_frame.copy()
            combine_frame[y1:y2, x1:x2] = res_frame

        blended_frames.append(combine_frame)
        blend_times.append(time.time() - blend_start)

    avg_blend_time = np.mean(blend_times) * 1000
    print(f"   평균 블렌딩 시간: {avg_blend_time:.1f}ms/프레임")

    # 비디오 생성 (ffmpeg로 H.264 인코딩 - 브라우저 호환)
    print("   비디오 인코딩 중...")
    output_video = output_dir / "output_temp.mp4"
    output_with_audio = output_dir / "output_with_audio.mp4"

    # 임시로 mp4v로 저장 후 ffmpeg로 H.264 변환
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (frame_width, frame_height))

    for frame in tqdm(blended_frames, desc="   인코딩"):
        video_writer.write(frame)
    video_writer.release()

    # ffmpeg로 H.264 변환 + 오디오 합성 (브라우저 호환)
    os.system(f'ffmpeg -y -v warning -i "{output_video}" -i "{args.audio_path}" -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -c:a aac -shortest "{output_with_audio}"')

    # 임시 파일 삭제
    if output_video.exists():
        output_video.unlink()

    # ===== 최종 요약 =====
    total_realtime = avg_frame_time + avg_blend_time
    final_fps = 1000 / total_realtime

    print("\n" + "=" * 70)
    print("실시간 립싱크 완료!")
    print("=" * 70)
    print(f"  - 출력: {output_with_audio}")
    print(f"\n  === 실시간 성능 예측 ===")
    print(f"  - UNet 추론: {avg_frame_time:.1f}ms/프레임")
    print(f"  - 블렌딩: {avg_blend_time:.1f}ms/프레임")
    print(f"  - 총합: {total_realtime:.1f}ms/프레임")
    print(f"  - 예상 FPS: {final_fps:.1f}")
    print(f"  - 실시간 가능: {'[OK] YES' if final_fps >= 25 else '[X] NO'}")
    print("=" * 70)

    return output_with_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실시간 립싱크 추론")
    parser.add_argument("--precomputed", type=str, required=True, help="사전 계산 파일 경로")
    parser.add_argument("--audio_path", type=str, required=True, help="오디오 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./results/realtime", help="출력 디렉토리")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")

    args = parser.parse_args()

    os.chdir("c:/NewAvata/NewAvata/realtime-interview-avatar")
    realtime_lipsync(args)
