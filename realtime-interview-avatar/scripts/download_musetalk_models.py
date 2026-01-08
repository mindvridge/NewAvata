"""
MuseTalk 공식 모델 다운로드 스크립트

공식 저장소: https://github.com/TMElyralab/MuseTalk
HuggingFace: https://huggingface.co/TMElyralab/MuseTalk

사용법:
    python scripts/download_musetalk_models.py
    python scripts/download_musetalk_models.py --use-mirror  # 중국 미러 사용
    python scripts/download_musetalk_models.py --models-dir ./custom/path
"""

import os
import sys
import argparse
import subprocess
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
import time
import urllib.request
import ssl

# Windows 콘솔 UTF-8 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 색상 출력
try:
    import colorama
    colorama.init()
    USE_COLOR = True
except ImportError:
    USE_COLOR = False


class Colors:
    if USE_COLOR:
        RED = '\033[0;31m'
        GREEN = '\033[0;32m'
        YELLOW = '\033[1;33m'
        BLUE = '\033[0;34m'
        CYAN = '\033[0;36m'
        NC = '\033[0m'
    else:
        RED = GREEN = YELLOW = BLUE = CYAN = NC = ''


def print_header(text: str):
    print(f"\n{Colors.BLUE}{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}{Colors.NC}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}[OK] {text}{Colors.NC}")


def print_error(text: str):
    print(f"{Colors.RED}[ERROR] {text}{Colors.NC}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.NC}")


def print_info(text: str):
    print(f"{Colors.CYAN}[INFO] {text}{Colors.NC}")


def get_file_size_str(size_bytes: int) -> str:
    """파일 크기를 읽기 좋은 형태로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_file_with_progress(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """
    파일 다운로드 (진행률 표시)
    """
    print_info(f"{desc}")
    print_info(f"URL: {url}")
    print_info(f"저장: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # SSL 컨텍스트 (인증서 검증 무시 옵션)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # tqdm 사용 시도
        try:
            from tqdm import tqdm

            response = urllib.request.urlopen(url, context=ctx)
            total_size = int(response.headers.get('content-length', 0))

            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        except ImportError:
            # tqdm 없으면 기본 다운로드
            print_info("다운로드 중... (tqdm이 없어 진행률 표시 불가)")
            urllib.request.urlretrieve(url, output_path)

        if output_path.exists():
            size = output_path.stat().st_size
            print_success(f"다운로드 완료: {output_path.name} ({get_file_size_str(size)})")
            return True
        else:
            print_error("다운로드 실패: 파일이 생성되지 않음")
            return False

    except Exception as e:
        print_error(f"다운로드 실패: {e}")
        return False


def download_from_huggingface(
    repo_id: str,
    local_dir: Path,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    use_mirror: bool = False
) -> bool:
    """
    HuggingFace Hub에서 모델 다운로드
    """
    print_info(f"HuggingFace 다운로드: {repo_id}")
    print_info(f"저장 경로: {local_dir}")

    try:
        from huggingface_hub import snapshot_download, hf_hub_download

        # 미러 설정
        if use_mirror:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print_info("HuggingFace 미러 사용: hf-mirror.com")

        local_dir.mkdir(parents=True, exist_ok=True)

        # 다운로드 옵션
        kwargs = {
            'repo_id': repo_id,
            'local_dir': str(local_dir),
            'local_dir_use_symlinks': False,
            'resume_download': True,
        }

        if allow_patterns:
            kwargs['allow_patterns'] = allow_patterns
        if ignore_patterns:
            kwargs['ignore_patterns'] = ignore_patterns

        snapshot_download(**kwargs)

        print_success(f"다운로드 완료: {repo_id}")
        return True

    except ImportError:
        print_error("huggingface_hub가 설치되지 않았습니다.")
        print_info("설치: pip install huggingface_hub")
        return False

    except Exception as e:
        print_error(f"다운로드 실패: {e}")
        return False


def download_musetalk_models(models_dir: Path, use_mirror: bool = False) -> bool:
    """
    MuseTalk 메인 모델 다운로드 (musetalk.json, pytorch_model.bin)
    """
    print_header("1. MuseTalk 모델 다운로드")

    musetalk_dir = models_dir / "musetalk"

    # 이미 존재하는지 확인
    required_files = ["musetalk.json", "pytorch_model.bin"]
    existing = [f for f in required_files if (musetalk_dir / f).exists()]

    if len(existing) == len(required_files):
        print_info("MuseTalk 모델이 이미 존재합니다.")
        for f in required_files:
            size = (musetalk_dir / f).stat().st_size
            print_success(f"  {f}: {get_file_size_str(size)}")
        return True

    return download_from_huggingface(
        "TMElyralab/MuseTalk",
        musetalk_dir,
        allow_patterns=["musetalk.json", "pytorch_model.bin", "*.json"],
        use_mirror=use_mirror
    )


def download_vae_model(models_dir: Path, use_mirror: bool = False) -> bool:
    """
    SD VAE (sd-vae-ft-mse) 다운로드
    """
    print_header("2. SD-VAE-FT-MSE 모델 다운로드")

    vae_dir = models_dir / "sd-vae-ft-mse"

    # 필수 파일 확인
    required_files = ["config.json", "diffusion_pytorch_model.safetensors"]
    alt_files = ["config.json", "diffusion_pytorch_model.bin"]

    if all((vae_dir / f).exists() for f in required_files):
        print_info("VAE 모델이 이미 존재합니다 (safetensors).")
        return True

    if all((vae_dir / f).exists() for f in alt_files):
        print_info("VAE 모델이 이미 존재합니다 (bin).")
        return True

    return download_from_huggingface(
        "stabilityai/sd-vae-ft-mse",
        vae_dir,
        ignore_patterns=["*.md", "*.txt"],
        use_mirror=use_mirror
    )


def download_whisper_model(models_dir: Path) -> bool:
    """
    Whisper tiny 모델 다운로드
    """
    print_header("3. Whisper Tiny 모델 다운로드")

    whisper_dir = models_dir / "whisper"
    whisper_path = whisper_dir / "tiny.pt"

    if whisper_path.exists():
        size = whisper_path.stat().st_size
        print_info(f"Whisper 모델이 이미 존재합니다: {get_file_size_str(size)}")
        return True

    # OpenAI 공식 URL
    whisper_url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"

    return download_file_with_progress(whisper_url, whisper_path, "Whisper tiny 모델 다운로드")


def download_dwpose_model(models_dir: Path, use_mirror: bool = False) -> bool:
    """
    DWPose 모델 다운로드
    """
    print_header("4. DWPose 모델 다운로드")

    dwpose_dir = models_dir / "dwpose"

    # 필수 파일 확인
    required_files = ["dw-ll_ucoco_384.onnx"]

    if all((dwpose_dir / f).exists() for f in required_files):
        print_info("DWPose 모델이 이미 존재합니다.")
        return True

    return download_from_huggingface(
        "yzd-v/DWPose",
        dwpose_dir,
        allow_patterns=["*.onnx", "*.pth"],
        use_mirror=use_mirror
    )


def download_face_parse_model(models_dir: Path) -> bool:
    """
    Face Parse BiSeNet 모델 다운로드
    """
    print_header("5. Face Parse BiSeNet 모델 다운로드")

    face_parse_dir = models_dir / "face-parse-bisent"
    face_parse_dir.mkdir(parents=True, exist_ok=True)

    # 79999_iter.pth
    bisenet_path = face_parse_dir / "79999_iter.pth"
    if not bisenet_path.exists():
        # Google Drive에서 직접 다운로드는 어려우므로 HuggingFace 대체 사용
        print_info("BiSeNet 모델 다운로드 중...")

        # HuggingFace 대체 소스
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id="jonathandinu/face-parsing",
                filename="79999_iter.pth",
                local_dir=str(face_parse_dir),
                local_dir_use_symlinks=False
            )
            print_success("79999_iter.pth 다운로드 완료")
        except Exception as e:
            print_warning(f"HuggingFace에서 다운로드 실패: {e}")
            print_info("수동 다운로드 필요:")
            print_info("https://github.com/zllrunning/face-parsing.PyTorch")

    # resnet18 backbone
    resnet_path = face_parse_dir / "resnet18-5c106cde.pth"
    if not resnet_path.exists():
        resnet_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        download_file_with_progress(resnet_url, resnet_path, "ResNet18 다운로드")

    return bisenet_path.exists()


def download_gfpgan_model(models_dir: Path) -> bool:
    """
    GFPGAN 모델 다운로드 (선택사항)
    """
    print_header("6. GFPGAN 모델 다운로드 (선택)")

    gfpgan_dir = models_dir / "gfpgan"
    gfpgan_path = gfpgan_dir / "GFPGANv1.4.pth"

    if gfpgan_path.exists():
        print_info("GFPGAN 모델이 이미 존재합니다.")
        return True

    try:
        response = input("GFPGAN 모델을 다운로드하시겠습니까? (얼굴 향상용, ~330MB) [y/N]: ").strip().lower()
    except EOFError:
        response = 'n'

    if response != 'y':
        print_info("GFPGAN 다운로드 건너뜀")
        return True

    gfpgan_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    return download_file_with_progress(gfpgan_url, gfpgan_path, "GFPGAN v1.4 다운로드")


def verify_downloads(models_dir: Path) -> Dict[str, bool]:
    """
    다운로드된 모델 파일 검증
    """
    print_header("다운로드 검증")

    results = {}

    # MuseTalk 모델
    musetalk_files = [
        models_dir / "musetalk" / "musetalk.json",
        models_dir / "musetalk" / "pytorch_model.bin",
    ]
    results["MuseTalk"] = all(f.exists() for f in musetalk_files)

    # VAE 모델
    vae_files = [
        models_dir / "sd-vae-ft-mse" / "config.json",
    ]
    vae_model = (
        (models_dir / "sd-vae-ft-mse" / "diffusion_pytorch_model.safetensors").exists() or
        (models_dir / "sd-vae-ft-mse" / "diffusion_pytorch_model.bin").exists()
    )
    results["VAE"] = all(f.exists() for f in vae_files) and vae_model

    # Whisper
    results["Whisper"] = (models_dir / "whisper" / "tiny.pt").exists()

    # DWPose
    dwpose_dir = models_dir / "dwpose"
    results["DWPose"] = (
        (dwpose_dir / "dw-ll_ucoco_384.onnx").exists() or
        (dwpose_dir / "dw-ll_ucoco_384.pth").exists()
    )

    # Face Parse
    results["FaceParse"] = (models_dir / "face-parse-bisent" / "79999_iter.pth").exists()

    # GFPGAN (선택)
    results["GFPGAN"] = (models_dir / "gfpgan" / "GFPGANv1.4.pth").exists()

    # 결과 출력
    for name, status in results.items():
        if status:
            print_success(f"{name}: 설치됨")
        else:
            if name == "GFPGAN":
                print_warning(f"{name}: 미설치 (선택사항)")
            else:
                print_error(f"{name}: 미설치")

    return results


def create_config_yaml(models_dir: Path):
    """
    모델 경로 설정 파일 생성
    """
    print_header("설정 파일 생성")

    config_path = models_dir / "config.yaml"

    config_content = f"""# MuseTalk 모델 설정
# 자동 생성됨: {time.strftime('%Y-%m-%d %H:%M:%S')}

# 모델 경로
model_dir: {models_dir}

# MuseTalk 모델
musetalk:
  config: {models_dir / "musetalk" / "musetalk.json"}
  checkpoint: {models_dir / "musetalk" / "pytorch_model.bin"}

# VAE 모델
vae:
  path: {models_dir / "sd-vae-ft-mse"}
  type: stabilityai/sd-vae-ft-mse

# Whisper 모델
whisper:
  model: tiny
  checkpoint: {models_dir / "whisper" / "tiny.pt"}

# DWPose 모델
dwpose:
  path: {models_dir / "dwpose"}

# Face Parse 모델
face_parse:
  checkpoint: {models_dir / "face-parse-bisent" / "79999_iter.pth"}
  backbone: {models_dir / "face-parse-bisent" / "resnet18-5c106cde.pth"}

# GFPGAN (선택)
gfpgan:
  checkpoint: {models_dir / "gfpgan" / "GFPGANv1.4.pth"}
  enabled: {(models_dir / "gfpgan" / "GFPGANv1.4.pth").exists()}
"""

    config_path.write_text(config_content, encoding='utf-8')
    print_success(f"설정 파일 생성: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MuseTalk 공식 모델 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python scripts/download_musetalk_models.py
    python scripts/download_musetalk_models.py --use-mirror
    python scripts/download_musetalk_models.py --models-dir ./custom/models

필요한 모델:
    1. MuseTalk (musetalk.json, pytorch_model.bin) - ~500MB
    2. SD-VAE-FT-MSE (config.json, diffusion_pytorch_model.bin) - ~330MB
    3. Whisper tiny (tiny.pt) - ~75MB
    4. DWPose (dw-ll_ucoco_384.onnx) - ~200MB
    5. Face Parse BiSeNet (79999_iter.pth) - ~50MB
    6. GFPGAN (선택, GFPGANv1.4.pth) - ~330MB

총 용량: 약 1.5GB (GFPGAN 포함 시 ~1.8GB)
        """
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="모델 저장 디렉토리 (기본: ./models/musetalk)"
    )

    parser.add_argument(
        "--use-mirror",
        action="store_true",
        help="HuggingFace 중국 미러 사용 (다운로드 속도 향상)"
    )

    parser.add_argument(
        "--skip-gfpgan",
        action="store_true",
        help="GFPGAN 다운로드 건너뛰기"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="다운로드 없이 검증만 수행"
    )

    args = parser.parse_args()

    # 모델 디렉토리 설정
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        # 스크립트 위치 기준으로 프로젝트 루트 찾기
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        models_dir = project_root / "models" / "musetalk"

    print_header("MuseTalk 모델 다운로드")
    print_info(f"모델 디렉토리: {models_dir}")
    print_info(f"미러 사용: {args.use_mirror}")

    # 검증만 수행
    if args.verify_only:
        verify_downloads(models_dir)
        return 0

    # 디렉토리 생성
    models_dir.mkdir(parents=True, exist_ok=True)

    # huggingface_hub 설치 확인
    try:
        import huggingface_hub
        print_success("huggingface_hub 설치됨")
    except ImportError:
        print_warning("huggingface_hub가 설치되지 않았습니다.")
        print_info("설치 중...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

    # 모델 다운로드
    success = True

    success &= download_musetalk_models(models_dir, args.use_mirror)
    success &= download_vae_model(models_dir, args.use_mirror)
    success &= download_whisper_model(models_dir)
    success &= download_dwpose_model(models_dir, args.use_mirror)
    success &= download_face_parse_model(models_dir)

    if not args.skip_gfpgan:
        download_gfpgan_model(models_dir)

    # 검증
    results = verify_downloads(models_dir)

    # 설정 파일 생성
    create_config_yaml(models_dir)

    # 결과 요약
    print_header("다운로드 완료")

    required_models = ["MuseTalk", "VAE", "Whisper", "DWPose", "FaceParse"]
    missing = [m for m in required_models if not results.get(m, False)]

    if missing:
        print_error(f"누락된 필수 모델: {', '.join(missing)}")
        print_info("위의 모델을 수동으로 다운로드해주세요.")
        return 1
    else:
        print_success("모든 필수 모델이 설치되었습니다!")
        print()
        print(f"모델 위치: {models_dir}")
        print(f"설정 파일: {models_dir / 'config.yaml'}")
        print()
        print("다음 명령어로 테스트할 수 있습니다:")
        print(f"  python -m src.avatar.musetalk_wrapper --model-dir {models_dir}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
