"""
MuseTalk 모델 및 의존성 자동 설치 스크립트 (Python 버전)

사용법:
    python scripts/setup_musetalk.py
    python scripts/setup_musetalk.py --use-mirror
    python scripts/setup_musetalk.py --models-only
"""

import os
import sys
import argparse
import subprocess
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
import time

# 색상 코드 (Windows 호환)
try:
    import colorama
    colorama.init()
    USE_COLOR = True
except ImportError:
    USE_COLOR = False


class Colors:
    """터미널 색상"""
    if USE_COLOR:
        RED = '\033[0;31m'
        GREEN = '\033[0;32m'
        YELLOW = '\033[1;33m'
        BLUE = '\033[0;34m'
        NC = '\033[0m'
    else:
        RED = GREEN = YELLOW = BLUE = NC = ''


def print_header(text: str):
    """헤더 출력"""
    print(f"\n{Colors.BLUE}{'=' * 70}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 70}{Colors.NC}\n")


def print_success(text: str):
    """성공 메시지"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text: str):
    """에러 메시지"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_warning(text: str):
    """경고 메시지"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")


def print_info(text: str):
    """정보 메시지"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.NC}")


def check_command(command: str) -> bool:
    """명령어 존재 확인"""
    return shutil.which(command) is not None


def run_command(command: list, capture_output: bool = False) -> Tuple[bool, str]:
    """
    명령어 실행

    Args:
        command: 실행할 명령어 리스트
        capture_output: 출력 캡처 여부

    Returns:
        (성공 여부, 출력)
    """
    try:
        if capture_output:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout.strip()
        else:
            subprocess.run(command, check=True)
            return True, ""
    except subprocess.CalledProcessError as e:
        return False, str(e)


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """
    파일 다운로드 (진행률 표시)

    Args:
        url: 다운로드 URL
        output_path: 저장 경로
        desc: 설명
    """
    print_info(f"{desc}: {url}")

    try:
        # tqdm이 있으면 진행률 표시
        try:
            from tqdm import tqdm

            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get('content-length', 0))

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        except ImportError:
            # tqdm이 없으면 기본 다운로드
            urllib.request.urlretrieve(url, output_path)

        print_success(f"다운로드 완료: {output_path.name}")
        return True

    except Exception as e:
        print_error(f"다운로드 실패: {e}")
        return False


def download_from_hf(
    repo_id: str,
    local_dir: Path,
    allow_patterns: Optional[str] = None,
    use_mirror: bool = False
):
    """
    Hugging Face에서 모델 다운로드

    Args:
        repo_id: Hugging Face 저장소 ID
        local_dir: 로컬 저장 디렉토리
        allow_patterns: 다운로드할 파일 패턴
        use_mirror: 중국 미러 사용 여부
    """
    print_info(f"다운로드 중: {repo_id}")

    try:
        from huggingface_hub import snapshot_download

        # 미러 설정
        if use_mirror:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print_info("Hugging Face 미러 사용")

        # 다운로드
        kwargs = {
            'repo_id': repo_id,
            'local_dir': str(local_dir),
            'resume_download': True
        }

        if allow_patterns:
            kwargs['allow_patterns'] = allow_patterns

        snapshot_download(**kwargs)

        print_success(f"다운로드 완료: {repo_id}")
        return True

    except Exception as e:
        print_error(f"다운로드 실패: {e}")
        return False


def check_environment():
    """환경 확인"""
    print_header("환경 확인")

    # Python 버전
    python_version = sys.version.split()[0]
    print_success(f"Python 버전: {python_version}")

    if sys.version_info < (3, 8):
        print_error("Python 3.8 이상이 필요합니다")
        return False

    # Git
    if check_command('git'):
        success, version = run_command(['git', '--version'], capture_output=True)
        if success:
            print_success(f"Git: {version}")
    else:
        print_warning("Git이 설치되어 있지 않습니다")

    # pip
    try:
        import pip
        print_success("pip 설치됨")
    except ImportError:
        print_error("pip이 설치되어 있지 않습니다")
        return False

    return True


def install_huggingface_hub():
    """Hugging Face CLI 설치"""
    print_header("Hugging Face CLI 설치")

    try:
        import huggingface_hub
        print_info("huggingface_hub 이미 설치됨")
        return True
    except ImportError:
        print_info("huggingface_hub 설치 중...")

        success, _ = run_command([
            sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'
        ])

        if success:
            print_success("huggingface_hub 설치 완료")
            return True
        else:
            print_error("huggingface_hub 설치 실패")
            return False


def clone_musetalk_repo(project_root: Path, skip_clone: bool = False):
    """MuseTalk 저장소 클론"""
    print_header("MuseTalk 저장소 클론")

    repo_dir = project_root / "third_party" / "MuseTalk"

    if repo_dir.exists() and skip_clone:
        print_info("저장소 클론 건너뛰기 (이미 존재함)")
        return True

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if repo_dir.exists():
        print_warning("기존 저장소 삭제 중...")
        shutil.rmtree(repo_dir)

    print_info("MuseTalk 저장소 클론 중...")

    success, _ = run_command([
        'git', 'clone',
        'https://github.com/TMElyralab/MuseTalk.git',
        str(repo_dir)
    ])

    if success:
        print_success("저장소 클론 완료")
        return True
    else:
        print_error("저장소 클론 실패")
        return False


def download_vae(models_dir: Path, use_mirror: bool = False):
    """VAE 모델 다운로드"""
    print_header("1. VAE 모델 다운로드")

    vae_dir = models_dir / "sd-vae-ft-mse"

    if (vae_dir / "diffusion_pytorch_model.safetensors").exists():
        print_info("VAE 모델이 이미 존재합니다. 건너뜁니다.")
        return True

    return download_from_hf(
        "stabilityai/sd-vae-ft-mse",
        vae_dir,
        allow_patterns="*.safetensors",
        use_mirror=use_mirror
    )


def download_whisper(models_dir: Path):
    """Whisper 모델 다운로드"""
    print_header("2. Whisper 모델 다운로드")

    whisper_dir = models_dir / "whisper"

    if (whisper_dir / "tiny.pt").exists():
        print_info("Whisper 모델이 이미 존재합니다. 건너뜁니다.")
        return True

    whisper_dir.mkdir(parents=True, exist_ok=True)

    print_info("Whisper tiny 모델 다운로드 중...")

    try:
        import whisper

        # tiny 모델 다운로드
        model = whisper.load_model('tiny')
        print_success("Whisper tiny 모델 다운로드 완료")

        # 캐시에서 모델 파일 복사
        cache_dir = Path.home() / ".cache" / "whisper"
        tiny_path = cache_dir / "tiny.pt"

        if tiny_path.exists():
            shutil.copy(tiny_path, whisper_dir / "tiny.pt")
            print_success("모델 파일 복사 완료")
            return True
        else:
            print_warning("캐시 파일을 찾을 수 없습니다")
            return False

    except Exception as e:
        print_warning(f"Whisper 다운로드 중 오류: {e}")
        print_info("수동으로 다운로드해주세요:")
        print_info("https://openaipublic.azureedge.net/main/whisper/models/tiny.pt")
        return False


def download_dwpose(models_dir: Path, use_mirror: bool = False):
    """DWPose 모델 다운로드"""
    print_header("3. DWPose 모델 다운로드")

    dwpose_dir = models_dir / "dwpose"

    if dwpose_dir.exists() and list(dwpose_dir.glob("*.onnx")):
        print_info("DWPose 모델이 이미 존재합니다. 건너뜁니다.")
        return True

    return download_from_hf(
        "yzd-v/DWPose",
        dwpose_dir,
        use_mirror=use_mirror
    )


def download_face_parse(models_dir: Path, use_mirror: bool = False):
    """Face Parse BiSeNet 모델 다운로드"""
    print_header("4. Face Parse BiSeNet 모델 다운로드")

    face_parse_dir = models_dir / "face-parse-bisent"

    if face_parse_dir.exists() and list(face_parse_dir.glob("*.pth")):
        print_info("Face Parse 모델이 이미 존재합니다. 건너뜁니다.")
        return True

    return download_from_hf(
        "jonathandinu/face-parsing",
        face_parse_dir,
        allow_patterns="*.pth",
        use_mirror=use_mirror
    )


def download_musetalk_checkpoint(models_dir: Path, use_mirror: bool = False):
    """MuseTalk 체크포인트 다운로드"""
    print_header("5. MuseTalk 체크포인트 다운로드")

    checkpoint_dir = models_dir / "musetalk"

    if checkpoint_dir.exists() and list(checkpoint_dir.glob("*.pth")):
        print_info("MuseTalk 체크포인트가 이미 존재합니다. 건너뜁니다.")
        return True

    return download_from_hf(
        "TMElyralab/MuseTalk",
        checkpoint_dir,
        use_mirror=use_mirror
    )


def download_gfpgan(models_dir: Path):
    """GFPGAN 모델 다운로드 (선택사항)"""
    print_header("6. GFPGAN 모델 다운로드 (얼굴 향상)")

    gfpgan_dir = models_dir / "gfpgan"
    gfpgan_path = gfpgan_dir / "GFPGANv1.4.pth"

    if gfpgan_path.exists():
        print_info("GFPGAN 모델이 이미 존재합니다. 건너뜁니다.")
        return True

    # 사용자 확인
    try:
        response = input("GFPGAN 모델을 다운로드하시겠습니까? (y/N): ").strip().lower()
    except:
        response = 'n'

    if response != 'y':
        print_info("GFPGAN 다운로드 건너뜀")
        return True

    gfpgan_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

    return download_file(gfpgan_url, gfpgan_path, "GFPGAN v1.4 다운로드")


def create_config_file(models_dir: Path):
    """모델 경로 설정 파일 생성"""
    print_header("모델 경로 설정 파일 생성")

    config_file = models_dir / "model_paths.yaml"

    vae_dir = models_dir / "sd-vae-ft-mse"
    whisper_dir = models_dir / "whisper"
    dwpose_dir = models_dir / "dwpose"
    face_parse_dir = models_dir / "face-parse-bisent"
    checkpoint_dir = models_dir / "musetalk"
    gfpgan_dir = models_dir / "gfpgan"

    gfpgan_enabled = (gfpgan_dir / "GFPGANv1.4.pth").exists()

    config_content = f"""# MuseTalk 모델 경로 설정
# 자동 생성됨: {time.strftime('%Y-%m-%d %H:%M:%S')}

vae:
  path: {vae_dir}
  checkpoint: diffusion_pytorch_model.safetensors

whisper:
  path: {whisper_dir}
  checkpoint: tiny.pt

dwpose:
  path: {dwpose_dir}
  det_checkpoint: yolox_l.onnx
  pose_checkpoint: dw-ll_ucoco_384.onnx

face_parse:
  path: {face_parse_dir}
  checkpoint: 79999_iter.pth

musetalk:
  path: {checkpoint_dir}
  unet_checkpoint: musetalk_unet.pth
  audio_processor: audio_processor.pth

gfpgan:
  path: {gfpgan_dir}
  checkpoint: GFPGANv1.4.pth
  enabled: {str(gfpgan_enabled).lower()}
"""

    config_file.write_text(config_content, encoding='utf-8')

    print_success(f"설정 파일 생성: {config_file}")
    return True


def validate_models(models_dir: Path):
    """모델 파일 검증"""
    print_header("모델 파일 검증")

    validation_passed = True

    # VAE 검증
    vae_file = models_dir / "sd-vae-ft-mse" / "diffusion_pytorch_model.safetensors"
    if vae_file.exists():
        size = vae_file.stat().st_size / (1024 * 1024)  # MB
        print_success(f"VAE: {size:.1f} MB")
    else:
        print_error("VAE 모델 파일이 없습니다")
        validation_passed = False

    # Whisper 검증
    whisper_file = models_dir / "whisper" / "tiny.pt"
    if whisper_file.exists():
        size = whisper_file.stat().st_size / (1024 * 1024)  # MB
        print_success(f"Whisper: {size:.1f} MB")
    else:
        print_warning("Whisper 모델 파일이 없습니다")

    # DWPose 검증
    dwpose_dir = models_dir / "dwpose"
    if dwpose_dir.exists() and list(dwpose_dir.glob("*.onnx")):
        print_success("DWPose: 설치됨")
    else:
        print_error("DWPose 모델이 없습니다")
        validation_passed = False

    # Face Parse 검증
    face_parse_dir = models_dir / "face-parse-bisent"
    if face_parse_dir.exists() and list(face_parse_dir.glob("*.pth")):
        print_success("Face Parse: 설치됨")
    else:
        print_error("Face Parse 모델이 없습니다")
        validation_passed = False

    # MuseTalk 검증
    musetalk_dir = models_dir / "musetalk"
    if musetalk_dir.exists() and list(musetalk_dir.glob("*.pth")):
        print_success("MuseTalk: 설치됨")
    else:
        print_error("MuseTalk 체크포인트가 없습니다")
        validation_passed = False

    return validation_passed


def run_tests():
    """설치 테스트 실행"""
    print_header("테스트 실행")

    try:
        response = input("설치 테스트를 실행하시겠습니까? (y/N): ").strip().lower()
    except:
        response = 'n'

    if response != 'y':
        print_info("테스트 건너뜀")
        return True

    print_info("테스트 스크립트 실행 중...\n")
    print("=== Python 환경 테스트 ===")

    # 필수 패키지 확인
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('diffusers', 'diffusers'),
        ('transformers', 'transformers'),
        ('opencv-python', 'cv2'),
        ('mediapipe', 'mediapipe'),
        ('whisper', 'whisper')
    ]

    missing_packages = []

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print_success(package_name)
        except ImportError:
            print_error(f"{package_name} (설치 필요)")
            missing_packages.append(package_name)

    if missing_packages:
        print_warning("\n다음 패키지를 설치해주세요:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False

    print_success("\n모든 필수 패키지가 설치되어 있습니다")

    # PyTorch 정보
    try:
        import torch

        print("\n=== PyTorch 정보 ===")
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            print(f"MPS 사용 가능: {torch.backends.mps.is_available()}")

    except Exception as e:
        print_warning(f"PyTorch 정보 조회 실패: {e}")

    print_success("\n테스트 완료!")
    return True


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="MuseTalk 모델 및 의존성 자동 설치"
    )
    parser.add_argument(
        '--use-mirror',
        action='store_true',
        help='중국 Hugging Face 미러 사용 (다운로드 속도 향상)'
    )
    parser.add_argument(
        '--skip-clone',
        action='store_true',
        help='저장소 클론 건너뛰기 (이미 클론된 경우)'
    )
    parser.add_argument(
        '--models-only',
        action='store_true',
        help='모델만 다운로드 (저장소 클론 안함)'
    )

    args = parser.parse_args()

    # 프로젝트 루트 디렉토리
    project_root = Path(__file__).parent.parent.absolute()
    models_dir = project_root / "models" / "musetalk"

    print_header("MuseTalk 모델 설치")
    print_info(f"프로젝트 루트: {project_root}")
    print_info(f"모델 디렉토리: {models_dir}")

    # 환경 확인
    if not check_environment():
        print_error("환경 확인 실패")
        return 1

    # 디렉토리 생성
    models_dir.mkdir(parents=True, exist_ok=True)

    # Hugging Face CLI 설치
    if not install_huggingface_hub():
        return 1

    # MuseTalk 저장소 클론
    if not args.models_only:
        if not clone_musetalk_repo(project_root, args.skip_clone):
            print_warning("저장소 클론 실패 (계속 진행)")

    # 모델 다운로드
    success = True

    success &= download_vae(models_dir, args.use_mirror)
    success &= download_whisper(models_dir)
    success &= download_dwpose(models_dir, args.use_mirror)
    success &= download_face_parse(models_dir, args.use_mirror)
    success &= download_musetalk_checkpoint(models_dir, args.use_mirror)
    download_gfpgan(models_dir)  # 선택사항

    # 설정 파일 생성
    create_config_file(models_dir)

    # 검증
    validation_passed = validate_models(models_dir)

    # 테스트
    run_tests()

    # 완료
    print_header("설치 완료!")

    print(f"{Colors.GREEN}")
    print("MuseTalk 모델 설치가 완료되었습니다.")
    print()
    print(f"모델 위치: {models_dir}")
    print(f"설정 파일: {models_dir / 'model_paths.yaml'}")
    print()
    print("다음 명령어로 아바타를 사용할 수 있습니다:")
    print("  python -m src.avatar.example_usage")
    print(f"{Colors.NC}")

    if not validation_passed:
        print_warning("일부 모델이 누락되었습니다. 위의 에러를 확인해주세요.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
