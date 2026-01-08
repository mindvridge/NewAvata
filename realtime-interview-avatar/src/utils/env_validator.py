"""
환경 변수 검증 유틸리티

.env 파일의 환경 변수를 검증하고 누락된 항목을 알려줍니다.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnvVar:
    """환경 변수 정의"""
    name: str
    required: bool = True
    default: Optional[str] = None
    description: str = ""
    validation: Optional[callable] = None


# 필수 환경 변수 정의
REQUIRED_ENV_VARS = [
    # 서버
    EnvVar("SERVER_HOST", required=False, default="0.0.0.0", description="서버 호스트"),
    EnvVar("SERVER_PORT", required=False, default="8000", description="서버 포트"),
    EnvVar("API_KEY", required=True, description="API 인증 키"),
    
    # 외부 API (최소 하나는 필수)
    EnvVar("OPENAI_API_KEY", required=False, description="OpenAI API 키"),
    EnvVar("ANTHROPIC_API_KEY", required=False, description="Anthropic API 키"),
    
    # STT
    EnvVar("STT_PROVIDER", required=False, default="deepgram", description="STT 제공자"),
    EnvVar("DEEPGRAM_API_KEY", required=False, description="Deepgram API 키"),
    
    # TTS
    EnvVar("TTS_PROVIDER", required=False, default="edge", description="TTS 제공자"),
    
    # GPU
    EnvVar("CUDA_VISIBLE_DEVICES", required=False, default="0", description="사용할 GPU ID"),
]


def validate_env_vars() -> Dict[str, List[str]]:
    """
    환경 변수 검증
    
    Returns:
        Dict with 'missing', 'warnings', 'info' keys
    """
    results = {
        "missing": [],
        "warnings": [],
        "info": []
    }
    
    for env_var in REQUIRED_ENV_VARS:
        value = os.getenv(env_var.name)
        
        if value is None:
            if env_var.required:
                results["missing"].append(
                    f"❌ {env_var.name}: {env_var.description} (필수)"
                )
            elif env_var.default:
                results["info"].append(
                    f"ℹ️  {env_var.name}: 기본값 '{env_var.default}' 사용"
                )
                os.environ[env_var.name] = env_var.default
            else:
                results["warnings"].append(
                    f"⚠️  {env_var.name}: {env_var.description} (권장)"
                )
        else:
            # 값이 있으면 검증
            if env_var.validation and not env_var.validation(value):
                results["warnings"].append(
                    f"⚠️  {env_var.name}: 유효하지 않은 값"
                )
    
    # 특수 검증: LLM API 키는 하나 이상 필요
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        results["missing"].append(
            "❌ OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 중 하나는 필수입니다"
        )
    
    # STT 제공자에 따른 API 키 검증
    stt_provider = os.getenv("STT_PROVIDER", "deepgram")
    if stt_provider == "deepgram" and not os.getenv("DEEPGRAM_API_KEY"):
        results["warnings"].append(
            "⚠️  DEEPGRAM_API_KEY: Deepgram STT를 사용하려면 필요합니다"
        )
    
    # TTS 제공자에 따른 API 키 검증
    tts_provider = os.getenv("TTS_PROVIDER", "edge")
    if tts_provider == "elevenlabs" and not os.getenv("ELEVENLABS_API_KEY"):
        results["warnings"].append(
            "⚠️  ELEVENLABS_API_KEY: ElevenLabs TTS를 사용하려면 필요합니다"
        )
    
    return results


def print_validation_results(results: Dict[str, List[str]]):
    """검증 결과 출력"""
    if results["missing"]:
        logger.error("\n=== 누락된 필수 환경 변수 ===")
        for msg in results["missing"]:
            logger.error(msg)
        logger.error("\n.env 파일을 확인하고 필수 환경 변수를 설정해주세요.")
        logger.error("예시: cp .env.example .env\n")
        return False
    
    if results["warnings"]:
        logger.warning("\n=== 경고 ===")
        for msg in results["warnings"]:
            logger.warning(msg)
    
    if results["info"]:
        logger.info("\n=== 정보 ===")
        for msg in results["info"]:
            logger.info(msg)
    
    logger.info("\n✅ 환경 변수 검증 완료\n")
    return True


def check_security():
    """보안 설정 확인"""
    warnings = []
    
    # DEBUG 모드 확인
    if os.getenv("DEBUG", "false").lower() == "true":
        warnings.append("⚠️  프로덕션 환경에서는 DEBUG=false로 설정하세요")
    
    # API 키 기본값 확인
    api_key = os.getenv("API_KEY", "")
    if api_key in ["", "your_secret_api_key_here", "test_api_key"]:
        warnings.append("⚠️  API_KEY를 실제 값으로 변경하세요")
    
    # JWT 시크릿 확인
    jwt_secret = os.getenv("JWT_SECRET", "")
    if jwt_secret in ["", "your-secret-jwt-key-here"]:
        warnings.append("⚠️  JWT_SECRET을 실제 값으로 변경하세요")
    
    # CORS 설정 확인
    cors_origins = os.getenv("CORS_ORIGINS", "")
    if "*" in cors_origins or "0.0.0.0" in cors_origins:
        warnings.append("⚠️  CORS_ORIGINS에 특정 도메인만 허용하세요 (프로덕션)")
    
    if warnings:
        logger.warning("\n=== 보안 경고 ===")
        for warning in warnings:
            logger.warning(warning)
        logger.warning("")
    
    return len(warnings) == 0


def load_env_file(env_file: str = ".env"):
    """
    .env 파일 로드
    
    Args:
        env_file: .env 파일 경로
    """
    if not os.path.exists(env_file):
        logger.warning(f"⚠️  {env_file} 파일을 찾을 수 없습니다")
        logger.warning("예시 파일을 복사하세요: cp .env.example .env")
        return False
    
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=True)
        logger.info(f"✅ {env_file} 로드 완료")
        return True
    except ImportError:
        logger.error("❌ python-dotenv 패키지가 설치되지 않았습니다")
        logger.error("설치: pip install python-dotenv")
        return False


def validate_all(env_file: str = ".env") -> bool:
    """
    전체 검증 실행
    
    Args:
        env_file: .env 파일 경로
    
    Returns:
        검증 통과 여부
    """
    print("\n" + "="*60)
    print("🔍 환경 변수 검증 시작")
    print("="*60 + "\n")
    
    # .env 파일 로드
    if not load_env_file(env_file):
        return False
    
    # 환경 변수 검증
    results = validate_env_vars()
    valid = print_validation_results(results)
    
    if not valid:
        return False
    
    # 보안 검증
    check_security()
    
    print("="*60)
    print("✅ 검증 완료")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    import sys
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    
    # 검증 실행
    success = validate_all()
    
    # 검증 실패 시 종료 코드 1
    sys.exit(0 if success else 1)
