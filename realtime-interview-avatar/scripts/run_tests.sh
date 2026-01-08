#!/bin/bash

# ============================================================================
# 테스트 실행 스크립트
# ============================================================================
# 다양한 테스트 시나리오를 쉽게 실행할 수 있는 유틸리티 스크립트
# ============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 배너 출력
print_banner() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║            테스트 실행 스크립트                            ║"
    echo "║         Interview Avatar System Tests                     ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

# 도움말 출력
print_help() {
    cat << EOF
사용법: $0 [옵션]

옵션:
  all           모든 테스트 실행
  unit          단위 테스트만 실행
  integration   통합 테스트만 실행
  fast          빠른 테스트만 (slow 제외)
  slow          느린 테스트 포함
  benchmark     성능 벤치마크 실행
  coverage      커버리지 리포트 생성
  watch         watch 모드 (변경 감지 시 자동 재실행)
  ci            CI 환경용 테스트 (병렬, 커버리지)
  help          이 도움말 출력

예제:
  $0 all              # 모든 테스트
  $0 unit             # 단위 테스트만
  $0 fast             # 빠른 테스트만
  $0 integration      # 통합 테스트
  $0 coverage         # 커버리지 리포트

EOF
}

# 환경 확인
check_env() {
    log_info "환경 확인 중..."

    # Python 확인
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3이 설치되어 있지 않습니다."
        exit 1
    fi

    # pytest 확인
    if ! python3 -m pytest --version &> /dev/null; then
        log_warning "pytest가 설치되어 있지 않습니다. 설치 중..."
        pip install pytest pytest-asyncio pytest-cov
    fi

    log_success "환경 확인 완료"
}

# 로그 디렉토리 생성
setup_logs() {
    mkdir -p tests/logs
    mkdir -p htmlcov
}

# 전체 테스트
run_all_tests() {
    log_info "전체 테스트 실행 중..."

    pytest \
        -v \
        --tb=short \
        --maxfail=5 \
        --durations=10

    log_success "전체 테스트 완료"
}

# 단위 테스트
run_unit_tests() {
    log_info "단위 테스트 실행 중..."

    pytest \
        -v \
        -m "unit" \
        --tb=short

    log_success "단위 테스트 완료"
}

# 통합 테스트
run_integration_tests() {
    log_info "통합 테스트 실행 중..."

    pytest \
        -v \
        -s \
        -m "integration" \
        --tb=short \
        --durations=10

    log_success "통합 테스트 완료"
}

# 빠른 테스트
run_fast_tests() {
    log_info "빠른 테스트 실행 중 (slow 제외)..."

    pytest \
        -v \
        -m "not slow" \
        --tb=short

    log_success "빠른 테스트 완료"
}

# 느린 테스트 포함
run_all_with_slow() {
    log_info "느린 테스트 포함 실행 중..."

    pytest \
        -v \
        -s \
        --tb=short \
        --durations=20

    log_success "모든 테스트 완료"
}

# 벤치마크
run_benchmark() {
    log_info "성능 벤치마크 실행 중..."

    pytest \
        -v \
        -s \
        -m "benchmark" \
        --tb=short \
        --durations=0

    log_success "벤치마크 완료"
}

# 커버리지 리포트
run_coverage() {
    log_info "커버리지 리포트 생성 중..."

    pytest \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-report=xml \
        --cov-branch \
        -v

    log_success "커버리지 리포트 생성 완료"
    log_info "HTML 리포트: htmlcov/index.html"

    # 자동으로 HTML 리포트 열기
    if command -v xdg-open &> /dev/null; then
        xdg-open htmlcov/index.html
    elif command -v open &> /dev/null; then
        open htmlcov/index.html
    elif command -v start &> /dev/null; then
        start htmlcov/index.html
    fi
}

# Watch 모드
run_watch() {
    log_info "Watch 모드 시작 (pytest-watch 필요)..."

    # pytest-watch 설치 확인
    if ! python3 -m ptw --version &> /dev/null; then
        log_warning "pytest-watch가 설치되어 있지 않습니다. 설치 중..."
        pip install pytest-watch
    fi

    ptw -- -v -m "not slow"
}

# CI 환경용
run_ci() {
    log_info "CI 환경용 테스트 실행 중..."

    # 병렬 실행 (pytest-xdist 필요)
    if ! python3 -m pytest --version | grep -q xdist; then
        log_warning "pytest-xdist가 설치되어 있지 않습니다. 설치 중..."
        pip install pytest-xdist
    fi

    pytest \
        -v \
        -n auto \
        -m "not slow and not expensive" \
        --cov=src \
        --cov-report=xml \
        --cov-report=term \
        --tb=short \
        --maxfail=3 \
        --junitxml=junit.xml

    log_success "CI 테스트 완료"
    log_info "Coverage XML: coverage.xml"
    log_info "JUnit XML: junit.xml"
}

# 특정 테스트 실행
run_specific_test() {
    local test_path=$1

    if [ -z "$test_path" ]; then
        log_error "테스트 경로를 지정해주세요."
        echo "예: $0 specific tests/test_integration.py::test_full_pipeline"
        exit 1
    fi

    log_info "특정 테스트 실행: $test_path"

    pytest \
        -v \
        -s \
        "$test_path" \
        --tb=short

    log_success "테스트 완료"
}

# 실패한 테스트만 재실행
run_failed() {
    log_info "실패한 테스트만 재실행 중..."

    pytest \
        -v \
        --lf \
        --tb=short

    log_success "재실행 완료"
}

# 메인 함수
main() {
    print_banner

    # 인자가 없으면 도움말 출력
    if [ $# -eq 0 ]; then
        print_help
        exit 0
    fi

    check_env
    setup_logs

    case "$1" in
        all)
            run_all_tests
            ;;
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        fast)
            run_fast_tests
            ;;
        slow)
            run_all_with_slow
            ;;
        benchmark)
            run_benchmark
            ;;
        coverage)
            run_coverage
            ;;
        watch)
            run_watch
            ;;
        ci)
            run_ci
            ;;
        specific)
            run_specific_test "$2"
            ;;
        failed)
            run_failed
            ;;
        help)
            print_help
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            print_help
            exit 1
            ;;
    esac

    echo ""
    log_success "✨ 테스트 실행 완료!"
    echo ""
}

# 스크립트 실행
main "$@"
