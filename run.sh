#!/usr/bin/env bash
# 경사하강법 시뮬레이션 — 의존성 자동 설치(OS별) 후 실행
#
# 사용 예:
#   ./run.sh                                  # GUI (DISPLAY 있을 때) 또는 자동 헤드리스
#   ./run.sh --headless                       # 강제 헤드리스 (gif 저장)
#   ./run.sh --data data_convex.csv           # CSV 입력
#   ./run.sh --headless --output run.gif      # 헤드리스 + 출력 경로 지정
#
# 처음 실행 시 python3 / venv / pip / (필요하면) tkinter 를 OS에 맞춰 설치합니다.
# Linux: apt / dnf / pacman / apk  중 사용 가능한 것 자동 사용 (sudo 필요)
# macOS: brew 사용 (Homebrew 미설치면 안내)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# ── 권한 헬퍼 ────────────────────────────────────────────────────────────────
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
fi

run_priv() {
    if [ -n "$SUDO" ]; then
        echo "  $ $SUDO $*"
        $SUDO "$@"
    else
        echo "  $ $*"
        "$@"
    fi
}

# ── OS 감지 ──────────────────────────────────────────────────────────────────
detect_os() {
    case "$(uname -s)" in
        Darwin) echo "macos"; return ;;
        Linux)  : ;;
        *)      echo "unsupported"; return ;;
    esac
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case "${ID:-}${ID_LIKE:-}" in
            *debian*|*ubuntu*) echo "debian" ;;
            *fedora*|*rhel*|*centos*) echo "rhel" ;;
            *arch*)            echo "arch" ;;
            *alpine*)          echo "alpine" ;;
            *)                 echo "linux-unknown" ;;
        esac
    else
        echo "linux-unknown"
    fi
}

OS="$(detect_os)"

need_sudo_or_die() {
    if [ -z "$SUDO" ] && [ "$(id -u)" -ne 0 ]; then
        echo "오류: 시스템 패키지 설치에 sudo 또는 루트 권한이 필요합니다." >&2
        exit 1
    fi
}

# ── 시스템 패키지 설치 ────────────────────────────────────────────────────────
install_python() {
    echo "[*] python3 / venv / pip 설치 중... ($OS)"
    case "$OS" in
        macos)
            if ! command -v brew >/dev/null 2>&1; then
                echo "오류: Homebrew가 필요합니다. https://brew.sh 에서 설치 후 다시 실행하세요." >&2
                exit 1
            fi
            brew install python3
            ;;
        debian)
            need_sudo_or_die
            run_priv apt-get update -qq
            run_priv apt-get install -y python3 python3-venv python3-pip
            ;;
        rhel)
            need_sudo_or_die
            if command -v dnf >/dev/null 2>&1; then
                run_priv dnf install -y python3 python3-pip
            else
                run_priv yum install -y python3 python3-pip
            fi
            ;;
        arch)
            need_sudo_or_die
            run_priv pacman -Sy --noconfirm python python-pip
            ;;
        alpine)
            need_sudo_or_die
            run_priv apk add --no-cache python3 py3-pip
            ;;
        *)
            echo "오류: 자동 설치 미지원 OS ($OS). python3 / pip 를 직접 설치해 주세요." >&2
            exit 1
            ;;
    esac
}

install_tk() {
    echo "[*] tkinter 설치 중... ($OS)"
    case "$OS" in
        macos)
            # python.org 빌드와 brew python3 는 tkinter 포함
            ;;
        debian)
            need_sudo_or_die
            run_priv apt-get install -y python3-tk
            ;;
        rhel)
            need_sudo_or_die
            if command -v dnf >/dev/null 2>&1; then
                run_priv dnf install -y python3-tkinter
            else
                run_priv yum install -y python3-tkinter
            fi
            ;;
        arch)
            need_sudo_or_die
            run_priv pacman -Sy --noconfirm tk
            ;;
        alpine)
            need_sudo_or_die
            run_priv apk add --no-cache python3-tkinter
            ;;
        *)
            echo "경고: tkinter 자동 설치 불가 — GUI 모드 실패 시 직접 설치하세요." >&2
            ;;
    esac
}

# ── GUI 모드 여부 (tkinter 필요한지 판단) ────────────────────────────────────
needs_gui() {
    for arg in "$@"; do
        [ "$arg" = "--headless" ] && return 1
    done
    if [ "$(uname -s)" = "Linux" ]; then
        # DISPLAY/WAYLAND_DISPLAY 둘 다 없으면 헤드리스 자동 진입 → tkinter 불필요
        if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
            return 1
        fi
    fi
    return 0
}

# ── 1) python3 확인 / 설치 ───────────────────────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
    install_python
fi

# ── 2) venv 모듈 확인 / 설치 (Debian 계열은 별도 패키지) ─────────────────────
if ! python3 -c "import venv" >/dev/null 2>&1; then
    install_python
fi

# ── 3) (필요 시) tkinter 확인 / 설치 ─────────────────────────────────────────
if needs_gui "$@"; then
    if ! python3 -c "import tkinter" >/dev/null 2>&1; then
        install_tk
    fi
fi

# ── 4) venv 생성 ─────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] 가상환경 생성 중... ($VENV_DIR)"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 5) pip 의존성 확인 / 설치 ────────────────────────────────────────────────
if ! python -c "import numpy, matplotlib, PIL" 2>/dev/null; then
    echo "[*] pip 의존성 설치 중..."
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements.txt
fi

# ── 6) 실행 ──────────────────────────────────────────────────────────────────
exec python simulation.py "$@"
