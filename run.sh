#!/usr/bin/env bash
# Gradient Descent Simulation - auto-install dependencies (per OS) and run.
#
# Usage:
#   ./run.sh                                  # GUI (when DISPLAY exists) or auto-headless
#   ./run.sh --headless                       # Force headless (saves a gif)
#   ./run.sh --data data_convex.csv           # CSV input
#   ./run.sh --headless --output run.gif      # Headless + custom output path
#
# On first run, installs python3 / venv / pip / (if needed) tkinter for the host OS.
# Linux: uses whichever of apt / dnf / pacman / apk is available (sudo required).
# macOS: uses brew (will exit with a hint if Homebrew is missing).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# ── privilege helper ─────────────────────────────────────────────────────────
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

# ── OS detection ─────────────────────────────────────────────────────────────
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
        echo "Error: installing system packages requires sudo or root." >&2
        exit 1
    fi
}

# ── system package installs ──────────────────────────────────────────────────
install_python() {
    echo "[*] Installing python3 / venv / pip... ($OS)"
    case "$OS" in
        macos)
            if ! command -v brew >/dev/null 2>&1; then
                echo "Error: Homebrew is required. Install it from https://brew.sh and re-run." >&2
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
            echo "Error: auto-install not supported for OS ($OS). Please install python3 / pip manually." >&2
            exit 1
            ;;
    esac
}

has_korean_font() {
    if [ "$(uname -s)" = "Darwin" ]; then
        return 0   # macOS ships with AppleGothic
    fi
    command -v fc-list >/dev/null 2>&1 \
        && fc-list :lang=ko 2>/dev/null | grep -q .
}

install_korean_font() {
    if [ "$OS" = "macos" ]; then
        return
    fi
    if [ -z "$SUDO" ] && [ "$(id -u)" -ne 0 ]; then
        echo "  Warning: skipping Korean font install (no sudo). Korean characters may render as boxes." >&2
        return
    fi
    echo "[*] Installing Korean font... ($OS)"
    case "$OS" in
        debian)
            run_priv apt-get install -y fonts-nanum 2>/dev/null \
                || run_priv apt-get install -y fonts-noto-cjk 2>/dev/null \
                || echo "  Warning: Korean font package install failed." >&2
            ;;
        rhel)
            if command -v dnf >/dev/null 2>&1; then
                run_priv dnf install -y google-noto-sans-cjk-fonts 2>/dev/null \
                    || run_priv dnf install -y nhn-nanum-fonts-common 2>/dev/null \
                    || echo "  Warning: Korean font package install failed." >&2
            fi
            ;;
        arch)
            run_priv pacman -Sy --noconfirm noto-fonts-cjk 2>/dev/null \
                || echo "  Warning: Korean font package install failed." >&2
            ;;
        alpine)
            run_priv apk add --no-cache font-noto-cjk 2>/dev/null \
                || echo "  Warning: Korean font package install failed." >&2
            ;;
        *)
            echo "  Warning: Korean font auto-install not supported for OS ($OS)." >&2
            ;;
    esac
    # refresh font caches so newly installed fonts get picked up
    command -v fc-cache >/dev/null 2>&1 && fc-cache -f >/dev/null 2>&1 || true
    rm -rf "$HOME/.cache/matplotlib" 2>/dev/null || true
}

install_tk() {
    echo "[*] Installing tkinter... ($OS)"
    case "$OS" in
        macos)
            # python.org and brew python3 ship with tkinter
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
            echo "Warning: cannot auto-install tkinter for this OS - install it manually if GUI mode fails." >&2
            ;;
    esac
}

# ── decide whether GUI mode is needed (so we can install tkinter only if so) ─
needs_gui() {
    for arg in "$@"; do
        [ "$arg" = "--headless" ] && return 1
    done
    if [ "$(uname -s)" = "Linux" ]; then
        # No DISPLAY and no WAYLAND_DISPLAY -> headless auto-mode -> no tkinter
        if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
            return 1
        fi
    fi
    return 0
}

# ── 1) ensure python3 ────────────────────────────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
    install_python
fi

# ── 2) ensure venv module (Debian splits pip wheels into python3-venv) ──────
# `import venv` alone is not enough on Ubuntu/Debian -- the actual venv build
# step needs the wheels shipped by `python3-venv`, otherwise it leaves an
# incomplete .venv/ without bin/activate. Best test: try a real creation.
test_venv_creation() {
    local tmp ok=1
    tmp="$(mktemp -d)"
    if python3 -m venv "$tmp/probe" >/dev/null 2>&1 \
       && [ -f "$tmp/probe/bin/activate" ]; then
        ok=0
    fi
    rm -rf "$tmp"
    return $ok
}

if ! test_venv_creation; then
    install_python
fi

# ── 3) ensure tkinter when GUI mode is needed ────────────────────────────────
if needs_gui "$@"; then
    if ! python3 -c "import tkinter" >/dev/null 2>&1; then
        install_tk
    fi
fi

# ── 4) ensure a Korean font (best-effort) ────────────────────────────────────
if ! has_korean_font; then
    install_korean_font
fi

# ── 5) create venv (recreate if a previous run left it incomplete) ──────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    if [ -d "$VENV_DIR" ]; then
        echo "[*] Removing incomplete venv at $VENV_DIR"
        rm -rf "$VENV_DIR"
    fi
    echo "[*] Creating virtual environment... ($VENV_DIR)"
    if ! python3 -m venv "$VENV_DIR" || [ ! -f "$VENV_DIR/bin/activate" ]; then
        # Most common cause on Ubuntu/Debian: python3-venv not installed.
        # Try installing system packages and retry once.
        echo "[*] venv creation failed -- installing missing system packages and retrying"
        rm -rf "$VENV_DIR"
        install_python
        python3 -m venv "$VENV_DIR"
    fi
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "Error: cannot create a working venv at $VENV_DIR." >&2
        echo "  On Ubuntu/Debian, install python3-venv manually and re-run:" >&2
        echo "    sudo apt install -y python3-venv python3-pip" >&2
        exit 1
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 6) install pip dependencies ──────────────────────────────────────────────
if ! python -c "import numpy, matplotlib, PIL" 2>/dev/null; then
    echo "[*] Installing pip dependencies..."
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements.txt
fi

# ── 7) run ───────────────────────────────────────────────────────────────────
exec python simulation.py "$@"
