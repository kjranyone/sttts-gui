#!/usr/bin/env bash
# Development launcher for sttts-gui: the same commands as dev.ps1, for macOS / Linux.
#
#   ./dev.sh                    # choose what to do (macOS / Linux: builds sttts-say)
#   ./dev.sh say                # build target/release/sttts-say
#   ./dev.sh gui -Mode real     # build and launch the GUI (Windows only, e.g. from Git Bash)
#   ./dev.sh say -DebugBuild    # debug profile (target/debug)
#
# Options (as in dev.ps1; --mode / --debug-build also work):
#   -Mode mock|real   GUI only. mock: no model download / real: Irodori + ASR (models download on first run)
#   -DebugBuild       use the debug profile instead of release
#   anything else     passed through to the GUI executable
#
# cargo build runs every time (only changed crates are recompiled). The GUI is Windows-only;
# on macOS / Linux only sttts-say is built (experimental).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

fail() {
    printf '\033[31m[dev.sh] ERROR: %s\033[0m\n' "$1" >&2
    exit 1
}
step() {
    printf '\033[36m[dev.sh] %s\033[0m\n' "$1"
}
usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) WINDOWS=1; EXE_SUFFIX=.exe ;;
    *) WINDOWS=0; EXE_SUFFIX= ;;
esac

COMMAND=
MODE=
DEBUG_BUILD=0
APP_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h | --help | -Help) usage; exit 0 ;;
        -Mode | -mode | --mode)
            [ $# -ge 2 ] || fail "$1 needs mock or real"
            MODE="$2"
            shift 2
            ;;
        -DebugBuild | -debugbuild | --debug-build) DEBUG_BUILD=1; shift ;;
        gui | say)
            if [ -z "$COMMAND" ] && [ ${#APP_ARGS[@]} -eq 0 ]; then COMMAND="$1"; else APP_ARGS+=("$1"); fi
            shift
            ;;
        *)
            [ -n "$COMMAND" ] || fail "unknown command: $1 (gui | say)"
            APP_ARGS+=("$1")
            shift
            ;;
    esac
done
case "$MODE" in
    '' | mock | real) ;;
    *) fail "-Mode must be mock or real: $MODE" ;;
esac

# --- 0) Prerequisites
command -v cargo >/dev/null 2>&1 || fail 'cargo not found. Please install Rust (rustup).'

# --- 0.5) What to do (ask if not given)
select_task() {
    # Ask what to do (empty = GUI with the real engine)
    printf '\n\033[36mChoose what to do:\033[0m\n' >&2
    echo "  [1] real : launch the GUI with the real engine (Irodori + ASR; models download on first run; default)" >&2
    echo "  [2] mock : launch the GUI without models (for checking the UI and wiring)" >&2
    echo "  [3] say  : build the CLI (sttts-say)" >&2
    while true; do
        printf 'Choice [1/2/3] (empty = 1): ' >&2
        read -r choice || fail 'no choice given'
        case "$(printf '%s' "$choice" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')" in
            '' | 1 | real) echo real; return ;;
            2 | mock) echo mock; return ;;
            3 | say) echo say; return ;;
            *) printf '\033[33m  Enter 1, 2 or 3 (real / mock / say)\033[0m\n' >&2 ;;
        esac
    done
}
if [ -z "$COMMAND" ]; then
    if [ -n "$MODE" ]; then
        COMMAND=gui
    elif [ "$WINDOWS" = 1 ]; then
        TASK="$(select_task)"
        if [ "$TASK" = say ]; then COMMAND=say; else COMMAND=gui; MODE="$TASK"; fi
    else
        # The GUI is Windows-only, so there is nothing to choose
        COMMAND=say
    fi
fi
if [ "$COMMAND" = say ] && { [ -n "$MODE" ] || [ ${#APP_ARGS[@]} -gt 0 ]; }; then
    fail '-Mode and extra arguments are for the GUI. "say" only builds sttts-say.'
fi
if [ "$COMMAND" = gui ] && [ "$WINDOWS" = 0 ]; then
    fail 'The GUI runs on Windows only. On macOS / Linux, build the CLI with ./dev.sh say'
fi
if [ "$DEBUG_BUILD" = 1 ]; then CONFIG=debug; else CONFIG=release; fi

build() {
    # One package per cargo call: building the GUI and sttts-say together would unify their features
    # and put the microphone / ASR code into sttts-say
    step "Running cargo build -p $1 ($CONFIG) (a few minutes the first time, incremental afterwards)"
    if [ "$DEBUG_BUILD" = 1 ]; then
        (cd "$ROOT" && cargo build -p "$1") || fail 'cargo build failed'
    else
        (cd "$ROOT" && cargo build --release -p "$1") || fail 'cargo build failed'
    fi
}

# --- 1) CLI: build only
if [ "$COMMAND" = say ]; then
    build sttts-say
    # A background sttts-say process of the previous build is replaced by itself on the next call
    step "Built $ROOT/target/$CONFIG/sttts-say$EXE_SUFFIX"
    exit 0
fi

# --- 2) GUI: build every time (cargo's incremental build finishes in seconds when nothing changed), then launch
MODE="${MODE:-real}"
build sttts-gui
step "Launching sttts-gui (mode=$MODE, build=$CONFIG)"
exec "$ROOT/target/$CONFIG/sttts-gui$EXE_SUFFIX" "--$MODE" ${APP_ARGS[@]+"${APP_ARGS[@]}"}
