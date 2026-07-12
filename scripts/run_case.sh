#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${CASE_VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

cd "$PROJECT_ROOT"

if [[ -x "$PYTHON_BIN" ]]; then
    exec "$PYTHON_BIN" main.py "$@"
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    exec python main.py "$@"
fi

echo "CASE virtual environment not found at: $VENV_DIR" >&2
exit 1
