#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_SRC="$PROJECT_ROOT/deployment/systemd/case.service"
SERVICE_DST="/etc/systemd/system/case.service"
ENABLE=false
START=false

usage() {
    cat <<'EOF'
Usage: install_service.sh [--enable] [--start]

Installs the CASE systemd unit to /etc/systemd/system/case.service.
Without flags, the script only installs and reloads systemd.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --enable)
            ENABLE=true
            ;;
        --start)
            START=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [[ ! -f "$SERVICE_SRC" ]]; then
    echo "Service file not found: $SERVICE_SRC" >&2
    exit 1
fi

install -Dm644 "$SERVICE_SRC" "$SERVICE_DST"
systemctl daemon-reload

if [[ "$ENABLE" == true ]]; then
    systemctl enable case.service
fi

if [[ "$START" == true ]]; then
    systemctl start case.service
fi

echo "Installed case.service"
