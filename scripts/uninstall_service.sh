#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="case.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

if systemctl is-active --quiet "$SERVICE_NAME"; then
    systemctl stop "$SERVICE_NAME"
fi

if systemctl is-enabled --quiet "$SERVICE_NAME"; then
    systemctl disable "$SERVICE_NAME"
fi

if [[ -f "$SERVICE_PATH" ]]; then
    rm -f "$SERVICE_PATH"
fi

systemctl daemon-reload
systemctl reset-failed "$SERVICE_NAME" >/dev/null 2>&1 || true

echo "Removed ${SERVICE_NAME}"
