#!/usr/bin/env python3
"""Check CASE runtime model/audio artifacts for the selected STT profile."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import defaults  # noqa: E402
from src.stt_backends.stt_profile import resolve_stt_profile  # noqa: E402


SENSEVOICE_MODEL = (
    Path(defaults.SHERPA_SENSEVOICE_MODEL_DIR) / "model.int8.onnx"
)
SENSEVOICE_TOKENS = Path(defaults.SHERPA_SENSEVOICE_MODEL_DIR) / "tokens.txt"

ALWAYS_REQUIRED = (
    "models/wakewords/hey_case_v2.onnx",
    "models/wakewords/hey_case_v2.onnx.data",
    defaults.VOSK_SMALL_MODEL_PATH,
    defaults.PIPER_MODEL_PATH,
    defaults.PIPER_CONFIG_PATH,
    "assets/audio/wake_ack/generated/yes.wav",
    "assets/audio/wake_ack/generated/im_listening.wav",
)

SENSEVOICE_REQUIRED = (
    str(SENSEVOICE_MODEL),
    str(SENSEVOICE_TOKENS),
)

OPTIONAL_ADVANCED = {
    "smart_turn": defaults.SMART_TURN_MODEL_PATH,
}


@dataclass
class CheckLine:
    status: str
    classification: str
    path: str


def resolve_artifact(path: str) -> Path:
    resolved = ROOT / path
    if path == defaults.VOSK_LGRAPH_MODEL_PATH and not resolved.exists():
        legacy = ROOT / Path(path).name
        if legacy.exists():
            return legacy
    return resolved


def exists(path: str) -> bool:
    return resolve_artifact(path).exists()


def line(status: str, classification: str, path: str) -> CheckLine:
    return CheckLine(status, classification, path)


def check(profile: str, final_backend: str) -> tuple[int, dict[str, object], list[CheckLine]]:
    plan = resolve_stt_profile(profile, final_backend)
    lines: list[CheckLine] = []
    missing_always = [path for path in ALWAYS_REQUIRED if not exists(path)]
    for path in missing_always:
        lines.append(line("MISSING", "REQUIRED", path))

    degraded_reasons: list[str] = []
    fallback = ""
    accuracy = ""

    if plan.profile in {"balanced", "accuracy"}:
        if not exists(defaults.SILERO_VAD_MODEL_PATH):
            lines.append(line("MISSING", "PROFILE_REQUIRED", defaults.SILERO_VAD_MODEL_PATH))
            degraded_reasons.append("silero_vad")

    wants_lgraph = "vosk_lgraph" in plan.final_chain or plan.profile in {"balanced", "accuracy"}
    if wants_lgraph:
        if exists(defaults.VOSK_LGRAPH_MODEL_PATH):
            if "sensevoice" in plan.final_chain:
                lines.append(
                    line("OK", "FALLBACK_AVAILABLE", defaults.VOSK_LGRAPH_MODEL_PATH)
                )
        else:
            lines.append(line("MISSING", "PROFILE_REQUIRED", defaults.VOSK_LGRAPH_MODEL_PATH))
            degraded_reasons.append("vosk_lgraph")
            fallback = "vosk_small"
            if plan.profile == "balanced":
                accuracy = "lower"

    wants_sensevoice = "sensevoice" in plan.final_chain
    if wants_sensevoice:
        missing_sensevoice = [path for path in SENSEVOICE_REQUIRED if not exists(path)]
        for path in missing_sensevoice:
            lines.append(line("MISSING", "PROFILE_REQUIRED", path))
        if missing_sensevoice:
            degraded_reasons.append("sensevoice")
            fallback = "vosk_lgraph" if exists(defaults.VOSK_LGRAPH_MODEL_PATH) else "vosk_small"

    for name, path in OPTIONAL_ADVANCED.items():
        if not exists(path):
            lines.append(line("MISSING", f"OPTIONAL_{name}", path))

    if missing_always:
        result = "cannot_run"
        exit_code = 2
    elif degraded_reasons:
        result = "degraded"
        exit_code = 1
    else:
        result = "ok"
        exit_code = 0

    payload = {
        "profile": plan.profile,
        "final_backend": plan.final_backend,
        "final_chain": list(plan.final_chain),
        "result": result,
        "fallback": fallback,
        "accuracy": accuracy,
        "missing_required": missing_always,
        "degraded_reasons": degraded_reasons,
        "lines": [entry.__dict__ for entry in lines],
    }
    return exit_code, payload, lines


def print_text(payload: dict[str, object], lines: list[CheckLine]) -> None:
    for entry in lines:
        if entry.classification == "OPTIONAL_smart_turn":
            print(
                "OPTIONAL_smart_turn missing; CASE will use VAD/timing "
                "turn-ending fallback."
            )
            continue
        print(f"{entry.status:<7} {entry.classification} {entry.path}")

    result = str(payload["result"])
    if result == "ok":
        print("RESULT: ok")
        return
    if result == "cannot_run":
        print("RESULT: cannot_run")
        return

    parts = ["RESULT: degraded"]
    if payload.get("fallback"):
        parts.append(f"fallback={payload['fallback']}")
    if payload.get("accuracy"):
        parts.append(f"accuracy={payload['accuracy']}")
    print(parts[0] + (", " + ", ".join(parts[1:]) if len(parts) > 1 else ""))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check CASE runtime artifacts for the selected STT profile."
    )
    parser.add_argument(
        "--profile",
        choices=("fast", "balanced", "accuracy"),
        default=os.getenv("CASE_STT_PROFILE", defaults.CASE_STT_PROFILE),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    exit_code, payload, lines = check(
        args.profile,
        os.getenv("CASE_STT_FINAL_BACKEND", defaults.CASE_STT_FINAL_BACKEND),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text(payload, lines)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
