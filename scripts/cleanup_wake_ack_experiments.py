#!/usr/bin/env python3
"""Archive recorded wake-ack experiment assets without deleting user files."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAKE_ACK_DIR = ROOT / "assets" / "audio" / "wake_ack"
ARCHIVE_DIR = WAKE_ACK_DIR / "_archive_recorded_experiment"
EXPERIMENT_ITEMS = (
    "recorded_raw",
    "recorded_processed",
    "recorded",
    "recorded_selection.json",
)


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.name}_{stamp}")


def archive_recorded_experiment() -> list[tuple[Path, Path]]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    for item in EXPERIMENT_ITEMS:
        source = WAKE_ACK_DIR / item
        if not source.exists():
            continue
        destination = _unique_destination(ARCHIVE_DIR / item)
        shutil.move(str(source), str(destination))
        moved.append((source, destination))
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive recorded wake-ack experiment folders/files."
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Move recorded wake-ack experiment assets into the archive folder.",
    )
    args = parser.parse_args()

    if not args.archive:
        parser.print_help()
        return 0

    moved = archive_recorded_experiment()
    if not moved:
        print("WAKE_ACK_CLEANUP: no recorded experiment assets found")
        return 0
    for source, destination in moved:
        print(
            "WAKE_ACK_CLEANUP: moved "
            f"{source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}"
        )
    print(f"WAKE_ACK_CLEANUP: archive={ARCHIVE_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
