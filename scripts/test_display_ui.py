#!/usr/bin/env python3
"""Standalone CASE display UI demo."""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from display.display_manager import DisplayManager


class DemoScenario:
    def __init__(self, manager: DisplayManager) -> None:
        self._manager = manager
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="case-display-demo", daemon=True)
        self._response = (
            "I am online. I can track conversation state, system metrics, and live typing without any robot back end attached."
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        phrases = [
            ("YOU", "CASE, status check."),
            ("YOU", "Report your systems."),
            ("YOU", "Keep the console alive."),
        ]
        cycle = 0
        while not self._stop_event.is_set():
            self._manager.set_status("LISTENING")
            self._manager.clear_stream()
            self._manager.update_system_metrics(network_status="OK", mic_state="LIVE", speaker_state="READY")
            self._manager.append_message(*phrases[cycle % len(phrases)])
            self._tick_metrics(cycle, listening=True)
            self._sleep(2.0)
            if self._stop_event.is_set():
                break

            self._manager.set_status("THINKING")
            self._tick_metrics(cycle, thinking=True)
            self._sleep(1.6)
            if self._stop_event.is_set():
                break

            self._manager.set_status("SPEAKING")
            self._stream_response()
            self._manager.finish_stream()
            self._tick_metrics(cycle, speaking=True)
            self._sleep(1.5)
            cycle += 1

    def _stream_response(self) -> None:
        streamed = ""
        for character in self._response:
            if self._stop_event.is_set():
                return
            streamed += character
            self._manager.update_stream(streamed)
            self._tick_metrics(len(streamed), streaming=True)
            self._sleep(0.04)

    def _tick_metrics(self, tick: int, *, listening: bool = False, thinking: bool = False, speaking: bool = False, streaming: bool = False) -> None:
        phase = tick / 8.0
        cpu = 16.0 + 12.0 * (0.5 + 0.5 * math.sin(phase))
        ram = 28.0 + 4.0 * (0.5 + 0.5 * math.cos(phase / 2.0))
        temperature = 49.0 + 7.0 * (0.5 + 0.5 * math.sin(phase / 1.4))
        network = "OK" if tick % 7 else "CHECK"
        mic = "LIVE" if listening or thinking else "QUIET"
        speaker = "READY" if not speaking else "ACTIVE"
        if streaming:
            speaker = "ACTIVE"
        self._manager.update_system_metrics(
            cpu_percent=cpu,
            ram_percent=ram,
            temperature=temperature,
            network_status=network,
            mic_state=mic,
            speaker_state=speaker,
        )

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)


def main() -> int:
    manager = DisplayManager()
    try:
        manager.start()
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    scenario = DemoScenario(manager)
    scenario.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        scenario.stop()
        manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
