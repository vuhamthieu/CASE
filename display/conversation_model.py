"""Thread-safe conversation state for the CASE display."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, RLock, Thread
from time import sleep, time
from typing import Literal

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    psutil = None  # type: ignore[assignment]

Speaker = Literal["YOU", "CASE", "SYSTEM"]


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    speaker: Speaker
    text: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class SystemMetrics:
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    temperature: float = 0.0
    network_status: str = "UNKNOWN"
    mic_state: str = "UNKNOWN"
    speaker_state: str = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    status: str
    messages: tuple[ConversationMessage, ...]
    current_stream_text: str
    metrics: SystemMetrics
    last_update: float


class ConversationModel:
    """Single source of truth for the display state."""

    def __init__(self, max_messages: int = 8) -> None:
        self._lock = RLock()
        self._max_messages = max_messages
        self._status = "IDLE"
        self._messages: list[ConversationMessage] = []
        self._current_stream_text = ""
        self._metrics = SystemMetrics()
        self._last_update = time()
        self._telemetry_stop = Event()
        self._telemetry_thread: Thread | None = None

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._last_update = time()

    def append_message(self, speaker: Speaker, text: str) -> None:
        with self._lock:
            self._messages.append(ConversationMessage(speaker=speaker, text=text, timestamp=time()))
            if len(self._messages) > self._max_messages:
                self._messages = self._messages[-self._max_messages :]
            self._last_update = time()

    def update_stream(self, text: str) -> None:
        with self._lock:
            self._current_stream_text = text
            self._last_update = time()

    def finish_stream(self, text: str | None = None) -> None:
        with self._lock:
            final_text = self._current_stream_text if text is None else text
            if final_text:
                self._messages.append(ConversationMessage(speaker="CASE", text=final_text, timestamp=time()))
                if len(self._messages) > self._max_messages:
                    self._messages = self._messages[-self._max_messages :]
            self._current_stream_text = ""
            self._last_update = time()

    def clear_stream(self) -> None:
        with self._lock:
            self._current_stream_text = ""
            self._last_update = time()

    def update_system_metrics(
        self,
        *,
        cpu_percent: float | None = None,
        ram_percent: float | None = None,
        temperature: float | None = None,
        network_status: str | None = None,
        mic_state: str | None = None,
        speaker_state: str | None = None,
    ) -> None:
        with self._lock:
            metrics = self._metrics
            self._metrics = replace(
                metrics,
                cpu_percent=metrics.cpu_percent if cpu_percent is None else cpu_percent,
                ram_percent=metrics.ram_percent if ram_percent is None else ram_percent,
                temperature=metrics.temperature if temperature is None else temperature,
                network_status=metrics.network_status if network_status is None else network_status,
                mic_state=metrics.mic_state if mic_state is None else mic_state,
                speaker_state=metrics.speaker_state if speaker_state is None else speaker_state,
            )
            self._last_update = time()

    def start_telemetry(self) -> None:
        if self._telemetry_thread is not None and self._telemetry_thread.is_alive():
            return
        self._telemetry_stop.clear()
        self._telemetry_thread = Thread(target=self._telemetry_loop, name="case-display-telemetry", daemon=True)
        self._telemetry_thread.start()

    def stop_telemetry(self) -> None:
        self._telemetry_stop.set()
        if self._telemetry_thread is not None and self._telemetry_thread.is_alive():
            self._telemetry_thread.join(timeout=2.0)

    def _telemetry_loop(self) -> None:
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        while not self._telemetry_stop.is_set():
            cpu_percent = self._read_cpu_percent()
            ram_percent = self._read_ram_percent()
            temperature = self._read_temperature_celsius()
            self.update_system_metrics(
                cpu_percent=cpu_percent,
                ram_percent=ram_percent,
                temperature=temperature,
            )
            if self._telemetry_stop.wait(1.0):
                break

    @staticmethod
    def _read_cpu_percent() -> float:
        if psutil is None:
            return 0.0
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return 0.0

    @staticmethod
    def _read_ram_percent() -> float:
        if psutil is None:
            return 0.0
        try:
            return float(psutil.virtual_memory().percent)
        except Exception:
            return 0.0

    @staticmethod
    def _read_temperature_celsius() -> float:
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        try:
            raw_value = temp_path.read_text(encoding="utf-8").strip()
            return float(raw_value) / 1000.0
        except Exception:
            return 0.0

    def snapshot(self) -> ConversationSnapshot:
        with self._lock:
            return ConversationSnapshot(
                status=self._status,
                messages=tuple(self._messages),
                current_stream_text=self._current_stream_text,
                metrics=self._metrics,
                last_update=self._last_update,
            )

    @property
    def max_messages(self) -> int:
        return self._max_messages
