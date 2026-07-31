"""High-level state manager for the CASE display subsystem."""

from __future__ import annotations

from .conversation_model import ConversationModel, Speaker
from .renderer import DisplayRenderer, RendererConfig
from .theme import Theme, theme as default_theme


class DisplayManager:
    """Owns the model and rendering thread, but no render logic."""

    def __init__(self, *, theme: Theme = default_theme, max_messages: int = 8, config: RendererConfig | None = None) -> None:
        self.model = ConversationModel(max_messages=max_messages)
        self._renderer = DisplayRenderer(self.model, theme=theme, config=config)

    def start(self) -> None:
        self.model.start_telemetry()
        self._renderer.start()

    def stop(self) -> None:
        self.model.stop_telemetry()
        self._renderer.stop()

    def is_running(self) -> bool:
        return self._renderer.is_running()

    def set_status(self, status: str) -> None:
        self.model.set_status(status)

    def append_message(self, speaker: Speaker, text: str) -> None:
        self.model.append_message(speaker, text)

    def update_stream(self, text: str) -> None:
        self.model.update_stream(text)

    def finish_stream(self, text: str | None = None) -> None:
        self.model.finish_stream(text)

    def clear_stream(self) -> None:
        self.model.clear_stream()

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
        self.model.update_system_metrics(
            cpu_percent=cpu_percent,
            ram_percent=ram_percent,
            temperature=temperature,
            network_status=network_status,
            mic_state=mic_state,
            speaker_state=speaker_state,
        )
