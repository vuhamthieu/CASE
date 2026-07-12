"""Footer metrics rendering for the CASE display."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import pygame
except ModuleNotFoundError:  # pragma: no cover - runtime dependency guard
    pygame = None  # type: ignore[assignment]

from ..conversation_model import SystemMetrics
from ..theme import Theme


@dataclass(frozen=True, slots=True)
class FooterLayout:
    x: int = 20
    y: int = 406
    width: int = 760
    row_spacing: int = 6


class Footer:
    def __init__(self, font: pygame.font.Font, small_font: pygame.font.Font, theme: Theme, layout: FooterLayout | None = None) -> None:
        self._font = font
        self._small_font = small_font
        self._theme = theme
        self._layout = layout or FooterLayout()

    def render(self, surface: pygame.Surface, metrics: SystemMetrics) -> None:
        layout = self._layout
        first_row = f"CPU {metrics.cpu_percent:.0f}%   RAM {metrics.ram_percent:.0f}%   TEMP {metrics.temperature:.0f}C"
        second_row = f"MIC {metrics.mic_state}   NET {metrics.network_status}   SPEAKER {metrics.speaker_state}   VOICE READY"

        first_surface = self._font.render(first_row, True, self._theme.foreground)
        second_surface = self._small_font.render(second_row, True, self._theme.muted)
        surface.blit(first_surface, (layout.x, layout.y))
        surface.blit(second_surface, (layout.x, layout.y + self._font.get_linesize() + layout.row_spacing))
        surface.fill(self._theme.separator, (layout.x, layout.y - 12, layout.width, 1))
