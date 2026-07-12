"""Top status bar rendering for the CASE display."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import pygame
except ModuleNotFoundError:  # pragma: no cover - runtime dependency guard
    pygame = None  # type: ignore[assignment]

from ..theme import Theme


@dataclass(frozen=True, slots=True)
class StatusBarLayout:
    x: int = 20
    y: int = 18
    width: int = 760


class StatusBar:
    def __init__(self, font: pygame.font.Font, theme: Theme, layout: StatusBarLayout | None = None) -> None:
        self._font = font
        self._theme = theme
        self._layout = layout or StatusBarLayout()

    def render(self, surface: pygame.Surface, status: str) -> None:
        layout = self._layout
        title = f"CASE {status.upper()}"
        title_surface = self._font.render(title, True, self._theme.foreground)
        surface.blit(title_surface, (layout.x, layout.y))

        y = layout.y + self._font.get_linesize() + 8
        surface.fill(self._theme.separator, (layout.x, y, layout.width, 1))
