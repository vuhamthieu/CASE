"""Centralized theme definitions for the CASE display."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Theme:
    background: tuple[int, int, int] = (0, 0, 0)
    foreground: tuple[int, int, int] = (245, 245, 245)
    muted: tuple[int, int, int] = (170, 170, 170)
    accent: tuple[int, int, int] = (255, 255, 255)
    warning: tuple[int, int, int] = (255, 196, 0)
    success: tuple[int, int, int] = (140, 255, 140)
    error: tuple[int, int, int] = (255, 110, 110)
    panel_line: tuple[int, int, int] = (55, 55, 55)
    separator: tuple[int, int, int] = (80, 80, 80)
    cursor: tuple[int, int, int] = (245, 245, 245)


theme = Theme()
