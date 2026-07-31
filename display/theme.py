"""Centralized theme definitions for the CASE display."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Theme:
    background: str = "black"
    foreground: str = "white"
    muted: str = "bright_black"
    accent: str = "white"
    warning: str = "yellow"
    success: str = "green"
    error: str = "red"
    panel_line: str = "bright_black"
    separator: str = "bright_black"
    cursor: str = "white"


theme = Theme()
