"""Message history rendering for the CASE display."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    import pygame
except ModuleNotFoundError:  # pragma: no cover - runtime dependency guard
    pygame = None  # type: ignore[assignment]

from ..conversation_model import ConversationMessage
from ..theme import Theme


@dataclass(frozen=True, slots=True)
class MessageViewLayout:
    x: int = 20
    y: int = 72
    width: int = 760
    height: int = 300
    line_spacing: int = 6
    speaker_gap: int = 10


def _wrap_text(text: str, font: pygame.font.Font, width: int) -> list[str]:
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [text]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


class MessageView:
    def __init__(self, font: pygame.font.Font, small_font: pygame.font.Font, theme: Theme, layout: MessageViewLayout | None = None) -> None:
        self._font = font
        self._small_font = small_font
        self._theme = theme
        self._layout = layout or MessageViewLayout()

    def render(
        self,
        surface: pygame.Surface,
        messages: Iterable[ConversationMessage],
        current_stream_text: str,
        streaming: bool,
        cursor_visible: bool,
    ) -> None:
        layout = self._layout
        max_width = layout.width
        max_height = layout.height
        line_height = self._font.get_linesize() + layout.line_spacing
        speaker_height = self._small_font.get_linesize() + 2

        prepared: list[tuple[str, str, bool]] = []
        for message in messages:
            prepared.append((message.speaker, message.text, False))
        if current_stream_text:
            prepared.append(("CASE", current_stream_text, streaming))

        line_groups: list[tuple[str, list[str], bool]] = []
        for speaker, text, is_streaming in prepared:
            prefix = f"[{speaker}]"
            available_width = max_width - self._font.size(prefix + " ")[0]
            wrapped = _wrap_text(text, self._font, max(1, available_width))
            line_groups.append((prefix, wrapped, is_streaming))

        rendered_groups: list[tuple[str, list[str], bool]] = []
        total_height = 0
        for group in reversed(line_groups):
            group_height = speaker_height + len(group[1]) * line_height + layout.speaker_gap
            if total_height + group_height > max_height and rendered_groups:
                break
            rendered_groups.append(group)
            total_height += group_height
            if total_height > max_height:
                break

        rendered_groups.reverse()

        y = layout.y
        for prefix, wrapped_lines, is_streaming in rendered_groups:
            speaker_surface = self._small_font.render(prefix, True, self._theme.muted)
            surface.blit(speaker_surface, (layout.x, y))
            y += speaker_height

            for index, line in enumerate(wrapped_lines):
                display_line = line
                if is_streaming and cursor_visible and index == len(wrapped_lines) - 1:
                    display_line = f"{line}▋"
                text_surface = self._font.render(display_line, True, self._theme.foreground)
                surface.blit(text_surface, (layout.x, y))
                y += line_height

            y += layout.speaker_gap
