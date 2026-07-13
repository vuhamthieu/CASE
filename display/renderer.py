"""Rich-based terminal renderer for the CASE display."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from time import monotonic, sleep

from rich.align import Align
from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from .conversation_model import ConversationMessage, ConversationModel, ConversationSnapshot, SystemMetrics
from .theme import Theme, theme as default_theme


@dataclass(frozen=True, slots=True)
class RendererConfig:
    fps: int = 30
    title: str = "CASE"
    cursor_blink_seconds: float = 0.5
    history_turns: int = 4
    min_terminal_width: int = 80


class DisplayRenderer:
    """Render-only thread that reads snapshots from the conversation model."""

    def __init__(self, model: ConversationModel, theme: Theme = default_theme, config: RendererConfig | None = None) -> None:
        self._model = model
        self._theme = theme
        self._config = config or RendererConfig()
        self._stop_event = Event()
        self._ready_event = Event()
        self._thread: Thread | None = None
        self._last_signature: tuple[object, ...] | None = None
        self._console = Console(force_terminal=True, color_system=None, highlight=False)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = Thread(target=self._run, name="case-display-renderer", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=5.0)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        self._ready_event.set()
        last_redraw = 0.0
        redraw_interval = 1.0 / max(1, self._config.fps)
        while not self._stop_event.is_set():
            snapshot = self._model.snapshot()
            cursor_visible = self._stream_cursor_visible(snapshot)
            signature = self._build_signature(snapshot, cursor_visible)
            now = monotonic()
            if signature != self._last_signature or (now - last_redraw) >= redraw_interval:
                self._render_snapshot(snapshot, cursor_visible)
                self._last_signature = signature
                last_redraw = now
            sleep(0.02)

    def _stream_cursor_visible(self, snapshot: ConversationSnapshot) -> bool:
        if not snapshot.current_stream_text:
            return False
        elapsed = monotonic() % (self._config.cursor_blink_seconds * 2.0)
        return elapsed < self._config.cursor_blink_seconds

    def _build_signature(self, snapshot: ConversationSnapshot, cursor_visible: bool) -> tuple[object, ...]:
        return (
            snapshot.status,
            snapshot.messages,
            snapshot.current_stream_text,
            snapshot.metrics,
            cursor_visible,
        )

    def _render_snapshot(self, snapshot: ConversationSnapshot, cursor_visible: bool) -> None:
        self._console.clear(home=True)
        self._console.print(self._render_header(snapshot.status))
        self._console.print(Rule(style=self._style_for_line(snapshot.status)))
        self._console.print(self._render_messages(snapshot.messages, snapshot.current_stream_text, cursor_visible))
        self._console.print(Rule(style=self._theme.panel_line))
        self._console.print(self._render_footer(snapshot.metrics))

    def _render_header(self, status: str):
        header = Text(f"CASE {status.upper()}", style=self._style_for_line(status), justify="left")
        return Align.left(header)

    def _render_messages(self, messages: tuple[ConversationMessage, ...], current_stream_text: str, cursor_visible: bool):
        rendered = Text(justify="left")
        for speaker, text, is_streaming in self._select_turns(messages, current_stream_text):
            rendered.append(f"[{speaker}]\n", style=self._theme.muted)
            rendered.append(self._wrap_text(text + ("▋" if is_streaming and cursor_visible else ""), width=self._body_width()))
            rendered.append("\n")
        return Align.left(rendered)

    def _render_footer(self, metrics: SystemMetrics):
        footer = Text(justify="left")
        footer.append(
            f"CPU {metrics.cpu_percent:.0f}%   RAM {metrics.ram_percent:.0f}%   TEMP {metrics.temperature:.0f}C\n",
            style=self._theme.foreground,
        )
        footer.append(
            f"MIC {metrics.mic_state}   NET {metrics.network_status}   SPEAKER {metrics.speaker_state}   VOICE READY",
            style=self._theme.muted,
        )
        return Align.left(footer)

    def _select_turns(self, messages: tuple[ConversationMessage, ...], current_stream_text: str) -> list[tuple[str, str, bool]]:
        selected = list(messages[-self._config.history_turns :])
        turns: list[tuple[str, str, bool]] = [(message.speaker, message.text, False) for message in selected]
        if current_stream_text:
            turns.append(("CASE", current_stream_text, True))
        return turns

    def _wrap_text(self, text: str, width: int) -> Text:
        width = max(10, width)
        wrapped = Text()
        words = text.split()
        if not words:
            wrapped.append("\n")
            return wrapped
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if len(candidate) <= width:
                line = candidate
            else:
                wrapped.append(f"{line}\n", style=self._theme.foreground)
                line = word
        wrapped.append(line, style=self._theme.foreground)
        return wrapped

    def _body_width(self) -> int:
        return max(self._config.min_terminal_width, self._console.width or self._config.min_terminal_width) - 4

    def _style_for_line(self, status: str) -> str:
        status_name = status.upper()
        if status_name in {"SPEAKING", "ACTIVE"}:
            return "bold white"
        if status_name in {"THINKING"}:
            return "bold bright_white"
        if status_name in {"LISTENING"}:
            return "bold white"
        return "white"
