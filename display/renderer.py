"""Pygame renderer thread for the CASE display."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from time import monotonic

try:
    import pygame
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    pygame = None  # type: ignore[assignment]
    _PYGAME_IMPORT_ERROR = exc
else:
    _PYGAME_IMPORT_ERROR = None

from .conversation_model import ConversationModel, ConversationSnapshot
from .theme import Theme, theme as default_theme
from .widgets.footer import Footer
from .widgets.message_view import MessageView
from .widgets.status_bar import StatusBar


@dataclass(frozen=True, slots=True)
class RendererConfig:
    size: tuple[int, int] = (640, 480)
    fps: int = 30
    title: str = "CASE"
    cursor_blink_seconds: float = 0.5


class DisplayRenderer:
    """Render-only thread that reads snapshots from the conversation model."""

    def __init__(self, model: ConversationModel, theme: Theme = default_theme, config: RendererConfig | None = None) -> None:
        self._model = model
        self._theme = theme
        self._config = config or RendererConfig()
        self._stop_event = Event()
        self._ready_event = Event()
        self._thread: Thread | None = None
        self._screen: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._status_bar: StatusBar | None = None
        self._message_view: MessageView | None = None
        self._footer: Footer | None = None
        self._last_signature: tuple[object, ...] | None = None
        self._clock = pygame.time.Clock()
        self._font_cache: dict[tuple[str, int, bool], pygame.font.Font] = {}

    def start(self) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required to start the CASE display renderer") from _PYGAME_IMPORT_ERROR
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = Thread(target=self._run, name="case-display-renderer", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=5.0)

    def stop(self) -> None:
        if pygame is None:
            return
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        pygame.init()
        pygame.font.init()
        try:
            flags = pygame.FULLSCREEN | pygame.DOUBLEBUF
            self._screen = pygame.display.set_mode(self._config.size, flags, vsync=1)
            pygame.display.set_caption(self._config.title)
            self._font = self._get_font(24)
            self._small_font = self._get_font(18)
            screen_width, screen_height = self._config.size
            content_width = max(1, screen_width - 40)
            status_layout = self._build_status_layout(content_width)
            footer_layout = self._build_footer_layout(screen_width, screen_height, content_width)
            message_layout = self._build_message_layout(screen_width, footer_layout.y, content_width)
            self._status_bar = StatusBar(self._font, self._theme, status_layout)
            self._message_view = MessageView(self._font, self._small_font, self._theme, message_layout)
            self._footer = Footer(self._small_font, self._small_font, self._theme, footer_layout)
            self._ready_event.set()
            while not self._stop_event.is_set():
                self._handle_events()
                snapshot = self._model.snapshot()
                cursor_visible = self._stream_cursor_visible(snapshot)
                signature = self._build_signature(snapshot, cursor_visible)
                if signature != self._last_signature:
                    self._render_snapshot(snapshot, cursor_visible)
                    self._last_signature = signature
                self._clock.tick(self._config.fps)
        finally:
            pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._stop_event.set()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._stop_event.set()

    def _get_font(self, size: int, *, bold: bool = False) -> pygame.font.Font:
        key = ("monospace", size, bold)
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        candidates = ["DejaVu Sans Mono", "Liberation Mono", "Monospace", "Courier New"]
        font = None
        for candidate in candidates:
            font = pygame.font.SysFont(candidate, size, bold=bold)
            if font is not None:
                break
        if font is None:
            font = pygame.font.Font(None, size)
        self._font_cache[key] = font
        return font

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
        if self._screen is None or self._font is None or self._small_font is None or self._status_bar is None or self._message_view is None or self._footer is None:
            return

        self._screen.fill(self._theme.background)
        self._status_bar.render(self._screen, snapshot.status)
        self._message_view.render(
            self._screen,
            snapshot.messages,
            snapshot.current_stream_text,
            bool(snapshot.current_stream_text),
            cursor_visible,
        )
        self._footer.render(self._screen, snapshot.metrics)
        pygame.display.flip()

    def _build_status_layout(self, content_width: int):
        from .widgets.status_bar import StatusBarLayout

        return StatusBarLayout(x=20, y=18, width=content_width)

    def _build_message_layout(self, screen_width: int, footer_y: int, content_width: int):
        from .widgets.message_view import MessageViewLayout

        message_y = 72
        message_height = max(1, footer_y - message_y - 24)
        return MessageViewLayout(x=20, y=message_y, width=content_width, height=message_height)

    def _build_footer_layout(self, screen_width: int, screen_height: int, content_width: int):
        from .widgets.footer import FooterLayout

        footer_y = max(18, screen_height - 74)
        return FooterLayout(x=20, y=footer_y, width=content_width)
