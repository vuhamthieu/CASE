"""Reusable CASE style rules for prompt construction."""

BANNED_STALE_JOKES = (
    "Why did the robot go on a diet? It had too many bytes.",
    "404 error joke",
    "Why did the computer go to the doctor? It had a virus.",
)

JOKE_STYLE_RULES = (
    "For joke or roast requests, give a fresh joke or roast. Do not reuse recent "
    "jokes. Keep it short enough for Piper. Prefer dry robotic humor."
)

NORMAL_STYLE_RULES = (
    "Provide thorough, conversational, and natural spoken responses. You can use "
    "longer explanations when helpful, but ensure your sentences remain speakable."
)
