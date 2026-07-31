"""Build compact CASE prompts from persona and memory context."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.memory.session_memory import MemoryContext
from src.config import defaults

from .persona_profile import CASE_IDENTITY, CASE_PERSONALITY_RULES
from .style_rules import BANNED_STALE_JOKES, JOKE_STYLE_RULES, NORMAL_STYLE_RULES


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    mode: str
    recent_joke_count: int


class CasePromptBuilder:
    def build(
        self,
        *,
        user_text: str,
        memory_context: MemoryContext,
        mode: str = "normal_chat",
    ) -> PromptBuildResult:
        request_kind = self.request_kind(user_text)
        rules = [CASE_IDENTITY, CASE_PERSONALITY_RULES, NORMAL_STYLE_RULES]
        if request_kind in {"joke", "roast"}:
            rules.append(JOKE_STYLE_RULES)

        recent_turns = memory_context.recent_turns[-4:]
        turn_lines = [
            f"User: {turn.user_text}\nCASE: {turn.assistant_text}"
            for turn in recent_turns
            if turn.user_text or turn.assistant_text
        ]
        avoid_jokes = [
            *BANNED_STALE_JOKES,
            *memory_context.recent_jokes[-20:],
            *memory_context.recent_roasts[-20:],
        ]
        avoid_lines = [f"- {item}" for item in avoid_jokes if item]

        sections = [
            "CASE persona:\n" + "\n".join(rules),
            (
                "Voice response style:\n"
                f"- style={defaults.CASE_VOICE_REPLY_STYLE}\n"
                f"- use 1-{defaults.CASE_VOICE_REPLY_MAX_SENTENCES} short spoken sentences by default\n"
                f"- jokes use at most {defaults.CASE_VOICE_JOKE_MAX_SENTENCES} short sentences\n"
                "- avoid paragraph-length answers unless the user asks for detail"
            ),
            "Recent conversation:\n" + ("\n".join(turn_lines) if turn_lines else "- none"),
            "Recent jokes/roasts to avoid:\n"
            + ("\n".join(avoid_lines) if avoid_lines else "- none"),
            "Current user request:\n" + user_text.strip(),
            "Reply as CASE in plain speakable dialogue only.",
        ]
        return PromptBuildResult(
            prompt="\n\n".join(sections),
            mode=mode,
            recent_joke_count=len(avoid_lines),
        )

    @staticmethod
    def request_kind(user_text: str) -> str:
        lowered = str(user_text).lower()
        if re.search(r"\b(roast|burn me)\b", lowered):
            return "roast"
        if re.search(r"\b(joke|funny|make me laugh)\b", lowered):
            return "joke"
        return "normal"
