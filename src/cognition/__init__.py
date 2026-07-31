"""Local cognition helpers that run before cloud personality handling."""

from .intent_router import IntentRouter, IntentType, LocalIntent

__all__ = ["IntentRouter", "IntentType", "LocalIntent"]
