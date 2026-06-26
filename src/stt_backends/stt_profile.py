"""Profile policy for CASE local STT runtime selection."""

from __future__ import annotations

from dataclasses import dataclass


VALID_STT_PROFILES = ("fast", "balanced", "accuracy")
VALID_FINAL_BACKENDS = ("auto", "vosk_small", "vosk_lgraph", "sensevoice")


@dataclass(frozen=True)
class SttProfilePlan:
    profile: str
    final_backend: str
    final_chain: tuple[str, ...]

    @property
    def preferred_final_mode(self) -> str:
        return self.final_chain[0]

    @property
    def requested_label(self) -> str:
        if self.final_backend != "auto":
            return self.final_backend
        return self.profile


def normalize_stt_profile(value: str | None) -> str:
    profile = str(value or "balanced").strip().lower()
    if profile not in VALID_STT_PROFILES:
        return "balanced"
    return profile


def normalize_final_backend(value: str | None) -> str:
    backend = str(value or "auto").strip().lower()
    if backend == "sherpa_sensevoice":
        backend = "sensevoice"
    if backend in {"local_vosk_fast", "vosk"}:
        backend = "vosk_small"
    if backend not in VALID_FINAL_BACKENDS:
        return "auto"
    return backend


def resolve_stt_profile(
    profile: str | None = None,
    final_backend: str | None = None,
) -> SttProfilePlan:
    normalized_profile = normalize_stt_profile(profile)
    normalized_backend = normalize_final_backend(final_backend)

    if normalized_backend == "sensevoice":
        chain = ("sensevoice", "vosk_lgraph", "vosk_small")
    elif normalized_backend == "vosk_lgraph":
        chain = ("vosk_lgraph", "vosk_small")
    elif normalized_backend == "vosk_small":
        chain = ("vosk_small",)
    elif normalized_profile == "fast":
        chain = ("vosk_small",)
    elif normalized_profile == "accuracy":
        chain = ("sensevoice", "vosk_lgraph", "vosk_small")
    else:
        chain = ("vosk_lgraph", "vosk_small")

    return SttProfilePlan(
        profile=normalized_profile,
        final_backend=normalized_backend,
        final_chain=chain,
    )
