"""Environment loading and validated override helpers."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("CONFIG: invalid boolean %s; using default=%s", name, default)
    return default


def get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("CONFIG: invalid float %s; using default=%s", name, default)
        return default


def get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("CONFIG: invalid integer %s; using default=%s", name, default)
        return default


def mask_secret(value: str) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...****"


def log_secret_status(name: str, value: str) -> None:
    logger.debug("CONFIG: %s loaded: %s", name, mask_secret(value))
