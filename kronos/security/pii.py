"""PII masking for logs, audit records, and observability payloads.

This is a baseline regex guard, not enterprise DLP. It deliberately avoids
masking personal names because false positives would hurt agent usefulness.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

try:  # Optional at import time for lightweight security utility tests.
    from langchain_core.messages import BaseMessage
except Exception:  # pragma: no cover - langchain_core is a runtime dependency
    BaseMessage = ()  # type: ignore[assignment]


EMAIL_MASK = "***@***.com"
RU_PHONE_MASK = "+7***___**__"
INT_PHONE_MASK = "+***"
PASSPORT_RU_MASK = "**** ******"
IP_MASK = "***.***.***.***"

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_CARD_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_RU_PHONE_RE = re.compile(
    r"(?<!\d)\+?7[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
)
_INT_PHONE_RE = re.compile(r"(?<!\d)\+\d{1,3}[\s-]?\d{4,14}(?!\d)")
_PASSPORT_RU_RE = re.compile(r"\b\d{4}\s?\d{6}\b")
_IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)


def mask_pii(text: str) -> str:
    """Mask common PII patterns in a string."""
    if not text:
        return text

    masked = _EMAIL_RE.sub(EMAIL_MASK, text)
    masked = _CARD_RE.sub(_mask_card, masked)
    masked = _RU_PHONE_RE.sub(RU_PHONE_MASK, masked)
    masked = _INT_PHONE_RE.sub(INT_PHONE_MASK, masked)
    masked = _PASSPORT_RU_RE.sub(PASSPORT_RU_MASK, masked)
    return _IP_RE.sub(IP_MASK, masked)


def mask_pii_object(value: Any) -> Any:
    """Recursively mask strings in common structured observability payloads."""
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, Mapping):
        return {key: mask_pii_object(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_pii_object(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_pii_object(item) for item in value)
    if isinstance(value, set):
        return {mask_pii_object(item) for item in value}
    if BaseMessage and isinstance(value, BaseMessage):
        content = mask_pii_object(value.content)
        if hasattr(value, "model_copy"):
            return value.model_copy(update={"content": content})
        return value.copy(update={"content": content})
    if hasattr(value, "generations"):
        return _mask_llm_result(value)
    return value


def _mask_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return f"****-****-****-{digits[-4:]}"


def _mask_llm_result(value: Any) -> Any:
    """Mask LangChain LLMResult-like objects without mutating live responses."""
    try:
        cloned = deepcopy(value)
    except Exception:
        return value

    for generation_group in getattr(cloned, "generations", []) or []:
        for generation in generation_group:
            if hasattr(generation, "text") and isinstance(generation.text, str):
                generation.text = mask_pii(generation.text)
            if hasattr(generation, "message"):
                generation.message = mask_pii_object(generation.message)
    return cloned
