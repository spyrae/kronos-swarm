"""Logging helpers for KAOS runtime processes."""

from __future__ import annotations

import logging
from typing import Any

from kronos.security.pii import mask_pii, mask_pii_object

_original_record_factory = None


class PIIFilter(logging.Filter):
    """Mask PII before log records are formatted or streamed."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_pii(str(record.msg))
        if record.args:
            record.args = _mask_args(record.args)
        if record.exc_text:
            record.exc_text = mask_pii(record.exc_text)
        return True


def add_pii_filter(target: logging.Filterer) -> None:
    """Attach a PII filter once to a logger or handler."""
    if not any(isinstance(item, PIIFilter) for item in target.filters):
        target.addFilter(PIIFilter())


def install_pii_filter(logger: logging.Logger | None = None) -> None:
    """Install PII masking on the root logger and current handlers."""
    _install_record_factory()
    target = logger or logging.getLogger()
    add_pii_filter(target)
    for handler in target.handlers:
        add_pii_filter(handler)


def _install_record_factory() -> None:
    global _original_record_factory
    if _original_record_factory is not None:
        return

    _original_record_factory = logging.getLogRecordFactory()
    pii_filter = PIIFilter()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = _original_record_factory(*args, **kwargs)
        pii_filter.filter(record)
        return record

    logging.setLogRecordFactory(factory)


def _mask_args(args: Any) -> Any:
    if isinstance(args, dict):
        return {key: mask_pii_object(value) for key, value in args.items()}
    if isinstance(args, tuple):
        return tuple(mask_pii_object(value) for value in args)
    return mask_pii_object(args)
