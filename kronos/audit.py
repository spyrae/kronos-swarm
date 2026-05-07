"""Audit and cost logging — tracks every request with approximate cost.

Logs to JSONL files:
- audit.jsonl: full request/response audit trail
- cost.jsonl: cost tracking per request
- tool_calls.jsonl: durable tool-call trail
"""

import json
import logging
import math
import re
import time
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

from kronos.config import settings
from kronos.security.pii import mask_pii

log = logging.getLogger("kronos.audit")

# DeepSeek V3 pricing (per 1M tokens)
COST_TABLE = {
    "lite": {"input": 0.27, "output": 1.10},
    "standard": {"input": 0.27, "output": 1.10},  # same model for now
    "blocked": {"input": 0, "output": 0},
}

_audit_dir: Path | None = None
_tool_audit_context: ContextVar[dict[str, str]] = ContextVar("tool_audit_context", default={})
_SECRET_FIELD_NAMES = {"token", "secret", "password", "api_key", "apikey", "key", "hash", "authorization"}
_SECRET_PATTERNS = (
    (re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}"), "Bearer ***REDACTED***"),
    (re.compile(r"sk-[A-Za-z0-9_-]{12,}"), "sk-***REDACTED***"),
    (re.compile(r"(api[_-]?key=)[^&\s]+", re.IGNORECASE), r"\1***REDACTED***"),
    (re.compile(r"(token=)[^&\s]+", re.IGNORECASE), r"\1***REDACTED***"),
)


def _get_audit_dir() -> Path:
    global _audit_dir
    target = Path(settings.db_path).parent / "logs"
    if _audit_dir != target:
        _audit_dir = target
        _audit_dir.mkdir(parents=True, exist_ok=True)
    return _audit_dir


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3.5 chars per token for mixed RU/EN."""
    return math.ceil(len(text) / 3.5)


def set_tool_audit_context(**context: str) -> Token:
    """Set per-request context for durable tool-call audit events."""
    clean = {key: str(value) for key, value in context.items() if value is not None}
    return _tool_audit_context.set(clean)


def reset_tool_audit_context(token: Token) -> None:
    _tool_audit_context.reset(token)


def _redact_string(value: str, *, max_len: int = 500) -> str:
    redacted = value
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    redacted = mask_pii(redacted)
    if len(redacted) > max_len:
        return f"{redacted[:max_len - 3]}..."
    return redacted


def redact_tool_payload(value: Any, key: str = "") -> Any:
    """Redact secret-like fields before tool args/results reach storage or UI."""
    key_name = key.lower().replace("-", "_")
    if key_name in _SECRET_FIELD_NAMES or key_name.endswith(("_token", "_secret", "_password", "_api_key", "_key")):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {str(k): redact_tool_payload(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_tool_payload(item) for item in value[:20]]
    if isinstance(value, tuple):
        return [redact_tool_payload(item) for item in value[:20]]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _summarize_payload(value: Any, *, max_len: int = 500) -> str:
    redacted = redact_tool_payload(value)
    if isinstance(redacted, str):
        return _redact_string(redacted, max_len=max_len)
    try:
        rendered = json.dumps(redacted, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        rendered = str(redacted)
    return _redact_string(rendered, max_len=max_len)


def _infer_tool_capability(tool_name: str) -> str:
    name = tool_name.lower()
    if name.startswith("delegate_to_"):
        return "delegation"
    if name.startswith("mcp_"):
        return "mcp"
    if name in {"load_skill", "load_skill_reference", "approve_skill", "import_skill_from_source"}:
        return "skills"
    if "memory" in name or "session_search" in name:
        return "memory"
    if "browser" in name or "search" in name or "fetch" in name or "exa" in name or "brave" in name:
        return "research"
    if "expense" in name or "budget" in name or "tranche" in name:
        return "finance"
    if "server" in name or "ssh" in name:
        return "server_ops"
    if "dynamic" in name or "create_new_tool" in name:
        return "dynamic_tools"
    return "tools"


def _tool_event_status(event: str, payload: dict[str, Any]) -> str:
    if event == "tool_call":
        return "called"
    content = str(payload.get("content", ""))
    if content.startswith("[BLOCKED]") or content.lower().startswith("blocked:"):
        return "blocked"
    if payload.get("ok") is False or content.startswith("[ERROR]"):
        return "error"
    return "ok"


def log_tool_event(event: str, payload: dict[str, Any]) -> None:
    """Persist a tool-call lifecycle event to ``tool_calls.jsonl``.

    This stores only redacted summaries. Raw args/results never leave memory.
    """
    try:
        name = str(payload.get("name") or "unknown")
        status = _tool_event_status(event, payload)
        context = _tool_audit_context.get()
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            "status": status,
            "tool": name,
            "capability": str(payload.get("capability") or _infer_tool_capability(name)),
            "approval_status": "blocked" if status == "blocked" else "not_required",
            "call_id": str(payload.get("call_id") or payload.get("id") or ""),
            "turn": payload.get("turn"),
            "agent": context.get("agent", settings.agent_name),
            "thread_id": context.get("thread_id", ""),
            "session_id": context.get("session_id", ""),
            "user_id": context.get("user_id", ""),
            "source_kind": context.get("source_kind", ""),
            "args_summary": _summarize_payload(payload.get("args", {})),
            "result_summary": _summarize_payload(payload.get("content", "")) if event == "tool_result" else "",
            "error": status in {"error", "blocked"},
            "duration_ms": payload.get("duration_ms"),
            "cost_usd": payload.get("cost_usd"),
            "input_tokens": payload.get("input_tokens"),
            "output_tokens": payload.get("output_tokens"),
        }

        audit_dir = _get_audit_dir()
        with open(audit_dir / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        log.debug("Tool audit logging failed: %s", e)


def log_request(
    *,
    user_id: str,
    session_id: str,
    tier: str,
    input_text: str,
    output_text: str,
    duration_ms: int,
    agent_path: str = "",
    blocked: bool = False,
) -> None:
    """Log a request to audit and cost JSONL files."""
    try:
        input_tokens = _estimate_tokens(input_text)
        output_tokens = _estimate_tokens(output_text)
        costs = COST_TABLE.get(tier, COST_TABLE["standard"])
        approx_cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        audit_dir = _get_audit_dir()

        # Audit log (detailed)
        audit_entry = {
            "ts": ts,
            "user_id": user_id,
            "session_id": session_id,
            "tier": tier,
            "agent_path": agent_path,
            "blocked": blocked,
            "duration_ms": duration_ms,
            "input_len": len(input_text),
            "output_len": len(output_text),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "approx_cost_usd": round(approx_cost, 6),
            "input_preview": mask_pii(input_text)[:100],
            "output_preview": mask_pii(output_text)[:100],
        }

        with open(audit_dir / "audit.jsonl", "a") as f:
            f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")

        # Cost log (compact, for aggregation)
        cost_entry = {
            "ts": ts,
            "tier": tier,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(approx_cost, 6),
        }

        with open(audit_dir / "cost.jsonl", "a") as f:
            f.write(json.dumps(cost_entry) + "\n")

        log.debug(
            "Audit: tier=%s, tokens=%d+%d, cost=$%.6f, duration=%dms",
            tier, input_tokens, output_tokens, approx_cost, duration_ms,
        )

    except Exception as e:
        log.error("Audit logging failed: %s", e)


def get_daily_cost() -> dict:
    """Get today's cost summary from cost.jsonl."""
    today = time.strftime("%Y-%m-%d")
    total_cost = 0.0
    total_requests = 0
    total_input_tokens = 0
    total_output_tokens = 0

    cost_file = _get_audit_dir() / "cost.jsonl"
    if not cost_file.exists():
        return {"date": today, "cost_usd": 0, "requests": 0, "input_tokens": 0, "output_tokens": 0}

    try:
        with open(cost_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ts", "").startswith(today):
                    total_cost += entry.get("cost_usd", 0)
                    total_requests += 1
                    total_input_tokens += entry.get("input_tokens", 0)
                    total_output_tokens += entry.get("output_tokens", 0)
    except Exception as e:
        log.error("Cost aggregation failed: %s", e)

    return {
        "date": today,
        "cost_usd": round(total_cost, 4),
        "requests": total_requests,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }
