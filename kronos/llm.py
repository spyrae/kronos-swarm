"""LLM factory with configurable provider chains and cooldown tracking.

Default resolution order:
  standard: Kimi K2.5 via Fireworks -> DeepSeek V3
  lite:     DeepSeek V3 -> Kimi K2.5 via Fireworks

Users can override the chains and provider details from .env without editing
Python code. Most hosted and local providers are covered through the generic
OpenAI-compatible adapter.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from kronos.config import settings
from kronos.security.pii import mask_pii, mask_pii_object

log = logging.getLogger("kronos.llm")

COOLDOWN_SECONDS = 300  # 5 minutes


class ModelTier(str, Enum):
    LITE = "lite"
    STANDARD = "standard"


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider configuration."""

    provider_id: str
    adapter: str
    model: str
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    max_tokens: int = 4096
    temperature: float = 0.5
    api_key_required: bool = True

    @property
    def ready(self) -> bool:
        return bool(self.model) and (bool(self.api_key) or not self.api_key_required)

    @property
    def display(self) -> str:
        base = f"{self.provider_id}:{self.model}"
        if self.base_url:
            base = f"{base} @ {self.base_url}"
        return base

    @property
    def signature(self) -> tuple:
        return (
            self.provider_id,
            self.adapter,
            self.model,
            self.base_url,
            bool(self.api_key),
            self.api_key_env,
            self.max_tokens,
            self.temperature,
            self.api_key_required,
        )


class _ProviderState:
    """Tracks per-provider health and cached model instances."""

    def __init__(self):
        self._cooldowns: dict[str, float] = {}
        self._models: dict[str, tuple[tuple, BaseChatModel]] = {}

    def is_available(self, provider: str) -> bool:
        cooldown_until = self._cooldowns.get(provider, 0)
        return time.time() >= cooldown_until

    def mark_failed(self, provider: str) -> None:
        self._cooldowns[provider] = time.time() + COOLDOWN_SECONDS
        log.warning("Provider '%s' entered cooldown for %ds", provider, COOLDOWN_SECONDS)

    def mark_success(self, provider: str) -> None:
        self._cooldowns.pop(provider, None)

    def get_or_create(self, provider: str) -> BaseChatModel | None:
        config = resolve_provider_config(provider)
        if not config or not config.ready:
            return None

        cached = self._models.get(provider)
        if cached and cached[0] == config.signature:
            return cached[1]

        model = _create_model(config)
        if model:
            self._models[provider] = (config.signature, model)
        return model

    def reset_cooldown(self, provider: str) -> None:
        self._cooldowns.pop(provider, None)

    def clear_cache(self) -> None:
        self._models.clear()


_state = _ProviderState()
_callback_signature: tuple[str, bool, str] | None = None
_callback_cache: list[BaseCallbackHandler] = []


class _PIIMaskingCallbackHandler(BaseCallbackHandler):
    """Forward LangChain callback events after masking PII payloads."""

    def __init__(self, inner: BaseCallbackHandler):
        super().__init__()
        self._inner = inner

    def _forward(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._inner, method_name, None)
        if not method:
            return None
        masked_args = mask_pii_object(args)
        masked_kwargs = mask_pii_object(kwargs)
        return method(*masked_args, **masked_kwargs)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        return self._forward("on_llm_start", serialized, prompts, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        return self._forward("on_chat_model_start", serialized, messages, **kwargs)

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        return self._forward("on_llm_end", response, **kwargs)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        return self._forward("on_llm_error", RuntimeError(mask_pii(str(error))), **kwargs)

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> Any:
        return self._forward("on_chain_start", serialized, inputs, **kwargs)

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> Any:
        return self._forward("on_chain_end", outputs, **kwargs)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        return self._forward("on_tool_start", serialized, input_str, **kwargs)

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        return self._forward("on_tool_end", output, **kwargs)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        return self._forward("on_text", text, **kwargs)


_PRESETS: dict[str, dict[str, object]] = {
    "kimi": {
        "adapter": "openai-compatible",
        "model": "accounts/fireworks/routers/kimi-k2p5-turbo",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "max_tokens": 8192,
    },
    "fireworks": {
        "adapter": "openai-compatible",
        "model": "accounts/fireworks/routers/kimi-k2p5-turbo",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "max_tokens": 8192,
    },
    "deepseek": {
        "adapter": "deepseek",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "max_tokens": 4096,
    },
    "openai": {
        "adapter": "openai-compatible",
        "model": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 4096,
    },
    "openrouter": {
        "adapter": "openai-compatible",
        "model": "openai/gpt-4.1-mini",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "max_tokens": 4096,
    },
    "groq": {
        "adapter": "openai-compatible",
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "max_tokens": 4096,
    },
    "together": {
        "adapter": "openai-compatible",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "max_tokens": 4096,
    },
    "litellm": {
        "adapter": "openai-compatible",
        "model": "gpt-4.1-mini",
        "base_url": "http://127.0.0.1:4000/v1",
        "api_key_env": "LITELLM_API_KEY",
        "max_tokens": 4096,
        "api_key_required": False,
    },
    "ollama": {
        "adapter": "openai-compatible",
        "model": "llama3.1",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key_env": "OLLAMA_API_KEY",
        "max_tokens": 4096,
        "api_key_required": False,
    },
    "local": {
        "adapter": "openai-compatible",
        "model": "llama3.1",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key_env": "LOCAL_LLM_API_KEY",
        "max_tokens": 4096,
        "api_key_required": False,
    },
}


def get_model(tier: ModelTier = ModelTier.STANDARD) -> BaseChatModel:
    """Get the first available chat model for the given tier."""
    chain = provider_chain(tier)

    for provider in chain:
        if not _has_key(provider):
            continue
        if not _state.is_available(provider):
            continue
        model = _state.get_or_create(provider)
        if model:
            return model

    # Fallback: ignore cooldowns, try anything configured.
    for provider in chain:
        if not _has_key(provider):
            continue
        model = _state.get_or_create(provider)
        if model:
            log.warning("All providers in cooldown, using '%s' anyway", provider)
            return model

    configured = ", ".join(chain) or "(empty chain)"
    raise RuntimeError(f"No API keys configured for {tier.value} tier: {configured}")


def get_fallback_model() -> BaseChatModel:
    """Get a fallback model using the union of configured chains."""
    seen: set[str] = set()
    for tier in (ModelTier.LITE, ModelTier.STANDARD):
        for provider in provider_chain(tier):
            if provider in seen:
                continue
            seen.add(provider)
            if not _has_key(provider):
                continue
            if not _state.is_available(provider):
                continue
            model = _state.get_or_create(provider)
            if model:
                return model

    raise RuntimeError("No fallback LLM providers configured")


def invoke_with_fallback(
    messages: list[BaseMessage],
    tier: ModelTier = ModelTier.STANDARD,
    tools: list | None = None,
) -> BaseMessage:
    """Invoke LLM with automatic fallback chain and cooldown tracking."""
    chain = provider_chain(tier)
    last_error = None

    for provider in chain:
        if not _has_key(provider):
            continue
        if not _state.is_available(provider):
            log.debug("Skipping '%s' (cooldown)", provider)
            continue

        model = _state.get_or_create(provider)
        if not model:
            continue

        try:
            if tools:
                model = model.bind_tools(tools)
            response = model.invoke(messages)
            _state.mark_success(provider)
            return response
        except Exception as e:
            last_error = e
            _state.mark_failed(provider)
            log.error("Provider '%s' failed: %s", provider, e)
            continue

    # Last resort: retry first configured provider ignoring cooldowns.
    for provider in chain:
        if not _has_key(provider):
            continue
        model = _state.get_or_create(provider)
        if not model:
            continue
        try:
            if tools:
                model = model.bind_tools(tools)
            response = model.invoke(messages)
            _state.mark_success(provider)
            return response
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


def is_runtime_llm_configured() -> bool:
    """Return whether any provider in either configured chain is ready."""
    return any(
        _has_key(provider)
        for tier in (ModelTier.STANDARD, ModelTier.LITE)
        for provider in provider_chain(tier)
    )


def provider_chain(tier: ModelTier = ModelTier.STANDARD) -> list[str]:
    """Return normalized provider ids for a tier."""
    raw = (
        settings.kaos_standard_provider_chain
        if tier == ModelTier.STANDARD
        else settings.kaos_lite_provider_chain
    )
    fallback = "kimi,deepseek" if tier == ModelTier.STANDARD else "deepseek,kimi"
    return _parse_chain(raw or fallback)


def describe_provider_chain(tier: ModelTier = ModelTier.STANDARD) -> list[dict[str, object]]:
    """Return provider state for CLI/docs without constructing models."""
    rows: list[dict[str, object]] = []
    for provider in provider_chain(tier):
        config = resolve_provider_config(provider)
        rows.append({
            "provider": provider,
            "configured": bool(config and config.ready),
            "model": config.model if config else "",
            "base_url": config.base_url if config else "",
            "api_key_env": config.api_key_env if config else "",
            "api_key_required": config.api_key_required if config else True,
            "adapter": config.adapter if config else "",
        })
    return rows


def resolve_provider_config(provider: str) -> ProviderConfig | None:
    """Resolve a provider preset plus KAOS_PROVIDER_<ID>_* overrides."""
    provider_id = _normalize_provider_id(provider)
    preset = dict(_PRESETS.get(provider_id, {}))

    prefix = f"KAOS_PROVIDER_{_env_key(provider_id)}_"
    adapter = str(_env(prefix + "ADAPTER", _env(prefix + "TYPE", preset.get("adapter", "openai-compatible"))))
    model = str(_env(prefix + "MODEL", preset.get("model", "")))
    base_url = str(_env(prefix + "BASE_URL", preset.get("base_url", ""))).rstrip("/")
    api_key_env = str(_env(prefix + "API_KEY_ENV", preset.get("api_key_env", "")))
    api_key = str(_env(prefix + "API_KEY", ""))
    max_tokens = _int_env(prefix + "MAX_TOKENS", int(preset.get("max_tokens", 4096)))
    temperature = _float_env(prefix + "TEMPERATURE", float(preset.get("temperature", 0.5)))
    api_key_required = _bool_env(prefix + "API_KEY_REQUIRED", bool(preset.get("api_key_required", True)))

    if not api_key and api_key_env:
        api_key = _api_key_from_env(api_key_env)

    if not model:
        return None

    return ProviderConfig(
        provider_id=provider_id,
        adapter=adapter,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key_required=api_key_required,
    )


def reset_provider_state() -> None:
    """Clear cached model instances and cooldowns. Intended for tests."""
    global _callback_cache, _callback_signature
    _state._cooldowns.clear()
    _state.clear_cache()
    _callback_signature = None
    _callback_cache = []


def _has_key(provider: str) -> bool:
    config = resolve_provider_config(provider)
    return bool(config and config.ready)


def _create_model(config: ProviderConfig) -> BaseChatModel | None:
    try:
        callbacks = _observability_callbacks()
        if config.adapter in {"openai", "openai-compatible", "openai_compatible"}:
            from langchain_openai import ChatOpenAI

            api_key = config.api_key or "not-needed"
            kwargs = {
                "model": config.model,
                "api_key": api_key,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url
            if callbacks:
                kwargs["callbacks"] = callbacks
            return ChatOpenAI(**kwargs)

        if config.adapter == "deepseek":
            from langchain_deepseek import ChatDeepSeek

            kwargs = {
                "model": config.model,
                "api_key": config.api_key,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            }
            if callbacks:
                kwargs["callbacks"] = callbacks
            return ChatDeepSeek(**kwargs)

        log.error("Unsupported LLM adapter '%s' for provider '%s'", config.adapter, config.provider_id)
    except Exception as e:
        log.error("Failed to create '%s' model: %s", config.provider_id, e)
    return None


def _observability_callbacks() -> list[BaseCallbackHandler]:
    """Create optional Langfuse callbacks with PII masking."""
    global _callback_cache, _callback_signature
    signature = (
        settings.langfuse_public_key,
        bool(settings.langfuse_secret_key),
        settings.langfuse_host,
    )
    if signature == _callback_signature:
        return _callback_cache

    _callback_signature = signature
    _callback_cache = []
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return _callback_cache

    try:
        from langfuse.callback import CallbackHandler

        kwargs = {
            "public_key": settings.langfuse_public_key,
            "secret_key": settings.langfuse_secret_key,
        }
        if settings.langfuse_host:
            kwargs["host"] = settings.langfuse_host
        _callback_cache = [_PIIMaskingCallbackHandler(CallbackHandler(**kwargs))]
    except Exception as e:
        log.warning("Langfuse callback disabled: %s", e)
        _callback_cache = []

    return _callback_cache


def _parse_chain(value: str) -> list[str]:
    providers = [
        _normalize_provider_id(item)
        for item in re.split(r"[, ]+", value.strip())
        if item.strip()
    ]
    return providers


def _normalize_provider_id(provider: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", provider.strip().lower()).strip("_")


def _env_key(provider: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_")


def _env(name: str, default: object = "") -> object:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        log.warning("Invalid integer for %s=%r; using %s", name, value, default)
        return default


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        log.warning("Invalid float for %s=%r; using %s", name, value, default)
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _api_key_from_env(name: str) -> str:
    if name == "FIREWORKS_API_KEY":
        return settings.fireworks_api_key or os.environ.get(name, "")
    if name == "DEEPSEEK_API_KEY":
        return settings.deepseek_api_key or os.environ.get(name, "")
    if name == "OPENAI_API_KEY":
        return settings.openai_api_key or os.environ.get(name, "")
    if name == "GROQ_API_KEY":
        return settings.groq_api_key or os.environ.get(name, "")
    if name == "LITELLM_API_KEY":
        return os.environ.get(name) or settings.litellm_admin_key
    if name == "LITELLM_ADMIN_KEY":
        return settings.litellm_admin_key or os.environ.get(name, "")
    return os.environ.get(name, "")
