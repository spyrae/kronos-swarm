import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kronos.config import settings
from kronos.vision import analyze_image_bytes, is_supported_image_mime


def test_supported_image_mime_types():
    assert is_supported_image_mime("image/jpeg")
    assert is_supported_image_mime("image/png; charset=binary")
    assert is_supported_image_mime("image/webp")
    assert not is_supported_image_mime("application/pdf")


@pytest.mark.asyncio
async def test_analyze_image_bytes_sends_responses_image_input(monkeypatch):
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="Текст на картинке: hello")

    class FakeAsyncOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "kaos_vision_provider", "openai-api")
    monkeypatch.setattr(settings, "kaos_vision_model", "gpt-5.2-codex")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    result = await analyze_image_bytes(
        b"fake-image",
        mime_type="image/png",
        context="Сделай OCR",
    )

    assert result.text == "Текст на картинке: hello"
    assert result.model == "gpt-5.2-codex"
    request = calls[0]
    assert request["model"] == "gpt-5.2-codex"
    content = request["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert "Сделай OCR" in content[0]["text"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert base64.b64decode(content[1]["image_url"].split(",", 1)[1]) == b"fake-image"


@pytest.mark.asyncio
async def test_analyze_image_bytes_requires_configuration(monkeypatch):
    monkeypatch.setattr(settings, "kaos_vision_provider", "openai-api")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "kaos_vision_model", "gpt-5.2-codex")

    with pytest.raises(RuntimeError, match="OpenAI API vision is not configured"):
        await analyze_image_bytes(b"fake-image", mime_type="image/jpeg")


@pytest.mark.asyncio
async def test_analyze_image_bytes_uses_codex_cli_without_api_key(monkeypatch):
    from kronos import vision

    calls = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            output_path = calls[0][calls[0].index("--output-last-message") + 1]
            Path(output_path).write_text("Detected text: halo", encoding="utf-8")
            return b"", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        calls.append(list(args))
        return FakeProcess()

    monkeypatch.setattr(settings, "kaos_vision_provider", "codex-cli")
    monkeypatch.setattr(settings, "kaos_codex_command", "codex")
    monkeypatch.setattr(settings, "kaos_vision_model", "gpt-5.2-codex")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(vision.shutil, "which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.setattr(vision.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = await analyze_image_bytes(b"fake-image", mime_type="image/jpeg", context="OCR")

    assert result.text == "Detected text: halo"
    assert result.model == "gpt-5.2-codex"
    assert calls[0][:2] == ["codex", "exec"]
    assert "--images" in calls[0]
    assert "--ephemeral" in calls[0]
    assert "OPENAI_API_KEY" not in " ".join(calls[0])


def test_bridge_image_helpers_detect_photo_and_compose_prompt():
    from kronos import bridge

    event = SimpleNamespace(
        message=SimpleNamespace(
            media=object(),
            photo=object(),
        ),
    )

    assert bridge._is_image_message(event)
    assert bridge._image_mime_type(event) == "image/jpeg"

    prompt = bridge._compose_image_agent_message("Что тут написано?", "Detected text: hello")
    assert "Что тут написано?" in prompt
    assert "[Vision analysis]" in prompt
    assert "Detected text: hello" in prompt
    assert "Если пользователь просит OCR" in prompt
