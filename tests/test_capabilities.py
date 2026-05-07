import pytest

from kronos.config import Settings, settings


def test_empty_telegram_allowlist_blocks_by_default():
    cfg = Settings(_env_file=None, allowed_users="", allow_all_users=False)

    assert cfg.allowed_user_ids == set()
    assert cfg.is_telegram_user_allowed(123) is False
    assert "NONE" in cfg.telegram_access_description


def test_allow_all_users_is_explicit():
    cfg = Settings(_env_file=None, allowed_users="", allow_all_users=True)

    assert cfg.is_telegram_user_allowed(123) is True
    assert "ALLOW_ALL_USERS=true" in cfg.telegram_access_description


def test_allowed_users_take_precedence():
    cfg = Settings(_env_file=None, allowed_users="123, 456", allow_all_users=False)

    assert cfg.allowed_user_ids == {123, 456}
    assert cfg.is_telegram_user_allowed(123) is True
    assert cfg.is_telegram_user_allowed(999) is False


def test_allowed_users_ignores_blank_comment_placeholder():
    cfg = Settings(
        _env_file=None,
        allowed_users="# comma-separated Telegram user IDs",
        allow_all_users=False,
    )

    assert cfg.allowed_user_ids == set()
    assert cfg.invalid_allowed_user_tokens == ()
    assert cfg.is_telegram_user_allowed(123) is False


def test_allowed_users_reports_invalid_tokens():
    cfg = Settings(_env_file=None, allowed_users="123, not-a-user", allow_all_users=False)

    assert cfg.allowed_user_ids == {123}
    assert cfg.invalid_allowed_user_tokens == ("not-a-user",)


@pytest.mark.asyncio
async def test_dynamic_tool_creation_is_disabled_by_default(monkeypatch):
    from kronos.tools.dynamic import create_tool, load_persisted_tools
    from kronos.tools.dynamic_tools import get_dynamic_management_tools

    monkeypatch.setattr(settings, "enable_dynamic_tools", False)

    tool, message = await create_tool("sample_tool", "Return a sample string")

    assert tool is None
    assert "disabled" in message.lower()
    assert load_persisted_tools() == []
    assert get_dynamic_management_tools() == []


@pytest.mark.asyncio
async def test_dynamic_tool_sandbox_fails_closed(monkeypatch):
    from kronos.tools import sandbox

    monkeypatch.setattr(settings, "require_dynamic_tool_sandbox", True)
    monkeypatch.setattr(sandbox, "_docker_available", lambda: False)

    stdout, stderr = await sandbox.execute_sandboxed("print('unsafe')")

    assert stdout == ""
    assert "Docker is required" in stderr


@pytest.mark.asyncio
async def test_dynamic_tool_sandbox_fails_closed_without_image(monkeypatch):
    from kronos.tools import sandbox

    monkeypatch.setattr(settings, "require_dynamic_tool_sandbox", True)
    monkeypatch.setattr(sandbox, "_docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "_docker_image_available", lambda image=sandbox.SANDBOX_IMAGE: False)

    stdout, stderr = await sandbox.execute_sandboxed("print('unsafe')")

    assert stdout == ""
    assert "sandbox image" in stderr.lower()


def test_sandbox_command_uses_restrictive_docker_flags():
    from kronos.tools import sandbox

    cmd = sandbox.build_sandbox_command("/tmp/kronos-sandbox-test", memory_limit="128m")

    assert "--network=none" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--pids-limit=50" in cmd
    assert "--user=10001:10001" in cmd
    assert "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m" in cmd
    assert "kronos-sandbox:latest" in cmd
    assert cmd[-2:] == ["python", "/sandbox/runner.py"]


def test_dynamic_tool_validation_rejects_top_level_execution():
    from kronos.tools.dynamic import validate_code

    valid, reason = validate_code(
        "print('side effect')\n\n"
        "async def sample_tool() -> str:\n"
        "    return 'ok'\n"
    )

    assert valid is False
    assert "Top-level statements" in reason


@pytest.mark.asyncio
async def test_dynamic_tool_creation_requires_ready_sandbox_image(monkeypatch):
    from kronos.tools import dynamic, sandbox

    class FakeModel:
        def invoke(self, _messages):
            return type("Response", (), {
                "content": (
                    "async def hello_tool(name: str) -> str:\n"
                    "    \"\"\"Say hello.\"\"\"\n"
                    "    return f'hello {name}'\n"
                )
            })()

    monkeypatch.setattr(settings, "enable_dynamic_tools", True)
    monkeypatch.setattr(settings, "require_dynamic_tool_sandbox", True)
    monkeypatch.setattr(dynamic, "get_model", lambda _tier: FakeModel())
    monkeypatch.setattr(sandbox, "sandbox_ready", lambda: False)
    monkeypatch.setattr(
        sandbox,
        "sandbox_unavailable_message",
        lambda: "Sandbox image kronos-sandbox:latest is missing. Run `scripts/build-sandbox.sh`.",
    )

    tool, message = await dynamic.create_tool("hello_tool", "Say hello")

    assert tool is None
    assert "sandbox image" in message.lower()


@pytest.mark.asyncio
async def test_dynamic_tool_can_register_from_ast_metadata_without_required_sandbox(monkeypatch, tmp_path):
    from kronos.tools import dynamic, sandbox

    class FakeModel:
        def invoke(self, _messages):
            return type("Response", (), {
                "content": (
                    "async def hello_tool(name: str) -> str:\n"
                    "    \"\"\"Say hello.\"\"\"\n"
                    "    return f'hello {name}'\n"
                )
            })()

    monkeypatch.setattr(settings, "enable_dynamic_tools", True)
    monkeypatch.setattr(settings, "require_dynamic_tool_sandbox", False)
    monkeypatch.setattr(dynamic, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(dynamic, "get_model", lambda _tier: FakeModel())
    monkeypatch.setattr(sandbox, "sandbox_ready", lambda: False)

    tool, message = await dynamic.create_tool("hello_tool", "Say hello")

    assert tool is not None
    assert "created successfully" in message
    assert await tool.ainvoke({"name": "Ada"}) == "hello Ada"
    assert (tmp_path / "hello_tool.py").exists()
