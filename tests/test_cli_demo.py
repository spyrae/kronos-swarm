import os

from kronos.cli import _demo_reply, main
from kronos.config import settings


def test_demo_runs_offline_without_llm_or_telegram(capsys):
    result = main(["demo"])

    out = capsys.readouterr().out
    assert result == 0
    assert "KAOS safe demo" in out
    assert "No Telegram, Docker, server registry, or LLM key is required" in out
    assert "For a real LLM-backed demo: kaos demo --live" in out


def test_demo_forces_risky_capabilities_off(monkeypatch, capsys):
    monkeypatch.setattr(settings, "enable_dynamic_tools", True)
    monkeypatch.setattr(settings, "enable_mcp_gateway_management", True)
    monkeypatch.setattr(settings, "enable_dynamic_mcp_servers", True)
    monkeypatch.setattr(settings, "enable_server_ops", True)

    result = main(["demo"])

    capsys.readouterr()
    assert result == 0
    assert settings.enable_dynamic_tools is False
    assert settings.enable_mcp_gateway_management is False
    assert settings.enable_dynamic_mcp_servers is False
    assert settings.enable_server_ops is False
    assert settings.require_dynamic_tool_sandbox is True


def test_demo_reply_mentions_relevant_kaos_module():
    assert "Memory demo" in _demo_reply("how does memory work?")
    assert "Skill demo" in _demo_reply("show skill flow")
    assert "Tool demo" in _demo_reply("mcp tools")
    assert "Swarm demo" in _demo_reply("sub agents")


def test_chat_prompt_without_runtime_llm_fails_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(settings, "fireworks_api_key", "")
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "kaos_standard_provider_chain", "kimi,deepseek")
    monkeypatch.setattr(settings, "kaos_lite_provider_chain", "deepseek,kimi")
    for name in list(os.environ):
        if name.startswith("KAOS_PROVIDER_") or name in {
            "FIREWORKS_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "LITELLM_API_KEY",
            "OLLAMA_API_KEY",
        }:
            monkeypatch.delenv(name, raising=False)

    result = main(["chat", "--prompt", "hello"])

    out = capsys.readouterr().out
    assert result == 1
    assert "KAOS chat requires at least one configured LLM provider" in out
    assert "kaos demo" in out


def test_chat_parser_accepts_prompt_and_no_memory(monkeypatch, capsys):
    import kronos.cli as cli

    async def fake_run_cli(**kwargs):
        assert kwargs["prompt"] == "hello"
        assert kwargs["enable_memory"] is False
        return 0

    monkeypatch.setattr(cli, "run_cli", fake_run_cli)

    result = cli.main(["chat", "--prompt", "hello", "--no-memory"])

    capsys.readouterr()
    assert result == 0


def test_doctor_allows_fresh_clone_before_workspace_init(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fresh-kaos'\n", encoding="utf-8")
    monkeypatch.setattr(settings, "agent_name", "kronos")
    monkeypatch.setattr(settings, "workspace_path", "")
    monkeypatch.setattr(settings, "db_dir", "data/kronos")
    monkeypatch.setattr(settings, "enable_dynamic_tools", False)
    monkeypatch.setattr(settings, "enable_mcp_gateway_management", False)
    monkeypatch.setattr(settings, "enable_dynamic_mcp_servers", False)
    monkeypatch.setattr(settings, "enable_server_ops", False)
    monkeypatch.setattr(settings, "allowed_users", "")
    monkeypatch.setattr(settings, "allow_all_users", False)

    result = main(["doctor"])

    out = capsys.readouterr().out
    assert result == 0
    assert "[WARN] Workspace: No workspace for AGENT_NAME=kronos yet" in out
    assert "kaos init kronos" in out
    assert "No hard blockers found." in out


def test_doctor_fails_when_dynamic_tools_enabled_without_sandbox_image(monkeypatch, tmp_path, capsys):
    from kronos.tools import sandbox

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fresh-kaos'\n", encoding="utf-8")
    monkeypatch.setattr(settings, "agent_name", "kronos")
    monkeypatch.setattr(settings, "workspace_path", "")
    monkeypatch.setattr(settings, "db_dir", "data/kronos")
    monkeypatch.setattr(settings, "enable_dynamic_tools", True)
    monkeypatch.setattr(settings, "require_dynamic_tool_sandbox", True)
    monkeypatch.setattr(settings, "enable_mcp_gateway_management", False)
    monkeypatch.setattr(settings, "enable_dynamic_mcp_servers", False)
    monkeypatch.setattr(settings, "enable_server_ops", False)
    monkeypatch.setattr(settings, "allowed_users", "")
    monkeypatch.setattr(settings, "allow_all_users", False)
    monkeypatch.setattr(sandbox, "sandbox_status", lambda: {
        "docker_available": True,
        "image": "kronos-sandbox:latest",
        "image_available": False,
        "build_script": "scripts/build-sandbox.sh",
    })

    result = main(["doctor"])

    out = capsys.readouterr().out
    assert result == 1
    assert "sandbox image is missing; run scripts/build-sandbox.sh" in out


def test_tool_event_printer_redacts_secret_args(capsys):
    from kronos.cli import _make_tool_event_printer

    printer = _make_tool_event_printer()
    printer("tool_call", {
        "name": "mcp_add_server",
        "args": {
            "server": "local",
            "api_key": "sk-real-secret",
            "keyword": "agent runtime",
        },
    })
    printer("tool_result", {
        "name": "mcp_add_server",
        "ok": True,
        "content": "registered",
    })

    err = capsys.readouterr().err
    assert "[tool] mcp_add_server" in err
    assert "[tool:ok] mcp_add_server registered" in err
    assert "sk-real-secret" not in err
    assert '"api_key": "***REDACTED***"' in err
    assert '"keyword": "agent runtime"' in err


def test_dashboard_command_routes_to_runner(monkeypatch, capsys):
    import kronos.cli as cli

    called = False

    def fake_dashboard():
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(cli, "run_dashboard_command", fake_dashboard)

    result = cli.main(["dashboard"])

    capsys.readouterr()
    assert result == 0
    assert called is True


def test_sessions_backfill_command_routes_to_runner(monkeypatch, capsys):
    import kronos.cli as cli

    called = False

    async def fake_backfill(agent: str = ""):
        nonlocal called
        called = True
        assert agent == "kronos"
        return 0

    monkeypatch.setattr(cli, "_run_sessions_backfill_search", fake_backfill)

    result = cli.main(["sessions", "backfill-search", "--agent", "kronos"])

    capsys.readouterr()
    assert result == 0
    assert called is True


def test_cli_version(capsys):
    import pytest

    import kronos.cli as cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])

    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert out.startswith("kaos ")
