"""Tests for persona loading system (Three-Space layout)."""

from pathlib import Path

from kronos.persona import MAX_USER_MODEL_CHARS, build_system_prompt, load_persona, load_user_model
from kronos.workspace import Workspace


def _make_workspace(tmp: Path, files: dict[str, str]) -> Workspace:
    """Create a temporary Three-Space workspace with given files."""
    for name, content in files.items():
        filepath = tmp / name
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
    return Workspace(tmp)


class TestLoadPersona:
    def test_loads_core_files(self, tmp_path, monkeypatch):
        test_ws = _make_workspace(tmp_path, {
            "self/IDENTITY.md": "# Identity\nI am Kronos",
            "self/SOUL.md": "# Soul\nBoundaries here",
        })
        monkeypatch.setattr("kronos.workspace.ws", test_ws)
        result = load_persona()
        assert "I am Kronos" in result
        assert "Boundaries here" in result

    def test_missing_files_graceful(self, tmp_path, monkeypatch):
        test_ws = _make_workspace(tmp_path, {
            "self/IDENTITY.md": "# Identity",
        })
        monkeypatch.setattr("kronos.workspace.ws", test_ws)
        result = load_persona()
        assert "Identity" in result


class TestBuildSystemPrompt:
    def test_full_prompt(self, tmp_path, monkeypatch):
        test_ws = _make_workspace(tmp_path, {
            "self/IDENTITY.md": "# Kronos INTJ",
            "self/SOUL.md": "# Boundaries",
            "notes/user/MEMORY.md": "User prefers Russian",
        })
        monkeypatch.setattr("kronos.workspace.ws", test_ws)
        prompt = build_system_prompt()
        assert "Kronos INTJ" in prompt
        assert "Boundaries" in prompt
        assert "User prefers Russian" in prompt

    def test_user_model_injected_after_memory(self, tmp_path, monkeypatch):
        test_ws = _make_workspace(tmp_path, {
            "self/IDENTITY.md": "# Kronos",
            "notes/user/MEMORY.md": "Durable facts",
            "notes/user/USER-MODEL.md": "## Beliefs\n- [0.9] Prefers concise technical answers",
        })
        monkeypatch.setattr("kronos.workspace.ws", test_ws)
        prompt = build_system_prompt()

        memory_pos = prompt.index("Durable facts")
        model_pos = prompt.index("Prefers concise technical answers")
        assert "# Dialectic User Model" in prompt
        assert memory_pos < model_pos

    def test_user_model_is_truncated(self, tmp_path, monkeypatch):
        test_ws = _make_workspace(tmp_path, {
            "notes/user/USER-MODEL.md": "x" * (MAX_USER_MODEL_CHARS + 100),
        })
        monkeypatch.setattr("kronos.workspace.ws", test_ws)

        result = load_user_model()

        assert len(result) < MAX_USER_MODEL_CHARS + 100
        assert "[User model truncated]" in result

    def test_handoff_loaded_first(self, tmp_path, monkeypatch):
        test_ws = _make_workspace(tmp_path, {
            "self/IDENTITY.md": "# Kronos",
            "ops/sessions/handoff.md": "Previous session context",
        })
        monkeypatch.setattr("kronos.workspace.ws", test_ws)
        prompt = build_system_prompt()
        # Handoff should appear before persona
        handoff_pos = prompt.index("Previous session context")
        kronos_pos = prompt.index("Kronos")
        assert handoff_pos < kronos_pos
