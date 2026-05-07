import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeModel:
    def __init__(self, content: str, captured: dict | None = None):
        self._content = content
        self._captured = captured

    def invoke(self, messages):
        if self._captured is not None:
            self._captured["prompt"] = messages[0].content
        return FakeResponse(self._content)


class FailingModel:
    def invoke(self, _messages):
        raise RuntimeError("provider unavailable")


class FakeSwarm:
    def search_sessions(self, query: str, **_kwargs):
        if query == "предпочитаю":
            return [{"role": "user", "content": "Я предпочитаю короткие технические ответы с причиной."}]
        return []


@pytest.fixture
def user_model_workspace(tmp_path, monkeypatch):
    from kronos import workspace
    from kronos.config import settings

    root = tmp_path / "workspace"
    ws = workspace.Workspace(root)
    ws.ensure_dirs()
    db_path = tmp_path / "data" / "kronos" / "session.db"
    (db_path.parent / "logs").mkdir(parents=True)

    monkeypatch.setattr(workspace, "ws", ws)
    monkeypatch.setattr(settings, "workspace_path", str(root))
    monkeypatch.setattr(settings, "db_path", str(db_path))
    monkeypatch.setattr(settings, "agent_name", "kronos")
    return ws


def _write_audit(db_path, entries):
    audit = db_path.parent / "logs" / "audit.jsonl"
    audit.write_text(
        "\n".join(json.dumps(entry) for entry in entries),
        encoding="utf-8",
    )


def _entries(count: int = 6):
    now = datetime.now(UTC)
    rows = []
    for idx in range(count):
        user_text = "Дай краткий технический план"
        if idx == count - 1:
            user_text = "Не так, переделай короче и с причиной"
        rows.append({
            "ts": (now - timedelta(hours=idx)).isoformat(),
            "session_id": f"s-{idx}",
            "tier": "standard" if idx % 2 else "lite",
            "duration_ms": 20_000 if idx == 0 else 500 + idx,
            "approx_cost_usd": 0.0001 * (idx + 1),
            "input_len": len(user_text),
            "output_len": 90 + idx,
            "input_preview": user_text,
            "output_preview": "Ответил структурированно",
            "tool_calls_count": 6 if idx == 1 else 1,
        })
    return rows


@pytest.mark.asyncio
async def test_run_user_model_writes_model_with_passive_signals(user_model_workspace, monkeypatch):
    import kronos.cron.user_model as user_model
    import kronos.swarm_store as swarm_store
    from kronos.config import settings

    captured = {}
    sent = []
    model_text = """## Beliefs (confidence: 0.0-1.0)
- [0.90] Пользователь предпочитает краткие технические ответы.
  - Evidence: просит короче и с причиной.
  - Tensions: иногда просит глубокий аудит.

## Motivations
- Быстро закрывать инженерные решения без ручной рутины.

## Decision Patterns
- Сначала аудит, затем маленькая закрываемая задача.

## Tensions (unresolved)
- Хочет автономности, но сохраняет контроль над рискованными решениями.

## Evolution
- 2026-05: реакции заменены на пассивные сигналы качества.
"""
    _write_audit(Path(settings.db_path), _entries())
    monkeypatch.setattr(user_model, "get_model", lambda _tier: FakeModel(model_text, captured))
    monkeypatch.setattr(user_model, "send_bot_api", lambda text, topic_id=None: sent.append((text, topic_id)))
    monkeypatch.setattr(swarm_store, "get_swarm", lambda: FakeSwarm())

    await user_model.run_user_model()

    prompt = captured["prompt"]
    assert "Пассивные сигналы качества" in prompt
    assert "Correction/refinement requests: 1" in prompt
    assert "Satisfaction rate" not in prompt
    assert "предпочитаю короткие технические ответы" in prompt

    raw_model = user_model_workspace.user_model.read_text(encoding="utf-8")
    assert "# User Model" in raw_model
    assert "## Beliefs" in raw_model
    patterns = user_model_workspace.user_patterns.read_text(encoding="utf-8")
    assert "## Passive Quality Signals" in patterns
    assert "Tool-heavy sessions" in patterns
    assert sent and "## Evolution" in sent[0][0]


@pytest.mark.asyncio
async def test_run_user_model_keeps_previous_model_when_llm_fails(user_model_workspace, monkeypatch):
    import kronos.cron.user_model as user_model
    import kronos.swarm_store as swarm_store
    from kronos.config import settings

    _write_audit(Path(settings.db_path), _entries())
    user_model_workspace.user_model.write_text("existing model", encoding="utf-8")
    monkeypatch.setattr(user_model, "get_model", lambda _tier: FailingModel())
    monkeypatch.setattr(user_model, "send_bot_api", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(swarm_store, "get_swarm", lambda: FakeSwarm())

    await user_model.run_user_model()

    assert user_model_workspace.user_model.read_text(encoding="utf-8") == "existing model"


def test_collect_passive_signals_summarizes_implicit_quality():
    from kronos.cron.user_model import _collect_passive_signals

    signals = _collect_passive_signals(_entries())

    assert "Correction/refinement requests: 1" in signals
    assert "Slow responses" in signals
    assert "Tool-heavy sessions (>=5 calls): 1" in signals
