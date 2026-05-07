import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from kronos.config import settings
from kronos.session import SessionStore


@pytest.fixture
def isolated_session_search(tmp_path, monkeypatch):
    swarm_path = tmp_path / "swarm.db"
    db_dir = tmp_path / "agent"
    db_dir.mkdir()
    session_path = db_dir / "session.db"

    monkeypatch.setattr(settings, "swarm_db_path", str(swarm_path))
    monkeypatch.setattr(settings, "db_dir", str(db_dir))
    monkeypatch.setattr(settings, "db_path", str(session_path))

    from kronos import db as _db

    _db._instances.clear()
    import kronos.swarm_store as ss

    ss._singleton = None

    from kronos.swarm_store import get_swarm

    return session_path, get_swarm()


@pytest.mark.asyncio
async def test_session_save_indexes_messages_without_duplicates(isolated_session_search):
    session_path, swarm = isolated_session_search
    store = SessionStore(str(session_path), agent_name="kronos")
    messages = [
        HumanMessage(content="We discussed Kimi migration details"),
        AIMessage(content="Decision: use the Kimi router"),
    ]

    await store.save("thread-1", messages)
    await store.save("thread-1", messages)

    results = swarm.search_sessions(
        query="migration",
        agent_name="kronos",
        days=30,
        limit=10,
    )
    assert len(results) == 1
    assert results[0]["thread_id"] == "thread-1"
    assert results[0]["role"] == "user"


@pytest.mark.asyncio
async def test_backfill_existing_sessions_is_idempotent(isolated_session_search):
    session_path, swarm = isolated_session_search
    writer = SessionStore(str(session_path))
    messages = [
        HumanMessage(content="Arbitration bug notes"),
        AIMessage(content="Fix claim winner ordering"),
    ]
    await writer.save("thread-2", messages)

    store = SessionStore(str(session_path), agent_name="kronos")
    assert await store.backfill_swarm_fts() == 2
    assert await store.backfill_swarm_fts() == 0

    results = swarm.search_sessions(query="arbitration", days=30, limit=10)
    assert len(results) == 1
    assert results[0]["agent_name"] == "kronos"


def test_session_search_filters_by_agent_and_days(isolated_session_search):
    _, swarm = isolated_session_search
    swarm.index_session_message(
        agent_name="kronos",
        thread_id="recent",
        role="user",
        content="Kimi migration discussion",
    )
    swarm.index_session_message(
        agent_name="nexus",
        thread_id="other-agent",
        role="user",
        content="Kimi migration discussion from another agent",
    )
    swarm.index_session_message(
        agent_name="kronos",
        thread_id="old",
        role="user",
        content="Legacy launch discussion",
    )
    ancient = time.time() - 40 * 86400
    swarm._db.write(
        "UPDATE session_messages SET created_at = ? WHERE thread_id = ?",
        (ancient, "old"),
    )

    kronos_results = swarm.search_sessions(
        query="Kimi",
        agent_name="kronos",
        days=30,
        limit=10,
    )
    assert [r["thread_id"] for r in kronos_results] == ["recent"]

    assert swarm.search_sessions(query="Legacy", days=30, limit=10) == []
    assert swarm.search_sessions(query="Legacy", days=90, limit=10)[0]["thread_id"] == "old"


def test_session_search_falls_back_to_like_when_fts_fails(isolated_session_search, monkeypatch):
    _, swarm = isolated_session_search
    swarm.index_session_message(
        agent_name="kronos",
        thread_id="fallback",
        role="assistant",
        content="Fallback search can still find session notes",
    )

    original_read = swarm._db.read

    def flaky_read(sql, params=()):
        if "session_messages_fts" in sql:
            raise RuntimeError("fts unavailable")
        return original_read(sql, params)

    monkeypatch.setattr(swarm._db, "read", flaky_read)

    results = swarm.search_sessions(query="Fallback search", days=30, limit=10)
    assert len(results) == 1
    assert results[0]["thread_id"] == "fallback"
