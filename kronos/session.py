"""Session store — persistent conversation history per thread_id.

Replaces LangGraph's AsyncSqliteSaver checkpointer.
Stores messages as JSON in SQLite, keyed by thread_id.
"""

import hashlib
import json
import logging
from contextlib import asynccontextmanager

import aiosqlite
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

log = logging.getLogger("kronos.session")

# Max messages to keep in history (oldest are dropped on save).
# Keep small — large history causes LLM to copy prior patterns
# (including hallucinated tool calls) instead of using tools.
MAX_HISTORY = 30


def _session_fts_fingerprint(
    *,
    agent_name: str,
    thread_id: str,
    position: int,
    role: str,
    content: str,
) -> str:
    """Stable key for idempotent cross-session FTS indexing."""
    payload = json.dumps(
        [agent_name, thread_id, position, role, content],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize_message(msg: BaseMessage) -> dict:
    """Serialize a LangChain message to a JSON-safe dict."""
    data = {
        "type": msg.__class__.__name__,
        "content": msg.content,
    }
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = msg.tool_calls
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        data["tool_call_id"] = msg.tool_call_id
    return data


def _deserialize_message(data: dict) -> BaseMessage:
    """Deserialize a dict back to a LangChain message."""
    msg_type = data.get("type", "HumanMessage")
    content = data.get("content", "")

    if msg_type == "HumanMessage":
        return HumanMessage(content=content)
    elif msg_type == "AIMessage":
        msg = AIMessage(content=content)
        if data.get("tool_calls"):
            msg.tool_calls = data["tool_calls"]
        return msg
    elif msg_type == "SystemMessage":
        return SystemMessage(content=content)
    elif msg_type == "ToolMessage":
        return ToolMessage(
            content=content,
            tool_call_id=data.get("tool_call_id", ""),
        )
    else:
        return HumanMessage(content=content)


class SessionStore:
    """Async SQLite-based session store for conversation history."""

    def __init__(self, db_path: str, agent_name: str = ""):
        self.db_path = db_path
        self._agent_name = agent_name
        self._initialized = False

    @asynccontextmanager
    async def _open_db(self):
        """Open a connection with WAL mode and generous busy timeout."""
        async with aiosqlite.connect(self.db_path, timeout=30) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=30000")
            await db.execute("PRAGMA wal_autocheckpoint=100")
            yield db

    async def _ensure_table(self, db: aiosqlite.Connection) -> None:
        """Create sessions table if it doesn't exist."""
        if not self._initialized:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    thread_id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL DEFAULT '[]',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()
            self._initialized = True

    async def load(self, thread_id: str) -> list[BaseMessage]:
        """Load conversation history for a thread."""
        async with self._open_db() as db:
            await self._ensure_table(db)
            cursor = await db.execute(
                "SELECT messages FROM sessions WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()

        if not row:
            return []

        try:
            data = json.loads(row[0])
            return [_deserialize_message(d) for d in data]
        except (json.JSONDecodeError, KeyError) as e:
            log.error("Failed to deserialize session %s: %s", thread_id, e)
            return []

    async def save(self, thread_id: str, messages: list[BaseMessage]) -> None:
        """Save conversation history, keeping only the last MAX_HISTORY messages."""
        # Trim to max history (keep most recent)
        trimmed = messages[-MAX_HISTORY:] if len(messages) > MAX_HISTORY else messages

        data = json.dumps(
            [_serialize_message(m) for m in trimmed],
            ensure_ascii=False,
        )

        async with self._open_db() as db:
            await self._ensure_table(db)
            await db.execute(
                """INSERT INTO sessions (thread_id, messages, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(thread_id) DO UPDATE SET
                     messages = excluded.messages,
                     updated_at = excluded.updated_at""",
                (thread_id, data),
            )
            await db.commit()

        self._index_to_swarm_fts(thread_id, trimmed)

    def _index_to_swarm_fts(
        self, thread_id: str, messages: list[BaseMessage],
    ) -> int:
        """Index session messages into swarm FTS. Non-blocking, non-fatal."""
        if not self._agent_name:
            return 0
        try:
            from kronos.swarm_store import get_swarm

            swarm = get_swarm()
            indexed = 0
            for position, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                else:
                    continue
                if msg.content and isinstance(msg.content, str) and len(msg.content) > 5:
                    inserted = swarm.index_session_message(
                        agent_name=self._agent_name,
                        thread_id=thread_id,
                        role=role,
                        content=msg.content,
                        fingerprint=_session_fts_fingerprint(
                            agent_name=self._agent_name,
                            thread_id=thread_id,
                            position=position,
                            role=role,
                            content=msg.content,
                        ),
                    )
                    if inserted:
                        indexed += 1
            return indexed
        except Exception as e:
            log.warning("FTS indexing failed (non-fatal): %s", e)
            return 0

    async def backfill_swarm_fts(self) -> int:
        """Index existing session rows into the shared session-search FTS store.

        This is idempotent when the target swarm database has fingerprints.
        """
        if not self._agent_name:
            log.info("Skipping session FTS backfill: agent_name is empty")
            return 0

        rows: list[tuple[str, str]] = []
        async with self._open_db() as db:
            await self._ensure_table(db)
            cursor = await db.execute("SELECT thread_id, messages FROM sessions")
            rows = await cursor.fetchall()

        indexed = 0
        for thread_id, raw_messages in rows:
            try:
                data = json.loads(raw_messages)
                messages = [_deserialize_message(d) for d in data]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                log.warning("Skipping malformed session %s during FTS backfill: %s", thread_id, e)
                continue
            indexed += self._index_to_swarm_fts(thread_id, messages)

        log.info("Session FTS backfill complete: %d new messages indexed", indexed)
        return indexed

    async def clear(self, thread_id: str) -> int:
        """Clear conversation history for a thread. Returns rows deleted."""
        async with self._open_db() as db:
            await self._ensure_table(db)
            # Clear new sessions table
            cursor = await db.execute(
                "DELETE FROM sessions WHERE thread_id = ?",
                (thread_id,),
            )
            deleted = cursor.rowcount

            # Also clear legacy LangGraph checkpoint tables if they exist
            for table in ("checkpoints", "writes"):
                try:
                    cursor = await db.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?",
                        (thread_id,),
                    )
                    deleted += cursor.rowcount
                except Exception:
                    pass  # table may not exist

            await db.commit()
            return deleted
