"""Shared cross-agent ledger — one SQLite file, six concurrent writers.

This is the shared-state substrate that lets the 6 agent processes
coordinate without a pub/sub bus. Two tables:

``swarm_messages``
    Every message observed in a group chat (user, agent, or system). Used
    for cross-agent visibility, debugging, retention analysis, and root-
    message lookup by the router.

``reply_claims``
    Coordination ledger. An agent inserts a ``claimed`` row when it decides
    to reply to a message. Before actually sending, it runs an IMMEDIATE
    transaction to check that it is still the winner (lowest tier, earliest
    eta); if so it flips to ``sent``, otherwise it cancels. This replaces
    the pub/sub bus we considered in Phase 3 of the original plan.

Winner rule: ``ORDER BY tier ASC, eta_ts ASC, agent_name ASC``.

Access goes through :class:`SwarmStore`, a thin facade over the ``SafeDB``
helper (WAL mode, single-connection-with-lock, auto-reconnect). The
underlying file lives at ``settings.swarm_db_path``.

Metrics counters (``addressing_violations``, ``duplicate_replies``) are
written here too — this is the natural spot because every agent sees the
same ledger and can agree on "who did what".
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from kronos.db import get_db

log = logging.getLogger("kronos.swarm")


# Feedback emoji classification
POSITIVE_EMOJI = {"👍", "❤️", "🔥", "🎉", "💯", "⚡", "🏆", "👏", "❤"}
NEGATIVE_EMOJI = {"👎", "💩", "🤮", "😡"}

# Claim states
CLAIM_STATE_CLAIMED = "claimed"
CLAIM_STATE_SENT = "sent"
CLAIM_STATE_CANCELLED = "cancelled"
CLAIM_STATE_EXPIRED = "expired"

# Claims older than this many seconds without transitioning to ``sent`` are
# considered expired (agent crashed / lost power). Lazy cleanup.
CLAIM_EXPIRY_SECONDS = 120

# Retention for swarm_messages (used by a cron job, not enforced here).
MESSAGE_RETENTION_DAYS = 90

# Hard cap on non-explicit substantive replies to one root user message.
# Applies to Tier 2 and Tier 3; Tier 1 is exempt (explicit addressing wins).
DEFAULT_MAX_IMPLICIT_REPLIES = 2


def _schema(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS swarm_messages (
            chat_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL DEFAULT 0,
            msg_id INTEGER NOT NULL,
            reply_to_msg_id INTEGER,
            sender_id INTEGER NOT NULL,
            sender_type TEXT NOT NULL CHECK (sender_type IN ('user', 'agent', 'system')),
            agent_name TEXT,
            text TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (chat_id, topic_id, msg_id)
        );
        CREATE INDEX IF NOT EXISTS idx_swarm_messages_recent
            ON swarm_messages(chat_id, topic_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_swarm_messages_replies
            ON swarm_messages(chat_id, topic_id, reply_to_msg_id);
        CREATE INDEX IF NOT EXISTS idx_swarm_messages_agent
            ON swarm_messages(agent_name, created_at DESC);

        CREATE TABLE IF NOT EXISTS reply_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL DEFAULT 0,
            root_msg_id INTEGER NOT NULL,
            trigger_msg_id INTEGER NOT NULL,
            agent_name TEXT NOT NULL,
            tier INTEGER NOT NULL,
            eta_ts REAL NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('claimed','sent','cancelled','expired')),
            reason TEXT,
            reply_msg_id INTEGER,
            created_at REAL NOT NULL,
            UNIQUE (chat_id, topic_id, trigger_msg_id, agent_name)
        );
        CREATE INDEX IF NOT EXISTS idx_reply_claims_active
            ON reply_claims(chat_id, topic_id, root_msg_id, state);
        CREATE INDEX IF NOT EXISTS idx_reply_claims_winner
            ON reply_claims(chat_id, topic_id, root_msg_id, tier, eta_ts, agent_name);

        CREATE TABLE IF NOT EXISTS swarm_metrics (
            metric TEXT PRIMARY KEY,
            value INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL
        );

        -- Shared user facts: one view of the user for all agents.
        -- Classification rule (v1 heuristic): facts derived from USER messages
        -- land here; facts derived from the agent's own reflections stay in
        -- the per-agent Mem0 collection. No LLM classifier in v1.
        CREATE TABLE IF NOT EXISTS shared_user_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            source_agent TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed_at REAL NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE (user_id, fact)
        );
        CREATE INDEX IF NOT EXISTS idx_shared_user_facts_user
            ON shared_user_facts(user_id, last_accessed_at DESC);

        -- FTS5 keyword index over facts. Uses external content to stay in
        -- sync via triggers; falls back to raw storage if FTS5 unavailable.
        CREATE VIRTUAL TABLE IF NOT EXISTS shared_user_facts_fts
            USING fts5(fact, content='shared_user_facts', content_rowid='id');

        CREATE TRIGGER IF NOT EXISTS shared_user_facts_ai
            AFTER INSERT ON shared_user_facts BEGIN
                INSERT INTO shared_user_facts_fts(rowid, fact)
                VALUES (new.id, new.fact);
            END;
        CREATE TRIGGER IF NOT EXISTS shared_user_facts_ad
            AFTER DELETE ON shared_user_facts BEGIN
                INSERT INTO shared_user_facts_fts(shared_user_facts_fts, rowid, fact)
                VALUES ('delete', old.id, old.fact);
            END;
        CREATE TRIGGER IF NOT EXISTS shared_user_facts_au
            AFTER UPDATE ON shared_user_facts BEGIN
                INSERT INTO shared_user_facts_fts(shared_user_facts_fts, rowid, fact)
                VALUES ('delete', old.id, old.fact);
                INSERT INTO shared_user_facts_fts(rowid, fact)
                VALUES (new.id, new.fact);
            END;

        -- Session messages: cross-agent FTS5 search over conversation history.
        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            fingerprint TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_session_messages_thread
            ON session_messages(agent_name, thread_id, created_at);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_session_messages_fingerprint
            ON session_messages(fingerprint)
            WHERE fingerprint IS NOT NULL;

        CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts
            USING fts5(content, tokenize='unicode61', content='session_messages', content_rowid='id');

        CREATE TRIGGER IF NOT EXISTS session_messages_fts_ai
            AFTER INSERT ON session_messages BEGIN
                INSERT INTO session_messages_fts(rowid, content)
                VALUES (new.id, new.content);
            END;
        CREATE TRIGGER IF NOT EXISTS session_messages_fts_ad
            AFTER DELETE ON session_messages BEGIN
                INSERT INTO session_messages_fts(session_messages_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;

        -- Feedback: Telegram reactions → RL signal for self-improvement.
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            chat_id INTEGER NOT NULL,
            msg_id INTEGER NOT NULL,
            reaction TEXT NOT NULL,
            emoji TEXT NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(chat_id, msg_id, agent_name)
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_agent
            ON feedback(agent_name, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_feedback_reaction
            ON feedback(reaction, created_at DESC);
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(session_messages)").fetchall()
    }
    if "fingerprint" not in columns:
        conn.execute("ALTER TABLE session_messages ADD COLUMN fingerprint TEXT")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_session_messages_fingerprint
                ON session_messages(fingerprint)
                WHERE fingerprint IS NOT NULL
            """
        )


@dataclass
class ClaimOutcome:
    """Result of attempting to claim / check a reply slot."""

    won: bool
    reason: str = ""


class SwarmStore:
    """Facade over the shared swarm ledger."""

    def __init__(self):
        self._db = get_db("swarm")
        self._db.init_schema(_schema)

    # ------------------------------------------------------------------
    # swarm_messages
    # ------------------------------------------------------------------

    def record_inbound_message(
        self,
        *,
        chat_id: int,
        topic_id: int | None,
        msg_id: int,
        reply_to_msg_id: int | None,
        sender_id: int,
        sender_type: str,
        agent_name: str | None,
        text: str,
    ) -> None:
        """Record any observed message. Idempotent per PRIMARY KEY."""
        self._db.write(
            """
            INSERT OR IGNORE INTO swarm_messages
                (chat_id, topic_id, msg_id, reply_to_msg_id,
                 sender_id, sender_type, agent_name, text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                topic_id or 0,
                msg_id,
                reply_to_msg_id,
                sender_id,
                sender_type,
                agent_name,
                text,
                time.time(),
            ),
        )

    def record_outbound_message(
        self,
        *,
        chat_id: int,
        topic_id: int | None,
        msg_id: int,
        reply_to_msg_id: int | None,
        agent_name: str,
        text: str,
    ) -> None:
        """Record a message this agent just sent. Stable via INSERT OR IGNORE."""
        # We fabricate a sender_id of -1 for agent rows we posted ourselves
        # because Telethon only yields the proper bot sender_id on next poll.
        self._db.write(
            """
            INSERT OR IGNORE INTO swarm_messages
                (chat_id, topic_id, msg_id, reply_to_msg_id,
                 sender_id, sender_type, agent_name, text, created_at)
            VALUES (?, ?, ?, ?, ?, 'agent', ?, ?, ?)
            """,
            (
                chat_id,
                topic_id or 0,
                msg_id,
                reply_to_msg_id,
                -1,
                agent_name,
                text,
                time.time(),
            ),
        )

    def get_recent_messages(
        self,
        *,
        chat_id: int,
        topic_id: int | None,
        limit: int = 20,
    ) -> list[dict]:
        """Return most recent messages in chat/topic, newest first."""
        rows = self._db.read(
            """
            SELECT msg_id, reply_to_msg_id, sender_id, sender_type,
                   agent_name, text, created_at
            FROM swarm_messages
            WHERE chat_id = ? AND topic_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (chat_id, topic_id or 0, limit),
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # reply_claims — coordination
    # ------------------------------------------------------------------

    def claim_reply(
        self,
        *,
        chat_id: int,
        topic_id: int | None,
        root_msg_id: int,
        trigger_msg_id: int,
        agent_name: str,
        tier: int,
        eta_ts: float,
        reason: str = "",
    ) -> None:
        """Insert a claim row. Idempotent per (trigger_msg_id, agent_name)."""
        self._db.write(
            """
            INSERT OR IGNORE INTO reply_claims
                (chat_id, topic_id, root_msg_id, trigger_msg_id,
                 agent_name, tier, eta_ts, state, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'claimed', ?, ?)
            """,
            (
                chat_id,
                topic_id or 0,
                root_msg_id,
                trigger_msg_id,
                agent_name,
                tier,
                eta_ts,
                reason,
                time.time(),
            ),
        )

    def can_send_claim(
        self,
        *,
        chat_id: int,
        topic_id: int | None,
        root_msg_id: int,
        agent_name: str,
        tier: int,
        max_implicit_replies: int = DEFAULT_MAX_IMPLICIT_REPLIES,
    ) -> ClaimOutcome:
        """Atomically decide whether this agent may send now.

        Runs under an IMMEDIATE transaction so that when two agents race to
        check/mark_sent at the same moment, SQLite serialises them and
        exactly one wins.

        Rules:
          * Tier 1 (explicit address) always wins — no cap, no peer check.
          * Otherwise, count ``sent`` replies to this root_msg_id across all
            agents and tiers; reject if already at cap.
          * Otherwise, confirm this agent's claim is still the winner under
            ``ORDER BY tier ASC, eta_ts ASC, agent_name ASC`` among active
            ``claimed`` rows for the same root_msg_id.
        """
        now = time.time()

        def _tx(conn):
            # Lazy-expire stale claims
            conn.execute(
                """
                UPDATE reply_claims
                SET state = 'expired'
                WHERE state = 'claimed'
                  AND (? - created_at) > ?
                """,
                (now, CLAIM_EXPIRY_SECONDS),
            )

            # Tier 1 bypasses arbitration & cap.
            if tier == 1:
                return ClaimOutcome(True, "tier-1 explicit")

            # Anti-flood cap across all agents.
            (sent_count,) = conn.execute(
                """
                SELECT COUNT(*) FROM reply_claims
                WHERE chat_id = ? AND topic_id = ? AND root_msg_id = ?
                  AND state = 'sent' AND tier > 1
                """,
                (chat_id, topic_id or 0, root_msg_id),
            ).fetchone()
            if sent_count >= max_implicit_replies:
                return ClaimOutcome(False, f"cap reached ({sent_count}>={max_implicit_replies})")

            # Winner lookup.
            winner = conn.execute(
                """
                SELECT agent_name FROM reply_claims
                WHERE chat_id = ? AND topic_id = ? AND root_msg_id = ?
                  AND state = 'claimed'
                ORDER BY tier ASC, eta_ts ASC, agent_name ASC
                LIMIT 1
                """,
                (chat_id, topic_id or 0, root_msg_id),
            ).fetchone()
            if winner is None:
                return ClaimOutcome(False, "no active claim")
            if winner[0] != agent_name:
                return ClaimOutcome(False, f"lost to {winner[0]}")
            return ClaimOutcome(True, "winner")

        return self._db.write_tx(_tx)

    def mark_sent(
        self,
        *,
        chat_id: int,
        topic_id: int | None,
        trigger_msg_id: int,
        agent_name: str,
        reply_msg_id: int | None,
    ) -> None:
        self._db.write(
            """
            UPDATE reply_claims
            SET state = 'sent', reply_msg_id = ?
            WHERE chat_id = ? AND topic_id = ?
              AND trigger_msg_id = ? AND agent_name = ?
            """,
            (reply_msg_id, chat_id, topic_id or 0, trigger_msg_id, agent_name),
        )

    def cancel_claim(
        self,
        *,
        chat_id: int,
        topic_id: int | None,
        trigger_msg_id: int,
        agent_name: str,
        reason: str = "",
    ) -> None:
        self._db.write(
            """
            UPDATE reply_claims
            SET state = 'cancelled', reason = COALESCE(NULLIF(?, ''), reason)
            WHERE chat_id = ? AND topic_id = ?
              AND trigger_msg_id = ? AND agent_name = ?
              AND state = 'claimed'
            """,
            (reason, chat_id, topic_id or 0, trigger_msg_id, agent_name),
        )

    def count_sent_replies(
        self,
        *,
        chat_id: int,
        topic_id: int | None,
        root_msg_id: int,
    ) -> int:
        row = self._db.read_one(
            """
            SELECT COUNT(*) AS c FROM reply_claims
            WHERE chat_id = ? AND topic_id = ? AND root_msg_id = ?
              AND state = 'sent'
            """,
            (chat_id, topic_id or 0, root_msg_id),
        )
        return int(row["c"]) if row else 0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def incr_metric(self, metric: str, delta: int = 1) -> None:
        self._db.write(
            """
            INSERT INTO swarm_metrics (metric, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(metric) DO UPDATE
                SET value = value + excluded.value,
                    updated_at = excluded.updated_at
            """,
            (metric, delta, time.time()),
        )

    def get_metrics(self) -> dict[str, int]:
        rows = self._db.read("SELECT metric, value FROM swarm_metrics")
        return {r["metric"]: int(r["value"]) for r in rows}

    # ------------------------------------------------------------------
    # Shared user facts — cross-agent view of the user
    # ------------------------------------------------------------------

    def add_shared_fact(
        self,
        *,
        user_id: str,
        fact: str,
        source_agent: str,
    ) -> bool:
        """Insert a user-derived fact. Returns True if new, False if duplicate."""
        fact = fact.strip()
        if not fact:
            return False
        now = time.time()
        cursor = self._db.write(
            """
            INSERT OR IGNORE INTO shared_user_facts
                (user_id, fact, source_agent, created_at, last_accessed_at, access_count)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (user_id, fact, source_agent, now, now),
        )
        return bool(cursor and cursor.rowcount)

    def search_shared_facts(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[str]:
        """FTS5 keyword search over shared facts for a user.

        Falls back to a plain LIKE match if FTS5 is unavailable or the
        query contains FTS5 special characters that we cannot safely
        escape for the MATCH operator.
        """
        query = query.strip()
        if not query:
            return []
        safe_query = " ".join(
            f'"{tok}"' for tok in query.split() if tok.strip()
        )
        if not safe_query:
            return []
        try:
            rows = self._db.read(
                """
                SELECT f.id, f.fact
                FROM shared_user_facts_fts fts
                JOIN shared_user_facts f ON f.id = fts.rowid
                WHERE fts.fact MATCH ?
                  AND f.user_id = ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, user_id, limit),
            )
        except Exception as e:
            log.warning("Shared facts FTS5 search failed, falling back: %s", e)
            like = f"%{query}%"
            rows = self._db.read(
                """
                SELECT id, fact FROM shared_user_facts
                WHERE user_id = ? AND fact LIKE ?
                ORDER BY last_accessed_at DESC
                LIMIT ?
                """,
                (user_id, like, limit),
            )

        if not rows:
            return []
        ids = tuple(int(r["id"]) for r in rows)
        # Touch accessed facts (read + recency bump) in one transaction.
        placeholders = ",".join("?" * len(ids))
        self._db.write(
            f"""
            UPDATE shared_user_facts
            SET access_count = access_count + 1,
                last_accessed_at = ?
            WHERE id IN ({placeholders})
            """,
            (time.time(), *ids),
        )
        return [r["fact"] for r in rows]

    def all_shared_facts(self, *, user_id: str, limit: int = 100) -> list[str]:
        rows = self._db.read(
            """
            SELECT fact FROM shared_user_facts
            WHERE user_id = ?
            ORDER BY last_accessed_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return [r["fact"] for r in rows]

    # ------------------------------------------------------------------
    # Session messages — cross-agent FTS5 search
    # ------------------------------------------------------------------

    def index_session_message(
        self,
        *,
        agent_name: str,
        thread_id: str,
        role: str,
        content: str,
        fingerprint: str = "",
    ) -> bool:
        """Index a single message from a session into the cross-agent FTS store.

        ``fingerprint`` is optional for callers that do not have a stable
        per-session position. When absent, we derive a content-based key to
        keep repeated backfills from duplicating rows.
        """
        clean_content = content.strip()
        if not clean_content:
            return False
        fingerprint = fingerprint or hashlib.sha256(
            f"{agent_name}\0{thread_id}\0{role}\0{clean_content}".encode()
        ).hexdigest()
        cursor = self._db.write(
            """
            INSERT OR IGNORE INTO session_messages
                (agent_name, thread_id, role, content, created_at, fingerprint)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (agent_name, thread_id, role, clean_content, time.time(), fingerprint),
        )
        return bool(cursor and cursor.rowcount)

    def search_sessions(
        self,
        *,
        query: str,
        agent_name: str = "",
        days: int = 30,
        limit: int = 10,
    ) -> list[dict]:
        """FTS5 search over all session messages.

        Returns list of dicts with keys:
        agent_name, thread_id, role, content, created_at.
        """
        query = query.strip()
        if not query:
            return []
        safe_query = " ".join(f'"{tok}"' for tok in query.split() if tok.strip())
        if not safe_query:
            return []

        cutoff = time.time() - (days * 86400)

        try:
            if agent_name:
                rows = self._db.read(
                    """
                    SELECT sm.agent_name, sm.thread_id, sm.role,
                           sm.content, sm.created_at
                    FROM session_messages_fts fts
                    JOIN session_messages sm ON sm.id = fts.rowid
                    WHERE fts.content MATCH ?
                      AND sm.agent_name = ?
                      AND sm.created_at > ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, agent_name, cutoff, limit),
                )
            else:
                rows = self._db.read(
                    """
                    SELECT sm.agent_name, sm.thread_id, sm.role,
                           sm.content, sm.created_at
                    FROM session_messages_fts fts
                    JOIN session_messages sm ON sm.id = fts.rowid
                    WHERE fts.content MATCH ?
                      AND sm.created_at > ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, cutoff, limit),
                )
        except Exception as e:
            log.warning("Session FTS search failed, falling back to LIKE: %s", e)
            like = f"%{query}%"
            agent_filter = ""
            params: list = [like, cutoff, limit]
            if agent_name:
                agent_filter = "AND agent_name = ?"
                params = [like, agent_name, cutoff, limit]
            rows = self._db.read(
                f"""
                SELECT agent_name, thread_id, role, content, created_at
                FROM session_messages
                WHERE content LIKE ? {agent_filter}
                  AND created_at > ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                tuple(params),
            )

        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Feedback — Telegram reactions as RL signal
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_emoji(emoji: str) -> str:
        """Classify emoji into sentiment."""
        if emoji in POSITIVE_EMOJI:
            return "positive"
        if emoji in NEGATIVE_EMOJI:
            return "negative"
        return "neutral"

    def add_feedback(
        self,
        *,
        agent_name: str,
        chat_id: int,
        msg_id: int,
        emoji: str,
    ) -> bool:
        """Record a reaction as feedback. Returns True if new."""
        reaction = self._classify_emoji(emoji)
        cursor = self._db.write(
            """
            INSERT OR REPLACE INTO feedback
                (agent_name, chat_id, msg_id, reaction, emoji, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (agent_name, chat_id, msg_id, reaction, emoji, time.time()),
        )
        return bool(cursor and cursor.rowcount)

    def get_feedback(
        self,
        *,
        agent_name: str = "",
        reaction: str = "",
        days: int = 30,
        limit: int = 50,
    ) -> list[dict]:
        """Get feedback records with optional filters."""
        cutoff = time.time() - (days * 86400)
        conditions = ["created_at > ?"]
        params: list = [cutoff]

        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        if reaction:
            conditions.append("reaction = ?")
            params.append(reaction)

        where = " AND ".join(conditions)
        params.append(limit)

        rows = self._db.read(
            f"""
            SELECT agent_name, chat_id, msg_id, reaction, emoji, created_at
            FROM feedback
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [dict(r) for r in rows]

    def get_satisfaction_rate(
        self,
        *,
        agent_name: str = "",
        days: int = 7,
    ) -> dict:
        """Calculate satisfaction metrics."""
        cutoff = time.time() - (days * 86400)
        agent_filter = "AND agent_name = ?" if agent_name else ""
        params = (cutoff, agent_name) if agent_name else (cutoff,)

        rows = self._db.read(
            f"""
            SELECT reaction, COUNT(*) as cnt
            FROM feedback
            WHERE created_at > ? {agent_filter}
            GROUP BY reaction
            """,
            params,
        )

        counts = {r["reaction"]: int(r["cnt"]) for r in rows}
        total = sum(counts.values())
        positive = counts.get("positive", 0)
        negative = counts.get("negative", 0)

        rate = (positive / total * 100) if total > 0 else 0.0

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "neutral": counts.get("neutral", 0),
            "satisfaction_rate": round(rate, 1),
            "days": days,
        }

    # ------------------------------------------------------------------
    # Retention (called by a periodic job — not wired in this step)
    # ------------------------------------------------------------------

    def prune_old_messages(self, older_than_days: int = MESSAGE_RETENTION_DAYS) -> int:
        cutoff = time.time() - older_than_days * 86400
        cursor = self._db.write(
            "DELETE FROM swarm_messages WHERE created_at < ?", (cutoff,),
        )
        deleted = cursor.rowcount if cursor is not None else 0
        log.info("Pruned %d swarm_messages older than %d days", deleted, older_than_days)
        return deleted


_singleton: SwarmStore | None = None


def get_swarm() -> SwarmStore:
    """Process-wide singleton so schema init and lock are shared."""
    global _singleton
    if _singleton is None:
        _singleton = SwarmStore()
    return _singleton
