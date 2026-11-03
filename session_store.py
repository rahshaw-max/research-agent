"""SQLite-backed session store (stdlib sqlite3 + asyncio.to_thread)."""
import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone

# Fields that are JSON-serializable and worth persisting across restarts.
PERSIST_KEYS = [
    "idea", "questions", "answers", "depth", "refinements", "cycle_count",
    "evidence", "final_research", "last_gap_report", "prev_coverage_score",
    "consecutive_small_improvements", "coverage_history", "agent_outputs",
    "pptx_path", "phase", "interrogator_questions", "vcounts",
    "recommendation_reason", "auto_approve_recommended", "per_sub_question_scores",
    "awaiting_review_data",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot(state: dict) -> dict:
    return {k: state.get(k) for k in PERSIST_KEYS}


class SessionStore:
    """Thread-safe SQLite session store. Call await store.init() before use."""

    def __init__(self, db_path: str = "sessions.db"):
        self._db_path = db_path
        self._lock = asyncio.Lock()

    # ── internal sync helpers (run via asyncio.to_thread) ──────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sync(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'clarifying',
                state_json  TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.commit()
        conn.close()

    def _create_sync(self, session_id: str, state: dict) -> None:
        now = _now()
        conn = self._connect()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at, updated_at, status, state_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, now, now, state.get("phase", "clarifying"),
             json.dumps(_snapshot(state))),
        )
        conn.commit()
        conn.close()

    def _save_sync(self, session_id: str, state: dict) -> None:
        now = _now()
        conn = self._connect()
        conn.execute(
            "UPDATE sessions SET updated_at=?, status=?, state_json=? WHERE id=?",
            (now, state.get("phase", "researching"), json.dumps(_snapshot(state)), session_id),
        )
        conn.commit()
        conn.close()

    def _get_sync(self, session_id: str) -> dict | None:
        conn = self._connect()
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        conn.close()
        if row is None:
            return None
        d = dict(row)
        d["state"] = json.loads(d["state_json"])
        return d

    def _mark_interrupted_sync(self) -> int:
        conn = self._connect()
        cur = conn.execute(
            "UPDATE sessions SET status='interrupted', updated_at=?"
            " WHERE status IN ('researching', 'architecting')",
            (_now(),),
        )
        count = cur.rowcount
        conn.commit()
        conn.close()
        return count

    def _delete_expired_sync(self, ttl_hours: int) -> int:
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM sessions WHERE updated_at < datetime('now', ?)",
            (f"-{ttl_hours} hours",),
        )
        count = cur.rowcount
        conn.commit()
        conn.close()
        return count

    def _list_recent_sync(self, limit: int) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, created_at, updated_at, status"
            " FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── public async API ────────────────────────────────────────────────────

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    async def create(self, session_id: str, state: dict) -> None:
        async with self._lock:
            await asyncio.to_thread(self._create_sync, session_id, state)

    async def save(self, session_id: str, state: dict) -> None:
        async with self._lock:
            await asyncio.to_thread(self._save_sync, session_id, state)

    async def get(self, session_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_sync, session_id)

    async def mark_interrupted(self) -> int:
        """Mark sessions that were mid-run as interrupted (called on server start)."""
        return await asyncio.to_thread(self._mark_interrupted_sync)

    async def delete_expired(self, ttl_hours: int = 72) -> int:
        """Delete sessions not updated within ttl_hours."""
        async with self._lock:
            return await asyncio.to_thread(self._delete_expired_sync, ttl_hours)

    async def list_recent(self, limit: int = 20) -> list[dict]:
        return await asyncio.to_thread(self._list_recent_sync, limit)
