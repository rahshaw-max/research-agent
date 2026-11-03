"""Tests for session_store.SessionStore."""
import asyncio
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta

import pytest

from session_store import SessionStore, PERSIST_KEYS


# ── Helpers ─────────────────────────────────────────────────────────────────

def _tmp_store(tmp_path) -> SessionStore:
    return SessionStore(str(tmp_path / "test.db"))


def _make_session(**overrides) -> dict:
    base = {
        "idea": "AI invoice processing",
        "questions": ["Q1?", "Q2?"],
        "answers": {"Q1?": "SMBs", "Q2?": "reduce cost"},
        "depth": "standard",
        "refinements": [],
        "cycle_count": 1,
        "evidence": [{"url": "https://example.com", "title": "Example"}],
        "final_research": "Some research text.",
        "last_gap_report": {"coverage_score": 0.72, "gaps": [], "sub_question_scores": []},
        "prev_coverage_score": None,
        "consecutive_small_improvements": 0,
        "coverage_history": [],
        "agent_outputs": {},
        "pptx_path": None,
        "phase": "clarifying",
        "interrogator_questions": [],
        "vcounts": {"total": 10, "supported": 8, "partial": 1, "unsupported": 1, "repaired": 1, "removed": 0},
        "recommendation_reason": "first cycle",
        "auto_approve_recommended": False,
        "per_sub_question_scores": [],
        "awaiting_review_data": None,
    }
    base.update(overrides)
    return base


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_and_get_roundtrip(tmp_path):
    store = _tmp_store(tmp_path)
    await store.init()
    session = _make_session(phase="clarifying")
    await store.create("sess-1", session)

    record = await store.get("sess-1")
    assert record is not None
    assert record["status"] == "clarifying"
    state = record["state"]
    assert state["idea"] == "AI invoice processing"
    assert state["questions"] == ["Q1?", "Q2?"]
    assert state["answers"]["Q1?"] == "SMBs"
    assert state["depth"] == "standard"
    assert state["cycle_count"] == 1
    assert state["evidence"][0]["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_save_updates_status(tmp_path):
    store = _tmp_store(tmp_path)
    await store.init()
    session = _make_session(phase="clarifying")
    await store.create("sess-2", session)

    session["phase"] = "awaiting_review"
    session["final_research"] = "Detailed research report."
    session["last_gap_report"] = {"coverage_score": 0.8, "gaps": [], "sub_question_scores": []}
    session["awaiting_review_data"] = {
        "cycle": 1,
        "coverage_score": 0.8,
        "gap_report": {},
        "questions": ["What is the market size?"],
    }
    await store.save("sess-2", session)

    record = await store.get("sess-2")
    assert record["status"] == "awaiting_review"
    state = record["state"]
    assert state["phase"] == "awaiting_review"
    assert state["final_research"] == "Detailed research report."
    assert state["last_gap_report"]["coverage_score"] == 0.8
    assert state["awaiting_review_data"]["cycle"] == 1


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(tmp_path):
    store = _tmp_store(tmp_path)
    await store.init()
    assert await store.get("does-not-exist") is None


@pytest.mark.asyncio
async def test_delete_expired(tmp_path):
    """Sessions older than ttl_hours should be deleted."""
    db_path = str(tmp_path / "test.db")
    store = SessionStore(db_path)
    await store.init()

    session = _make_session()
    await store.create("old-sess", session)
    await store.create("new-sess", session)

    # Backdate old-sess to 100 hours ago
    conn = sqlite3.connect(db_path)
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (old_ts, "old-sess"))
    conn.commit()
    conn.close()

    deleted = await store.delete_expired(ttl_hours=72)
    assert deleted == 1
    assert await store.get("old-sess") is None
    assert await store.get("new-sess") is not None


@pytest.mark.asyncio
async def test_mark_interrupted(tmp_path):
    """Sessions in researching/architecting become interrupted; others are unchanged."""
    store = _tmp_store(tmp_path)
    await store.init()

    for sid, phase in [("r", "researching"), ("a", "architecting"),
                       ("aw", "awaiting_review"), ("d", "done")]:
        s = _make_session(phase=phase)
        await store.create(sid, s)
        await store.save(sid, s)   # save sets status from phase

    count = await store.mark_interrupted()
    assert count == 2

    assert (await store.get("r"))["status"] == "interrupted"
    assert (await store.get("a"))["status"] == "interrupted"
    assert (await store.get("aw"))["status"] == "awaiting_review"
    assert (await store.get("d"))["status"] == "done"


@pytest.mark.asyncio
async def test_awaiting_review_data_survives_roundtrip(tmp_path):
    """The awaiting_review_data payload is preserved across create/save/get."""
    store = _tmp_store(tmp_path)
    await store.init()

    ard = {
        "critique": "",
        "questions": ["How big is the market?"],
        "cycle": 1,
        "gap_report": {"coverage_score": 0.65, "gaps": [{"type": "stale", "severity": 3}]},
        "coverage_score": 0.65,
        "auto_approve_recommended": False,
        "recommendation_reason": "first cycle — score 0.65",
        "per_sub_question_scores": [],
        "verification_counts": {"total": 5, "supported": 4},
    }
    session = _make_session(phase="awaiting_review", awaiting_review_data=ard)
    await store.create("resume-sess", session)
    await store.save("resume-sess", session)

    record = await store.get("resume-sess")
    assert record["status"] == "awaiting_review"
    restored = record["state"]["awaiting_review_data"]
    assert restored["cycle"] == 1
    assert restored["coverage_score"] == 0.65
    assert restored["gap_report"]["gaps"][0]["type"] == "stale"


@pytest.mark.asyncio
async def test_list_recent(tmp_path):
    store = _tmp_store(tmp_path)
    await store.init()

    for i in range(5):
        s = _make_session(phase="done")
        await store.create(f"s{i}", s)

    recent = await store.list_recent(limit=3)
    assert len(recent) == 3
    for r in recent:
        assert "id" in r and "status" in r


@pytest.mark.asyncio
async def test_only_persist_keys_are_saved(tmp_path):
    """Non-PERSIST_KEYS fields (e.g. ws, hitl_queue) must not appear in state_json."""
    store = _tmp_store(tmp_path)
    await store.init()

    session = _make_session()
    session["ws"] = object()
    session["ws_ready"] = object()
    session["hitl_queue"] = object()

    await store.create("ws-sess", session)
    record = await store.get("ws-sess")
    state = record["state"]
    assert "ws" not in state
    assert "ws_ready" not in state
    assert "hitl_queue" not in state
    assert "idea" in state


@pytest.mark.asyncio
async def test_resume_without_llm_calls(tmp_path):
    """Simulates the full resume flow: save awaiting_review state, reload, check data."""
    store = _tmp_store(tmp_path)
    await store.init()

    # Simulate pipeline reaching HITL gate and saving
    ard = {"cycle": 1, "coverage_score": 0.78, "questions": ["Q?"], "gap_report": {}}
    session = _make_session(
        phase="awaiting_review",
        final_research="The research report text.",
        cycle_count=1,
        awaiting_review_data=ard,
    )
    await store.create("resume-test", session)
    await store.save("resume-test", session)

    # Simulate server restart: mark interrupted (only researching/architecting are marked)
    # awaiting_review stays as-is
    interrupted = await store.mark_interrupted()
    assert interrupted == 0  # none should change

    # Load from store (as WS reconnect would do)
    record = await store.get("resume-test")
    assert record["status"] == "awaiting_review"
    state = record["state"]
    assert state["phase"] == "awaiting_review"
    assert state["final_research"] == "The research report text."
    assert state["awaiting_review_data"]["cycle"] == 1
    assert state["awaiting_review_data"]["coverage_score"] == 0.78
