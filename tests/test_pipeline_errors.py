"""
Tests for pipeline error handling.

Cases:
  (a) sub-researcher timeout then success on retry
  (b) all sub-researchers fail → StageFailed raised, critique NOT called
  (c) partial failure → proceeds with warning + thin_coverage gaps emitted
  (d) _validate rejects empty / too-short output
  (e) retry_stage action re-runs only the failed stage
"""

import asyncio
import pytest
import anthropic

from unittest.mock import AsyncMock, MagicMock, patch, call
from agents.errors import StageFailed
from pipeline.orchestrator import _validate


# ── (d) _validate helper ──────────────────────────────────────────────────────

def test_validate_rejects_none():
    with pytest.raises(StageFailed) as exc_info:
        _validate("researcher", None)
    assert exc_info.value.stage == "researcher"
    assert "None" in exc_info.value.reason


def test_validate_rejects_empty_string():
    with pytest.raises(StageFailed):
        _validate("researcher", "")


def test_validate_rejects_whitespace_string():
    with pytest.raises(StageFailed):
        _validate("researcher", "   \n  ")


def test_validate_rejects_too_short():
    with pytest.raises(StageFailed) as exc_info:
        _validate("architect_v1", "short", min_chars=200)
    assert "too short" in exc_info.value.reason


def test_validate_accepts_long_enough():
    _validate("researcher", "x" * 200, min_chars=200)  # no exception


def test_validate_rejects_empty_dict():
    with pytest.raises(StageFailed):
        _validate("auditor", {})


def test_validate_rejects_empty_list():
    with pytest.raises(StageFailed):
        _validate("auditor", [])


def test_validate_accepts_non_empty_dict():
    _validate("auditor", {"title": "foo"})  # no exception


# ── (a) sub-researcher timeout then success on retry ──────────────────────────

@pytest.mark.asyncio
async def test_subquestion_retries_on_timeout_then_succeeds():
    """_research_subquestion should retry on APITimeoutError and succeed on second attempt."""
    from agents import researcher

    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise anthropic.APITimeoutError(request=MagicMock())
        # Second call succeeds with a minimal response
        msg = MagicMock()
        msg.stop_reason = "end_turn"
        msg.usage = MagicMock(input_tokens=10, output_tokens=20,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0)
        msg.content = []
        return msg

    sem = asyncio.Semaphore(1)
    with patch.object(researcher.client.messages, "create", side_effect=fake_create), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        result = await researcher._research_subquestion(
            "What is the market size?",
            queries=["market size query"],
            semaphore=sem,
        )

    assert not result.get("failed"), f"Expected success, got: {result}"
    assert call_count == 2  # one failure + one success


# ── (b) all sub-researchers fail → StageFailed, critique NOT called ───────────

@pytest.mark.asyncio
async def test_all_subquestions_fail_raises_stage_failed():
    """When every sub-question fails, researcher.run() must raise StageFailed."""
    from agents import researcher

    # A plan with 2 sub-questions
    fake_plan = [
        {"subquestion": "Market size?", "queries": ["market size fintech"]},
        {"subquestion": "Competitors?",  "queries": ["fintech competitors"]},
    ]

    failed_result = {
        "subquestion": "x", "raw_evidence": [], "summary": "",
        "failed": True, "error": "APIConnectionError(status=None)",
    }

    answers = {"0": "test answer"}

    with patch.object(researcher, "_plan_research", new=AsyncMock(return_value=fake_plan)), \
         patch.object(researcher, "_research_subquestion", new=AsyncMock(return_value=failed_result)), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(StageFailed) as exc_info:
            async for _ in researcher.run("Build a fintech app", answers):
                pass

    assert exc_info.value.stage == "researcher"
    assert "failed" in exc_info.value.reason.lower() or "source" in exc_info.value.reason.lower()


# ── (c) partial failure → proceeds with warning + thin_coverage gaps ──────────

@pytest.mark.asyncio
async def test_partial_failure_emits_thin_coverage_gaps():
    """When some sub-questions fail but ≥50% succeed and MIN_EVIDENCE met,
    run() should yield a thin_coverage_gaps chunk instead of raising."""
    from agents import researcher

    # We'll mock the parallel _research_subquestion calls directly
    good_result = {
        "subquestion": "Q1",
        "raw_evidence": [{"url": f"http://example.com/{i}", "title": f"Source {i}"}
                         for i in range(10)],
        "summary": "Good research content " * 20,
        "failed": False,
    }
    bad_result = {
        "subquestion": "Q2",
        "raw_evidence": [],
        "summary": "",
        "failed": True,
        "error": "APITimeoutError(status=None)",
    }

    async def fake_subq(subquestion, **kwargs):
        if "Q1" in subquestion or subquestion == list(kwargs.get("_all", []))[0]:
            return good_result
        return bad_result

    # Patch _research_subquestion to return mixed results
    # We need at least 50% success rate and >= MIN_EVIDENCE unique sources
    with patch.object(researcher, "_research_subquestion",
                      side_effect=[good_result] * 5 + [bad_result] * 1):
        # Also patch _synthesize_research so it doesn't actually call the API
        async def fake_synthesis(*a, **kw):
            yield "Synthesized research content " * 50

        with patch.object(researcher, "_stream_synthesis", return_value=fake_synthesis()):
            chunks = []
            async for chunk in researcher.run("Build a fintech app", {"0": "SMB owners"}):
                chunks.append(chunk)

    thin_gaps = [c for c in chunks if isinstance(c, dict) and c.get("__type__") == "thin_coverage_gaps"]
    assert thin_gaps, "Expected thin_coverage_gaps chunk when some sub-questions fail"
    assert len(thin_gaps[0]["gaps"]) >= 1


# ── (e) retry_stage re-runs only the failed stage ────────────────────────────

@pytest.mark.asyncio
async def test_retry_stage_reruns_stage():
    """_await_retry_or_abort should loop the caller back when action == 'retry_stage'."""
    import asyncio
    from pipeline.orchestrator import _await_retry_or_abort

    exc = StageFailed("researcher", "only 3 unique sources gathered (need 8)")

    ws = AsyncMock()
    sessions = {"sid": {"hitl_queue": asyncio.Queue()}}
    await sessions["sid"]["hitl_queue"].put({"action": "retry_stage"})

    with patch("pipeline.orchestrator.log_error"):
        # Should return without raising (caller can loop)
        await _await_retry_or_abort(exc, "sid", sessions, ws, "runs/sid/errors.log")

    # stage_failed event was sent
    ws.send_json.assert_called_once()
    call_data = ws.send_json.call_args[0][0]
    assert call_data["event"] == "stage_failed"
    assert call_data["data"]["stage"] == "researcher"


@pytest.mark.asyncio
async def test_abort_reraises_exception():
    """_await_retry_or_abort should re-raise when action == 'abort'."""
    import asyncio
    from pipeline.orchestrator import _await_retry_or_abort

    exc = StageFailed("researcher", "critical failure")

    ws = AsyncMock()
    sessions = {"sid": {"hitl_queue": asyncio.Queue()}}
    await sessions["sid"]["hitl_queue"].put({"action": "abort"})

    with patch("pipeline.orchestrator.log_error"):
        with pytest.raises(StageFailed) as exc_info:
            await _await_retry_or_abort(exc, "sid", sessions, ws, "runs/sid/errors.log")

    assert exc_info.value is exc
