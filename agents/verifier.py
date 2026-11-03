"""
agents/verifier.py — fast claim verification against retrieved evidence.

Flow (all MODEL_FAST, target <30 s):
  1. Extract cited bullets from report
  2. Build evidence map from passed-in evidence list (fallback: parse ## Sources)
  3. Concurrent batch verification — BATCH_SIZE claims/call, asyncio.gather
  4. Single no-tools repair call for unsupported claims (MODEL_FAST, tool-forced)
  5. [Optional] VERIFIER_ALLOW_SEARCH=true: max 2 parallel web searches with 60-s
     hard timeout, only for claims still unresolved after no-tools repair
  6. Patch report: annotate partial, replace/remove unsupported
"""

import asyncio
import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

from agents.config import MODEL_FAST
BATCH_SIZE            = int(os.getenv("VERIFIER_BATCH_SIZE", "12"))
VERIFIER_ALLOW_SEARCH = os.getenv("VERIFIER_ALLOW_SEARCH", "false").lower() == "true"
_MAX_EXTRA_SEARCHES   = 2
_SEARCH_TIMEOUT       = 60.0

client = anthropic.AsyncAnthropic()

_WS_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": _MAX_EXTRA_SEARCHES},
]

# ── Structured-output tools ───────────────────────────────────────────────

_VERIFY_TOOL = {
    "name": "submit_verdicts",
    "description": "Submit verification verdicts for a batch of claims.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index":   {"type": "integer", "description": "1-based claim index in the batch"},
                        "verdict": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                        "reason":  {"type": "string", "description": "One sentence explanation"},
                    },
                    "required": ["index", "verdict", "reason"],
                },
            }
        },
        "required": ["verdicts"],
    },
}

_REPAIR_TOOL = {
    "name": "submit_repairs",
    "description": "Submit rewritten or removed versions of unsupported claims.",
    "input_schema": {
        "type": "object",
        "properties": {
            "repairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "original":    {"type": "string", "description": "Original claim text verbatim"},
                        "replacement": {
                            "type": "string",
                            "description": (
                                "Rewritten claim with [EN] citation if evidence supports a corrected version, "
                                "or the literal string REMOVE if no evidence supports it."
                            ),
                        },
                    },
                    "required": ["original", "replacement"],
                },
            }
        },
        "required": ["repairs"],
    },
}

_VERIFY_SYSTEM = """\
You are a strict fact-checking agent verifying whether cited sources plausibly support specific claims.

Verdicts:
- "supported": the cited source type and title are exactly the right match for this specific claim,
  and the specifics (numbers, names, dates) are consistent with what such a source would report.
- "partial": the source is topically related but the claim overstates, adds specificity the source
  cannot confirm, or mixes source facts with assumptions.
- "unsupported": the source appears unrelated, OR the claim contains invented statistics, names,
  or figures the cited source would not plausibly contain.

When in doubt between "supported" and "partial", choose "partial". Be strict about specific numbers."""

_REPAIR_SYSTEM = """\
You are a research editor with access to an evidence list. For each unsupported claim:
  (a) If an evidence item plausibly supports a corrected version of the claim, rewrite it
      accurately with the matching [EN] citation from the list.
  (b) If no evidence item supports it, set replacement to exactly "REMOVE".

Rules: keep the same bullet prose style; do not invent new facts; do not fetch URLs."""


# ── Report parsing ────────────────────────────────────────────────────────

def _extract_claims(report: str) -> list[dict]:
    """Return [{text, citations:[int], has_citation}] for every bullet outside Sources."""
    claims = []
    in_sources = False
    for line in report.splitlines():
        s = line.strip()
        if s.startswith("## Sources"):
            in_sources = True
            continue
        if in_sources and s.startswith("## "):
            in_sources = False
        if in_sources:
            continue
        if not (s.startswith("- ") or s.startswith("* ")):
            continue
        content = s.lstrip("-* ").strip()
        if not content:
            continue
        cites = [int(c) for c in re.findall(r"\[E?(\d+)\]", content)]
        claims.append({"text": content, "citations": cites, "has_citation": bool(cites)})
    return claims


def _build_evidence_map(evidence: list[dict] | None, report: str) -> dict[int, dict]:
    """
    Build {int_id: {title, url}} from the passed-in evidence list.
    Falls back to parsing ## Sources if no evidence list is provided.
    """
    ev_map: dict[int, dict] = {}
    if evidence:
        for e in evidence:
            raw_id = e.get("id", "")
            m = re.match(r"E?(\d+)$", str(raw_id))
            if m:
                ev_map[int(m.group(1))] = {
                    "title": e.get("title", ""),
                    "url":   e.get("url",   ""),
                }
        if ev_map:
            return ev_map

    # Fallback: parse ## Sources section
    in_sources = False
    for line in report.splitlines():
        s = line.strip()
        if s.startswith("## Sources"):
            in_sources = True
            continue
        if in_sources:
            if s.startswith("## "):
                break
            m = re.match(r"\[E?(\d+)\]\s+(.+?)\s+[—–\-]+\s+(https?://\S+)", s)
            if m:
                n = int(m.group(1))
                ev_map[n] = {
                    "title": m.group(2).strip(),
                    "url":   m.group(3).strip().rstrip("),. "),
                }
    return ev_map


# ── Concurrent batch verification ─────────────────────────────────────────

async def _verify_batch(
    batch: list[dict],
    ev_map: dict[int, dict],
    _usage: dict | None = None,
) -> list[dict]:
    """One MODEL_FAST call verifying a batch. Returns [{claim_text, verdict, reason}]."""
    lines = []
    for i, item in enumerate(batch, 1):
        src_parts = []
        for cn in item["citations"]:
            ev = ev_map.get(cn)
            if ev:
                src_parts.append(f"[E{cn}] {ev['title']} — {ev['url']}")
            else:
                src_parts.append(f"[E{cn}] (not in evidence list)")
        lines.append(
            f"Claim {i}:\n"
            f"  Text: \"{item['text']}\"\n"
            f"  Cites: {'; '.join(src_parts) or '(none)'}"
        )

    response = await client.messages.create(
        model=MODEL_FAST,
        max_tokens=1024,
        system=[{"type": "text", "text": _VERIFY_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "\n\n".join(lines) + "\n\nVerify each claim."}],
        tools=[_VERIFY_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdicts"},
    )
    if _usage is not None:
        _usage["input_tokens"]  = _usage.get("input_tokens",  0) + response.usage.input_tokens
        _usage["output_tokens"] = _usage.get("output_tokens", 0) + response.usage.output_tokens

    results = []
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_verdicts":
            for v in block.input.get("verdicts", []):
                idx = v.get("index", 1) - 1
                if 0 <= idx < len(batch):
                    results.append({
                        "claim_text": batch[idx]["text"],
                        "verdict":    v.get("verdict", "unsupported"),
                        "reason":     v.get("reason",  ""),
                    })
    return results


# ── Fast no-tools repair ──────────────────────────────────────────────────

async def _repair_fast(
    unsupported: list[dict],
    evidence: list[dict],
    _usage: dict | None = None,
) -> dict[str, str]:
    """
    Single MODEL_FAST call (no web searches).
    Returns {original_text: replacement_text_or_REMOVE}.
    """
    if not unsupported:
        return {}

    ev_lines = "\n".join(
        f"[{e['id']}] {e['title']} — {e['url']}"
        for e in evidence
    )
    claims_text = "\n".join(
        f"{i + 1}. {c['claim_text']}" for i, c in enumerate(unsupported)
    )
    user_msg_blocks = [
        {"type": "text", "text": f"Evidence list:\n{ev_lines}", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"Unsupported claims to repair:\n{claims_text}\n\nCall submit_repairs with one entry per claim."},
    ]

    response = await client.messages.create(
        model=MODEL_FAST,
        max_tokens=4096,
        system=[{"type": "text", "text": _REPAIR_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg_blocks}],
        tools=[_REPAIR_TOOL],
        tool_choice={"type": "tool", "name": "submit_repairs"},
    )
    if _usage is not None:
        _usage["input_tokens"]  = _usage.get("input_tokens",  0) + response.usage.input_tokens
        _usage["output_tokens"] = _usage.get("output_tokens", 0) + response.usage.output_tokens

    repairs_map: dict[str, str] = {}
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_repairs":
            for r in block.input.get("repairs", []):
                orig = r.get("original", "")
                repl = r.get("replacement", "REMOVE")
                if orig:
                    repairs_map[orig] = repl
    return repairs_map


# ── Optional targeted search (VERIFIER_ALLOW_SEARCH=true only) ───────────

async def _search_one(query: str, _usage: dict | None = None) -> list[dict]:
    """One web_search call; returns [{title, url}]. Hard 60-second timeout."""
    try:
        async with asyncio.timeout(_SEARCH_TIMEOUT):
            async with client.messages.stream(
                model=MODEL_FAST,
                max_tokens=256,
                system="Search for the query and stop.",
                messages=[{"role": "user", "content": f"Search: {query}"}],
                tools=_WS_TOOLS,
            ) as stream:
                async for _ in stream:
                    pass
                final = await stream.get_final_message()
                if _usage is not None:
                    _usage["input_tokens"]  = _usage.get("input_tokens",  0) + final.usage.input_tokens
                    _usage["output_tokens"] = _usage.get("output_tokens", 0) + final.usage.output_tokens
                    _usage["tool_calls"]    = _usage.get("tool_calls",    0) + 1
                results = []
                for block in final.content:
                    if getattr(block, "type", None) == "web_search_tool_result":
                        for item in getattr(block, "content", []):
                            if getattr(item, "type", None) == "web_search_result":
                                results.append({
                                    "title": getattr(item, "title", ""),
                                    "url":   getattr(item, "url",   ""),
                                })
                return results
    except (asyncio.TimeoutError, anthropic.APIError):
        return []


async def _targeted_searches(
    still_unresolved: list[dict],
    _usage: dict | None = None,
) -> list[dict]:
    """
    Run up to _MAX_EXTRA_SEARCHES parallel web searches for the most critical
    unresolved claims. Returns new evidence items for the next researcher cycle.
    """
    if not still_unresolved:
        return []
    to_search = still_unresolved[:_MAX_EXTRA_SEARCHES]
    queries = [f"verify: {c['claim_text'][:120]}" for c in to_search]
    results_lists = await asyncio.gather(
        *[_search_one(q, _usage) for q in queries],
        return_exceptions=True,
    )
    new_ev = []
    for r in results_lists:
        if isinstance(r, list):
            new_ev.extend(r)
    return new_ev


# ── Patch the report ──────────────────────────────────────────────────────

def _patch_report(
    report: str,
    partial_texts: set[str],
    repairs_map: dict[str, str],
) -> str:
    result_lines = []
    in_sources = False

    for line in report.splitlines():
        s = line.strip()
        if s.startswith("## Sources"):
            in_sources = True
        if in_sources:
            result_lines.append(line)
            continue

        is_bullet = s.startswith("- ") or s.startswith("* ")
        if not is_bullet:
            result_lines.append(line)
            continue

        indent_len    = len(line) - len(line.lstrip())
        bullet_prefix = line[indent_len: indent_len + 2]
        content       = s.lstrip("-* ").strip()

        if content in repairs_map:
            replacement = repairs_map[content]
            if replacement == "REMOVE":
                continue
            result_lines.append(" " * indent_len + bullet_prefix + replacement)
            continue

        if content in partial_texts and "(partially supported)" not in content:
            annotated = re.sub(r"(\[E?\d+\])", r"(partially supported) \1", content, count=1)
            result_lines.append(" " * indent_len + bullet_prefix + annotated)
            continue

        result_lines.append(line)

    return "\n".join(result_lines)


# ── Public API ────────────────────────────────────────────────────────────

async def run(
    report: str,
    idea: str,
    answers: dict,
    evidence: list[dict] | None = None,
    _usage: dict | None = None,
) -> dict:
    """
    Verify all cited claims and repair unsupported ones — no web searches by default.

    Args:
      report:   Research report markdown text.
      evidence: Evidence list from researcher [{id, title, url, page_age}].
                Falls back to parsing ## Sources if omitted.

    Returns:
      {
        "report": str,           # patched report
        "counts": {total, supported, partial, unsupported, repaired, removed},
      }
    """
    claims = _extract_claims(report)
    ev_map = _build_evidence_map(evidence, report)

    if not claims:
        return {
            "report": report,
            "counts": {
                "total": 0, "supported": 0, "partial": 0,
                "unsupported": 0, "repaired": 0, "removed": 0,
            },
        }

    # ── Auto-fail uncited; batch-verify cited concurrently ────────────────
    uncited = [c for c in claims if not c["has_citation"]]
    cited   = [c for c in claims if c["has_citation"]]

    auto_unsupported = [
        {"claim_text": c["text"], "verdict": "unsupported", "reason": "no inline citation"}
        for c in uncited
    ]
    all_results: list[dict] = list(auto_unsupported)

    if cited:
        batches      = [cited[i: i + BATCH_SIZE] for i in range(0, len(cited), BATCH_SIZE)]
        batch_results = await asyncio.gather(
            *[_verify_batch(b, ev_map, _usage) for b in batches],
            return_exceptions=True,
        )
        for r in batch_results:
            if isinstance(r, list):
                all_results.extend(r)

    supported   = [r for r in all_results if r["verdict"] == "supported"]
    partial     = [r for r in all_results if r["verdict"] == "partial"]
    unsupported = [r for r in all_results if r["verdict"] == "unsupported"]

    # ── Fast no-tools repair ──────────────────────────────────────────────
    ev_list     = evidence or []
    repairs_map = await _repair_fast(unsupported, ev_list, _usage)

    # ── Optional targeted search (default: off) ───────────────────────────
    if VERIFIER_ALLOW_SEARCH:
        still_unresolved = [
            c for c in unsupported
            if repairs_map.get(c["claim_text"], "REMOVE") == "REMOVE"
        ]
        # New evidence returned here is available for the caller to surface
        # to the next researcher cycle via gap_queries.
        await _targeted_searches(still_unresolved, _usage)

    repaired = sum(1 for v in repairs_map.values() if v != "REMOVE")
    removed  = sum(1 for v in repairs_map.values() if v == "REMOVE")

    partial_texts = {r["claim_text"] for r in partial}
    patched = _patch_report(report, partial_texts, repairs_map)

    return {
        "report": patched,
        "counts": {
            "total":       len(all_results),
            "supported":   len(supported),
            "partial":     len(partial),
            "unsupported": len(unsupported),
            "repaired":    repaired,
            "removed":     removed,
        },
    }
