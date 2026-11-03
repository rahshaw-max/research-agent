"""
agents/researcher.py — parallel research pipeline.

Step A  MODEL_FAST  (1 call, no tools) : decompose idea into 5-6 sub-questions × 2-3 queries
Step B  MODEL_FAST  (concurrent tasks) : one sub-researcher per sub-question, semaphore-limited,
                                         each with a small search+fetch budget; real-time status
Step C  MODEL_SYNTHESIS (1 streaming call, no tools) : write the structured report from merged,
                                         deduplicated evidence with [E#] citations

Evidence record shape:
  {id, title, url, page_age, source_type, quote, retrieved_text_excerpt}
  - quote: cited_text from web_search_result_location citations on text blocks (≤150 chars)
  - retrieved_text_excerpt: first 2000 chars of web_fetch body (source.data for text pages)

On refinement cycles (prev_research supplied):
  - Prior evidence is carried via the _evidence list already containing previous items
  - Sub-questions target only gap_queries
  - Synthesis uses SYSTEM_PROMPT_REVISION and receives the prior report

web_fetch max_content_tokens: Anthropic's web_fetch_20250910 tool does not expose a
max_content_tokens parameter. Content is bounded by the model's max_tokens budget
(SUBQ_MAX_TOKENS per sub-researcher call).
"""

import asyncio
import json
import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.AsyncAnthropic()

from agents.config import MODEL_FAST, MODEL_SYNTHESIS
from evals.domains import source_type_of

RESEARCH_CONCURRENCY    = int(os.getenv("RESEARCH_CONCURRENCY",        "4"))
SEARCHES_PER_SUBQ       = int(os.getenv("RESEARCH_SEARCHES_PER_SUBQ",  "3"))
FETCHES_PER_SUBQ        = int(os.getenv("RESEARCH_FETCHES_PER_SUBQ",   "2"))
RESEARCH_TIMEOUT        = float(os.getenv("RESEARCH_TIMEOUT_SECONDS",  "150"))
RESEARCH_EVIDENCE_LIMIT = int(os.getenv("RESEARCH_EVIDENCE_LIMIT",     "40"))
SUBQ_MAX_TOKENS         = 1536   # bounds fetch content processed per sub-researcher turn
_EXCERPT_LIMIT          = 2000   # chars of fetched text to store per source
_SYNTH_QUOTE_LIMIT      = 300    # chars of quote/excerpt shown per evidence item in synthesis

# Source-type priority for synthesis selection (lower = higher priority)
_SOURCE_PRIORITY = {"primary": 0, "analyst": 1, "news": 2, "blog": 3, "other": 4}

# ── System prompts ─────────────────────────────────────────────────────────

_CITATION_RULES = """
CITATION RULES (mandatory):
- You are given an evidence list with IDs [E1], [E2], etc. Each item shows its source text or excerpt.
- Attach an inline citation [EN] to EVERY factual bullet — use the exact ID from the evidence list.
- A claim MUST be supported by text visible in the quoted excerpt for that evidence item.
- A single bullet may carry multiple [EN] citations.
- If a claim cannot be matched to any evidence item's text, write "(unverified)" — do not guess.
- End your entire response with a "## Sources" section listing every cited evidence ID as:
  [E1] Title — URL
  (one line per source, ordered by evidence ID)
"""

SYSTEM_PROMPT = f"""You are an expert research analyst. Research findings and a numbered evidence list have been provided — use them to write the structured report.

{_CITATION_RULES}

Produce a structured research report using EXACTLY this template:

## Market Landscape
- [finding with inline citation] [E1] — 2–4 sentences with evidence and source context
- [finding] [E2]
- [finding] [E3]
- [finding] [E4]
(3–5 bullets)

## Use Cases
- [use case]: [who benefits and how, with specific examples] [EN]
(3–4 use cases)

## Existing Solutions
- **[solution name]**: [what it does, its key differentiator, and its gap] [EN]
(3–4 solutions)

## Key Challenges
- [challenge]: [why it matters, what makes it hard, real-world evidence] [EN]
(3–4 challenges)

## Opportunities
- [opportunity]: [why now / why underserved, with market evidence] [EN]
(2–3 opportunities)

## Sources
[E1] Title — URL
(every source cited above, in evidence ID order)

RULES:
- EVERY bullet MUST end with at least one [EN] citation — if a bullet has no [EN] it will be rejected.
- Each bullet should be 2–4 sentences — include evidence, context, and the cited source.
- Aim for 1500–2500 words total (excluding Sources).
- Do NOT search the web — all research has already been done and is in the provided findings."""

SYSTEM_PROMPT_REVISION = f"""You are an expert research analyst producing a cumulative updated research report. New research findings have been provided to fill specific gaps in the prior report.

{_CITATION_RULES}

REVISION RULES:
- You are given the PREVIOUS RESEARCH REPORT. Retain all findings that carry inline citations [EN].
- You are given NEW FINDINGS from targeted gap research. Incorporate them: add new bullets, expand existing ones, or correct contradicted claims.
- Do NOT remove cited findings unless new evidence from a primary source contradicts them.
- Keep citation numbering from the previous report; new sources have higher IDs.

Produce the COMPLETE updated report using EXACTLY this template:

## Research Updates
- [gap addressed] → [what new evidence was found and where] [EN] — 2–3 sentences

## Market Landscape
- [finding] [EN] — 2–4 sentences with evidence and source context
(3–5 bullets, retaining valid prior findings)

## Use Cases
- [use case]: [who benefits and how, with specific examples] [EN]
(3–4 use cases)

## Existing Solutions
- **[solution name]**: [what it does, its key differentiator, and its gap] [EN]
(3–4 solutions)

## Key Challenges
- [challenge]: [why it matters, what makes it hard, real-world evidence] [EN]
(3–4 challenges)

## Opportunities
- [opportunity]: [why now / why underserved, with market evidence] [EN]
(2–3 opportunities)

## Sources
[E1] Title — URL
(all sources in evidence ID order)

RULES:
- Aim for 1500–2500 words total (excluding Sources).
- Mark anything unmatched as "(unverified)"."""

_PLAN_SYSTEM = """\
You are a research planning assistant. Decompose a research idea into focused sub-questions.

Return ONLY valid JSON (no markdown fences, no explanation) in this exact format:
[
  {
    "subquestion": "Precise, answerable research question",
    "queries": ["search query 1", "search query 2", "search query 3"]
  }
]

Rules:
- 5-6 sub-questions covering: market size/growth, target users and pain points, existing solutions,
  technical/adoption challenges, fraud/risk factors, and market opportunities.
- If gap_queries are provided, include sub-questions that DIRECTLY target those gaps; reduce
  coverage of areas already well-researched in the prior cycle.
- 2-3 search queries per sub-question — specific and non-overlapping.
- No two sub-questions should overlap in scope."""

_SUBQ_SYSTEM = """\
You are a focused research assistant with live web access. Answer ONE specific research question.

Instructions:
1. Run web_search calls using the provided priority queries (add your own if useful).
2. For the most relevant pages, run web_fetch to read the full content.
3. Write a concise summary (150-250 words) of the most important findings.

Style: lead with numbers and source names when possible. Example:
  "According to Market.us, the market is $2.8B in 2024 growing at 32.6% CAGR. Goldman Sachs data
   shows manual invoice costs of $22.26/invoice for small businesses..."

Be specific, factual, and dense with data. Mention source domains or titles for each key fact."""


# ── Evidence helpers ───────────────────────────────────────────────────────

def _extract_evidence_from_blocks(content_blocks) -> dict[str, dict]:
    """
    Extract evidence from all block types in one response turn.
    Returns {url: {title, url, page_age, source_type, quote, retrieved_text_excerpt}}.

    Sources:
    - web_search_tool_result  → title, url, page_age (text is encrypted — not readable)
    - web_fetch_tool_result   → retrieved_text_excerpt from source.data (text pages only)
    - text block citations     → cited_text as quote (web_search_result_location only)
    """
    by_url: dict[str, dict] = {}

    def _ensure(url: str, title: str = "", page_age=None) -> dict:
        if url not in by_url:
            by_url[url] = {
                "title":                 title,
                "url":                   url,
                "page_age":              page_age,
                "source_type":           source_type_of(url),
                "quote":                 "",
                "retrieved_text_excerpt": "",
            }
        return by_url[url]

    for block in content_blocks:
        btype = getattr(block, "type", None)

        # ── web_search results ──────────────────────────────────────────────
        if btype == "web_search_tool_result":
            bc = getattr(block, "content", [])
            if not isinstance(bc, list):
                continue
            for item in bc:
                if getattr(item, "type", None) != "web_search_result":
                    continue
                url = (getattr(item, "url", None) or "").strip()
                if not url:
                    continue
                title    = (getattr(item, "title",    None) or "").strip()
                page_age = getattr(item, "page_age", None)
                rec = _ensure(url, title, page_age)
                # Back-fill title/page_age if the record was created by a citation first
                if not rec["title"] and title:
                    rec["title"] = title
                if not rec["page_age"] and page_age:
                    rec["page_age"] = page_age

        # ── web_fetch results ───────────────────────────────────────────────
        elif btype == "web_fetch_tool_result":
            bc = getattr(block, "content", None)
            if bc is None or getattr(bc, "type", None) != "web_fetch_result":
                continue
            url = (getattr(bc, "url", None) or "").strip()
            if not url:
                continue
            doc    = getattr(bc, "content", None)
            title  = ""
            excerpt = ""
            if doc:
                title = (getattr(doc, "title", None) or "").strip()
                src   = getattr(doc, "source", None)
                # Only read text pages (not base64-encoded PDFs)
                if src and getattr(src, "type", None) == "text":
                    data   = getattr(src, "data", "") or ""
                    excerpt = data[:_EXCERPT_LIMIT]

            rec = _ensure(url, title)
            if excerpt and not rec["retrieved_text_excerpt"]:
                rec["retrieved_text_excerpt"] = excerpt
            if title and not rec["title"]:
                rec["title"] = title

        # ── text blocks → cited_text from web_search_result_location ────────
        elif btype == "text":
            for citation in (getattr(block, "citations", None) or []):
                if getattr(citation, "type", None) != "web_search_result_location":
                    continue
                url        = (getattr(citation, "url",        None) or "").strip()
                cited_text = (getattr(citation, "cited_text", None) or "").strip()
                title      = (getattr(citation, "title",      None) or "").strip()
                if not url or not cited_text:
                    continue
                rec = _ensure(url, title)
                if not rec["quote"]:
                    rec["quote"] = cited_text
                if title and not rec["title"]:
                    rec["title"] = title

    return by_url


def _enrich(base: dict, new: dict) -> None:
    """Merge `new` into `base` for fields that are empty in base."""
    for field in ("quote", "retrieved_text_excerpt", "title", "page_age"):
        if new.get(field) and not base.get(field):
            base[field] = new[field]


def _dedup_assign_ids(raw_items: list[dict], start_n: int = 0) -> list[dict]:
    """Deduplicate by URL and assign sequential E{n} IDs."""
    seen: set[str] = set()
    result: list[dict] = []
    n = start_n
    for item in raw_items:
        url = item.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        n += 1
        result.append({**item, "id": f"E{n}"})
    return result


def _select_for_synthesis(evidence: list[dict], limit: int) -> list[dict]:
    """
    Select up to `limit` evidence items for the synthesis prompt.
    Priority: primary > analyst > news > blog > other.
    Within each tier, items with text (quote or excerpt) come first;
    otherwise original discovery order is preserved (stable sort).
    """
    def _key(e):
        has_text = bool(e.get("quote") or e.get("retrieved_text_excerpt"))
        return (_SOURCE_PRIORITY.get(e.get("source_type", "other"), 4), 0 if has_text else 1)
    return sorted(evidence, key=_key)[:limit]


def _evidence_list_text(evidence: list[dict]) -> str:
    """Format evidence for the synthesis prompt, including quote/excerpt."""
    lines = []
    for e in evidence:
        line = f"[{e['id']}] {e['title']} — {e['url']}"
        if e.get("page_age"):
            line += f" ({e['page_age']})"
        text = e.get("quote") or (e.get("retrieved_text_excerpt", "")[:_SYNTH_QUOTE_LIMIT])
        if text:
            line += f'\n    Excerpt: "{text}"'
        lines.append(line)
    return "\n".join(lines)


# ── Step A: Research planning ──────────────────────────────────────────────

async def _plan_research(
    idea: str,
    answers: dict,
    gap_queries: list[str] | None,
    latest_refinement: dict | None,
    _usage: dict | None = None,
) -> list[dict]:
    """Generate 5-6 sub-questions with 2-3 queries each (MODEL_FAST, no tools)."""
    ctx_parts = []
    if answers:
        ctx_parts.append("Requester context:\n" + "\n".join(f"- {k}: {v}" for k, v in answers.items()))
    if latest_refinement:
        if latest_refinement.get("refined_prompt"):
            ctx_parts.append(f"Refined focus: {latest_refinement['refined_prompt']}")
        if latest_refinement.get("additional_context"):
            ctx_parts.append(f"Additional context: {latest_refinement['additional_context']}")

    if gap_queries:
        gap_block = "Specific gaps to address (priority):\n" + "\n".join(f"- {q}" for q in gap_queries)
        instruction = "Focus sub-questions on filling these gaps while maintaining minimal overall coverage."
    else:
        gap_block = ""
        instruction = "Cover the full landscape: market size, use cases, existing solutions, challenges, opportunities."

    user_msg = "\n\n".join(filter(None, [
        f"Idea: {idea}",
        "\n".join(ctx_parts) if ctx_parts else "",
        gap_block,
        instruction,
        "Generate 5-6 focused sub-questions with 2-3 search queries each.",
    ]))

    response = await client.messages.create(
        model=MODEL_FAST,
        max_tokens=1024,
        system=_PLAN_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    if _usage is not None:
        _usage["input_tokens"]  = _usage.get("input_tokens",  0) + response.usage.input_tokens
        _usage["output_tokens"] = _usage.get("output_tokens", 0) + response.usage.output_tokens

    text = (response.content[0].text if response.content else "[]").strip()
    # Strip markdown fences if model adds them despite instructions
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        plan = json.loads(text)
        if isinstance(plan, list) and plan:
            return plan[:6]
    except Exception:
        pass

    # Fallback: generic plan
    topic = idea[:60]
    return [
        {"subquestion": f"What is the current market size and growth trajectory for {topic}?",
         "queries": [f"{topic} market size 2024 2025", f"{topic} growth statistics CAGR"]},
        {"subquestion": f"Who are the target users and what are their key pain points?",
         "queries": [f"{topic} SMB pain points", f"small business {topic} challenges statistics"]},
        {"subquestion": f"What existing solutions exist and what gaps do they leave?",
         "queries": [f"{topic} software solutions comparison 2024", f"best {topic} tools alternatives"]},
        {"subquestion": f"What are the key technical and adoption challenges?",
         "queries": [f"{topic} implementation challenges", f"{topic} adoption barriers 2025"]},
        {"subquestion": f"What fraud and risk factors are relevant?",
         "queries": [f"{topic} fraud statistics", f"{topic} security risks compliance"]},
        {"subquestion": f"What market opportunities are underserved?",
         "queries": [f"{topic} market opportunity gap", f"{topic} startup opportunity 2025"]},
    ]


# ── Step B: Sub-researcher worker ──────────────────────────────────────────

async def _research_subquestion(
    subquestion: str,
    queries: list[str],
    semaphore: asyncio.Semaphore,
    _usage: dict | None = None,
    searches: int | None = None,
    fetches: int | None = None,
) -> dict:
    """
    Run a mini research loop for one sub-question (MODEL_FAST, server-side tools).
    Returns {subquestion, raw_evidence: [{title, url, page_age, source_type, quote,
                                          retrieved_text_excerpt}], summary: str}.
    """
    async with semaphore:
        tools = [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": searches or SEARCHES_PER_SUBQ},
            {"type": "web_fetch_20250910",  "name": "web_fetch",  "max_uses": fetches  or FETCHES_PER_SUBQ},
        ]
        queries_text = "\n".join(f"- {q}" for q in queries)
        messages: list = [{"role": "user", "content": (
            f"Research question: {subquestion}\n\n"
            f"Priority search queries:\n{queries_text}"
        )}]

        # Dict-keyed by URL for enrichment across turns
        raw_ev_by_url: dict[str, dict] = {}
        full_text = ""

        while True:
            try:
                response = await client.messages.create(
                    model=MODEL_FAST,
                    max_tokens=SUBQ_MAX_TOKENS,
                    system=_SUBQ_SYSTEM,
                    messages=messages,
                    tools=tools,
                )
            except anthropic.APIError:
                break

            if _usage is not None:
                _usage["input_tokens"]  = _usage.get("input_tokens",  0) + response.usage.input_tokens
                _usage["output_tokens"] = _usage.get("output_tokens", 0) + response.usage.output_tokens
                _usage["tool_calls"]    = _usage.get("tool_calls",    0) + sum(
                    1 for b in response.content
                    if getattr(b, "type", None) == "server_tool_use"
                )

            # Collect and enrich evidence from all block types
            new_ev = _extract_evidence_from_blocks(response.content)
            for url, ev in new_ev.items():
                if url not in raw_ev_by_url:
                    raw_ev_by_url[url] = ev
                else:
                    _enrich(raw_ev_by_url[url], ev)

            # Collect text output
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    full_text += block.text

            if response.stop_reason != "pause_turn":
                break

            continuation = [
                b.model_dump() if hasattr(b, "model_dump") else b
                for b in response.content
            ]
            messages = [
                *messages,
                {"role": "assistant", "content": continuation},
                {"role": "user", "content": "Continue researching."},
            ]

        return {
            "subquestion":  subquestion,
            "raw_evidence": list(raw_ev_by_url.values()),
            "summary":      full_text.strip(),
        }


# ── Step C: Synthesis streaming ────────────────────────────────────────────

async def _stream_synthesis(
    system: str,
    user_prompt: str,
    evidence_text: str | None = None,
    _usage: dict | None = None,
):
    """
    Async generator: streams synthesis text chunks, no tools.
    `evidence_text` is placed in a cached block before the main prompt,
    so it can be reused across refinement cycles without re-paying full input cost.
    """
    if evidence_text:
        user_content = [
            {"type": "text", "text": f"Evidence list:\n{evidence_text}",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt

    async with client.messages.stream(
        model=MODEL_SYNTHESIS,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
        if _usage is not None:
            _usage["input_tokens"]  = _usage.get("input_tokens",  0) + final.usage.input_tokens
            _usage["output_tokens"] = _usage.get("output_tokens", 0) + final.usage.output_tokens


# ── Citation quality ───────────────────────────────────────────────────────

def count_citation_rate(report: str) -> tuple[int, int]:
    """Returns (cited_bullets, total_bullets) outside the Sources section."""
    cited = total = 0
    in_sources = False
    for line in report.splitlines():
        s = line.strip()
        if s.startswith("## Sources"):
            in_sources = True; continue
        if in_sources and s.startswith("## "):
            in_sources = False
        if in_sources:
            continue
        if not (s.startswith("- ") or s.startswith("* ")):
            continue
        content = s.lstrip("-* ").strip()
        if not content:
            continue
        total += 1
        if re.search(r"\[E?\d+\]", content):
            cited += 1
    return cited, total


async def _citation_rewrite(
    report: str,
    evidence: list[dict],
    _usage: dict | None = None,
) -> str:
    """MODEL_FAST single-pass rewrite: add [EN] IDs to uncited bullets or mark (unverified)."""
    ev_lines = "\n".join(f"[{e['id']}] {e['title']} — {e['url']}" for e in evidence)
    system = (
        "You are a citation editor. Given a research report and an evidence list, "
        "add the most appropriate [EN] inline citation to every bullet that currently "
        "lacks one. Do NOT change wording, add content, or remove bullets. "
        "If no evidence ID plausibly supports a bullet, append (unverified) instead. "
        "Return ONLY the complete updated report — no explanation, no fences."
    )
    response = await client.messages.create(
        model=MODEL_FAST,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": f"Evidence list:\n{ev_lines}\n\nReport:\n{report}"}],
    )
    if _usage is not None:
        _usage["input_tokens"]  = _usage.get("input_tokens",  0) + response.usage.input_tokens
        _usage["output_tokens"] = _usage.get("output_tokens", 0) + response.usage.output_tokens
    return response.content[0].text.strip()


# ── Utilities ──────────────────────────────────────────────────────────────

def _format_answers(answers: dict) -> str:
    if not answers:
        return "No additional context provided."
    return "\n".join(f"- {k}: {v}" for k, v in answers.items())


def _unique_domains(evidence: list[dict]) -> int:
    domains: set[str] = set()
    for e in evidence:
        url = e.get("url", "")
        if url:
            try:
                from urllib.parse import urlparse
                domains.add(urlparse(url).netloc)
            except Exception:
                pass
    return len(domains)


# ── Public API ─────────────────────────────────────────────────────────────

async def run(
    idea: str,
    answers: dict,
    prev_research: str | None = None,
    gap_queries: list[str] | None = None,
    latest_refinement: dict | None = None,
    _usage: dict | None = None,
    _evidence: list | None = None,
    max_subq: int = 6,
    searches_per_subq: int | None = None,
    fetches_per_subq: int | None = None,
):
    """
    Parallel research pipeline. Yields:
      str                          — streaming text chunks from synthesis
      {"__type__": "status", ...}  — progress events
      {"__type__": "citation_stat", "cited": int, "total": int}
      {"__type__": "report_final", "text": str}   — authoritative final text
    """
    # Prior evidence from previous cycles (already has IDs)
    prior_evidence: list[dict] = list(_evidence) if _evidence else []
    prior_count = len(prior_evidence)

    # ── Step A: Plan ──────────────────────────────────────────────────────
    yield {"__type__": "status", "message": "planning research sub-questions..."}
    plan_usage: dict = {}
    plan = await _plan_research(idea, answers, gap_queries, latest_refinement, _usage=plan_usage)
    if _usage is not None:
        _usage["input_tokens"]  = _usage.get("input_tokens",  0) + plan_usage.get("input_tokens",  0)
        _usage["output_tokens"] = _usage.get("output_tokens", 0) + plan_usage.get("output_tokens", 0)
    plan = plan[:max_subq]
    yield {"__type__": "status", "message": f"planned {len(plan)} sub-questions"}

    # ── Step B: Parallel sub-researchers ─────────────────────────────────
    semaphore = asyncio.Semaphore(RESEARCH_CONCURRENCY)

    tasks: list[asyncio.Task] = [
        asyncio.create_task(
            _research_subquestion(
                sp["subquestion"], sp["queries"], semaphore, _usage=_usage,
                searches=searches_per_subq, fetches=fetches_per_subq,
            )
        )
        for sp in plan
    ]

    subq_results: list[dict] = []
    pending = set(tasks)
    deadline = asyncio.get_event_loop().time() + RESEARCH_TIMEOUT

    while pending:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            for t in pending:
                t.cancel()
            yield {"__type__": "status",
                   "message": f"time budget reached — {len(subq_results)}/{len(tasks)} sub-questions completed"}
            break

        done, pending = await asyncio.wait(
            pending,
            timeout=min(remaining, RESEARCH_TIMEOUT),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in done:
            try:
                result = t.result()
                subq_results.append(result)
                n_ev = len(result.get("raw_evidence", []))
                label = result["subquestion"][:55]
                yield {"__type__": "status", "message": f"found {n_ev} sources: {label}"}
            except Exception:
                pass

    # ── Merge and deduplicate evidence ────────────────────────────────────
    # Build a URL-keyed dict across all sub-questions for enrichment
    all_raw_by_url: dict[str, dict] = {}
    for r in subq_results:
        for ev in r.get("raw_evidence", []):
            url = ev.get("url", "")
            if not url:
                continue
            if url not in all_raw_by_url:
                all_raw_by_url[url] = ev
            else:
                _enrich(all_raw_by_url[url], ev)

    # Enrich prior evidence items if the same URL was retrieved again
    prior_urls: set[str] = set()
    for e in prior_evidence:
        url = e.get("url", "")
        if url:
            prior_urls.add(url)
            if url in all_raw_by_url:
                _enrich(e, all_raw_by_url[url])

    # New evidence: items not in prior cycles
    fresh_raw = [ev for url, ev in all_raw_by_url.items() if url not in prior_urls]
    new_evidence = _dedup_assign_ids(fresh_raw, start_n=prior_count)

    # Full evidence list (prior + new)
    full_evidence = prior_evidence + new_evidence

    # Propagate to caller's list
    if _evidence is not None:
        _evidence.extend(new_evidence)

    n_domains = _unique_domains(full_evidence)
    n_with_text = sum(
        1 for e in full_evidence
        if e.get("quote") or e.get("retrieved_text_excerpt")
    )
    yield {"__type__": "status",
           "message": f"evidence: {len(full_evidence)} sources ({n_domains} domains, {n_with_text} with text)"}

    # ── Step C: Synthesis ────────────────────────────────────────────────
    # Select top items for the synthesis prompt
    synth_evidence = _select_for_synthesis(full_evidence, RESEARCH_EVIDENCE_LIMIT)
    ev_text = _evidence_list_text(synth_evidence)

    subq_block = "\n\n".join(
        f"### {r['subquestion']}\n{r['summary']}"
        for r in subq_results
        if r.get("summary")
    ) or "(no summaries — evidence list above contains all gathered material)"

    refinement_parts = []
    if latest_refinement:
        if latest_refinement.get("refined_prompt"):
            refinement_parts.append(f"Refined focus: {latest_refinement['refined_prompt']}")
        if latest_refinement.get("additional_context"):
            refinement_parts.append(f"Additional context: {latest_refinement['additional_context']}")
        for q, a in (latest_refinement.get("interrogator_answers") or {}).items():
            if a and a.strip():
                refinement_parts.append(f"  Q: {q}\n  A: {a}")
    refinement_text = ("\n\nHuman reviewer's refinement:\n" + "\n".join(refinement_parts)
                       if refinement_parts else "")

    if prev_research:
        system = SYSTEM_PROMPT_REVISION
        gap_block = ("\n".join(f"- {q}" for q in gap_queries)
                     if gap_queries else "(no specific queries provided)")
        user_prompt = f"""Idea: {idea}

Requester context:
{_format_answers(answers)}{refinement_text}

Gap-filling queries addressed in this cycle:
{gap_block}

Previous research report (retain all cited findings):
{prev_research}

Research findings by sub-topic (cite using IDs from the evidence list):
{subq_block}

Produce the cumulative updated research report. Every bullet MUST cite an [EN] whose excerpt supports the claim."""
    else:
        system = SYSTEM_PROMPT
        user_prompt = f"""Idea: {idea}

Requester context:
{_format_answers(answers)}{refinement_text}

Research findings by sub-topic (cite using IDs from the evidence list above):
{subq_block}

Produce the structured research report. Every bullet MUST end with at least one [EN] citation whose excerpt supports the claim."""

    synth_usage: dict = {}
    full_text = ""
    async for chunk in _stream_synthesis(system, user_prompt, evidence_text=ev_text, _usage=synth_usage):
        full_text += chunk
        yield chunk

    if _usage is not None:
        _usage["input_tokens"]  = _usage.get("input_tokens",  0) + synth_usage.get("input_tokens",  0)
        _usage["output_tokens"] = _usage.get("output_tokens", 0) + synth_usage.get("output_tokens", 0)

    # ── Citation post-check ───────────────────────────────────────────────
    cited, total = count_citation_rate(full_text)
    missing_rate = (total - cited) / total if total > 0 else 0.0
    final_text = full_text
    rewrite_usage: dict = {}

    if total > 0 and missing_rate > 0.20 and full_evidence:
        yield {"__type__": "status",
               "message": f"citation rate {cited}/{total} ({100*cited//total}%) — rewriting with evidence IDs"}
        final_text = await _citation_rewrite(full_text, full_evidence, _usage=rewrite_usage)
        if _usage is not None:
            _usage["input_tokens"]  = _usage.get("input_tokens",  0) + rewrite_usage.get("input_tokens",  0)
            _usage["output_tokens"] = _usage.get("output_tokens", 0) + rewrite_usage.get("output_tokens", 0)
        cited, total = count_citation_rate(final_text)
    else:
        yield {"__type__": "status",
               "message": f"citation rate {cited}/{total} ({100*cited//total if total else 0}%)"}

    yield {"__type__": "citation_stat", "cited": cited, "total": total}
    yield {"__type__": "report_final",  "text": final_text}
