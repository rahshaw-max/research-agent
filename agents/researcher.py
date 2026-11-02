import json
import os
from urllib.parse import url parse

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.AsyncAnthropic()

MAX_SEARCHES = int(os.getenv("MAX_SEARCHES", "8"))

# Server-side tools — API executes them; no client tool-result round-trip needed.
# web_fetch only works on URLs that appeared in prior context (e.g. search results).
_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_SEARCHES},
    {"type": "web_fetch_20250910", "name": "web_fetch",   "max_uses": MAX_SEARCHES},
]

_CITATION_RULES = """
CITATION RULES (mandatory):
- Before asserting any number, market size, company name, product name, or trend, run a web_search to verify it.
- After a search surfaces a relevant page, use web_fetch to read the full page — do not rely on snippets alone.
- Prefer primary sources: SEC/CMS filings, regulator websites, vendor documentation, peer-reviewed papers.
- Attach an inline citation [N] to EVERY factual bullet. A single bullet may carry multiple citations.
- If a claim cannot be verified by search, write "(unverified)" — do not guess or hallucinate.
- End your entire response with a "## Sources" section listing every cited source as:
  [N] Title — URL
  (one line per source, ordered by first appearance)
"""

SYSTEM_PROMPT = f"""You are an expert research analyst with live web access. Before writing any fact, search for it.

{_CITATION_RULES}

Produce a structured research report using EXACTLY this template:

## Market Landscape
- [finding with inline citation] [N] (max 25 words)
- [finding] [N]
- [finding] [N]
- [finding] [N]
(3–5 bullets)

## Use Cases
- [use case]: [who benefits and how] [N] (max 25 words each)
(3–4 use cases)

## Existing Solutions
- **[solution name]**: [what it does, its key gap] [N] (max 30 words)
(3–4 solutions)

## Key Challenges
- [challenge]: [why it matters] [N] (max 25 words each)
(3–4 challenges)

## Opportunities
- [opportunity]: [why now / why underserved] [N] (max 25 words each)
(2–3 opportunities)

## Sources
[N] Title — URL
(every source cited above, in order)

RULES:
- Search first, write second. Never state a number or name without a citation.
- Bullets only in body sections — no prose paragraphs.
- Total output (excluding Sources) must fit within ~700 words."""

SYSTEM_PROMPT_REVISION = f"""You are an expert research analyst with live web access conducting a targeted follow-up pass. A critique identified specific gaps in the prior research. Your job is to close those gaps with fresh searches — not to reword existing content.

{_CITATION_RULES}

For each gap named in the critique, run at least one dedicated search targeting that specific claim or question. Do not rely on prior search results.

Produce an improved report using EXACTLY this template:

## Critique Gaps Addressed
- [critique point] → [how new evidence resolves it] [N] (max 25 words, one bullet per gap)

## Market Landscape
- [finding] [N] (max 25 words)
(3–5 bullets)

## Use Cases
- [use case]: [who benefits and how] [N] (max 25 words each)
(3–4 use cases)

## Existing Solutions
- **[solution name]**: [what it does, its key gap] [N] (max 30 words)
(3–4 solutions)

## Key Challenges
- [challenge]: [why it matters] [N] (max 25 words each)
(3–4 challenges)

## Opportunities
- [opportunity]: [why now / why underserved] [N] (max 25 words each)
(2–3 opportunities)

## Sources
[N] Title — URL

RULES:
- Every bullet in "Critique Gaps Addressed" must cite a freshly fetched source.
- Do not repeat bullets unchanged from the prior report — add new specificity or evidence.
- Search first, write second. Mark anything unverifiable as "(unverified)"."""


def _format_answers(answers: dict) -> str:
    if not answers:
        return "No additional context provided."
    return "\n".join(f"- {k}: {v}" for k, v in answers.items())


async def _stream_with_tools(system: str, user_prompt: str):
    """
    Drive a server-tool loop with web_search + web_fetch.

    Yields:
      str  — text chunk to stream to the caller
      dict — {"__type__": "status", "message": str}  side-channel status event
    Handles pause_turn continuation and tool errors without crashing.
    """
    messages: list = [{"role": "user", "content": user_prompt}]

    while True:
        try:
            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                system=system,
                messages=messages,
                tools=_TOOLS,
            ) as stream:
                current_type: str | None = None
                current_name: str | None = None
                partial_input: str = ""

                async for event in stream:
                    t = event.type

                    if t == "content_block_start":
                        cb = event.content_block
                        current_type = getattr(cb, "type", None)
                        current_name = getattr(cb, "name", None)
                        partial_input = ""

                    elif t == "content_block_delta":
                        d = event.delta
                        dt = getattr(d, "type", None)
                        if current_type == "server_tool_use" and dt == "input_json_delta":
                            partial_input += getattr(d, "partial_json", "")
                        elif current_type == "text" and dt == "text_delta":
                            text = getattr(d, "text", "")
                            if text:
                                yield text

                    elif t == "content_block_stop":
                        if current_type == "server_tool_use" and partial_input:
                            try:
                                inp = json.loads(partial_input)
                                if current_name == "web_search":
                                    q = inp.get("query", "")
                                    if q:
                                        yield {"__type__": "status", "message": f"searching: {q}"}
                                elif current_name == "web_fetch":
                                    u = inp.get("url", "")
                                    if u:
                                        domain = urlparse(u).netloc or u[:60]
                                        yield {"__type__": "status", "message": f"reading: {domain}"}
                            except Exception:
                                pass  # malformed partial JSON — ignore

                final = await stream.get_final_message()

                if final.stop_reason != "pause_turn":
                    break

                # pause_turn: append the assistant turn and continue
                continuation: list = []
                for block in final.content:
                    if hasattr(block, "model_dump"):
                        continuation.append(block.model_dump())
                    else:
                        continuation.append(block)
                messages = [*messages, {"role": "assistant", "content": continuation}]

        except anthropic.RateLimitError as e:
            yield {"__type__": "status", "message": f"rate limited — retrying after pause"}
            import asyncio
            await asyncio.sleep(5)
            # retry the same messages
        except anthropic.APIError as e:
            yield {"__type__": "status", "message": f"API error (stopping search): {e}"}
            break


async def run(
    idea: str,
    answers: dict,
    prev_critique: str | None = None,
    latest_refinement: dict | None = None,
):
    if prev_critique:
        system = SYSTEM_PROMPT_REVISION

        refinement_parts = []
        if latest_refinement:
            if latest_refinement.get("refined_prompt"):
                refinement_parts.append(f"Refined focus: {latest_refinement['refined_prompt']}")
            if latest_refinement.get("additional_context"):
                refinement_parts.append(f"Additional context: {latest_refinement['additional_context']}")
            answers_qa = latest_refinement.get("interrogator_answers", {})
            if answers_qa:
                refinement_parts.append("Answers to clarifying questions:")
                for question, answer in answers_qa.items():
                    if answer and answer.strip():
                        refinement_parts.append(f"  Q: {question}\n  A: {answer}")

        refinement_text = ("\n\nHuman reviewer's refinement:\n" + "\n".join(refinement_parts)
                           if refinement_parts else "")

        user_prompt = f"""Idea: {idea}

Requester context:
{_format_answers(answers)}{refinement_text}

Previous critique (gaps to close with fresh research):
{prev_critique}

Produce the improved structured research report. Run new searches for each critique gap."""

    else:
        system = SYSTEM_PROMPT
        user_prompt = f"""Idea: {idea}

Requester context:
{_format_answers(answers)}

Produce the structured research report. Search before asserting any fact."""

    async for chunk in _stream_with_tools(system, user_prompt):
        yield chunk
