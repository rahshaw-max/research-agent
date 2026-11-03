import anthropic
from dotenv import load_dotenv

load_dotenv()

from agents.config import MODEL_SYNTHESIS

client = anthropic.AsyncAnthropic()

_SECTIONS = """
## System Overview
[3–4 sentences describing what is being built, its core value proposition, and the key design decisions]

## Core Components
| Component | Responsibility |
|-----------|----------------|
| [name] | [responsibility — what this component owns and how it interacts with others] |
(4–6 components total)

## Technology Stack
- Frontend: [technology + rationale]
- Backend: [technology + rationale]
- Database: [technology + rationale]
- Infrastructure: [technology + rationale]
- AI/ML: [technology + rationale] — if applicable

## Data Flow
- Step 1: [action → output]
- Step 2: [action → output]
- Step 3: [action → output]
(3–5 steps)

## Integration Points
- [service/API]: [purpose — what data is exchanged and why]
(3–4 integrations)

## Scalability
- [approach]: [what it addresses and at what scale]
(2–3 bullets)

## Security & Compliance
- [control]: [what it protects, which regulation, how implemented]
(3–4 bullets)

## Implementation Phases
- Phase 1 – MVP: [scope, deliverables, success criteria]
- Phase 2: [scope, deliverables]
- Phase 3: [scope, production readiness]"""

_WORD_TARGET = "Target ~900 words total. Be precise — one well-chosen technology per line, one tight sentence of rationale."

SYSTEM_PROMPT_V1 = f"""You are a senior solutions architect. Design a concrete technical architecture using EXACTLY the template below.

OUTPUT FORMAT:{_SECTIONS}

RULES:
- {_WORD_TARGET}
- Use the table and bullet format exactly — no prose paragraphs outside System Overview
- Name specific technologies (e.g. "PostgreSQL" not "a relational database") and justify each choice"""

SYSTEM_PROMPT_V1_REVISION = f"""You are a senior solutions architect producing an improved architecture after a critique review. Use EXACTLY the template below.

OUTPUT FORMAT:

## Changes From Prior Architecture
- [critique point] → [how this design resolves it]
(one bullet per critique point){_SECTIONS}

RULES:
- {_WORD_TARGET}
- The Changes section must explicitly address every architectural weakness from the critique
- Keep what worked; fix what was flagged"""

SYSTEM_PROMPT_V2 = f"""You are a senior solutions architect revising an architecture based on a critique. Use EXACTLY the template below.

OUTPUT FORMAT:

## Response to Critique
- [critique point] → [resolution or justified retention]
(one bullet per critique point){_SECTIONS}

Mark each changed section heading with [REVISED] or [ADDED] where applicable.

RULES:
- {_WORD_TARGET}
- Bullets and table only — no prose paragraphs outside System Overview and Response to Critique"""

SYSTEM_PROMPT_LITE = f"""You are a senior solutions architect. Design a complete, production-ready technical architecture in a single pass. Where unresolved research gaps have architectural implications, note them inline.

OUTPUT FORMAT:{_SECTIONS}

RULES:
- {_WORD_TARGET}
- Use the table and bullet format exactly — no prose paragraphs outside System Overview
- Name specific technologies and justify each choice"""


async def run(
    idea: str,
    research: str,
    prev_critique: str | None = None,
    prev_arch: str | None = None,
    _usage: dict | None = None,
):
    if prev_critique and prev_arch:
        system = SYSTEM_PROMPT_V1_REVISION
        variable_content = (
            f"Idea: {idea}\n\n"
            f"Previous Architecture:\n{prev_arch}\n\n"
            f"Critique of Previous Architecture:\n{prev_critique}\n\n"
            "Produce the improved structured architecture."
        )
    else:
        system = SYSTEM_PROMPT_V1
        variable_content = f"Idea: {idea}\n\nProduce the structured technical architecture."

    async with client.messages.stream(
        model=MODEL_SYNTHESIS,
        max_tokens=1200,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": research, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": variable_content},
        ]}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
        if _usage is not None:
            final = await stream.get_final_message()
            _usage["input_tokens"]         = _usage.get("input_tokens", 0) + final.usage.input_tokens
            _usage["output_tokens"]        = _usage.get("output_tokens", 0) + final.usage.output_tokens
            _usage["cache_read_tokens"]    = _usage.get("cache_read_tokens", 0) + getattr(final.usage, "cache_read_input_tokens", 0)
            _usage["cache_created_tokens"] = _usage.get("cache_created_tokens", 0) + getattr(final.usage, "cache_creation_input_tokens", 0)


async def run_revision(
    idea: str,
    research: str,
    arch_v1: str,
    critique: str,
    _usage: dict | None = None,
):
    async with client.messages.stream(
        model=MODEL_SYNTHESIS,
        max_tokens=1200,
        system=[{"type": "text", "text": SYSTEM_PROMPT_V2, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": research, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": (
                f"Idea: {idea}\n\nOriginal Architecture:\n{arch_v1}\n\n"
                f"Critique:\n{critique}\n\nProduce the revised structured architecture."
            )},
        ]}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
        if _usage is not None:
            final = await stream.get_final_message()
            _usage["input_tokens"]         = _usage.get("input_tokens", 0) + final.usage.input_tokens
            _usage["output_tokens"]        = _usage.get("output_tokens", 0) + final.usage.output_tokens
            _usage["cache_read_tokens"]    = _usage.get("cache_read_tokens", 0) + getattr(final.usage, "cache_creation_input_tokens", 0)
            _usage["cache_created_tokens"] = _usage.get("cache_created_tokens", 0) + getattr(final.usage, "cache_creation_input_tokens", 0)


async def run_lite(
    idea: str,
    research: str,
    gap_report: dict,
    _usage: dict | None = None,
):
    gaps = gap_report.get("gaps", [])
    gaps_text = "\n".join(
        f"- [sev {g['severity']}] {g['type']}: {g['description']}"
        for g in gaps
    ) if gaps else "(none)"

    async with client.messages.stream(
        model=MODEL_SYNTHESIS,
        max_tokens=1200,
        system=[{"type": "text", "text": SYSTEM_PROMPT_LITE, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": research, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": (
                f"Idea: {idea}\n\n"
                f"Unresolved research gaps (for architectural awareness):\n{gaps_text}\n\n"
                "Produce the structured technical architecture."
            )},
        ]}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
        if _usage is not None:
            final = await stream.get_final_message()
            _usage["input_tokens"]         = _usage.get("input_tokens", 0) + final.usage.input_tokens
            _usage["output_tokens"]        = _usage.get("output_tokens", 0) + final.usage.output_tokens
            _usage["cache_read_tokens"]    = _usage.get("cache_read_tokens", 0) + getattr(final.usage, "cache_read_input_tokens", 0)
            _usage["cache_created_tokens"] = _usage.get("cache_created_tokens", 0) + getattr(final.usage, "cache_creation_input_tokens", 0)
