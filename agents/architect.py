import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.AsyncAnthropic()

_SECTIONS = """
## System Overview
[2–3 sentences describing what is being built and its core value proposition]

## Core Components
| Component | Responsibility |
|-----------|----------------|
| [name] | [one-line responsibility, max 15 words] |
| [name] | [one-line responsibility] |
| [name] | [one-line responsibility] |
(4–6 components total)

## Technology Stack
- Frontend: [technology + rationale] (max 15 words)
- Backend: [technology + rationale] (max 15 words)
- Database: [technology + rationale] (max 15 words)
- Infrastructure: [technology + rationale] (max 15 words)
- AI/ML: [technology + rationale] (max 15 words) — if applicable

## Data Flow
- Step 1: [action → output] (max 15 words)
- Step 2: [action → output] (max 15 words)
- Step 3: [action → output] (max 15 words)
- Step 4: [action → output] (max 15 words)
(3–5 steps total)

## Integration Points
- [service/API]: [purpose] (max 15 words each)
(3–4 integrations)

## Scalability
- [approach]: [what it addresses] (max 15 words each)
(2–3 bullets)

## Security & Compliance
- [control]: [what it protects / regulation addressed] (max 15 words each)
(3–4 bullets)

## Implementation Phases
- Phase 1 – MVP: [scope, key deliverable] (max 20 words)
- Phase 2: [scope, key deliverable] (max 20 words)
- Phase 3: [scope, key deliverable] (max 20 words)"""

SYSTEM_PROMPT_V1 = f"""You are a senior solutions architect. Design a concrete technical architecture using EXACTLY the template below. Follow all length rules — they exist to prevent truncation.

OUTPUT FORMAT:{_SECTIONS}

RULES:
- Use the table and bullet format exactly as shown — no prose paragraphs outside System Overview
- Name specific technologies (e.g. "PostgreSQL" not "a relational database")
- Total output must fit within ~500 words"""

SYSTEM_PROMPT_V1_REVISION = f"""You are a senior solutions architect producing an improved architecture after a critique review. Use EXACTLY the template below.

OUTPUT FORMAT:

## Changes From Prior Architecture
- [critique point] → [how this design resolves it] (max 20 words each)
(one bullet per critique point — be exhaustive)
{_SECTIONS}

RULES:
- The "Changes" section must explicitly address every architectural weakness from the critique
- Keep what worked; fix what was flagged — do not discard sound decisions without justification
- Name specific technologies
- Total output must fit within ~600 words"""

SYSTEM_PROMPT_V2 = f"""You are a senior solutions architect revising an architecture based on a critique. Use EXACTLY the template below.

OUTPUT FORMAT:

## Response to Critique
- [critique point] → [resolution or justified retention] (max 20 words each)
(one bullet per critique point){_SECTIONS}

Mark each changed section heading with [REVISED] or [ADDED] where applicable.

RULES:
- Bullets and table only — no prose paragraphs outside System Overview and Response to Critique
- Total output must fit within ~600 words"""


async def run(
    idea: str,
    research: str,
    prev_critique: str | None = None,
    prev_arch: str | None = None,
):
    if prev_critique and prev_arch:
        system = SYSTEM_PROMPT_V1_REVISION
        user_prompt = f"""Idea: {idea}

Previous Architecture:
{prev_arch}

Critique of Previous Architecture:
{prev_critique}

Updated Research Report:
{research}

Produce the improved structured architecture."""
    else:
        system = SYSTEM_PROMPT_V1
        user_prompt = f"""Idea: {idea}

Research Report:
{research}

Produce the structured technical architecture."""

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def run_revision(idea: str, research: str, arch_v1: str, critique: str):
    user_prompt = f"""Idea: {idea}

Research Report:
{research}

Original Architecture:
{arch_v1}

Critique:
{critique}

Produce the revised structured architecture."""

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT_V2,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
