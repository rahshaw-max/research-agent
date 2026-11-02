import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.AsyncAnthropic()

SYSTEM_PROMPT = """You are a strategic research interrogator. You have just read a critique of a research report and technical architecture. Your job is to generate exactly 10 targeted questions that, if answered by the requester, would most significantly improve the next research and architecture iteration.

Generate questions across these categories:
- Use case assumptions flagged as unproven (2 questions)
- Missing or uncertain market facts (2 questions)
- Underspecified technical choices (2 questions)
- Security, compliance, or regulatory requirements that were unclear (2 questions)
- Business constraints that were implicit but unstated (2 questions)

Each question must:
- Be specific to the critique findings — not generic
- Be answerable by the requester (not asking for external research)
- Directly address a gap or weakness identified in the critique

Return ONLY a valid JSON array of exactly 10 question strings. No preamble, no explanation, no markdown fences."""


async def run(idea: str, research: str, arch_v1: str, critique: str) -> list[str]:
    user_prompt = f"""Idea: {idea}

Research Report:
{research}

Architecture:
{arch_v1}

Critique (gaps and weaknesses identified):
{critique}

Generate exactly 10 targeted clarifying questions."""

    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)
