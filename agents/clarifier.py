import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.AsyncAnthropic()

SYSTEM_PROMPT = """You are a research intake assistant. A user has submitted an idea or concept they want researched and turned into a presentation deck. Your job is to ask 3–5 targeted clarifying questions that will help scope the research properly.

Focus your questions on:
- The intended audience (technical, business, general public)
- The primary goal (pitch deck, internal exploration, feasibility study, product plan)
- The domain or industry context (if not already clear)
- Any known constraints (time horizon, geography, budget scale)
- What aspects they care about most (market, tech, competition, implementation)

Return ONLY a valid JSON array of question strings. No preamble. No explanation. No markdown fences.
Example: ["Who is the intended audience for this presentation?", "What is the primary goal — pitch, feasibility study, or internal exploration?"]"""


async def ask_clarifying_questions(idea: str) -> list[str]:
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"The user's idea is: {idea}"}],
    )
    text = response.content[0].text.strip()
    # Strip markdown fences if Claude added them anyway
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)
