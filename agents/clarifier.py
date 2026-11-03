from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agents.config import MODEL_FAST
from agents.llm import tool_forced_call

SYSTEM_PROMPT = """You are a research intake assistant. A user has submitted an idea or concept they want researched and turned into a presentation deck. Your job is to ask 3–5 targeted clarifying questions that will help scope the research properly.

Focus your questions on:
- The intended audience (technical, business, general public)
- The primary goal (pitch deck, internal exploration, feasibility study, product plan)
- The domain or industry context (if not already clear)
- Any known constraints (time horizon, geography, budget scale)
- What aspects they care about most (market, tech, competition, implementation)

Use the submit_questions tool to return your 3–5 questions."""

_TOOL = {
    "name": "submit_questions",
    "description": "Submit the clarifying questions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
            }
        },
        "required": ["questions"],
    },
}


class _Questions(BaseModel):
    questions: list[str]


_FALLBACK = [
    "Who is the intended audience for this presentation?",
    "What is the primary goal — pitch deck, feasibility study, or internal exploration?",
    "Are there any known constraints (budget, geography, or time horizon)?",
]


async def ask_clarifying_questions(idea: str) -> list[str]:
    try:
        result = await tool_forced_call(
            model=MODEL_FAST,
            max_tokens=512,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"The user's idea is: {idea}"}],
            tool=_TOOL,
            schema=_Questions,
        )
        return result.questions
    except Exception:
        return _FALLBACK
