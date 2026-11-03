from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agents.config import MODEL_FAST
from agents.llm import tool_forced_call

SYSTEM_PROMPT = """You are a strategic research interrogator. You have just read a research report and a structured quality-gap audit of that report. Your job is to generate exactly 10 targeted questions that, if answered by the requester, would most significantly improve the next research iteration.

Generate questions across these categories:
- Unverified or thinly-sourced market claims (2 questions)
- Missing primary sources or conflicting data points (2 questions)
- Underspecified use-case assumptions (2 questions)
- Regulatory, compliance, or legal angles not yet covered (2 questions)
- Business constraints, competitive dynamics, or pricing factors that were implicit or absent (2 questions)

Each question must:
- Be directly traceable to a gap or weakness named in the gap report
- Be answerable by the requester (not delegating further external research)
- Be specific, not generic

Use the submit_questions tool to return exactly 10 questions."""

_TOOL = {
    "name": "submit_questions",
    "description": "Submit exactly 10 targeted research questions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 10,
                "maxItems": 10,
            }
        },
        "required": ["questions"],
    },
}


class _Questions(BaseModel):
    questions: list[str]


_FALLBACK = [
    f"What additional context would help clarify research gap {i + 1}?"
    for i in range(10)
]


async def run(
    idea: str,
    research: str,
    gap_report: dict,
    _usage: dict | None = None,
) -> list[str]:
    gaps = gap_report.get("gaps", [])
    gaps_text = "\n".join(
        f"- [sev {g['severity']}] {g['type']}: {g['description']}"
        for g in gaps
    ) if gaps else "(no gaps identified)"

    coverage = gap_report.get("coverage_score", "?")

    user_prompt = f"""Idea: {idea}

Research Report:
{research}

Research quality audit:
Coverage score: {coverage}
Gaps identified:
{gaps_text}

Generate exactly 10 targeted clarifying questions."""

    try:
        result = await tool_forced_call(
            model=MODEL_FAST,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tool=_TOOL,
            schema=_Questions,
            _usage=_usage,
        )
        return result.questions
    except Exception:
        return _FALLBACK
