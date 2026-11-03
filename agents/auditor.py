import re
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agents.config import MODEL_FAST
from agents.llm import tool_forced_call

SYSTEM_PROMPT = """You are a presentation architect. Distill a research and architecture dossier into structured data for a PowerPoint presentation.

Use the submit_presentation tool. Field constraints:
- research_highlights: 4-6 items, each max 25 words
- use_cases: 3-5 items
- architecture_components: 4-8 items
- key_risks: 3-5 items
- revisions_made: 3-5 bullets
- title: max 8 words
- recommendation: max 40 words"""

_TOOL = {
    "name": "submit_presentation",
    "description": "Submit the structured presentation data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title":               {"type": "string"},
            "subtitle":            {"type": "string"},
            "executive_summary":   {"type": "string"},
            "research_highlights": {"type": "array", "items": {"type": "string"}},
            "use_cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
                    "required": ["title", "description"],
                },
            },
            "architecture_overview": {"type": "string"},
            "architecture_components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "role": {"type": "string"}},
                    "required": ["name", "role"],
                },
            },
            "key_risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"risk": {"type": "string"}, "mitigation": {"type": "string"}},
                    "required": ["risk", "mitigation"],
                },
            },
            "critique_summary": {"type": "string"},
            "revisions_made":   {"type": "array", "items": {"type": "string"}},
            "recommendation":   {"type": "string"},
        },
        "required": [
            "title", "subtitle", "executive_summary", "research_highlights",
            "use_cases", "architecture_overview", "architecture_components", "key_risks",
        ],
    },
}


class _UseCase(BaseModel):
    title: str
    description: str


class _ArchComponent(BaseModel):
    name: str
    role: str


class _Risk(BaseModel):
    risk: str
    mitigation: str


class _PresentationData(BaseModel):
    title: str
    subtitle: str
    executive_summary: str
    research_highlights: list[str]
    use_cases: list[_UseCase]
    architecture_overview: str
    architecture_components: list[_ArchComponent]
    key_risks: list[_Risk]
    # Optional — may be empty in lite mode when no separate critic pass ran
    critique_summary: str = ""
    revisions_made: list[str] = []
    recommendation: str = ""


def _format_answers(answers: dict) -> str:
    if not answers:
        return "No additional context provided."
    return "\n".join(f"- {k}: {v}" for k, v in answers.items())


def _top_sources(evidence: list[dict], researcher_text: str, n: int = 15) -> list[dict]:
    """Return the top n evidence items ranked by [E#] citation frequency."""
    counts: dict[int, int] = {}
    for m in re.finditer(r"\[E?(\d+)\]", researcher_text):
        idx = int(m.group(1))
        counts[idx] = counts.get(idx, 0) + 1

    scored = []
    seen_urls: set[str] = set()
    for item in evidence:
        raw_id = item.get("id", "")
        m = re.match(r"E?(\d+)$", str(raw_id))
        if not m:
            continue
        url = item.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        scored.append((counts.get(int(m.group(1)), 0), item))

    scored.sort(key=lambda x: -x[0])
    return [
        {
            "title":       e.get("title",       ""),
            "url":         e.get("url",         ""),
            "source_type": e.get("source_type", ""),
        }
        for _, e in scored[:n]
    ]


async def run(
    idea: str,
    answers: dict,
    researcher: str,
    architect_v1: str,
    critic: str,
    architect_v2: str,
    evidence: list[dict] | None = None,
    _usage: dict | None = None,
) -> dict:
    variable_content = (
        f"Idea: {idea}\n\n"
        f"Requester Context:\n{_format_answers(answers)}\n\n"
        f"Architecture v1:\n{architect_v1}\n\n"
        f"Critique:\n{critic}\n\n"
        f"Architecture v2 (Revised):\n{architect_v2}\n\n"
        "Submit the presentation data via the tool."
    )

    result = await tool_forced_call(
        model=MODEL_FAST,
        max_tokens=4096,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": researcher, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": variable_content},
        ]}],
        tool=_TOOL,
        schema=_PresentationData,
        _usage=_usage,
    )

    data = result.model_dump()
    data["sources"] = _top_sources(evidence or [], researcher)
    return data
