import anthropic
from dotenv import load_dotenv

load_dotenv()

from agents.config import MODEL_FAST, MODEL_SYNTHESIS

client = anthropic.AsyncAnthropic()

# ── Research quality audit ─────────────────────────────────────────────────

_GAP_TOOL = {
    "name": "report_gaps",
    "description": "Report structured research quality gaps and an overall coverage score.",
    "input_schema": {
        "type": "object",
        "properties": {
            "coverage_score": {
                "type": "number",
                "description": (
                    "Overall research coverage score 0.0–1.0. "
                    "0.0–0.3: many unsupported claims, major angles missing. "
                    "0.3–0.5: some verification but significant gaps. "
                    "0.5–0.7: decent coverage with minor gaps. "
                    "0.7–0.85: good, most claims sourced. "
                    "0.85–1.0: excellent, well-sourced, comprehensive."
                ),
            },
            "gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "no_primary_source",
                                "conflicting_claims",
                                "stale",
                                "unsupported_claim",
                                "missing_angle",
                                "thin_coverage",
                            ],
                        },
                        "description": {
                            "type": "string",
                            "description": "1–2 sentences identifying the specific gap and where it appears.",
                        },
                        "severity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "description": "5 = critical gap that undermines research value.",
                        },
                        "suggested_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 3,
                            "description": "2–3 specific web search queries that would fill this gap.",
                        },
                    },
                    "required": ["type", "description", "severity", "suggested_queries"],
                },
            },
        },
        "required": ["coverage_score", "gaps"],
    },
}

_RESEARCH_CRITIQUE_SYSTEM = """You are an expert research quality auditor. You will receive a research report on a given idea and requester context. Systematically identify every coverage gap and weakness.

For each gap:
- Classify by type: no_primary_source (claim lacks an authoritative source), conflicting_claims (sources contradict each other), stale (data is old for a fast-moving field), unsupported_claim (assertion made with no citation at all), missing_angle (important dimension of the topic not covered at all), thin_coverage (section exists but lacks depth or specificity).
- Describe the specific gap — name the section and the claim.
- Rate severity 1–5 (5 = critical).
- Provide 2–3 concrete, specific search queries that would fill this gap (not generic — queries that would yield verifiable primary-source data).

Judge the coverage score honestly against the idea and requester context. A report that cites Wikipedia or blog posts heavily scores lower than one citing SEC filings, regulator sites, or peer-reviewed papers."""


def _format_answers(answers: dict) -> str:
    if not answers:
        return "No additional context provided."
    return "\n".join(f"- {k}: {v}" for k, v in answers.items())


async def run_research_critique(idea: str, research: str, answers: dict, _usage: dict | None = None) -> dict:
    """
    Evaluate research quality. Returns a GapReport dict.
    Uses MODEL_FAST with tool-forcing and prompt caching on the research block.
    """
    answers_text = _format_answers(answers)
    response = await client.messages.create(
        model=MODEL_FAST,
        max_tokens=2048,
        system=[{"type": "text", "text": _RESEARCH_CRITIQUE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": research, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": (
                f"Idea: {idea}\n\nRequester context:\n{answers_text}\n\n"
                "Identify all coverage gaps and compute the overall coverage score."
            )},
        ]}],
        tools=[_GAP_TOOL],
        tool_choice={"type": "tool", "name": "report_gaps"},
    )

    if _usage is not None:
        _usage["input_tokens"]         = _usage.get("input_tokens", 0) + response.usage.input_tokens
        _usage["output_tokens"]        = _usage.get("output_tokens", 0) + response.usage.output_tokens
        _usage["cache_read_tokens"]    = _usage.get("cache_read_tokens", 0) + getattr(response.usage, "cache_read_input_tokens", 0)
        _usage["cache_created_tokens"] = _usage.get("cache_created_tokens", 0) + getattr(response.usage, "cache_creation_input_tokens", 0)

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "report_gaps":
            return block.input

    raise ValueError("run_research_critique: no tool_use block returned")


# ── Architecture critique ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a ruthless but constructive critic. Produce a structured critique in approximately 500 words using EXACTLY the template below.

OUTPUT FORMAT:

## 1. Use Case Critique
- **[Issue title]**: [finding — explain the problem, why it matters, evidence] → **Fix**: [specific actionable change]
- **[Issue title]**: [finding] → **Fix**: [specific action]
(2–3 bullets)

## 2. Market & Competitive Risks
- **[Risk title]**: [threat description — name specific competitors or market forces] → **Mitigation**: [specific action]
- **[Risk title]**: [threat description] → **Mitigation**: [specific action]
(2–3 bullets)

## 3. Architecture Weaknesses
- **[Weakness title]**: [what is wrong and why — cite component and failure mode] → **Fix**: [specific change]
- **[Weakness title]**: [what is wrong] → **Fix**: [specific change]
(2–4 bullets)

## 4. Security & Compliance Gaps
- **[Gap title]**: [exposure — identify specific regulation or threat vector] → **Fix**: [specific control]
- **[Gap title]**: [exposure] → **Fix**: [specific control]
(2–3 bullets)

## 5. Implementation Risks
- **[Risk title]**: [blocker or dependency — why this is a real risk] → **Mitigation**: [specific action]
- **[Risk title]**: [blocker] → **Mitigation**: [specific action]
(2–3 bullets)

## Priority Recommendations
1. [Most critical fix — specific and actionable, 1 sentence]
2. [Second most critical]
3. [Third most critical]
4. [Fourth]
5. [Fifth]

RULES:
- Target ~500 words total. Be concise — one clear sentence per bullet.
- Every bullet must name a specific issue AND provide a specific fix.
- No prose paragraphs — bullets and numbered list only."""


async def run(idea: str, research: str, arch_v1: str, _usage: dict | None = None):
    async with client.messages.stream(
        model=MODEL_SYNTHESIS,
        max_tokens=700,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": research, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"Idea: {idea}\n\nProposed Architecture:\n{arch_v1}\n\nProduce the structured critique."},
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
