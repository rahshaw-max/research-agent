import os

from agents.config import MODEL_CRITIQUE, MODEL_SYNTHESIS
from agents.llm import client

# ── Coverage-score computation config ─────────────────────────────────────
# Caps applied deterministically after the model scores each sub-question.
MIN_DOMAINS_FOR_FULL_SCORE = 2    # fewer independent domains → cap at DOMAIN_CAP
DOMAIN_CAP             = 0.5     # max score when n_independent_domains < MIN_DOMAINS_FOR_FULL_SCORE
NO_PRIMARY_CAP         = 0.7     # max score when has_primary_source is False
UNCITED_PENALTY_WEIGHT = 1.0     # fraction of uncited share applied as score penalty
COVERAGE_THRESHOLD     = float(os.getenv("COVERAGE_THRESHOLD", "0.75"))

# ── Research quality audit ─────────────────────────────────────────────────

_GAP_ITEMS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": [
                    "no_primary_source", "conflicting_claims", "stale",
                    "unsupported_claim", "missing_angle", "thin_coverage",
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
                "description": "2–3 specific queries that would fill this gap.",
            },
        },
        "required": ["type", "description", "severity", "suggested_queries"],
    },
}

_COVERAGE_TOOL = {
    "name": "report_coverage",
    "description": (
        "Score research coverage per sub-question and report overall gaps. "
        "Score each sub-question independently based on the evidence cited in the report."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_question_scores": {
                "type": "array",
                "description": (
                    "One entry per sub-question from the research plan. "
                    "If a sub-question has no matching content in the report, score it 0."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "sub_question": {
                            "type": "string",
                            "description": "The exact sub-question text.",
                        },
                        "score": {
                            "type": "number",
                            "description": (
                                "Coverage quality 0.0–1.0 (before deterministic caps). "
                                "0.0 = not addressed; 0.5 = partial, generic sources; "
                                "0.8 = well-sourced, specific; 1.0 = primary sources, complete."
                            ),
                        },
                        "n_independent_domains": {
                            "type": "integer",
                            "description": (
                                "Count of distinct source domains cited for claims "
                                "that answer this sub-question."
                            ),
                        },
                        "has_primary_source": {
                            "type": "boolean",
                            "description": (
                                "True if at least one primary source is cited for this "
                                "sub-question: government site, regulator, peer-reviewed "
                                "journal, SEC filing, official statistics."
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": "1–2 sentences explaining the score.",
                        },
                    },
                    "required": [
                        "sub_question", "score", "n_independent_domains",
                        "has_primary_source", "rationale",
                    ],
                },
            },
            "gaps": _GAP_ITEMS_SCHEMA,
        },
        "required": ["sub_question_scores", "gaps"],
    },
}

_RESEARCH_CRITIQUE_SYSTEM = """\
You are an expert research quality auditor. You will receive:
  1. The research idea and requester context
  2. The sub-questions the researcher was asked to answer
  3. The research report (already verified — remove unsupported claims may have been done)

TASK A — Score each sub-question:
For each sub-question, assess how well the report addresses it:
  - score: 0.0–1.0 quality rating BEFORE any caps (the system applies deterministic caps)
    · 0.0  = not addressed at all
    · 0.3  = mentioned briefly, no citations
    · 0.5  = covered but only secondary sources (blog posts, Wikipedia)
    · 0.7  = well covered with multiple cited sources
    · 0.9  = primary sources, specific data (numbers, dates, named entities)
    · 1.0  = primary sources + data + cross-verified
  - n_independent_domains: count distinct registered domains cited for THIS sub-question
  - has_primary_source: true only if a government site, regulator, peer-reviewed journal, or
    official statistics body is cited for THIS sub-question
  - rationale: 1–2 sentences explaining the score

TASK B — Identify gaps:
Report every coverage weakness (type, description, severity 1–5, 2–3 suggested_queries).
Types: no_primary_source, conflicting_claims, stale, unsupported_claim, missing_angle, thin_coverage.
Severity 5 = critical gap that undermines research value.

Be honest. A sub-question answered only with generic blog posts or Wikipedia earns ≤0.5 \
and has_primary_source=false."""


def _compute_coverage_score(
    sub_scores: list[dict],
    uncited_share: float,
) -> float:
    """
    Deterministic coverage score from model's per-sub-question assessments.

    Caps (applied independently, take the min):
      - n_independent_domains < MIN_DOMAINS_FOR_FULL_SCORE  → cap at DOMAIN_CAP
      - has_primary_source == False                         → cap at NO_PRIMARY_CAP

    Then weighted mean (uniform weights), then penalise uncited claims:
      final = mean * (1 - uncited_share * UNCITED_PENALTY_WEIGHT)
    """
    # Defensively filter non-dict items (model schema-compliance errors)
    valid = [s for s in sub_scores if isinstance(s, dict)]
    if not valid:
        return 0.0

    capped = []
    for s in valid:
        raw = max(0.0, min(1.0, float(s.get("score", 0.0))))
        if s.get("n_independent_domains", 0) < MIN_DOMAINS_FOR_FULL_SCORE:
            raw = min(raw, DOMAIN_CAP)
        if not s.get("has_primary_source", True):
            raw = min(raw, NO_PRIMARY_CAP)
        capped.append(raw)

    base = sum(capped) / len(capped)
    penalty = max(0.0, min(1.0, uncited_share)) * UNCITED_PENALTY_WEIGHT
    return round(max(0.0, base * (1.0 - penalty)), 3)


def _format_answers(answers: dict) -> str:
    if not answers:
        return "No additional context provided."
    return "\n".join(f"- {k}: {v}" for k, v in answers.items())


async def run_research_critique(
    idea: str,
    research: str,
    answers: dict,
    sub_questions: list[str] | None = None,
    evidence: list[dict] | None = None,
    uncited_share: float = 0.0,
    _usage: dict | None = None,
) -> dict:
    """
    Evaluate research quality with per-sub-question scoring.

    Returns a GapReport dict:
      {
        "coverage_score": float,           # deterministically computed
        "sub_question_scores": [...],      # model's per-sub-question assessments
        "gaps": [...],                     # same schema as before
      }

    Uses MODEL_CRITIQUE with tool-forcing and prompt caching on the research block.
    """
    answers_text = _format_answers(answers)

    subq_block = ""
    if sub_questions:
        subq_block = "Sub-questions the researcher was asked to answer:\n" + "\n".join(
            f"  {i+1}. {q}" for i, q in enumerate(sub_questions)
        )

    response = await client.messages.create(
        model=MODEL_CRITIQUE,
        max_tokens=4096,
        system=[{"type": "text", "text": _RESEARCH_CRITIQUE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": research, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": (
                f"Idea: {idea}\n\n"
                f"Requester context:\n{answers_text}\n\n"
                f"{subq_block}\n\n"
                "Score each sub-question and identify all coverage gaps."
            )},
        ]}],
        tools=[_COVERAGE_TOOL],
        tool_choice={"type": "tool", "name": "report_coverage"},
    )

    if _usage is not None:
        _usage["input_tokens"]         = _usage.get("input_tokens", 0) + response.usage.input_tokens
        _usage["output_tokens"]        = _usage.get("output_tokens", 0) + response.usage.output_tokens
        _usage["cache_read_tokens"]    = _usage.get("cache_read_tokens", 0) + getattr(response.usage, "cache_read_input_tokens", 0)
        _usage["cache_created_tokens"] = _usage.get("cache_created_tokens", 0) + getattr(response.usage, "cache_creation_input_tokens", 0)

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "report_coverage":
            raw = block.input if isinstance(block.input, dict) else {}
            sub_scores = [s for s in raw.get("sub_question_scores", []) if isinstance(s, dict)]
            gaps       = [g for g in raw.get("gaps", [])               if isinstance(g, dict)]
            score = _compute_coverage_score(sub_scores, uncited_share)
            return {
                "coverage_score":      score,
                "sub_question_scores": sub_scores,
                "gaps":                gaps,
            }

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
