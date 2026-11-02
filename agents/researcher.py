import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.AsyncAnthropic()

SYSTEM_PROMPT = """You are an expert research analyst. Produce a structured research report using EXACTLY the template below. Follow all length rules — they exist to prevent truncation.

OUTPUT FORMAT (use this exact structure):

## Market Landscape
- [market size / growth rate with specific numbers] (max 20 words)
- [2-3 dominant players and their positioning] (max 20 words)
- [maturity stage and trend direction] (max 20 words)
- [one notable market dynamic or shift] (max 20 words)

## Use Cases
- [primary use case]: [who benefits and how] (max 20 words each)
- [secondary use case]: [who benefits and how]
- [tertiary use case]: [who benefits and how]
(3–4 use cases total)

## Existing Solutions
- [solution/competitor name]: [what it does, its key gap] (max 25 words each)
- [solution/competitor name]: [what it does, its key gap]
- [solution/competitor name]: [what it does, its key gap]
(3–4 solutions total)

## Key Challenges
- [challenge]: [why it matters] (max 20 words each)
- [challenge]: [why it matters]
- [challenge]: [why it matters]
(3–4 challenges total)

## Opportunities
- [opportunity]: [why now / why underserved] (max 20 words each)
- [opportunity]: [why now / why underserved]
- [opportunity]: [why now / why underserved]
(2–3 opportunities total)

RULES:
- Use bullet points only — no prose paragraphs
- Each bullet must be self-contained and specific (name real companies, cite real numbers where possible)
- Do not add extra sections or commentary outside the template
- Total output must fit within ~600 words"""

SYSTEM_PROMPT_REVISION = """You are an expert research analyst conducting a targeted follow-up research pass. A previous round of research was critiqued. Produce an improved report using EXACTLY the template below.

OUTPUT FORMAT:

## Critique Gaps Addressed
- [critique point]: [how this report now resolves it] (max 20 words each)
(one bullet per critique point raised — be exhaustive)

## Market Landscape
- [finding] (max 20 words)
- [finding] (max 20 words)
- [finding] (max 20 words)
- [finding] (max 20 words)

## Use Cases
- [use case]: [who benefits and how] (max 20 words each)
(3–4 use cases)

## Existing Solutions
- [solution]: [what it does, its key gap] (max 25 words each)
(3–4 solutions)

## Key Challenges
- [challenge]: [why it matters] (max 20 words each)
(3–4 challenges)

## Opportunities
- [opportunity]: [why now / why underserved] (max 20 words each)
(2–3 opportunities)

RULES:
- Bullets only — no prose paragraphs
- The "Critique Gaps Addressed" section must explicitly name and close every gap from the critique
- Do not repeat information unchanged from the prior round — add new specificity or evidence
- Total output must fit within ~700 words"""


def _format_answers(answers: dict) -> str:
    if not answers:
        return "No additional context provided."
    return "\n".join(f"- {k}: {v}" for k, v in answers.items())


async def run(
    idea: str,
    answers: dict,
    prev_critique: str | None = None,
    latest_refinement: dict | None = None,
):
    if prev_critique:
        system = SYSTEM_PROMPT_REVISION
        refinement_text = ""
        if latest_refinement:
            parts = []
            if latest_refinement.get("refined_prompt"):
                parts.append(f"Refined focus: {latest_refinement['refined_prompt']}")
            if latest_refinement.get("additional_context"):
                parts.append(f"Additional context: {latest_refinement['additional_context']}")
            answers = latest_refinement.get("interrogator_answers", {})
            if answers:
                parts.append("\nAnswers to clarifying questions:")
                for question, answer in answers.items():
                    if answer and answer.strip():
                        parts.append(f"  Q: {question}\n  A: {answer}")
            if parts:
                refinement_text = "\n\nHuman reviewer's refinement:\n" + "\n".join(parts)

        user_prompt = f"""Idea: {idea}

Requester context:
{_format_answers(answers)}{refinement_text}

Previous critique (gaps to resolve):
{prev_critique}

Produce the improved structured research report."""
    else:
        system = SYSTEM_PROMPT
        user_prompt = f"""Idea: {idea}

Requester context:
{_format_answers(answers)}

Produce the structured research report."""

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
