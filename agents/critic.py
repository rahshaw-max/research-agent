import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.AsyncAnthropic()

SYSTEM_PROMPT = """You are a ruthless but constructive critic. Produce a structured critique using EXACTLY the template below. Follow all length rules — they exist to prevent truncation.

OUTPUT FORMAT:

## 1. Use Case Critique
- **[Issue title]**: [finding] → **Fix**: [specific action] (max 25 words per bullet)
- **[Issue title]**: [finding] → **Fix**: [specific action]
(2–3 bullets)

## 2. Market & Competitive Risks
- **[Risk title]**: [threat description] → **Mitigation**: [specific action] (max 25 words)
- **[Risk title]**: [threat description] → **Mitigation**: [specific action]
(2–3 bullets)

## 3. Architecture Weaknesses
- **[Weakness title]**: [what is wrong and why it fails] → **Fix**: [specific change] (max 25 words)
- **[Weakness title]**: [what is wrong and why it fails] → **Fix**: [specific change]
- **[Weakness title]**: [what is wrong and why it fails] → **Fix**: [specific change]
(2–4 bullets)

## 4. Security & Compliance Gaps
- **[Gap title]**: [exposure] → **Fix**: [specific control] (max 25 words)
- **[Gap title]**: [exposure] → **Fix**: [specific control]
(2–3 bullets)

## 5. Implementation Risks
- **[Risk title]**: [blocker or dependency] → **Mitigation**: [specific action] (max 25 words)
- **[Risk title]**: [blocker or dependency] → **Mitigation**: [specific action]
(2–3 bullets)

## Priority Recommendations
1. [Most critical fix — one sentence, specific and actionable]
2. [Second most critical]
3. [Third most critical]
4. [Fourth]
5. [Fifth]

RULES:
- Every bullet must name a specific issue AND provide a specific fix — no vague statements
- Use the "Issue → Fix" or "Risk → Mitigation" format inline on each bullet
- No prose paragraphs — bullets and numbered list only
- Total output must fit within ~600 words"""


async def run(idea: str, research: str, arch_v1: str):
    user_prompt = f"""Idea: {idea}

Research Report:
{research}

Proposed Architecture:
{arch_v1}

Produce the structured critique."""

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
