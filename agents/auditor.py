import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.AsyncAnthropic()

SYSTEM_PROMPT = """You are a presentation architect. You will receive a complete research and architecture dossier for a concept. Your task is to distill it into a structured JSON object that will be used to populate a PowerPoint presentation.

The JSON must conform EXACTLY to this schema:
{
  "title": "string — punchy 8-word max title for the concept",
  "subtitle": "string — one-line description with date, e.g. 'Market & Architecture Analysis  |  NOVEMBER 2026'",
  "executive_summary": "string — 2-3 sentence summary of the idea and its potential",
  "research_highlights": ["string", "string", "string"],
  "use_cases": [
    {"title": "string", "description": "string max 30 words"}
  ],
  "architecture_overview": "string — 2-3 sentence description of the final architecture",
  "architecture_components": [
    {"name": "string", "role": "string max 20 words"}
  ],
  "key_risks": [
    {"risk": "string max 15 words", "mitigation": "string max 20 words"}
  ],
  "critique_summary": "string — 2-3 sentences describing main weaknesses that were found",
  "revisions_made": ["string", "string"],
  "recommendation": "string — one strong concluding recommendation, max 40 words"
}

Constraints:
- research_highlights: 4-6 items, each max 25 words
- use_cases: 3-5 items
- architecture_components: 4-8 items
- key_risks: 3-5 items
- revisions_made: 3-5 bullets

Return ONLY valid JSON. No markdown fences. No explanation."""


def _format_answers(answers: dict) -> str:
    if not answers:
        return "No additional context provided."
    return "\n".join(f"- {k}: {v}" for k, v in answers.items())


async def run(
    idea: str,
    answers: dict,
    researcher: str,
    architect_v1: str,
    critic: str,
    architect_v2: str,
) -> dict:
    user_prompt = f"""Idea: {idea}

Requester Context:
{_format_answers(answers)}

Research Report:
{researcher}

Architecture v1:
{architect_v1}

Critique:
{critic}

Architecture v2 (Revised):
{architect_v2}

Generate the presentation JSON."""

    messages = [{"role": "user", "content": user_prompt}]

    for attempt in range(2):
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        text = response.content[0].text.strip()
        # Strip markdown fences if Claude added them
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == 0:
                messages.append({"role": "assistant", "content": response.content[0].text})
                messages.append({
                    "role": "user",
                    "content": "The JSON you returned was invalid. Please return ONLY valid JSON with no markdown fences or explanation.",
                })
            else:
                raise ValueError(f"Auditor failed to produce valid JSON after 2 attempts. Last output: {text[:500]}")
