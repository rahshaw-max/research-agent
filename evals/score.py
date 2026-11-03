#!/usr/bin/env python3
"""
Compute quality metrics for one eval run label.

Metrics per question:
  n_claims              factual bullets (verifier's definition)
  citation_rate         share of claims with an [E#] citation
  unique_sources        distinct URLs in the evidence list
  unique_domains        distinct registered domains
  primary_source_ratio  share of sources from regulators / .gov / .edu / journals
  numeric_claims_cited  share of digit-containing claims that carry a citation
  verified_claim_rate   verifier supported / total  (None if no verifier ran)
  wall_seconds          total wall time for the research cycle
  total_tokens          input + output tokens
  judge.*               LLM-as-judge scores 1-5 (specificity, factual_grounding,
                        completeness, decision_usefulness)

Usage:
    python evals/score.py --label head
    python evals/score.py --label baseline_74a47d0
"""
import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from agents.llm import tool_forced_call
from agents.config import MODEL_SYNTHESIS
from evals.domains import is_primary, domain_of

_EVALS_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Claim extraction (mirrors agents/verifier._extract_claims) ────────────────

def _extract_claims(report: str) -> list[dict]:
    claims = []
    in_sources = False
    for line in report.splitlines():
        s = line.strip()
        if s.startswith("## Sources"):
            in_sources = True
            continue
        if in_sources and s.startswith("## "):
            in_sources = False
        if in_sources:
            continue
        if not (s.startswith("- ") or s.startswith("* ")):
            continue
        content = s.lstrip("-* ").strip()
        if not content:
            continue
        cites = [int(c) for c in re.findall(r"\[E?(\d+)\]", content)]
        claims.append({"text": content, "has_citation": bool(cites)})
    return claims


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a rigorous research quality evaluator. You will receive:
  1. The research question (idea + requester context)
  2. The research report
  3. The evidence list (sources actually retrieved — empty for knowledge-only baselines)

Score the report on four dimensions using the submit_scores tool. Each dimension is scored 1–5:

**specificity**
  1 = only vague generalities, no numbers or named entities
  3 = some specific facts, companies, or figures
  5 = precise data (market sizes, percentages, dates, named vendors) for all key claims

**factual_grounding**
  1 = most claims unsupported; no citations or mismatched with evidence
  3 = major claims cited; some assertions unverified
  5 = every substantive claim carries an [E#] citation that matches the evidence list

**completeness**
  1 = major dimensions of the question are absent
  3 = core aspects covered; some angles thin or missing
  5 = market landscape, competitive analysis, technical depth, risks, and implementation all addressed

**decision_usefulness**
  1 = a decision-maker could not justify any budget decision from this
  3 = useful framing but lacks the specificity needed to act
  5 = concrete enough for a decision-maker to scope a project or select a vendor

Penalise heavily when:
  - The evidence list is empty but the report makes specific numerical claims
  - Citations are present but don't match any evidence list entry
  - The report is generic enough to apply to any company in the space

Provide a one-sentence justification for each score."""

_JUDGE_TOOL = {
    "name": "submit_scores",
    "description": "Submit quality scores for the research report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "specificity":              {"type": "integer", "minimum": 1, "maximum": 5},
            "specificity_reason":       {"type": "string"},
            "factual_grounding":        {"type": "integer", "minimum": 1, "maximum": 5},
            "factual_grounding_reason": {"type": "string"},
            "completeness":             {"type": "integer", "minimum": 1, "maximum": 5},
            "completeness_reason":      {"type": "string"},
            "decision_usefulness":      {"type": "integer", "minimum": 1, "maximum": 5},
            "decision_usefulness_reason": {"type": "string"},
        },
        "required": [
            "specificity", "specificity_reason",
            "factual_grounding", "factual_grounding_reason",
            "completeness", "completeness_reason",
            "decision_usefulness", "decision_usefulness_reason",
        ],
    },
}


class _JudgeScores(BaseModel):
    specificity: int
    specificity_reason: str
    factual_grounding: int
    factual_grounding_reason: str
    completeness: int
    completeness_reason: str
    decision_usefulness: int
    decision_usefulness_reason: str


async def _llm_judge(idea: str, answers: dict, report: str, evidence: list) -> dict:
    answers_text = "\n".join(f"- {k}: {v}" for k, v in answers.items()) or "(none)"
    ev_text = "\n".join(
        f"[E{i+1}] {e.get('title', '')} — {e.get('url', '')}"
        for i, e in enumerate(evidence[:60])
    ) or "(no evidence — knowledge-only run)"

    user_content = (
        f"Research question:\n{idea}\n\n"
        f"Requester context:\n{answers_text}\n\n"
        f"Evidence list:\n{ev_text}\n\n"
        f"Research report:\n{report}"
    )

    result = await tool_forced_call(
        model=MODEL_SYNTHESIS,
        max_tokens=1024,
        system=[{"type": "text", "text": _JUDGE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
        tool=_JUDGE_TOOL,
        schema=_JudgeScores,
    )
    return result.model_dump()


# ── Per-result scoring ────────────────────────────────────────────────────────

async def score_result(result: dict) -> dict:
    report   = result.get("report", "")
    evidence = result.get("evidence", [])
    vcounts  = result.get("verification_counts", {})
    timing   = result.get("timing", {})
    idea     = result.get("idea", "")
    answers  = result.get("answers", {})

    claims  = _extract_claims(report)
    n       = len(claims)
    cited   = sum(1 for c in claims if c["has_citation"])
    numeric = [c for c in claims if re.search(r"\d", c["text"])]
    num_cited = sum(1 for c in numeric if c["has_citation"])

    urls    = [e.get("url", "") for e in evidence if e.get("url")]
    uniq_urls = list(dict.fromkeys(urls))           # deduplicated, order-preserving
    domains   = [domain_of(u) for u in uniq_urls]
    uniq_dom  = list(dict.fromkeys(d for d in domains if d))

    primary_count = sum(1 for u in uniq_urls if is_primary(u))

    # Quote coverage: % of evidence items with a non-empty quote or excerpt
    items_with_text = sum(
        1 for e in evidence
        if e.get("quote") or e.get("retrieved_text_excerpt")
    )
    quote_coverage = round(items_with_text / max(len(evidence), 1), 3)

    total_v = vcounts.get("total", 0)
    verified_rate = (
        round(vcounts.get("supported", 0) / total_v, 3) if total_v else None
    )

    judge = {}
    if not result.get("error"):
        try:
            judge = await _llm_judge(idea, answers, report, evidence)
        except Exception as e:
            judge = {"error": str(e)}

    return {
        "id":                    result["id"],
        "n_claims":              n,
        "citation_rate":         round(cited / max(n, 1), 3),
        "unique_sources":        len(uniq_urls),
        "unique_domains":        len(uniq_dom),
        "primary_source_ratio":  round(primary_count / max(len(uniq_urls), 1), 3),
        "numeric_claims_cited":  round(num_cited / max(len(numeric), 1), 3),
        "verified_claim_rate":   verified_rate,
        "quote_coverage":        quote_coverage,
        "wall_seconds":          timing.get("wall_seconds", 0),
        "total_tokens":          (
            timing.get("total_input_tokens",  0) +
            timing.get("total_output_tokens", 0)
        ),
        "judge":                 judge,
        "error":                 result.get("error"),
    }


# ── Run scoring for a label ───────────────────────────────────────────────────

async def score_label(label: str) -> list[dict]:
    run_dir = os.path.join(_EVALS_DIR, "runs", label)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"No run directory: {run_dir}")

    result_files = sorted(f for f in os.listdir(run_dir) if f.endswith(".json")
                          and not f.endswith("_scores.json"))
    print(f"\nScoring {len(result_files)} results for label '{label}'...", flush=True)

    all_scores: list[dict] = []
    for fname in result_files:
        with open(os.path.join(run_dir, fname)) as f:
            result = json.load(f)
        print(f"  judging {result['id']}...", flush=True)
        scores = await score_result(result)
        all_scores.append(scores)

        score_path = os.path.join(run_dir, fname.replace(".json", "_scores.json"))
        with open(score_path, "w") as f:
            json.dump(scores, f, indent=2)

    # Print summary table
    _print_summary(label, all_scores)
    return all_scores


def _fmt(v) -> str:
    if v is None:    return "—"
    if isinstance(v, float): return f"{v:.3f}"
    return str(v)


def _print_summary(label: str, scores: list[dict]) -> None:
    ok = [s for s in scores if not s.get("error")]
    print(f"\n{'═'*70}")
    print(f"  SCORES — {label}  ({len(ok)}/{len(scores)} succeeded)")
    print(f"  {'ID':<25} {'claims':>6} {'cite%':>6} {'srcs':>5} {'doms':>5} "
          f"{'prim%':>6} {'num%':>6} {'ver%':>6} {'qtxt%':>6} {'spec':>5} {'grnd':>5} "
          f"{'comp':>5} {'decn':>5} {'tok':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*5} {'-'*5} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*7}")
    for s in scores:
        j = s.get("judge", {})
        print(
            f"  {s['id']:<25} {s['n_claims']:>6} "
            f"{s['citation_rate']:>6.1%} {s['unique_sources']:>5} {s['unique_domains']:>5} "
            f"{s['primary_source_ratio']:>6.1%} {s['numeric_claims_cited']:>6.1%} "
            f"{_fmt(s['verified_claim_rate']):>6} "
            f"{_fmt(s.get('quote_coverage')):>6} "
            f"{j.get('specificity','—'):>5} {j.get('factual_grounding','—'):>5} "
            f"{j.get('completeness','—'):>5} {j.get('decision_usefulness','—'):>5} "
            f"{s['total_tokens']:>7}"
        )
    if ok:
        def _mean(key, sub=None):
            vals = []
            for s in ok:
                v = s.get(key) if sub is None else (s.get(key) or {}).get(sub)
                if v is not None:
                    vals.append(v)
            return sum(vals) / len(vals) if vals else None
        print(f"  {'MEAN':<25} {_mean('n_claims'):>6.1f} "
              f"{_mean('citation_rate'):>6.1%} {_mean('unique_sources'):>5.1f} "
              f"{_mean('unique_domains'):>5.1f} "
              f"{_mean('primary_source_ratio'):>6.1%} {_mean('numeric_claims_cited'):>6.1%} "
              f"{_fmt(_mean('verified_claim_rate')):>6} "
              f"{_fmt(_mean('quote_coverage')):>6} "
              f"{_mean('judge','specificity') or 0:>5.1f} "
              f"{_mean('judge','factual_grounding') or 0:>5.1f} "
              f"{_mean('judge','completeness') or 0:>5.1f} "
              f"{_mean('judge','decision_usefulness') or 0:>5.1f} "
              f"{_mean('total_tokens'):>7.0f}")
    print(f"{'═'*70}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    asyncio.run(score_label(args.label))


if __name__ == "__main__":
    main()
