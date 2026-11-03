#!/usr/bin/env python3
"""
Print per-sub-question critic scores next to deterministic metrics for calibration.

Loads up to 3 saved result files from evals/runs/<label>/ and prints a table
so you can hand-score a few and check the model's numbers against your own judgment.

Usage:
    python evals/calibration.py --label head
    python evals/calibration.py --label head --n 3
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents import critic
from evals.domains import is_primary, domain_of

_EVALS_DIR = os.path.dirname(os.path.abspath(__file__))


def _deterministic_stats(evidence: list[dict]) -> dict:
    """Compute domain/primary stats from the evidence list (no API calls)."""
    urls = [e.get("url", "") for e in evidence if e.get("url")]
    uniq_urls = list(dict.fromkeys(urls))
    domains = list(dict.fromkeys(domain_of(u) for u in uniq_urls if domain_of(u)))
    primary_count = sum(1 for u in uniq_urls if is_primary(u))
    items_with_text = sum(1 for e in evidence if e.get("quote") or e.get("retrieved_text_excerpt"))
    return {
        "n_urls":        len(uniq_urls),
        "n_domains":     len(domains),
        "primary_count": primary_count,
        "primary_ratio": round(primary_count / max(len(uniq_urls), 1), 3),
        "quote_coverage": round(items_with_text / max(len(evidence), 1), 3),
    }


async def calibrate_result(result: dict) -> None:
    """Run critique on one saved result and print calibration table."""
    idea        = result.get("idea", "")
    answers     = result.get("answers", {})
    report      = result.get("report", "")
    evidence    = result.get("evidence", [])
    sub_questions = result.get("sub_questions", [])

    # If sub_questions weren't saved, try to pull them from gap_report
    if not sub_questions:
        gap = result.get("gap_report", {})
        sub_questions = [
            s.get("sub_question", "") for s in gap.get("sub_question_scores", [])
        ]

    vcounts     = result.get("verification_counts", {})
    uncited_share = vcounts.get("unsupported", 0) / max(vcounts.get("total", 0), 1)

    det = _deterministic_stats(evidence)

    print(f"\n{'━' * 70}")
    print(f"  ID: {result.get('id', '?')}")
    print(f"  Idea: {idea[:80]}")
    print(f"\n  Deterministic evidence stats:")
    print(f"    unique URLs:     {det['n_urls']}")
    print(f"    unique domains:  {det['n_domains']}")
    print(f"    primary sources: {det['primary_count']}  ({det['primary_ratio']:.0%})")
    print(f"    quote coverage:  {det['quote_coverage']:.0%}")
    print(f"    uncited share:   {uncited_share:.0%}  (verifier unsupported / total)")
    print(f"    sub-questions:   {len(sub_questions)}")
    print(f"  Running MODEL_CRITIQUE ({critic.MODEL_CRITIQUE}) critique...", flush=True)

    try:
        gap_report = await critic.run_research_critique(
            idea, report, answers,
            sub_questions=sub_questions,
            evidence=evidence,
            uncited_share=uncited_share,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    computed_score = gap_report.get("coverage_score", 0.0)
    sub_scores = gap_report.get("sub_question_scores", [])

    print(f"\n  Computed coverage_score (deterministic caps applied): {computed_score:.3f}")
    print(f"  COVERAGE_THRESHOLD: {critic.COVERAGE_THRESHOLD:.2f}  "
          f"({'PASS ✓' if computed_score >= critic.COVERAGE_THRESHOLD else 'FAIL ✗'})")
    print(f"  Config: domain_cap={critic.DOMAIN_CAP} (<{critic.MIN_DOMAINS_FOR_FULL_SCORE} domains), "
          f"no_primary_cap={critic.NO_PRIMARY_CAP}, uncited_weight={critic.UNCITED_PENALTY_WEIGHT}")

    print(f"\n  Per-sub-question scores:")
    header = f"  {'#':>3}  {'Score':>6}  {'→Capped':>8}  {'Doms':>5}  {'PrimSrc':>7}  Sub-question"
    print(header)
    print("  " + "-" * 80)

    for i, s in enumerate(sub_scores, 1):
        raw = float(s.get("score", 0))
        n_dom = s.get("n_independent_domains", 0)
        has_prim = s.get("has_primary_source", False)

        # Apply same caps as _compute_coverage_score
        capped = raw
        if n_dom < critic.MIN_DOMAINS_FOR_FULL_SCORE:
            capped = min(capped, critic.DOMAIN_CAP)
        if not has_prim:
            capped = min(capped, critic.NO_PRIMARY_CAP)

        subq_short = (s.get("sub_question", "") or "")[:55]
        flag_dom  = " ← dom_cap" if n_dom < critic.MIN_DOMAINS_FOR_FULL_SCORE and raw > capped else ""
        flag_prim = " ← no_prim_cap" if not has_prim and raw > capped else ""
        cap_str   = f"{capped:.2f}{flag_dom}{flag_prim}"
        prim_sym  = "✓" if has_prim else "✗"
        print(f"  {i:>3}  {raw:>6.2f}  {cap_str:<16}  {n_dom:>5}  {prim_sym:>7}  {subq_short}")
        if s.get("rationale"):
            for line in s["rationale"].split(". "):
                line = line.strip()
                if line:
                    print(f"       ↳ {line[:72]}")

    print(f"\n  Gaps ({len(gap_report.get('gaps', []))}):")
    for g in gap_report.get("gaps", []):
        print(f"    [{g.get('severity',0)}] {g.get('type','?'):22}  {(g.get('description',''))[:60]}")

    print(f"\n  ← Your hand-score (enter 0.0–1.0, or press Enter to skip): ", end="", flush=True)
    try:
        hand = input().strip()
        if hand:
            print(f"  Hand: {float(hand):.2f}  |  Model: {computed_score:.3f}  |  Δ: {float(hand) - computed_score:+.3f}")
    except (EOFError, ValueError):
        pass


async def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate per-sub-question critic scores")
    parser.add_argument("--label", required=True, help="Run label (directory under evals/runs/)")
    parser.add_argument("--n", type=int, default=3, help="Max number of results to calibrate (default 3)")
    args = parser.parse_args()

    run_dir = os.path.join(_EVALS_DIR, "runs", args.label)
    if not os.path.isdir(run_dir):
        print(f"No run directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        f for f in os.listdir(run_dir)
        if f.endswith(".json") and not f.endswith("_scores.json")
    )[: args.n]

    if not files:
        print(f"No result files in {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Calibrating {len(files)} result(s) from '{args.label}'")
    for fname in files:
        with open(os.path.join(run_dir, fname)) as f:
            result = json.load(f)
        await calibrate_result(result)

    print(f"\n{'━' * 70}\nDone.\n")


if __name__ == "__main__":
    asyncio.run(main())
