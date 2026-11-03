#!/usr/bin/env python3
"""Headless evaluation runner.

Runs the research pipeline (researcher → [critique ‖ verifier]) for each question
in questions.json and saves per-question artifacts to evals/runs/<label>/<id>.json.

Reuses the same agent calls as scripts/snapshot.py (research cycle only, no arch/pptx).

Usage:
    python evals/run_eval.py --label head
    python evals/run_eval.py --label baseline_74a47d0 --adapter baseline
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents import researcher, critic, verifier

DEPTH_QUICK = {"max_subq": 3, "searches": 2, "fetches": 1}
CONCURRENCY = 2

_EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
_QUESTIONS  = os.path.join(_EVALS_DIR, "questions.json")


async def _drain(generator) -> str:
    """Consume researcher's async generator; return the final report text."""
    text = ""
    final: str | None = None
    async for chunk in generator:
        if isinstance(chunk, dict):
            if chunk.get("__type__") == "report_final":
                final = chunk.get("text", "")
        else:
            text += chunk
    return final if final is not None else text


async def _run_head(q: dict) -> dict:
    """Full HEAD pipeline: researcher → [critique ‖ verifier]."""
    idea    = q["idea"]
    answers = q.get("answers", {})
    t0_wall = time.perf_counter()
    stages: list[dict] = []

    evidence: list = []
    ru: dict = {}
    t0 = time.perf_counter()
    report = await _drain(researcher.run(
        idea, answers,
        _usage=ru,
        _evidence=evidence,
        max_subq=DEPTH_QUICK["max_subq"],
        searches_per_subq=DEPTH_QUICK["searches"],
        fetches_per_subq=DEPTH_QUICK["fetches"],
    ))
    stages.append({"stage": "researcher", "seconds": round(time.perf_counter() - t0, 2), **ru})

    rcu: dict = {}
    vu:  dict = {}
    t0 = time.perf_counter()
    [gap_report, verify_result] = await asyncio.gather(
        critic.run_research_critique(idea, report, answers, _usage=rcu),
        verifier.run(report, idea, answers, evidence=evidence, _usage=vu),
    )
    t_par = round(time.perf_counter() - t0, 2)
    stages.append({"stage": "research_critique", "seconds": t_par, **rcu})
    stages.append({"stage": "verifier",          "seconds": t_par, **vu})
    report = verify_result["report"]

    return {
        "id":                   q["id"],
        "idea":                 idea,
        "answers":              answers,
        "report":               report,
        "evidence":             evidence,
        "gap_report":           gap_report,
        "verification_counts":  verify_result["counts"],
        "timing": {
            "wall_seconds":       round(time.perf_counter() - t0_wall, 2),
            "stages":             stages,
            "total_input_tokens": sum(s.get("input_tokens",  0) for s in stages),
            "total_output_tokens":sum(s.get("output_tokens", 0) for s in stages),
            "total_tool_calls":   sum(s.get("tool_calls",    0) for s in stages),
        },
        "error": None,
    }


async def _run_baseline(q: dict) -> dict:
    """Baseline adapter: calls 74a47d0 researcher (no tools, no evidence)."""
    from evals.adapters.baseline import run as _baseline

    idea    = q["idea"]
    answers = q.get("answers", {})
    t0_wall = time.perf_counter()

    report, evidence = await _baseline(idea, answers)

    return {
        "id":                   q["id"],
        "idea":                 idea,
        "answers":              answers,
        "report":               report,
        "evidence":             evidence,
        "gap_report":           {},
        "verification_counts":  {
            "total": 0, "supported": 0, "partial": 0,
            "unsupported": 0, "repaired": 0, "removed": 0,
        },
        "timing": {
            "wall_seconds":        round(time.perf_counter() - t0_wall, 2),
            "stages":              [],
            "total_input_tokens":  0,
            "total_output_tokens": 0,
            "total_tool_calls":    0,
        },
        "error": None,
    }


async def _run_one(q: dict, adapter: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        print(f"  → {q['id']}...", flush=True)
        try:
            result = await (_run_head(q) if adapter == "head" else _run_baseline(q))
            t = result["timing"]["wall_seconds"]
            print(f"  ✓ {q['id']}  ({t:.1f}s)", flush=True)
        except Exception as exc:
            print(f"  ✗ {q['id']}  ERROR: {exc}", flush=True)
            result = {
                "id": q["id"], "idea": q["idea"], "answers": q.get("answers", {}),
                "report": "", "evidence": [], "gap_report": {},
                "verification_counts": {
                    "total": 0, "supported": 0, "partial": 0,
                    "unsupported": 0, "repaired": 0, "removed": 0,
                },
                "timing": {
                    "wall_seconds": 0, "stages": [],
                    "total_input_tokens": 0, "total_output_tokens": 0, "total_tool_calls": 0,
                },
                "error": str(exc),
            }
    return result


async def run_eval(label: str, adapter: str) -> None:
    with open(_QUESTIONS) as f:
        questions = json.load(f)

    out_dir = os.path.join(_EVALS_DIR, "runs", label)
    os.makedirs(out_dir, exist_ok=True)

    sem     = asyncio.Semaphore(CONCURRENCY)
    tasks   = [_run_one(q, adapter, sem) for q in questions]

    print(f"\nRunning {len(questions)} questions  adapter={adapter}  concurrency={CONCURRENCY}", flush=True)
    results = await asyncio.gather(*tasks)

    for result in results:
        path = os.path.join(out_dir, f"{result['id']}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    ok   = sum(1 for r in results if not r["error"])
    fail = sum(1 for r in results if r["error"])
    print(f"\nDone: {ok} OK, {fail} failed → {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",   required=True, help="Run label (used as directory name)")
    parser.add_argument("--adapter", default="head", choices=["head", "baseline"],
                        help="'head' = current pipeline; 'baseline' = 74a47d0 researcher")
    args = parser.parse_args()
    asyncio.run(run_eval(args.label, args.adapter))


if __name__ == "__main__":
    main()
