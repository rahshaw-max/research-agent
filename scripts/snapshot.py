#!/usr/bin/env python3
"""Headless pipeline snapshot — no WebSocket, no HITL, auto-approve.

Cycle loop (repeats N times):
  researcher → [research_critique ‖ verifier] (concurrent) → interrogator

After all cycles (once):
  ARCH_MODE=lite : architect (single pass) → auditor → pptx
  ARCH_MODE=full : architect_v1 → arch_critic → architect_v2 → auditor → pptx

Saves to: snapshots/<label>_<timestamp>/

Usage:
    python scripts/snapshot.py --idea "Your idea here" --label "baseline"
    python scripts/snapshot.py --idea "Your idea here" --label "step5" --depth standard
    python scripts/snapshot.py --idea "..." --label "deep" --depth deep --cycles 2
    ARCH_MODE=full python scripts/snapshot.py --idea "..." --label "full"
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents import researcher, architect, critic, auditor, interrogator, verifier
from pptx_builder import builder

ARCH_MODE = os.getenv("ARCH_MODE", "lite")

DEPTH_CONFIGS = {
    "quick":    {"max_subq": 3, "searches": 2, "fetches": 1, "max_cycles": 1, "arch_mode": "lite"},
    "standard": {"max_subq": 6, "searches": 3, "fetches": 2, "max_cycles": 3, "arch_mode": "lite"},
    "deep":     {"max_subq": 8, "searches": 4, "fetches": 3, "max_cycles": 3, "arch_mode": "full"},
}


def _banner(title: str) -> None:
    print(f"\n{'─' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─' * 60}", flush=True)


async def _collect(label: str, generator, _evidence: list | None = None) -> str:
    """Stream an async generator to stdout and return the full text.

    Handles special sentinel dicts:
      {"__type__": "status"}        — print status line
      {"__type__": "citation_stat"} — print citation rate
      {"__type__": "report_final"}  — use as authoritative return value
    """
    _banner(label)
    text = ""
    final_text: str | None = None
    async for chunk in generator:
        if isinstance(chunk, dict):
            t = chunk.get("__type__")
            if t == "status":
                print(f"\n  [{chunk['message']}]", flush=True)
            elif t == "citation_stat":
                cited, total = chunk.get("cited", 0), chunk.get("total", 0)
                pct = 100 * cited // total if total else 0
                print(f"\n  [citation rate: {cited}/{total} ({pct}%)]", flush=True)
            elif t == "report_final":
                final_text = chunk.get("text", "")
        else:
            print(chunk, end="", flush=True)
            text += chunk
    print(flush=True)
    return final_text if final_text is not None else text


def _count_sources(researcher_md: str) -> list[str]:
    """Extract unique URLs from the ## Sources section of a researcher report."""
    import re
    urls = []
    in_sources = False
    for line in researcher_md.splitlines():
        if line.strip().startswith("## Sources"):
            in_sources = True
            continue
        if in_sources:
            if line.startswith("## "):
                break
            match = re.search(r"https?://\S+", line)
            if match:
                url = match.group(0).rstrip(")")
                if url not in urls:
                    urls.append(url)
    return urls


def _stage_row(stage: str, cycle: int, seconds: float, in_tok: int, out_tok: int, tc: int) -> str:
    cyc = f"c{cycle}" if cycle > 0 else "  "
    return (
        f"  {stage:<20} {cyc:<4}  {seconds:>7.1f}s  "
        f"{in_tok:>8} in  {out_tok:>7} out  {tc:>4} calls"
    )


async def run_snapshot(idea: str, label: str, out_dir: str, cycles: int = 1, depth: str = "standard") -> None:
    answers: dict = {}
    depth_cfg = DEPTH_CONFIGS.get(depth, DEPTH_CONFIGS["standard"])
    arch_mode = os.getenv("ARCH_MODE", depth_cfg["arch_mode"])

    print(f"\nIdea      : {idea}")
    print(f"Label     : {label}")
    print(f"Depth     : {depth}  (max_subq={depth_cfg['max_subq']}, searches={depth_cfg['searches']}, fetches={depth_cfg['fetches']})")
    print(f"Cycles    : {cycles}")
    print(f"ARCH_MODE : {arch_mode}")
    print(f"Output    : {out_dir}")

    final_research: str | None = None
    prev_coverage_score: float | None = None
    coverage_scores: list[float] = []
    last_gap_report: dict = {}
    timing_rows: list[dict] = []

    # ── CYCLE LOOP ────────────────────────────────────────────────────────────
    for cycle in range(1, cycles + 1):
        _banner(f"CYCLE {cycle} of {cycles}")

        gap_queries: list[str] | None = None
        if cycle > 1 and last_gap_report:
            gap_queries = []
            for gap in last_gap_report.get("gaps", []):
                gap_queries.extend(gap.get("suggested_queries", []))

        # ── Researcher ────────────────────────────────────────────────────────
        # Carry prior evidence forward so new IDs continue from previous cycle
        ru: dict = {}
        if cycle == 1:
            cycle_evidence: list = []
        # else: cycle_evidence already has prior evidence from cycle N-1
        t0 = time.perf_counter()
        final_research = await _collect(
            f"RESEARCHER (cycle {cycle})",
            researcher.run(
                idea, answers,
                prev_research=final_research if cycle > 1 else None,
                gap_queries=gap_queries,
                _usage=ru,
                _evidence=cycle_evidence,
                max_subq=depth_cfg["max_subq"],
                searches_per_subq=depth_cfg["searches"],
                fetches_per_subq=depth_cfg["fetches"],
            ),
        )
        timing_rows.append({"stage": "researcher", "cycle": cycle,
                             "seconds": round(time.perf_counter() - t0, 2),
                             "evidence_count": len(cycle_evidence), **ru})

        # ── Research quality audit + Verifier (concurrent) ───────────────────
        _banner(f"RESEARCH CRITIQUE + VERIFIER (cycle {cycle}) — concurrent")
        rcu: dict = {}
        vu: dict = {}
        t0_parallel = time.perf_counter()
        [last_gap_report, verify_result] = await asyncio.gather(
            critic.run_research_critique(idea, final_research, answers, _usage=rcu),
            verifier.run(final_research, idea, answers, evidence=cycle_evidence, _usage=vu),
        )
        t_parallel = round(time.perf_counter() - t0_parallel, 2)
        for stage, u in [("research_critique", rcu), ("verifier", vu)]:
            timing_rows.append({"stage": stage, "cycle": cycle, "seconds": t_parallel, **u})

        coverage_score: float = last_gap_report.get("coverage_score", 0.0)
        coverage_scores.append(coverage_score)

        print(f"\n  coverage_score = {coverage_score:.3f}", flush=True)
        if prev_coverage_score is not None:
            delta = coverage_score - prev_coverage_score
            flag = "✓ improved" if delta >= 0.03 else "⚠ did not improve by ≥0.03"
            print(f"  delta vs cycle {cycle-1} = {delta:+.3f}  {flag}", flush=True)
        prev_coverage_score = coverage_score

        gaps = last_gap_report.get("gaps") or []
        if not gaps and coverage_score < 0.85:
            print(f"  ⚠ WARNING: coverage_score={coverage_score:.2f} but gaps list is empty "
                  f"— likely truncated response (max_tokens too low)", flush=True)
        print(f"  gaps = {len(gaps)}", flush=True)
        for g in gaps:
            print(f"    [{g['severity']}] {g['type']}: {g['description'][:80]}", flush=True)

        final_research = verify_result["report"]
        vc = verify_result["counts"]
        print(
            f"  claims verified: {vc['total']}  "
            f"supported={vc['supported']}  partial={vc['partial']}  unsupported={vc['unsupported']}",
            flush=True,
        )
        if vc["repaired"] or vc["removed"]:
            print(f"  repaired={vc['repaired']}  removed={vc['removed']}", flush=True)

        # ── Interrogator (from gap report, not architecture) ──────────────────
        _banner(f"INTERROGATOR (cycle {cycle})")
        iu: dict = {}
        t0 = time.perf_counter()
        questions = await interrogator.run(idea, final_research, last_gap_report, _usage=iu)
        timing_rows.append({"stage": "interrogator", "cycle": cycle,
                             "seconds": round(time.perf_counter() - t0, 2), **iu})
        print(f"  {len(questions)} questions generated", flush=True)
        for i, q in enumerate(questions, 1):
            print(f"    {i}. {q[:100]}", flush=True)

    # ── Check coverage improvement between cycles ─────────────────────────────
    if len(coverage_scores) >= 2:
        print(f"\n{'─' * 60}", flush=True)
        print("  COVERAGE SCORE PROGRESSION", flush=True)
        for i, s in enumerate(coverage_scores, 1):
            print(f"    Cycle {i}: {s:.3f}", flush=True)
        overall_delta = coverage_scores[-1] - coverage_scores[0]
        status = "✓ improved" if overall_delta >= 0.03 else "✗ did NOT improve by ≥0.03 — investigate"
        print(f"  Total delta: {overall_delta:+.3f}  {status}", flush=True)
        print(f"{'─' * 60}", flush=True)

    # ── PHASE 2: Architecture (once, after cycles) ────────────────────────────
    if arch_mode == "lite":
        au1: dict = {}
        t0 = time.perf_counter()
        final_arch_v1 = await _collect(
            "ARCHITECT (lite)",
            architect.run_lite(idea, final_research, last_gap_report, _usage=au1),
        )
        timing_rows.append({"stage": "architect_v1", "cycle": 0,
                             "seconds": round(time.perf_counter() - t0, 2), **au1})
        final_arch_critique = ""
        arch_v2_out = final_arch_v1

    else:  # full
        au1: dict = {}
        t0 = time.perf_counter()
        final_arch_v1 = await _collect(
            "ARCHITECT v1",
            architect.run(idea, final_research, _usage=au1),
        )
        timing_rows.append({"stage": "architect_v1", "cycle": 0,
                             "seconds": round(time.perf_counter() - t0, 2), **au1})

        acu: dict = {}
        t0 = time.perf_counter()
        final_arch_critique = await _collect(
            "ARCH CRITIC",
            critic.run(idea, final_research, final_arch_v1, _usage=acu),
        )
        timing_rows.append({"stage": "arch_critic", "cycle": 0,
                             "seconds": round(time.perf_counter() - t0, 2), **acu})

        av2u: dict = {}
        t0 = time.perf_counter()
        arch_v2_out = await _collect(
            "ARCHITECT v2",
            architect.run_revision(idea, final_research, final_arch_v1, final_arch_critique, _usage=av2u),
        )
        timing_rows.append({"stage": "architect_v2", "cycle": 0,
                             "seconds": round(time.perf_counter() - t0, 2), **av2u})

    _banner("AUDITOR — building presentation JSON")
    adtu: dict = {}
    t0 = time.perf_counter()
    pptx_data = await auditor.run(
        idea,
        answers,
        researcher=final_research,
        architect_v1=final_arch_v1,
        critic=final_arch_critique,
        architect_v2=arch_v2_out,
        evidence=cycle_evidence,
        _usage=adtu,
    )
    timing_rows.append({"stage": "auditor", "cycle": 0,
                         "seconds": round(time.perf_counter() - t0, 2), **adtu})
    print(json.dumps(pptx_data, indent=2), flush=True)

    # ── Build PPTX ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    tmp_path = builder.build_presentation("snapshot-tmp", pptx_data, out_dir)
    pptx_elapsed = round(time.perf_counter() - t0, 2)
    timing_rows.append({"stage": "pptx_build", "cycle": 0, "seconds": pptx_elapsed,
                         "input_tokens": 0, "output_tokens": 0, "tool_calls": 0})
    deck_path = os.path.join(out_dir, "deck.pptx")
    shutil.move(tmp_path, deck_path)

    # ── Persist outputs ───────────────────────────────────────────────────────
    files = {
        "researcher.md":   final_research,
        "architect_v1.md": final_arch_v1,
        "critic.md":       final_arch_critique,
        "architect_v2.md": arch_v2_out,
    }
    for filename, content in files.items():
        with open(os.path.join(out_dir, filename), "w") as f:
            f.write(content)

    with open(os.path.join(out_dir, "pptx_data.json"), "w") as f:
        json.dump(pptx_data, f, indent=2)

    if last_gap_report:
        with open(os.path.join(out_dir, "gap_report.json"), "w") as f:
            json.dump(last_gap_report, f, indent=2)

    # ── Save timing JSONL ─────────────────────────────────────────────────────
    timing_path = os.path.join(out_dir, "timing.jsonl")
    with open(timing_path, "w") as f:
        for row in timing_rows:
            f.write(json.dumps(row) + "\n")

    sources = _count_sources(final_research)
    meta = {
        "idea": idea,
        "label": label,
        "cycles": cycles,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_dir": out_dir,
        "coverage_scores": coverage_scores,
        "files": list(files.keys()) + ["pptx_data.json", "gap_report.json", "deck.pptx", "timing.jsonl"],
        "unique_sources": len(sources),
        "sources": sources,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Print timing table ────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  STAGE TIMING")
    print(f"  {'Stage':<20} {'Cyc':<4}  {'Time':>8}  {'In tok':>8}  {'Out tok':>7}  {'Tools':>5}  {'Evid':>5}")
    print(f"  {'─'*20} {'─'*4}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*5}")
    for row in timing_rows:
        cyc = f"c{row['cycle']}" if row['cycle'] > 0 else "  "
        ev = row.get("evidence_count", "")
        ev_str = f"{ev:>5}" if isinstance(ev, int) else "     "
        print(
            f"  {row['stage']:<20} {cyc:<4}  {row.get('seconds', 0):>7.1f}s  "
            f"{row.get('input_tokens', 0):>8}  {row.get('output_tokens', 0):>7}  "
            f"{row.get('tool_calls', 0):>5}  {ev_str}",
            flush=True,
        )
    total_sec = sum(r.get("seconds", 0) for r in timing_rows)
    total_in  = sum(r.get("input_tokens", 0) for r in timing_rows)
    total_out = sum(r.get("output_tokens", 0) for r in timing_rows)
    total_tc  = sum(r.get("tool_calls", 0) for r in timing_rows)
    print(f"  {'─'*20} {'─'*4}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*5}")
    print(f"  {'TOTAL':<20} {'':4}  {total_sec:>7.1f}s  {total_in:>8}  {total_out:>7}  {total_tc:>5}")
    print(f"{'═' * 72}")

    print(f"\n{'═' * 60}")
    print(f"  Snapshot complete")
    print(f"  Directory    : {out_dir}")
    print(f"  Cycles run   : {cycles}")
    print(f"  Coverage     : {' → '.join(f'{s:.2f}' for s in coverage_scores)}")
    print(f"  Unique sources cited: {len(sources)}")
    if sources:
        for i, url in enumerate(sources, 1):
            print(f"    [{i}] {url}")
    print(f"{'═' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the research pipeline headlessly and save a snapshot."
    )
    parser.add_argument("--idea",   required=True, help="The idea to research")
    parser.add_argument("--label",  required=True, help="Label prefix for the snapshot directory")
    parser.add_argument("--cycles", type=int, default=1,
                        help="Number of research-critique cycles to run (default: 1)")
    parser.add_argument("--depth", choices=["quick", "standard", "deep"], default="standard",
                        help="Research depth preset (default: standard)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("snapshots", f"{args.label}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    asyncio.run(run_snapshot(args.idea, args.label, out_dir, cycles=args.cycles, depth=args.depth))


if __name__ == "__main__":
    main()
