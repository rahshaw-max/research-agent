#!/usr/bin/env python3
"""Headless pipeline snapshot — no WebSocket, no HITL, auto-approve, one cycle.

Runs: researcher → architect_v1 → critic → [auto-approve] → architect_v2 → auditor → pptx
Saves to: snapshots/<label>_<timestamp>/

Usage:
    python scripts/snapshot.py --idea "Your idea here" --label "baseline"
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime, timezone

# Resolve imports from the project root regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents import researcher, architect, critic, auditor
from pptx_builder import builder


def _banner(title: str) -> None:
    print(f"\n{'─' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─' * 60}", flush=True)


async def _collect(label: str, generator) -> str:
    """Stream an async generator to stdout and return the full text."""
    _banner(label)
    text = ""
    async for chunk in generator:
        print(chunk, end="", flush=True)
        text += chunk
    print(flush=True)
    return text


async def run_snapshot(idea: str, label: str, out_dir: str) -> None:
    answers: dict = {}  # no clarifying Q&A in headless mode

    print(f"\nIdea  : {idea}")
    print(f"Label : {label}")
    print(f"Output: {out_dir}")

    # ── Phase 1 ────────────────────────────────────────────────────────────
    researcher_out = await _collect(
        "RESEARCHER",
        researcher.run(idea, answers),
    )

    arch_v1_out = await _collect(
        "ARCHITECT v1",
        architect.run(idea, researcher_out),
    )

    critic_out = await _collect(
        "CRITIC",
        critic.run(idea, researcher_out, arch_v1_out),
    )

    print(f"\n[Auto-approved — HITL skipped]", flush=True)

    # ── Phase 2 ────────────────────────────────────────────────────────────
    arch_v2_out = await _collect(
        "ARCHITECT v2",
        architect.run_revision(idea, researcher_out, arch_v1_out, critic_out),
    )

    _banner("AUDITOR — building presentation JSON")
    pptx_data = await auditor.run(
        idea,
        answers,
        researcher=researcher_out,
        architect_v1=arch_v1_out,
        critic=critic_out,
        architect_v2=arch_v2_out,
    )
    print(json.dumps(pptx_data, indent=2), flush=True)

    # ── Build PPTX ─────────────────────────────────────────────────────────
    tmp_path = builder.build_presentation("snapshot-tmp", pptx_data, out_dir)
    deck_path = os.path.join(out_dir, "deck.pptx")
    shutil.move(tmp_path, deck_path)

    # ── Persist outputs ────────────────────────────────────────────────────
    files = {
        "researcher.md":    researcher_out,
        "architect_v1.md":  arch_v1_out,
        "critic.md":        critic_out,
        "architect_v2.md":  arch_v2_out,
    }
    for filename, content in files.items():
        with open(os.path.join(out_dir, filename), "w") as f:
            f.write(content)

    with open(os.path.join(out_dir, "pptx_data.json"), "w") as f:
        json.dump(pptx_data, f, indent=2)

    meta = {
        "idea": idea,
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_dir": out_dir,
        "files": list(files.keys()) + ["pptx_data.json", "deck.pptx"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  Snapshot complete")
    print(f"  Directory : {out_dir}")
    print(f"  Files     : {', '.join(meta['files'])}")
    print(f"{'═' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the research pipeline headlessly and save a snapshot."
    )
    parser.add_argument("--idea",  required=True, help="The idea to research")
    parser.add_argument("--label", required=True, help="Label prefix for the snapshot directory")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("snapshots", f"{args.label}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    asyncio.run(run_snapshot(args.idea, args.label, out_dir))


if __name__ == "__main__":
    main()
