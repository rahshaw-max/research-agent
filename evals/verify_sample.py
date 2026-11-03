#!/usr/bin/env python3
"""
Run one eval question and print 5 randomly sampled cited claims with their
evidence quotes and verifier verdicts — to confirm verification is real.
"""
import asyncio
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents import researcher, verifier

QUESTION = {
    "id": "invoice_smb",
    "idea": "AI-powered invoice processing for SMBs",
    "answers": {
        "audience": "CFOs and finance operations managers at small businesses (10-200 employees)",
        "goal": "Feasibility and vendor comparison for replacing manual AP with AI automation",
        "constraints": "QuickBooks/Xero integration required, under $500/month budget, minimal IT overhead",
        "focus_areas": "OCR accuracy, ERP integration, pricing models, leading vendors (Bill.com, Stampli, Rossum)",
    },
}

DEPTH = {"max_subq": 3, "searches": 2, "fetches": 1}


async def main():
    idea    = QUESTION["idea"]
    answers = QUESTION["answers"]

    print(f"\nRunning researcher for: {idea}", flush=True)
    evidence: list = []
    usage: dict = {}

    report = ""
    async for chunk in researcher.run(
        idea, answers,
        _usage=usage,
        _evidence=evidence,
        max_subq=DEPTH["max_subq"],
        searches_per_subq=DEPTH["searches"],
        fetches_per_subq=DEPTH["fetches"],
    ):
        if isinstance(chunk, dict):
            if chunk.get("__type__") == "status":
                print(f"  [status] {chunk['message']}", flush=True)
            elif chunk.get("__type__") == "report_final":
                report = chunk["text"]
        else:
            pass  # streaming text, collect via report_final

    if not report:
        print("No report generated.")
        return

    print(f"\nEvidence collected: {len(evidence)} items", flush=True)
    text_count = sum(1 for e in evidence if e.get("quote") or e.get("retrieved_text_excerpt"))
    print(f"  → {text_count} with text ({100*text_count//max(len(evidence),1)}%)", flush=True)

    # Show a sample of evidence items
    print("\n── Evidence sample (first 3 items with text) ─────────────────────────")
    shown = 0
    for e in evidence:
        if not (e.get("quote") or e.get("retrieved_text_excerpt")):
            continue
        text = e.get("quote") or e.get("retrieved_text_excerpt", "")[:200]
        print(f"\n[{e['id']}] {e['title']}")
        print(f"    URL: {e['url']}")
        print(f"    type: {e.get('source_type', '?')}")
        print(f"    text: {text[:200]!r}")
        shown += 1
        if shown >= 3:
            break

    # Run verifier
    print(f"\nRunning verifier...", flush=True)
    result = await verifier.run(report, idea, answers, evidence=evidence)
    counts = result["counts"]
    print(f"  Verdicts: {counts['supported']} supported / {counts['partial']} partial / "
          f"{counts['unsupported']} unsupported (total {counts['total']})", flush=True)

    # Extract cited claims with their verdicts (we need to re-run verifier logic partially)
    # Re-extract claims and match with verdicts by text
    from agents.verifier import _extract_claims, _build_evidence_map
    claims = _extract_claims(report)
    ev_map = _build_evidence_map(evidence, report)

    # Build verdict lookup by running a small fresh batch
    cited_claims = [c for c in claims if c["has_citation"]]
    random.shuffle(cited_claims)
    sample = cited_claims[:5]

    print("\n── 5 sampled cited claims with evidence text and verdicts ──────────────")
    for i, claim in enumerate(sample, 1):
        print(f"\n[Claim {i}]")
        print(f"  Text: {claim['text'][:200]}")
        print(f"  Cites: {claim['citations']}")
        for cn in claim["citations"]:
            ev = ev_map.get(cn)
            if ev:
                text = ev.get("text", "")
                print(f"  [E{cn}] {ev['title']}")
                print(f"         URL: {ev['url']}")
                if text:
                    print(f"         Evidence text: {text[:250]!r}")
                else:
                    print(f"         Evidence text: (none — title/URL only)")

    # Run verification on just the sample to get verdicts with supporting_quote
    if sample:
        from agents.verifier import _verify_batch
        print("\nRunning targeted verification on sample...")
        verdicts = await _verify_batch(sample, ev_map)
        for v in verdicts:
            print(f"\n  Claim: {v['claim_text'][:120]}")
            print(f"  Verdict: {v['verdict']}")
            print(f"  Reason: {v['reason']}")
            sq = v.get("supporting_quote", "")
            if sq:
                print(f"  Supporting quote: {sq!r}")

    # Save full result for inspection
    out = {
        "id":       QUESTION["id"],
        "report":   report,
        "evidence": evidence,
        "counts":   counts,
    }
    path = "evals/runs/_verify_sample.json"
    os.makedirs("evals/runs", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull result saved to {path}")


if __name__ == "__main__":
    asyncio.run(main())
