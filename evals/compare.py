#!/usr/bin/env python3
"""
Compare two eval run labels and print a markdown table.

Usage:
    python evals/compare.py --a baseline_74a47d0 --b head
"""
import argparse
import json
import os

_EVALS_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS = [
    ("n_claims",             "Claims",    ".1f"),
    ("citation_rate",        "Cite%",     ".1%"),
    ("unique_sources",       "Sources",   ".1f"),
    ("unique_domains",       "Domains",   ".1f"),
    ("primary_source_ratio", "Primary%",  ".1%"),
    ("numeric_claims_cited", "Num%",      ".1%"),
    ("verified_claim_rate",  "Verified%", ".1%"),
    ("wall_seconds",         "Secs",      ".1f"),
    ("total_tokens",         "Tokens",    ".0f"),
]

JUDGE_METRICS = [
    ("specificity",       "Spec"),
    ("factual_grounding", "Grnd"),
    ("completeness",      "Comp"),
    ("decision_usefulness","Decn"),
]


def _load_scores(label: str) -> dict[str, dict]:
    """Return {id: scores_dict} for all *_scores.json in the run dir."""
    run_dir = os.path.join(_EVALS_DIR, "runs", label)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"No run directory: {run_dir}")
    out = {}
    for fname in os.listdir(run_dir):
        if not fname.endswith("_scores.json"):
            continue
        qid = fname[:-len("_scores.json")]
        with open(os.path.join(run_dir, fname)) as f:
            out[qid] = json.load(f)
    return out


def _get(scores: dict, key: str, sub: str | None = None):
    if sub is None:
        return scores.get(key)
    return (scores.get(key) or {}).get(sub)


def _fmt(v, fmt: str) -> str:
    if v is None:
        return "—"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def compare(label_a: str, label_b: str) -> None:
    scores_a = _load_scores(label_a)
    scores_b = _load_scores(label_b)

    all_ids = sorted(set(scores_a) | set(scores_b))

    # ── Summary table ────────────────────────────────────────────────────────
    header_metrics = [m[1] for m in METRICS] + [j[1] for j in JUDGE_METRICS]
    col_w = 9

    def row(label, scores_map):
        vals = []
        for key, _, fmt in METRICS:
            v = _mean([_get(s, key) for s in scores_map.values()])
            vals.append(_fmt(v, fmt))
        for jkey, _ in JUDGE_METRICS:
            v = _mean([_get(s, "judge", jkey) for s in scores_map.values()])
            vals.append(_fmt(v, ".2f"))
        return vals

    print(f"\n## Mean scores — {label_a} vs {label_b}\n")
    hdr = f"| {'Label':<28} | " + " | ".join(f"{h:>{col_w}}" for h in header_metrics) + " |"
    sep = f"|{'-'*30}|" + "|".join("-" * (col_w + 2) for _ in header_metrics) + "|"
    print(hdr)
    print(sep)
    for lbl, sm in [(label_a, scores_a), (label_b, scores_b)]:
        if not sm:
            print(f"| {lbl:<28} | (no scores) |")
            continue
        vals = row(lbl, sm)
        print(f"| {lbl:<28} | " + " | ".join(f"{v:>{col_w}}" for v in vals) + " |")

    # ── Per-question delta table ─────────────────────────────────────────────
    print(f"\n## Per-question deltas ({label_b} − {label_a})\n")
    delta_metrics = [
        ("citation_rate",        "Cite%",  ".1%"),
        ("unique_sources",       "Srcs",   ".1f"),
        ("primary_source_ratio", "Prim%",  ".1%"),
        ("verified_claim_rate",  "Ver%",   ".1%"),
    ] + [(f"judge.{jk}", jl, ".2f") for jk, jl in JUDGE_METRICS]

    dhdr = f"| {'ID':<25} | " + " | ".join(f"{m[1]:>8}" for m in delta_metrics) + " |"
    dsep = f"|{'-'*27}|" + "|".join("-" * 10 for _ in delta_metrics) + "|"
    print(dhdr)
    print(dsep)

    for qid in all_ids:
        sa = scores_a.get(qid)
        sb = scores_b.get(qid)
        cells = []
        for key, _, fmt in delta_metrics:
            if key.startswith("judge."):
                jkey = key[len("judge."):]
                va = _get(sa, "judge", jkey) if sa else None
                vb = _get(sb, "judge", jkey) if sb else None
            else:
                va = _get(sa, key) if sa else None
                vb = _get(sb, key) if sb else None
            if va is None or vb is None:
                cells.append("    —")
            else:
                delta = vb - va
                sign = "+" if delta >= 0 else ""
                cells.append(f"{sign}{_fmt(delta, fmt):>7}")
        print(f"| {qid:<25} | " + " | ".join(f"{c:>8}" for c in cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="First (baseline) label")
    parser.add_argument("--b", required=True, help="Second (head) label")
    args = parser.parse_args()
    compare(args.a, args.b)


if __name__ == "__main__":
    main()
