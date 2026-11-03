import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from fastapi import WebSocket
from agents import researcher, architect, critic, auditor, interrogator, verifier
from pptx_builder import builder

DEPTH_CONFIGS = {
    "quick":    {"max_subq": 3, "searches": 2, "fetches": 1, "max_cycles": 1, "arch_mode": "lite"},
    "standard": {"max_subq": 6, "searches": 3, "fetches": 2, "max_cycles": 3, "arch_mode": "lite"},
    "deep":     {"max_subq": 8, "searches": 4, "fetches": 3, "max_cycles": 3, "arch_mode": "full"},
}
_DEFAULT_DEPTH = os.getenv("DEPTH", "standard")


def _build_idea_with_refinements(session: dict) -> str:
    parts = [session["idea"]]
    for ref in session["refinements"]:
        if ref.get("refined_prompt"):
            parts.append(f"\n[Refinement]: {ref['refined_prompt']}")
        if ref.get("additional_context"):
            parts.append(f"[Additional context]: {ref['additional_context']}")
    return "\n".join(parts)


def _timing_log_path(session_id: str) -> str:
    run_dir = os.path.join("runs", session_id)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "timing.jsonl")


def _write_timing(log_path: str, record: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


@asynccontextmanager
async def _stage_timer(
    stage: str,
    cycle: int,
    usage: dict,
    log_path: str,
    ws: WebSocket | None = None,
):
    t0 = time.perf_counter()
    try:
        yield usage
    finally:
        elapsed = round(time.perf_counter() - t0, 2)
        record = {
            "stage": stage,
            "cycle": cycle,
            "seconds": elapsed,
            "input_tokens":         usage.get("input_tokens", 0),
            "output_tokens":        usage.get("output_tokens", 0),
            "tool_calls":           usage.get("tool_calls", 0),
            "cache_read_tokens":    usage.get("cache_read_tokens", 0),
            "cache_created_tokens": usage.get("cache_created_tokens", 0),
        }
        _write_timing(log_path, record)
        if ws is not None:
            try:
                await ws.send_json({"event": "stage_timing", "agent": stage, "data": record})
            except Exception:
                pass


async def _stream_agent(name: str, generator, ws: WebSocket) -> str:
    await ws.send_json({"event": "agent_start", "agent": name, "data": {}})
    full_text = ""
    final_text: str | None = None
    async for chunk in generator:
        if isinstance(chunk, dict):
            t = chunk.get("__type__")
            if t == "status":
                await ws.send_json({"event": "agent_status", "agent": name, "data": {"message": chunk["message"]}})
            elif t == "citation_stat":
                cited, total = chunk.get("cited", 0), chunk.get("total", 0)
                pct = 100 * cited // total if total else 0
                await ws.send_json({"event": "agent_status", "agent": name, "data": {
                    "message": f"citation rate: {cited}/{total} ({pct}%)"
                }})
            elif t == "report_final":
                final_text = chunk.get("text", "")
        else:
            full_text += chunk
            await ws.send_json({"event": "agent_chunk", "agent": name, "data": {"text": chunk}})
    await ws.send_json({"event": "agent_complete", "agent": name, "data": {}})
    return final_text if final_text is not None else full_text


async def run_pipeline(session_id: str, sessions: dict) -> None:
    session = sessions[session_id]

    try:
        await asyncio.wait_for(session["ws_ready"].wait(), timeout=10.0)
    except asyncio.TimeoutError:
        return

    ws: WebSocket = session["ws"]
    log_path = _timing_log_path(session_id)
    all_timing: list[dict] = []

    depth_key = session.get("depth", _DEFAULT_DEPTH)
    depth = DEPTH_CONFIGS.get(depth_key, DEPTH_CONFIGS["standard"])
    max_cycles  = depth["max_cycles"]
    arch_mode   = depth["arch_mode"]

    final_research: str | None = None
    prev_coverage_score: float | None = None
    last_gap_report: dict = {}

    try:
        # ── CYCLE LOOP ────────────────────────────────────────────────────────
        while session["cycle_count"] <= max_cycles:
            cycle = session["cycle_count"]
            idea = _build_idea_with_refinements(session)
            latest_refinement = session["refinements"][-1] if session["refinements"] else None

            gap_queries: list[str] | None = None
            if cycle > 1 and last_gap_report:
                gap_queries = []
                for gap in last_gap_report.get("gaps", []):
                    gap_queries.extend(gap.get("suggested_queries", []))

            # ── Researcher ───────────────────────────────────────────────────
            ru: dict = {}
            cycle_evidence: list = list(session.get("evidence", []))
            async with _stage_timer("researcher", cycle, ru, log_path, ws) as ru:
                final_research = await _stream_agent(
                    "researcher",
                    researcher.run(
                        idea,
                        session["answers"],
                        prev_research=final_research if cycle > 1 else None,
                        gap_queries=gap_queries,
                        latest_refinement=latest_refinement,
                        _usage=ru,
                        _evidence=cycle_evidence,
                        max_subq=depth["max_subq"],
                        searches_per_subq=depth["searches"],
                        fetches_per_subq=depth["fetches"],
                    ),
                    ws,
                )
            session["evidence"] = cycle_evidence
            all_timing.append({"stage": "researcher", "cycle": cycle,
                                "evidence_count": len(cycle_evidence), **ru})

            # ── Critique + Verifier concurrently ─────────────────────────────
            # Both only need final_research (no inter-dependency).
            # Interrogator waits for critique; verifier result updates final_research.
            rcu: dict = {}
            vu: dict = {}
            t0_parallel = time.perf_counter()

            [last_gap_report, verify_result] = await asyncio.gather(
                critic.run_research_critique(idea, final_research, session["answers"], _usage=rcu),
                verifier.run(final_research, idea, session["answers"], evidence=cycle_evidence, _usage=vu),
            )
            t_parallel = round(time.perf_counter() - t0_parallel, 2)

            # Record both with shared wall time (they overlapped)
            for stage, u in [("research_critique", rcu), ("verifier", vu)]:
                record = {"stage": stage, "cycle": cycle, "seconds": t_parallel, **u}
                _write_timing(log_path, record)
                all_timing.append(record)
                try:
                    await ws.send_json({"event": "stage_timing", "agent": stage, "data": record})
                except Exception:
                    pass

            coverage_score: float = last_gap_report.get("coverage_score", 0.0)
            auto_approve_recommended = False
            if prev_coverage_score is not None:
                auto_approve_recommended = (coverage_score - prev_coverage_score) < 0.03
            prev_coverage_score = coverage_score

            final_research = verify_result["report"]
            vcounts = verify_result["counts"]
            if vcounts["total"] > 0:
                parts = [
                    f"{vcounts['supported']} ✓",
                    f"{vcounts['partial']} ~",
                    f"{vcounts['unsupported']} ✗",
                ]
                if vcounts["repaired"] or vcounts["removed"]:
                    parts.append(f"({vcounts['repaired']} repaired, {vcounts['removed']} removed)")
                try:
                    await ws.send_json({
                        "event": "agent_status",
                        "agent": "researcher",
                        "data": {"message": f"claims: {' · '.join(parts)}"},
                    })
                except Exception:
                    pass

            # ── Interrogator (needs gap report, serial) ───────────────────────
            iu: dict = {}
            async with _stage_timer("interrogator", cycle, iu, log_path, ws) as iu:
                try:
                    await ws.send_json({"event": "agent_start", "agent": "interrogator", "data": {}})
                except Exception:
                    pass
                questions = await interrogator.run(idea, final_research, last_gap_report, _usage=iu)
                try:
                    await ws.send_json({
                        "event": "agent_complete",
                        "agent": "interrogator",
                        "data": {"questions": questions},
                    })
                except Exception:
                    pass
            all_timing.append({"stage": "interrogator", "cycle": cycle, **iu})

            # ── HITL gate ────────────────────────────────────────────────────
            try:
                await ws.send_json({
                    "event": "awaiting_review",
                    "agent": "critic",
                    "data": {
                        "critique": "",
                        "questions": questions,
                        "cycle": cycle,
                        "gap_report": last_gap_report,
                        "coverage_score": coverage_score,
                        "auto_approve_recommended": auto_approve_recommended,
                        "verification_counts": vcounts,
                    },
                })
            except Exception:
                pass

            decision = await session["hitl_queue"].get()

            if decision["action"] == "refine" and cycle < max_cycles:
                session["refinements"].append({
                    "refined_prompt": decision.get("refined_prompt", ""),
                    "additional_context": decision.get("additional_context", ""),
                    "interrogator_answers": decision.get("interrogator_answers", {}),
                })
                session["cycle_count"] += 1
                try:
                    await ws.send_json({
                        "event": "cycle_start",
                        "agent": None,
                        "data": {"cycle": session["cycle_count"]},
                    })
                except Exception:
                    pass
                continue

            break  # approved or max cycles reached

        # ── PHASE 2: Architecture ─────────────────────────────────────────────
        idea = _build_idea_with_refinements(session)

        if arch_mode == "lite":
            au1: dict = {}
            async with _stage_timer("architect_v1", 0, au1, log_path, ws) as au1:
                arch_out = await _stream_agent(
                    "architect_v1",
                    architect.run_lite(idea, final_research, last_gap_report, _usage=au1),
                    ws,
                )
            all_timing.append({"stage": "architect_v1", "cycle": 0, **au1})
            final_arch_v1 = arch_out
            final_arch_critique = ""
            final_arch_v2 = arch_out

        else:  # full
            au1: dict = {}
            async with _stage_timer("architect_v1", 0, au1, log_path, ws) as au1:
                final_arch_v1 = await _stream_agent(
                    "architect_v1",
                    architect.run(idea, final_research, _usage=au1),
                    ws,
                )
            all_timing.append({"stage": "architect_v1", "cycle": 0, **au1})

            acu: dict = {}
            async with _stage_timer("arch_critic", 0, acu, log_path, ws) as acu:
                final_arch_critique = await _stream_agent(
                    "critic",
                    critic.run(idea, final_research, final_arch_v1, _usage=acu),
                    ws,
                )
            all_timing.append({"stage": "arch_critic", "cycle": 0, **acu})

            av2u: dict = {}
            async with _stage_timer("architect_v2", 0, av2u, log_path, ws) as av2u:
                final_arch_v2 = await _stream_agent(
                    "architect_v2",
                    architect.run_revision(idea, final_research, final_arch_v1, final_arch_critique, _usage=av2u),
                    ws,
                )
            all_timing.append({"stage": "architect_v2", "cycle": 0, **av2u})

        # ── Auditor ──────────────────────────────────────────────────────────
        adtu: dict = {}
        async with _stage_timer("auditor", 0, adtu, log_path, ws) as adtu:
            try:
                await ws.send_json({"event": "agent_start", "agent": "auditor", "data": {}})
            except Exception:
                pass
            pptx_data = await auditor.run(
                idea,
                session["answers"],
                researcher=final_research,
                architect_v1=final_arch_v1,
                critic=final_arch_critique,
                architect_v2=final_arch_v2,
                evidence=session.get("evidence", []),
                _usage=adtu,
            )
        all_timing.append({"stage": "auditor", "cycle": 0, **adtu})

        # ── PPTX build ───────────────────────────────────────────────────────
        pptxu: dict = {}
        async with _stage_timer("pptx_build", 0, pptxu, log_path, ws) as pptxu:
            pptx_path = builder.build_presentation(session_id, pptx_data, "output")
        all_timing.append({"stage": "pptx_build", "cycle": 0, **pptxu})

        session["agent_outputs"] = {
            "researcher":   final_research,
            "architect_v1": final_arch_v1,
            "critic":       final_arch_critique,
            "architect_v2": final_arch_v2,
        }
        session["pptx_path"] = pptx_path

        total_in   = sum(r.get("input_tokens",  0) for r in all_timing)
        total_out  = sum(r.get("output_tokens", 0) for r in all_timing)
        total_tc   = sum(r.get("tool_calls",    0) for r in all_timing)
        total_cr   = sum(r.get("cache_read_tokens",    0) for r in all_timing)
        total_cc   = sum(r.get("cache_created_tokens", 0) for r in all_timing)
        try:
            await ws.send_json({"event": "agent_complete", "agent": "auditor", "data": {}})
            await ws.send_json({
                "event": "pipeline_done",
                "agent": None,
                "data": {
                    "download_url": f"/api/download/{session_id}",
                    "timing": all_timing,
                    "timing_totals": {
                        "input_tokens":         total_in,
                        "output_tokens":        total_out,
                        "tool_calls":           total_tc,
                        "cache_read_tokens":    total_cr,
                        "cache_created_tokens": total_cc,
                    },
                },
            })
        except Exception:
            pass

    except Exception as exc:
        try:
            await ws.send_json({
                "event": "error",
                "agent": None,
                "data": {"message": str(exc)},
            })
        except Exception:
            pass
