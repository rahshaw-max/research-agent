import asyncio
from fastapi import WebSocket
from agents import researcher, architect, critic, auditor, interrogator
from pptx_builder import builder

MAX_CYCLES = 3


def _build_idea_with_refinements(session: dict) -> str:
    parts = [session["idea"]]
    for ref in session["refinements"]:
        if ref.get("refined_prompt"):
            parts.append(f"\n[Refinement]: {ref['refined_prompt']}")
        if ref.get("additional_context"):
            parts.append(f"[Additional context]: {ref['additional_context']}")
    return "\n".join(parts)


async def _stream_agent(name: str, generator, ws: WebSocket) -> str:
    await ws.send_json({"event": "agent_start", "agent": name, "data": {}})
    full_text = ""
    async for chunk in generator:
        if isinstance(chunk, dict) and chunk.get("__type__") == "status":
            await ws.send_json({"event": "agent_status", "agent": name, "data": {"message": chunk["message"]}})
        else:
            full_text += chunk
            await ws.send_json({"event": "agent_chunk", "agent": name, "data": {"text": chunk}})
    await ws.send_json({"event": "agent_complete", "agent": name, "data": {}})
    return full_text


async def run_pipeline(session_id: str, sessions: dict) -> None:
    session = sessions[session_id]

    try:
        # Wait for WebSocket to connect (up to 10s)
        await asyncio.wait_for(session["ws_ready"].wait(), timeout=10.0)
    except asyncio.TimeoutError:
        return

    ws: WebSocket = session["ws"]
    outputs = {}

    try:
        prev_outputs: dict = {}  # holds the outputs from the immediately preceding cycle

        while session["cycle_count"] <= MAX_CYCLES:
            cycle = session["cycle_count"]
            idea = _build_idea_with_refinements(session)

            # On cycle > 1, carry forward the previous critique and architecture so agents
            # can explicitly address the identified gaps rather than starting from scratch.
            prev_critique = prev_outputs.get("critic")
            prev_arch = prev_outputs.get("architect_v1")
            latest_refinement = session["refinements"][-1] if session["refinements"] else None

            # Phase 1 — Research cycle
            outputs["researcher"] = await _stream_agent(
                "researcher",
                researcher.run(
                    idea, session["answers"],
                    prev_critique=prev_critique,
                    latest_refinement=latest_refinement,
                ),
                ws,
            )
            outputs["architect_v1"] = await _stream_agent(
                "architect_v1",
                architect.run(
                    idea, outputs["researcher"],
                    prev_critique=prev_critique,
                    prev_arch=prev_arch,
                ),
                ws,
            )
            outputs["critic"] = await _stream_agent(
                "critic", critic.run(idea, outputs["researcher"], outputs["architect_v1"]), ws
            )

            # Interrogator — generates 10 clarifying questions from the critique (silent call)
            await ws.send_json({"event": "agent_start", "agent": "interrogator", "data": {}})
            questions = await interrogator.run(
                idea, outputs["researcher"], outputs["architect_v1"], outputs["critic"]
            )
            outputs["interrogator"] = questions
            await ws.send_json({"event": "agent_complete", "agent": "interrogator", "data": {"questions": questions}})

            # HITL gate — surface critique + questions, wait for human decision
            await ws.send_json({
                "event": "awaiting_review",
                "agent": "critic",
                "data": {"critique": outputs["critic"], "questions": questions, "cycle": cycle},
            })

            decision = await session["hitl_queue"].get()

            if decision["action"] == "refine" and cycle < MAX_CYCLES:
                session["refinements"].append({
                    "refined_prompt": decision.get("refined_prompt", ""),
                    "additional_context": decision.get("additional_context", ""),
                    "interrogator_answers": decision.get("interrogator_answers", {}),
                })
                prev_outputs = dict(outputs)  # snapshot this cycle for the next iteration
                session["cycle_count"] += 1
                await ws.send_json({
                    "event": "cycle_start",
                    "agent": None,
                    "data": {"cycle": session["cycle_count"]},
                })
                continue

            # Approved or max cycles reached
            break

        idea = _build_idea_with_refinements(session)

        # Phase 2 — Finalize
        outputs["architect_v2"] = await _stream_agent(
            "architect_v2",
            architect.run_revision(
                idea, outputs["researcher"], outputs["architect_v1"], outputs["critic"]
            ),
            ws,
        )

        # Auditor — silent (no streaming)
        await ws.send_json({"event": "agent_start", "agent": "auditor", "data": {}})
        pptx_data = await auditor.run(
            idea,
            session["answers"],
            researcher=outputs["researcher"],
            architect_v1=outputs["architect_v1"],
            critic=outputs["critic"],
            architect_v2=outputs["architect_v2"],
        )
        session["agent_outputs"] = outputs
        pptx_path = builder.build_presentation(session_id, pptx_data, "output")
        session["pptx_path"] = pptx_path
        await ws.send_json({"event": "agent_complete", "agent": "auditor", "data": {}})
        await ws.send_json({
            "event": "pipeline_done",
            "agent": None,
            "data": {"download_url": f"/api/download/{session_id}"},
        })

    except Exception as exc:
        await ws.send_json({
            "event": "error",
            "agent": None,
            "data": {"message": str(exc)},
        })
