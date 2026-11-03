import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

log = logging.getLogger(__name__)
os.makedirs("output", exist_ok=True)

# In-memory sessions: includes non-serialisable runtime fields (ws, ws_ready, hitl_queue).
# The store snapshots JSON-serialisable fields to SQLite at phase boundaries.
sessions: dict = {}

store = None   # SessionStore; created in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    from session_store import SessionStore
    store = SessionStore(os.getenv("SESSION_DB", "sessions.db"))
    await store.init()
    interrupted = await store.mark_interrupted()
    expired     = await store.delete_expired(ttl_hours=72)
    if interrupted or expired:
        log.info("Startup: %d sessions marked interrupted, %d expired deleted", interrupted, expired)
    yield


app = FastAPI(lifespan=lifespan)


# ── Request models ──────────────────────────────────────────────────────────

class ClarifyRequest(BaseModel):
    idea: str


class SubmitAnswersRequest(BaseModel):
    session_id: str
    answers: dict
    depth: str = "standard"   # "quick" | "standard" | "deep"


class ReviewRequest(BaseModel):
    session_id: str
    action: str  # "approve" | "refine" | "retry_stage" | "abort"
    refined_prompt: str = ""
    additional_context: str = ""
    interrogator_answers: dict = {}


# ── HTTP routes ─────────────────────────────────────────────────────────────

@app.post("/api/clarify")
async def clarify(req: ClarifyRequest):
    from agents.clarifier import ask_clarifying_questions

    session_id = str(uuid.uuid4())
    questions = await ask_clarifying_questions(req.idea)

    session = {
        "idea": req.idea,
        "questions": questions,
        "answers": {},
        "depth": "standard",
        "agent_outputs": {},
        "pptx_path": None,
        "ws": None,
        "ws_ready": asyncio.Event(),
        "hitl_queue": asyncio.Queue(),
        "cycle_count": 1,
        "refinements": [],
        "evidence": [],
        "phase": "clarifying",
    }
    sessions[session_id] = session

    if store is not None:
        await store.create(session_id, session)

    return {"session_id": session_id, "questions": questions}


@app.post("/api/submit-answers")
async def submit_answers(req: SubmitAnswersRequest, background_tasks: BackgroundTasks):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["answers"] = req.answers
    session["depth"]   = req.depth

    from pipeline.orchestrator import run_pipeline
    background_tasks.add_task(run_pipeline, req.session_id, sessions, store=store)

    return {"status": "started"}


@app.post("/api/review")
async def review(req: ReviewRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await session["hitl_queue"].put({
        "action": req.action,
        "refined_prompt": req.refined_prompt,
        "additional_context": req.additional_context,
        "interrogator_answers": req.interrogator_answers,
    })

    return {"status": "decision received"}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Return persisted session metadata for reconnection (omits non-serialisable fields)."""
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialised")
    record = await store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # Expose only the fields the frontend needs
    state = record.get("state", {})
    return {
        "session_id": session_id,
        "status": record["status"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "idea": state.get("idea", ""),
        "questions": state.get("questions", []),
        "answers": state.get("answers", {}),
        "depth": state.get("depth", "standard"),
        "cycle_count": state.get("cycle_count", 1),
        "pptx_path": state.get("pptx_path"),
        "coverage_score": (state.get("last_gap_report") or {}).get("coverage_score"),
        "awaiting_review_data": state.get("awaiting_review_data"),
    }


@app.get("/api/download/{session_id}")
async def download(session_id: str):
    session = sessions.get(session_id)
    pptx_path = session.get("pptx_path") if session else None

    # Fall back to stored path if not in memory
    if not pptx_path and store is not None:
        record = await store.get(session_id)
        if record:
            pptx_path = record.get("state", {}).get("pptx_path")

    if not pptx_path:
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(
        pptx_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="research_summary.pptx",
    )


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = sessions.get(session_id)

    # ── Resume from store if not in memory ────────────────────────────────
    resume_phase = "start"
    if session is None and store is not None:
        record = await store.get(session_id)
        if record is not None and record["status"] == "awaiting_review":
            state = record["state"]
            session = {
                **state,
                "ws": None,
                "ws_ready": asyncio.Event(),
                "hitl_queue": asyncio.Queue(),
            }
            sessions[session_id] = session
            resume_phase = "awaiting_review"
            from pipeline.orchestrator import run_pipeline
            asyncio.create_task(
                run_pipeline(session_id, sessions, store=store, resume_phase="awaiting_review")
            )

    if session is None:
        await websocket.close()
        return

    session["ws"] = websocket
    session["ws_ready"].set()

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        session["ws"] = None


# ── Static files ─────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
