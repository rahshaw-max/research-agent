import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

os.makedirs("output", exist_ok=True)

sessions: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


# ── Request models ──────────────────────────────────────────────────────────

class ClarifyRequest(BaseModel):
    idea: str


class SubmitAnswersRequest(BaseModel):
    session_id: str
    answers: dict


class ReviewRequest(BaseModel):
    session_id: str
    action: str  # "approve" | "refine"
    refined_prompt: str = ""
    additional_context: str = ""
    interrogator_answers: dict = {}


# ── HTTP routes ─────────────────────────────────────────────────────────────

@app.post("/api/clarify")
async def clarify(req: ClarifyRequest):
    from agents.clarifier import ask_clarifying_questions

    session_id = str(uuid.uuid4())
    questions = await ask_clarifying_questions(req.idea)

    sessions[session_id] = {
        "idea": req.idea,
        "questions": questions,
        "answers": {},
        "agent_outputs": {},
        "pptx_path": None,
        "ws": None,
        "ws_ready": asyncio.Event(),
        "hitl_queue": asyncio.Queue(),
        "cycle_count": 1,
        "refinements": [],
    }

    return {"session_id": session_id, "questions": questions}


@app.post("/api/submit-answers")
async def submit_answers(req: SubmitAnswersRequest, background_tasks: BackgroundTasks):
    session = sessions.get(req.session_id)
    if not session:
        return {"error": "Session not found"}, 404

    session["answers"] = req.answers

    from pipeline.orchestrator import run_pipeline
    background_tasks.add_task(run_pipeline, req.session_id, sessions)

    return {"status": "started"}


@app.post("/api/review")
async def review(req: ReviewRequest):
    session = sessions.get(req.session_id)
    if not session:
        return {"error": "Session not found"}, 404

    await session["hitl_queue"].put({
        "action": req.action,
        "refined_prompt": req.refined_prompt,
        "additional_context": req.additional_context,
        "interrogator_answers": req.interrogator_answers,
    })

    return {"status": "decision received"}


@app.get("/api/download/{session_id}")
async def download(session_id: str):
    session = sessions.get(session_id)
    if not session or not session.get("pptx_path"):
        return {"error": "File not ready"}, 404
    return FileResponse(
        session["pptx_path"],
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="research_summary.pptx",
    )


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = sessions.get(session_id)
    if not session:
        await websocket.close()
        return

    session["ws"] = websocket
    session["ws_ready"].set()

    try:
        # Keep the connection alive — the pipeline pushes events to us
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
