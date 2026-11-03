"""
Adapter for the baseline researcher at commit 74a47d0.

That version's researcher.run(idea, answers, prev_critique=None, latest_refinement=None)
is a plain streaming generator — no web tools, no evidence list, Sonnet-only.
Returns (report_text, evidence=[]).
"""
import importlib.util
import sys
import os


_BASELINE_ROOT = "/tmp/baseline_repo"


def _load_baseline_researcher():
    """Load agents/researcher.py from the 74a47d0 worktree without polluting sys.modules."""
    agent_path = os.path.join(_BASELINE_ROOT, "agents", "researcher.py")
    if not os.path.exists(agent_path):
        raise RuntimeError(
            f"Baseline worktree not found at {_BASELINE_ROOT}. "
            "Run: git worktree add --detach /tmp/baseline_repo 74a47d0"
        )
    # Give it a unique module name so it doesn't collide with agents.researcher
    spec = importlib.util.spec_from_file_location("_baseline_researcher", agent_path)
    mod  = importlib.util.module_from_spec(spec)
    # dotenv is already loaded by run_eval.py; anthropic client reads ANTHROPIC_API_KEY from env
    spec.loader.exec_module(mod)
    return mod


_researcher_mod = None


def _get_researcher():
    global _researcher_mod
    if _researcher_mod is None:
        _researcher_mod = _load_baseline_researcher()
    return _researcher_mod


async def run(idea: str, answers: dict) -> tuple[str, list]:
    """Run the 74a47d0 researcher and return (report_text, [])."""
    mod  = _get_researcher()
    text = ""
    async for chunk in mod.run(idea, answers):
        if isinstance(chunk, str):
            text += chunk
    return text, []
