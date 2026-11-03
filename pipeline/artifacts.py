"""Centralised file I/O helpers for pipeline artifacts."""
import json
import logging
import os
import traceback

log = logging.getLogger(__name__)


def run_dir(session_id: str) -> str:
    path = os.path.join("runs", session_id)
    os.makedirs(path, exist_ok=True)
    return path


def timing_path(session_id: str) -> str:
    return os.path.join(run_dir(session_id), "timing.jsonl")


def error_path(session_id: str) -> str:
    return os.path.join(run_dir(session_id), "errors.log")


def write_timing(log_path: str, record: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_error(session_id: str, exc: Exception) -> None:
    path = error_path(session_id)
    tb = traceback.format_exc()
    with open(path, "a") as f:
        f.write(f"\n{'=' * 60}\n{tb}\n")
    log.error("Stage error logged to %s: %s", path, exc)


def write_report(session_id: str, name: str, text: str) -> str:
    """Write a text report to runs/<session_id>/<name>.md and return path."""
    path = os.path.join(run_dir(session_id), f"{name}.md")
    with open(path, "w") as f:
        f.write(text)
    return path
