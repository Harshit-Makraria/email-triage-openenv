"""
Email Triage Environment — FastAPI server.

Endpoints
---------
POST /reset          Reset (or start) an episode.
POST /step           Submit email actions, receive reward + observation.
GET  /state          Inspect current session state.
GET  /tasks          List available tasks.
GET  /health         Liveness probe.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from email_triage_env import (
    Email,
    SingleEmailAction,
    StepResult,
    TriageObservation,
    ResetResult,
)
from tasks.data import TASKS
from tasks.graders import grade_task

app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment for RL agent evaluation.",
    version="1.0.0",
)

# ─────────────────────────────────────────────────────────────
# In-memory session store  (single-process; fine for eval runs)
# ─────────────────────────────────────────────────────────────
_sessions: Dict[str, "_Session"] = {}


class _Session:
    """Holds all mutable state for one episode."""

    def __init__(self, task_cfg: Dict[str, Any], session_id: str) -> None:
        self.session_id = session_id
        self.task_id: str = task_cfg["id"]
        self.task_name: str = task_cfg["name"]
        self.instructions: str = task_cfg["instructions"]
        self.max_steps: int = task_cfg["max_steps"]
        self.weights: Dict[str, float] = task_cfg["weights"]

        # Email data (strip ground truth from what the agent sees)
        self._raw_emails: List[Dict[str, Any]] = task_cfg["emails"]
        self.emails_by_id: Dict[str, Dict[str, Any]] = {
            e["id"]: e for e in self._raw_emails
        }
        self.ground_truth: Dict[str, Any] = task_cfg["ground_truth"]

        # Episode state
        self.step: int = 0
        self.processed: Dict[str, Dict[str, Any]] = {}  # email_id → last action
        self.step_rewards: List[float] = []
        self.prev_score: float = 0.0
        self.done: bool = False
        self.last_feedback: Optional[str] = None
        self.current_score: float = 0.0

    # ── properties ──────────────────────────────────────────

    @property
    def inbox_emails(self) -> List[Email]:
        return [
            Email(
                id=e["id"],
                sender=e["sender"],
                subject=e["subject"],
                body=e["body"],
                timestamp=e["timestamp"],
            )
            for e in self._raw_emails
        ]

    @property
    def remaining_ids(self) -> List[str]:
        return [e["id"] for e in self._raw_emails if e["id"] not in self.processed]

    @property
    def observation(self) -> TriageObservation:
        return TriageObservation(
            task_id=self.task_id,
            step=self.step,
            max_steps=self.max_steps,
            inbox=self.inbox_emails,
            remaining_email_ids=self.remaining_ids,
            processed_count=len(self.processed),
            total_emails=len(self._raw_emails),
            current_score=self.current_score,
            last_feedback=self.last_feedback,
            instructions=self.instructions,
        )


# ─────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "email-classify"
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    email_actions: List[SingleEmailAction]
    session_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_or_create_session(session_id: Optional[str], task_id: str) -> _Session:
    sid = session_id or "default"
    if sid not in _sessions or task_id != _sessions[sid].task_id:
        if task_id not in TASKS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}",
            )
        _sessions[sid] = _Session(TASKS[task_id], sid)
    return _sessions[sid]


def _get_session(session_id: Optional[str]) -> _Session:
    sid = session_id or "default"
    if sid not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found. Call /reset first.")
    return _sessions[sid]


def _compute_step(session: _Session, actions: List[SingleEmailAction]) -> StepResult:
    """Process one agent step: update processed emails, grade, compute reward."""

    # Record the actions (agent can revise earlier decisions)
    for act in actions:
        if act.email_id in session.emails_by_id:
            session.processed[act.email_id] = act.model_dump()

    session.step += 1

    # Grade ALL processed emails so far
    all_actions = list(session.processed.values())
    grade = grade_task(
        task_id=session.task_id,
        email_actions=all_actions,
        ground_truth=session.ground_truth,
        weights=session.weights,
    )

    current_score = grade["total_score"]
    # Reward = improvement in composite score (non-negative; no step penalty)
    reward = max(0.0, round(current_score - session.prev_score, 4))
    session.prev_score = current_score
    session.current_score = current_score
    session.step_rewards.append(reward)
    session.last_feedback = grade["feedback"]

    # Episode is done when all emails processed OR max_steps reached
    all_done = len(session.processed) >= len(session.emails_by_id)
    max_reached = session.step >= session.max_steps
    session.done = all_done or max_reached

    return StepResult(
        observation=session.observation,
        reward=reward,
        done=session.done,
        info={
            "component_scores": grade["component_scores"],
            "current_score": current_score,
            "step": session.step,
            "processed_count": len(session.processed),
        },
    )


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": t["id"],
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "max_steps": t["max_steps"],
                "num_emails": len(t["emails"]),
            }
            for t in TASKS.values()
        ]
    }


@app.post("/reset", response_model=ResetResult)
async def reset(body: ResetRequest = ResetRequest()) -> ResetResult:
    """
    Reset (or initialise) an episode.

    Body is optional — send ``{}`` to use defaults (task_id=email-classify).
    """
    task_id = body.task_id or "email-classify"
    sid = body.session_id or "default"

    # Always create a fresh session on reset
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}",
        )
    _sessions[sid] = _Session(TASKS[task_id], sid)
    session = _sessions[sid]

    return ResetResult(observation=session.observation, done=False)


@app.post("/step", response_model=StepResult)
async def step(body: StepRequest) -> StepResult:
    """Submit email actions and receive the next observation + reward."""
    session = _get_session(body.session_id)

    if session.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode.",
        )
    if not body.email_actions:
        raise HTTPException(status_code=400, detail="email_actions must not be empty.")

    return _compute_step(session, body.email_actions)


@app.get("/state")
async def state(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the full current state for a session (debug / introspection)."""
    session = _get_session(session_id)
    return {
        "session_id": session.session_id,
        "task_id": session.task_id,
        "step": session.step,
        "max_steps": session.max_steps,
        "done": session.done,
        "processed_count": len(session.processed),
        "total_emails": len(session.emails_by_id),
        "current_score": session.current_score,
        "step_rewards": session.step_rewards,
        "remaining_email_ids": session.remaining_ids,
        "last_feedback": session.last_feedback,
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()
