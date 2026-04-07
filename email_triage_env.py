"""
Email Triage Environment — Python client library.

Wraps the HTTP API exposed by the FastAPI server running inside the Docker
container (or a locally started server).

Usage
-----
    # From a running Docker image:
    env = await EmailTriageEnv.from_docker_image(IMAGE_NAME, task_id="email-classify")

    # Direct connection to a running server:
    env = EmailTriageEnv(base_url="http://localhost:7860", task_id="email-classify")

    result = await env.reset()
    while not result.done:
        action = TriageAction(email_actions=[...])
        result = await env.step(action)
    await env.close()
"""

from __future__ import annotations

import asyncio
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# Shared Pydantic models
# ─────────────────────────────────────────────────────────────

VALID_CATEGORIES = {"spam", "work", "personal", "newsletter", "urgent"}
VALID_ACTIONS = {"read", "archive", "delete", "respond", "flag"}


class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str


class SingleEmailAction(BaseModel):
    """Agent's decision for one email."""
    email_id: str
    category: str = Field(..., description="spam | work | personal | newsletter | urgent")
    priority: int = Field(..., ge=1, le=20, description="1=most urgent, N=least urgent")
    action: str = Field(..., description="read | archive | delete | respond | flag")
    response_draft: Optional[str] = Field(None, description="Draft reply (required for urgent/work that need responses)")


class TriageAction(BaseModel):
    """Full action payload: agent decisions for one or more emails."""
    email_actions: List[SingleEmailAction]
    session_id: Optional[str] = None


class TriageObservation(BaseModel):
    """What the agent observes at each step."""
    task_id: str
    step: int
    max_steps: int
    inbox: List[Email]
    remaining_email_ids: List[str]
    processed_count: int
    total_emails: int
    current_score: float
    last_feedback: Optional[str]
    instructions: str


class StepResult(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: TriageObservation
    done: bool


# ─────────────────────────────────────────────────────────────
# Environment client
# ─────────────────────────────────────────────────────────────

class EmailTriageEnv:
    """Async client for the Email Triage OpenEnv environment."""

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        task_id: str = "email-classify",
        session_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.task_id = task_id
        self.session_id = session_id or str(uuid.uuid4())
        self.timeout = timeout
        self._container_id: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None

    # ── lifecycle ──────────────────────────────────────────

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        task_id: str = "email-classify",
        port: int = 7860,
        startup_timeout: int = 90,
    ) -> "EmailTriageEnv":
        """
        Start a Docker container from *image_name* and return a connected env.

        The container is stopped/removed when ``close()`` is called.
        """
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "-p", f"{port}:{port}",
                "-e", f"PORT={port}",
                image_name,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed:\n{result.stderr}"
            )
        container_id = result.stdout.strip()
        base_url = f"http://localhost:{port}"

        # Poll until the server is ready
        env = cls(base_url=base_url, task_id=task_id)
        env._container_id = container_id
        await env._wait_for_server(startup_timeout)
        return env

    async def _wait_for_server(self, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{self.base_url}/health")
                    if resp.status_code == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
        raise TimeoutError(
            f"Server at {self.base_url} did not become ready within {timeout}s"
        )

    async def close(self) -> None:
        """Stop the Docker container (if we started one) and clean up."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._container_id:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
            )
            subprocess.run(
                ["docker", "rm", self._container_id],
                capture_output=True,
            )
            self._container_id = None

    # ── OpenEnv API ────────────────────────────────────────

    async def reset(self, task_id: Optional[str] = None) -> ResetResult:
        """Reset the environment and return the initial observation."""
        payload: Dict[str, Any] = {"session_id": self.session_id}
        if task_id:
            self.task_id = task_id
            payload["task_id"] = task_id
        else:
            payload["task_id"] = self.task_id

        data = await self._post("/reset", payload)
        return ResetResult(**data)

    async def step(self, action: TriageAction) -> StepResult:
        """Submit email actions and receive observation + reward."""
        action_dict = action.model_dump()
        action_dict["session_id"] = self.session_id
        data = await self._post("/step", action_dict)
        return StepResult(**data)

    async def state(self) -> Dict[str, Any]:
        """Return raw current state (useful for debugging)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/state",
                params={"session_id": self.session_id},
            )
            resp.raise_for_status()
            return resp.json()

    # ── helpers ────────────────────────────────────────────

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
