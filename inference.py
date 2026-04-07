"""
Inference Script — Email Triage OpenEnv Baseline
=================================================

Runs a language-model agent against all three tasks and emits structured
stdout logs in the exact [START] / [STEP] / [END] format required by the
OpenEnv evaluation harness.

Environment variables
---------------------
API_BASE_URL        LLM API endpoint  (default: https://router.huggingface.co/v1)
MODEL_NAME          Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN / API_KEY  API key for the LLM provider
IMAGE_NAME          Docker image name (use from_docker_image when set)
ENV_URL             Direct URL to a running env server (default: http://localhost:7860)

Stdout format (one episode = one task)
---------------------------------------
[START] task=<name> env=email-triage model=<model>
[STEP]  step=<n> action=<json_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from email_triage_env import EmailTriageEnv, SingleEmailAction, TriageAction

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

IMAGE_NAME: Optional[str] = os.getenv("IMAGE_NAME")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-key"
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK: str = "email-triage"

TASKS_TO_RUN: List[str] = ["email-classify", "email-prioritize", "email-triage-full"]
MAX_STEPS: int = 15
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 1500
SUCCESS_THRESHOLD: float = 0.5

# ─────────────────────────────────────────────────────────────
# Logging helpers  (strict stdout format)
# ─────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err_val = error if error else "null"
    done_val = str(done).lower()
    # Keep action on a single line (no embedded newlines)
    action_oneline = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_oneline} "
        f"reward={reward:.2f} done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────
# Prompt building
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert email triage assistant.

    Given a list of emails and task instructions, you must process every email
    and return a JSON object with the following structure:

    {
      "email_actions": [
        {
          "email_id": "<id>",
          "category": "<spam|work|personal|newsletter|urgent>",
          "priority": <integer — 1=most urgent, up to N=least urgent, no ties>,
          "action": "<read|archive|delete|respond|flag>",
          "response_draft": "<optional draft — required for emails needing a reply>"
        }
      ]
    }

    Rules:
    - Include an entry for EVERY email in the inbox.
    - Priorities must be unique integers (no two emails share the same priority).
    - Write a response_draft for every email whose action is "respond".
      The draft must be ≥20 words, professional, and address the email's content.
    - Output ONLY the JSON object — no markdown, no extra commentary.
    """
).strip()


def _build_user_prompt(
    observation: Any, step: int, last_reward: float
) -> str:
    inbox_lines = []
    remaining_set = set(observation.remaining_email_ids)
    for email in observation.inbox:
        status = "PENDING" if email.id in remaining_set else "processed"
        inbox_lines.append(
            f"[{email.id}] ({status})\n"
            f"  From: {email.sender}\n"
            f"  Subject: {email.subject}\n"
            f"  Body: {email.body}\n"
        )

    inbox_block = "\n".join(inbox_lines)
    feedback_block = (
        f"Last feedback: {observation.last_feedback}" if observation.last_feedback else ""
    )

    return textwrap.dedent(
        f"""
        Task: {observation.task_id}
        Step: {step} / {observation.max_steps}
        Processed: {observation.processed_count} / {observation.total_emails}
        Current score: {observation.current_score:.3f}
        Last step reward: {last_reward:.2f}
        {feedback_block}

        Instructions:
        {observation.instructions}

        Inbox ({observation.total_emails} emails):
        {inbox_block}

        Process ALL {len(observation.remaining_email_ids)} PENDING email(s) now.
        Return a single JSON object as specified.
        """
    ).strip()


# ─────────────────────────────────────────────────────────────
# LLM call + JSON parsing
# ─────────────────────────────────────────────────────────────

def _call_llm(client: OpenAI, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def _parse_action(raw: str, observation: Any) -> Optional[TriageAction]:
    """Extract a TriageAction from the LLM's raw text output."""
    if not raw:
        return None

    # Try direct parse first
    text = raw.strip()

    # Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Extract the outermost JSON object
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        text = brace_match.group(0)

    try:
        data = json.loads(text)
        actions = data.get("email_actions", [])
        parsed = [SingleEmailAction(**a) for a in actions]
        return TriageAction(email_actions=parsed)
    except Exception as exc:
        print(f"[DEBUG] JSON parse failed: {exc}. Raw: {raw[:200]}", flush=True)
        return None


def _fallback_action(observation: Any) -> TriageAction:
    """
    Deterministic fallback: classify all remaining emails as 'work',
    delete nothing, archive newsletters, delete spam.
    """
    remaining = {e.id: e for e in observation.inbox if e.id in set(observation.remaining_email_ids)}
    actions: List[SingleEmailAction] = []
    for rank, (eid, email) in enumerate(remaining.items(), start=1):
        subj_lower = (email.subject + " " + email.body).lower()
        if any(kw in subj_lower for kw in ["unsubscribe", "newsletter", "digest", "weekly"]):
            cat, act = "newsletter", "archive"
        elif any(kw in subj_lower for kw in ["congratul", "winner", "prize", "free gift", "flash sale"]):
            cat, act = "spam", "delete"
        elif any(kw in subj_lower for kw in ["urgent", "critical", "down", "immediate"]):
            cat, act = "urgent", "respond"
        else:
            cat, act = "work", "read"
        actions.append(
            SingleEmailAction(
                email_id=eid,
                category=cat,
                priority=rank,
                action=act,
            )
        )
    return TriageAction(email_actions=actions)


# ─────────────────────────────────────────────────────────────
# Single-task episode runner
# ─────────────────────────────────────────────────────────────

async def run_episode(env: EmailTriageEnv, task_id: str, client: OpenAI) -> float:
    """
    Run one complete episode for *task_id*.
    Emits [START] / [STEP]+ / [END] to stdout.
    Returns the final score ∈ [0, 1].
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_prompt = _build_user_prompt(obs, step, last_reward)
            raw = _call_llm(client, user_prompt)
            action = _parse_action(raw, obs)

            if action is None or not action.email_actions:
                action = _fallback_action(obs)
                last_error = "parse_error_used_fallback"
            else:
                last_error = None

            action_str = json.dumps(
                {"email_actions": [a.model_dump(exclude_none=True) for a in action.email_actions]},
                separators=(",", ":"),
            )

            result = await env.step(action)
            obs = result.observation
            reward = result.reward
            done = result.done

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

            if done:
                break

        score = obs.current_score  # final composite quality in [0, 1]
        score = round(max(0.0, min(1.0, score)), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

async def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        all_scores: Dict[str, float] = {}

        for task_id in TASKS_TO_RUN:
            # Create env client
            try:
                if IMAGE_NAME:
                    env = await EmailTriageEnv.from_docker_image(IMAGE_NAME, task_id=task_id)
                else:
                    env = EmailTriageEnv(base_url=ENV_URL, task_id=task_id)
            except Exception as exc:
                print(f"[DEBUG] env creation error for task {task_id}: {exc}", flush=True)
                continue

            try:
                score = await run_episode(env, task_id, client)
                all_scores[task_id] = score
            finally:
                try:
                    await env.close()
                except Exception as exc:
                    print(f"[DEBUG] env.close() error: {exc}", flush=True)

        # Summary across all tasks
        avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        print(
            f"\n[SUMMARY] tasks={len(all_scores)} "
            + " ".join(f"{k}={v:.3f}" for k, v in all_scores.items())
            + f" avg={avg:.3f}",
            flush=True,
        )
    except Exception as e:
        print(f"[FATAL] Unhandled exception in main: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
