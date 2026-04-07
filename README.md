---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - email
  - triage
  - rl
  - agent-evaluation
---

# Email Triage — OpenEnv Environment

A real-world email triage environment for training and evaluating AI agents.
Agents must **classify**, **prioritize**, and **respond to** a realistic inbox
of emails, mirroring one of the most common productivity tasks professionals
perform daily.

---

## Why Email Triage?

Email triage is a genuine, high-value task: the average knowledge worker
spends ~28% of their workday managing email. A capable agent that can triage
an inbox correctly has immediate, measurable economic value — unlike toy grid
worlds or synthetic puzzles. The graded structure (easy → hard) allows
systematic capability benchmarking.

---

## Environment Overview

| Field | Value |
|---|---|
| Domain | Productivity / Email Management |
| Tasks | 3 (easy, medium, hard) |
| Action space | Classify + prioritize + act + draft responses |
| Observation space | Full inbox + processing state + feedback |
| Reward | Dense (per-step quality improvement) |
| Episode termination | All emails processed **or** max steps reached |

---

## Observation Space

At each step the agent receives a `TriageObservation` object:

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Active task identifier |
| `step` | int | Current step (starts at 0 after reset) |
| `max_steps` | int | Episode step limit |
| `inbox` | List[Email] | All emails in the episode |
| `remaining_email_ids` | List[str] | IDs of unprocessed emails |
| `processed_count` | int | Emails handled so far |
| `total_emails` | int | Total emails in the episode |
| `current_score` | float ∈ [0,1] | Composite quality score so far |
| `last_feedback` | str \| None | Grader feedback from the previous step |
| `instructions` | str | Task-specific natural-language instructions |

Each `Email` has: `id`, `sender`, `subject`, `body`, `timestamp`.

---

## Action Space

At each step the agent submits a `TriageAction`:

```json
{
  "email_actions": [
    {
      "email_id": "e1",
      "category": "spam",
      "priority": 5,
      "action": "delete",
      "response_draft": null
    }
  ]
}
```

| Field | Values | Notes |
|---|---|---|
| `email_id` | string | Must match an inbox email ID |
| `category` | `spam` \| `work` \| `personal` \| `newsletter` \| `urgent` | Required |
| `priority` | integer 1–N | 1=most urgent; must be unique across the episode |
| `action` | `read` \| `archive` \| `delete` \| `respond` \| `flag` | Required |
| `response_draft` | string \| null | Required (≥20 words) when `action=respond` |

---

## Tasks

### Task 1 — `email-classify` (Easy)

- **Emails:** 5 (1 spam, 1 work, 1 personal, 1 newsletter, 1 urgent)
- **Max steps:** 5
- **Score weights:** classification 60% · action 20% · priority 20%
- **Goal:** Identify the category of each email correctly.
- **Expected baseline score:** ~0.55–0.75

### Task 2 — `email-prioritize` (Medium)

- **Emails:** 8 spanning all urgency levels
- **Max steps:** 8
- **Score weights:** priority ranking 60% · classification 25% · action 15%
- **Goal:** Rank all 8 emails by urgency (1=most urgent). Graded by Spearman rank correlation.
- **Expected baseline score:** ~0.45–0.65

### Task 3 — `email-triage-full` (Hard)

- **Emails:** 12 (mix of all categories, several requiring responses)
- **Max steps:** 12
- **Score weights:** classification 30% · priority 30% · action 20% · response quality 20%
- **Goal:** Full triage — classify, prioritize, choose actions, and draft professional responses.
- **Expected baseline score:** ~0.30–0.55

---

## Reward Function

```
reward_t = max(0, composite_score_t − composite_score_{t-1})
```

- **Dense:** every step that improves the triage quality yields a positive reward.
- **Non-negative:** partial progress is always rewarded; no step penalty.
- **Informative:** the `last_feedback` field in the observation explains *why* the score changed.
- **Final episode score:** `current_score` at termination (the composite quality, 0–1).

Response quality (Task 3) is assessed on: word count ≥20 (30%), keyword coverage (40%), professional greeting + sign-off (30%).

---

## Setup & Usage

### Local (no Docker)

```bash
pip install -r requirements.txt

# Start the server
python server.py           # listens on port 7860 by default
# or
PORT=8000 python server.py

# Run baseline inference (needs a running server)
ENV_URL=http://localhost:7860 \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=<your_token> \
python inference.py
```

### Docker

```bash
# Build
docker build -t email-triage-env .

# Run server
docker run -p 7860:7860 email-triage-env

# Run inference (server in Docker, inference on host)
IMAGE_NAME=email-triage-env \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=<your_token> \
python inference.py
```

### OpenEnv validation

```bash
pip install openenv-core
openenv validate
```

---

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe — returns `{"status":"ok"}` |
| `/tasks` | GET | List all available tasks with metadata |
| `/reset` | POST | Start/reset an episode. Body: `{"task_id":"email-classify","session_id":"optional"}` |
| `/step` | POST | Submit actions. Body: `{"email_actions":[...],"session_id":"..."}` |
| `/state` | GET | Inspect current session state (debug) |

---

## Baseline Scores (Qwen/Qwen2.5-72B-Instruct)

| Task | Difficulty | Score |
|---|---|---|
| email-classify | Easy | ~0.70 |
| email-prioritize | Medium | ~0.55 |
| email-triage-full | Hard | ~0.40 |
| **Average** | | **~0.55** |

*Scores are approximate and depend on the model and API temperature.*

---

## Project Structure

```
.
├── Dockerfile               Container definition
├── README.md                This file
├── openenv.yaml             OpenEnv metadata & task registry
├── requirements.txt         Python dependencies
├── inference.py             Baseline inference script (entry point)
├── server.py                FastAPI environment server
├── email_triage_env.py      Python client library + shared Pydantic models
└── tasks/
    ├── __init__.py
    ├── data.py              Email datasets with ground-truth labels
    └── graders.py           Deterministic scoring functions
```
