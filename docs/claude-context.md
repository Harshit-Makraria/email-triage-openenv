# Claude Context — Email Triage OpenEnv

Technical reference for continuing work on this project in future conversations.

---

## What This Project Is

An **OpenEnv competition submission** — a real-world RL environment where agents triage
email inboxes. Built to the OpenEnv spec with FastAPI server + Python client library +
3 graded tasks + Docker + HF Spaces deployment.

Competition requirements satisfied:
- Real-world task (not a game/toy)
- Full OpenEnv spec: typed Pydantic models, step()/reset()/state(), openenv.yaml
- 3 tasks: easy → medium → hard with deterministic graders scoring 0.0–1.0
- Dense reward function (per-step improvement signal, non-negative)
- inference.py using OpenAI client with exact [START]/[STEP]/[END] log format
- Dockerfile exposing port 7860 (HF Spaces standard)
- README.md with HF Spaces YAML frontmatter

---

## Architecture

```
server.py           FastAPI app — owns all episode state
  └── _sessions     Dict[session_id, _Session] — in-memory, single process
  └── _Session      Holds task config, emails, processed actions, scores
  
email_triage_env.py Shared Pydantic models + async HTTP client
  └── EmailTriageEnv    .reset() .step() .state() .close()
  └── from_docker_image() classmethod — starts container, polls /health

tasks/data.py       Static email datasets + ground truth labels
  └── TASK1_EMAILS / TASK1_GROUND_TRUTH  (5 emails, easy)
  └── TASK2_EMAILS / TASK2_GROUND_TRUTH  (8 emails, medium)
  └── TASK3_EMAILS / TASK3_GROUND_TRUTH  (12 emails, hard)
  └── TASKS dict  — registry used by server and inference

tasks/graders.py    Deterministic scoring functions
  └── score_classification()   — accuracy of category labels
  └── score_priority_ranking() — Spearman correlation (rescaled to [0,1])
  └── score_actions()          — action correctness (partial credit: read≈archive)
  └── score_responses()        — length + keyword + format (greeting/signoff)
  └── grade_task()             — weighted composite of all components

inference.py        Baseline script
  └── Runs all 3 tasks sequentially
  └── _fallback_action() — deterministic keyword-based fallback (no LLM needed)
  └── _parse_action()    — JSON extraction with markdown fence stripping
```

---

## Key Design Choices

**Reward function:**
```python
reward_t = max(0, current_score - prev_score)
```
Non-negative, dense. Agent can resubmit actions for already-processed emails (refine).
`current_score` is computed over ALL processed emails at each step.

**Episode done condition:**
```python
done = (len(processed) >= len(emails)) or (step >= max_steps)
```

**Score = composite quality at episode end** (not sum of rewards).
Logged as `score` in `[END]` line of inference.py.

**Task weights:**

| Component | Task 1 | Task 2 | Task 3 |
|-----------|--------|--------|--------|
| classification | 0.60 | 0.25 | 0.30 |
| priority | 0.20 | 0.60 | 0.30 |
| action | 0.20 | 0.15 | 0.20 |
| response | 0.00 | 0.00 | 0.20 |

**Response scorer** (Task 3 only, per urgent/work email needing a reply):
- length ≥ 20 words → up to 0.30
- keyword coverage (task-specific keywords in ground_truth) → up to 0.40
- greeting present → 0.15; sign-off present → 0.15

**Task 3 perfect score without responses = 0.80** (response weight 0.20 × 0.0 = 0).
Full 1.0 requires providing response_draft for all `needs_response=True` emails.

---

## API Contract

### POST /reset
```json
// request (all fields optional — empty {} works)
{"task_id": "email-classify", "session_id": "abc"}

// response
{"observation": {...TriageObservation...}, "done": false}
```

### POST /step
```json
// request
{
  "session_id": "abc",
  "email_actions": [
    {"email_id": "e1", "category": "spam", "priority": 10,
     "action": "delete", "response_draft": null}
  ]
}

// response
{"observation": {...}, "reward": 0.46, "done": false,
 "info": {"component_scores": {...}, "current_score": 0.46, "step": 1, "processed_count": 1}}
```

### GET /state?session_id=abc
Returns raw session internals (debug use).

### GET /health → {"status": "ok", "version": "1.0.0"}

### GET /tasks → list of task metadata dicts

---

## Pydantic Models (email_triage_env.py)

```python
class SingleEmailAction(BaseModel):
    email_id: str
    category: str          # spam|work|personal|newsletter|urgent
    priority: int          # 1–20, must be unique per episode
    action: str            # read|archive|delete|respond|flag
    response_draft: Optional[str]

class TriageAction(BaseModel):
    email_actions: List[SingleEmailAction]
    session_id: Optional[str]

class TriageObservation(BaseModel):
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
    info: Dict[str, Any]

class ResetResult(BaseModel):
    observation: TriageObservation
    done: bool
```

---

## Ground Truth Structure

Task 1/2 ground truth per email:
```python
{"category": "spam", "priority": 10, "action": "delete"}
```

Task 3 ground truth per email (extra fields for response grading):
```python
{
  "category": "urgent",
  "priority": 1,
  "action": "respond",
  "needs_response": True,
  "response_keywords": ["payment", "authorize", "patch", "approve"]
}
```

---

## Inference Script Key Variables

```python
IMAGE_NAME   = os.getenv("IMAGE_NAME")           # Docker image (triggers from_docker_image)
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")  # Direct server URL
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
MAX_STEPS    = 15
SUCCESS_THRESHOLD = 0.5  # score >= 0.5 → success=true in [END] line
```

---

## Validated Test Results

Integration tests pass (confirmed locally):

| Task | Perfect score | Empty score | Fallback (heuristic) |
|------|--------------|------------|----------------------|
| email-classify | 1.000 | 0.000 | ~0.64 |
| email-prioritize | 1.000 | 0.000 | ~0.85 |
| email-triage-full | 0.800* | 0.000 | ~0.56 |

*0.80 = perfect without response drafts; add good drafts to reach ~0.94+

Multi-step reward shaping confirmed: processing emails in 2 batches gives
positive reward at each step (not just at the end).

---

## Deployment Notes

**HF Spaces:**
- README.md has YAML frontmatter (`sdk: docker`, `tags: [openenv, ...]`)
- Dockerfile exposes port 7860 (HF Spaces standard)
- No ENV vars needed in the container itself (server has no external dependencies)

**Docker run:**
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
# PORT env var overrides the default 7860
docker run -p 8000:8000 -e PORT=8000 email-triage-env
```

**from_docker_image() behavior:**
- Runs `docker run -d -p PORT:PORT -e PORT=PORT IMAGE_NAME`
- Polls `/health` until 200 (up to 90s)
- `close()` runs `docker stop` + `docker rm`

---

## Potential Improvements

- Add session TTL / cleanup (current: sessions live forever in memory)
- Add `POST /tasks/{task_id}/reset` convenience endpoint
- Response grader is keyword-based — could be upgraded to semantic similarity
- Add a 4th task: email thread summarization
- Add adversarial emails (phishing that looks legitimate) for robustness testing
- Persist state to Redis for multi-process / HF Spaces restart resilience
