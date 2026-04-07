# Project Guide — Email Triage OpenEnv

A plain-English walkthrough of what was built, how it works, and how to use it.

---

## What Is This?

This is an **AI training environment** submitted to the OpenEnv competition.

The idea: instead of training AI agents on games like chess or Atari, this environment
teaches agents to do something useful — **triage an email inbox**. The agent reads emails,
classifies them (spam? work? urgent?), prioritizes them, and drafts replies.

OpenEnv is a standard interface (like OpenAI Gym but for real-world tasks) that lets
anyone plug an AI model in and measure how well it performs.

---

## The Three Tasks

Each task is a self-contained challenge the AI must solve.

### Task 1 — email-classify (Easy)
- **What:** 5 emails arrive. Classify each one: spam / work / personal / newsletter / urgent.
- **Scored on:** How many you get right (accuracy).
- **Example:** A "Congratulations you won $1M!" email should be spam. A "server is down" email should be urgent.

### Task 2 — email-prioritize (Medium)
- **What:** 8 emails. Rank them 1–8 by urgency (1 = handle first).
- **Scored on:** How close your ordering is to the correct order (Spearman rank correlation).
- **Example:** "Client meeting in 30 minutes" should rank higher than "coffee chat next week."

### Task 3 — email-triage-full (Hard)
- **What:** 12 emails. Classify + rank + decide action (reply/archive/delete) + write a draft reply for urgent ones.
- **Scored on:** All of the above, weighted equally (25% each component).
- **Example:** For a "payment service is down" email, the agent should mark it urgent, rank it #1, choose "respond", and write a professional reply authorizing the fix.

---

## How the Scoring Works

Every time the AI submits decisions, it gets a **reward** between 0 and 1.

- Reward = how much the score *improved* compared to last step
- If the AI processes emails one at a time, it gets small rewards along the way
- If it processes all at once, it gets one big reward
- Final **score** = the composite quality at the end of the episode

Score components per task:

| Component | Task 1 | Task 2 | Task 3 |
|-----------|--------|--------|--------|
| Classification | 60% | 25% | 30% |
| Priority ranking | 20% | 60% | 30% |
| Correct action | 20% | 15% | 20% |
| Response quality | — | — | 20% |

**Response quality** (Task 3 only) checks:
- Is the draft at least 20 words? (30%)
- Does it mention relevant keywords? (40%)
- Does it have a greeting and sign-off? (30%)

---

## How It All Connects

```
                  [AI Agent / LLM]
                       |
                 inference.py
                 (calls the AI,
                  formats actions)
                       |
            HTTP requests (JSON)
                       |
                  server.py
              (FastAPI web server)
                       |
           tasks/graders.py  ←  tasks/data.py
          (scores the actions)   (email datasets
                                  + ground truth)
```

The server runs inside a **Docker container** (or locally). The AI agent connects to it
and interacts through three endpoints:

- `POST /reset` — Start a new episode, get the inbox
- `POST /step` — Submit decisions, get back a score and feedback
- `GET /state` — Peek at the current state (for debugging)

---

## File Map

```
opencv/
├── server.py            The web server (brain of the environment)
├── email_triage_env.py  Python library to connect to the server
├── inference.py         The script that runs an AI model against all 3 tasks
├── openenv.yaml         Metadata file the OpenEnv platform reads
├── Dockerfile           How to package everything into a container
├── requirements.txt     Python packages needed
├── README.md            Public-facing documentation (shown on HuggingFace)
└── tasks/
    ├── data.py          The actual emails + correct answers (ground truth)
    └── graders.py       The math that converts answers into scores
```

---

## Running It Yourself

**Option A — Direct (no Docker):**
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (leave this running)
python server.py

# In another terminal, run the AI agent
ENV_URL=http://localhost:7860 \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=your_token_here \
python inference.py
```

**Option B — Docker:**
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
# Then run inference.py with ENV_URL=http://localhost:7860
```

**To test just the server manually:**
```bash
# Check it's running
curl http://localhost:7860/health

# Start a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "email-classify"}'

# Submit an answer
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "email_actions": [
      {"email_id": "e1", "category": "spam", "priority": 5, "action": "delete"}
    ]
  }'
```

---

## What the AI Output Looks Like

When `inference.py` runs, it prints structured logs:

```
[START] task=email-classify env=email-triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.64 done=false error=null
[STEP] step=2 action={...} reward=0.36 done=true error=null
[END] success=true steps=2 score=1.000 rewards=0.64,0.36

[START] task=email-prioritize env=email-triage model=Qwen/Qwen2.5-72B-Instruct
...
```

The OpenEnv evaluation platform reads exactly this format to score submissions.

---

## Key Design Decisions

**Why email triage?**
It's a real task with measurable correct answers. Unlike "write a story" (subjective),
email triage has ground truth: spam IS spam, a production outage IS more urgent than
a newsletter.

**Why dense rewards (not just end-of-episode)?**
Sparse rewards (only score at the end) make it hard for AI to learn. By giving reward
for each improvement step, the agent gets useful feedback throughout the episode.

**Why allow multi-step refinement?**
The agent can resubmit decisions to improve them. This mirrors how a human might
reconsider a classification after seeing all the emails. It also enables interesting
strategies (classify easy ones first, then tackle ambiguous ones).

**Why Spearman correlation for Task 2?**
It measures ranking quality properly — getting the top 3 emails in the right order
matters more than getting emails 6/7/8 slightly wrong. Spearman captures this better
than simple accuracy.
