---
title: IT Helpdesk OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - llm-evaluation
  - it-helpdesk
---

# 🎫 IT Helpdesk OpenEnv

**Real-world IT support ticket triage environment for LLM evaluation.**

An agent must handle IT helpdesk tickets across 4 progressive tasks, from simple category classification to writing professional user responses. All graders are deterministic — no randomness, no ML required at inference time.

---

## Environment Description

**Motivation:** IT helpdesk is a ubiquitous real-world task where language understanding matters. Agents must correctly interpret technical jargon, apply policy rules (what to escalate), and communicate professionally — a genuine test of language + reasoning capabilities.

**20 curated tickets** across 4 categories: `network`, `hardware`, `software`, `access`.

**Episode:** 5 tickets per task. Each ticket = 1 step. Score = total_reward / 5.

---

## Tasks

| # | Task Name | Difficulty | Action Required | Grader |
|---|-----------|-----------|-----------------|--------|
| 0 | `classify_category` | Easy | Classify into network/hardware/software/access | Exact match + adjacent partial |
| 1 | `suggest_steps` | Medium | List 2-4 troubleshooting steps | Keyword coverage + step quality (Jaccard) |
| 2 | `escalate_or_resolve` | Medium | Decide resolve or escalate_to_level2 | Exact match with policy rule |
| 3 | `draft_response` | Hard | Write professional user response | Keyword overlap + length + empathy score |

**Policy rule for Task 2:** Network tickets always require escalation. Hardware/software/access are resolved internally.

---

## Observation Space

```json
{
  "ticket":      "VPN keeps disconnecting every 10 minutes during work calls.",
  "task_name":   "classify_category",
  "task_id":     0,
  "step":        0,
  "total_steps": 5,
  "hint":        "Respond with JSON: {\"action\":\"classify\",\"category\":\"<network|hardware|software|access>\"}"
}
```

## Action Space

```json
// Task 0 — classify
{"action": "classify", "category": "network"}

// Task 1 — suggest_steps
{"action": "suggest_steps", "steps": ["Update VPN client", "Toggle network adapter", "Check firewall rules"]}

// Task 2 — decide
{"action": "decide", "decision": "escalate_to_level2"}

// Task 3 — draft
{"action": "draft", "draft": "We have received your report and our network team is investigating. Please restart your VPN client. We will follow up within 2 hours."}
```

## Reward Function

| Task | Scoring Logic |
|------|--------------|
| `classify_category` | 1.0 exact match · 0.3 adjacent category · 0.0 wrong |
| `suggest_steps` | 0.5 × step quality (Jaccard) + 0.5 × GT keyword coverage + 0.1 length bonus |
| `escalate_or_resolve` | 1.0 exact · 0.0 network wrongly resolved · 0.2 other soft mismatch |
| `draft_response` | 0.5 × keyword overlap + 0.2 × length (15-80 words) + 0.3 × empathy keywords |

All rewards are clipped to **[0.0, 1.0]**.

---

## Setup & Usage

### Docker (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/it-helpdesk-openenv
cd it-helpdesk-openenv

docker build -t helpdesk-openenv .
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_key \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  helpdesk-openenv
```

### Local

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run inference script

```bash
export HF_TOKEN=hf_your_key
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

To run a single task:
```bash
HELPDESK_TASK=classify_category python inference.py
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/tasks` | List all tasks |
| `POST` | `/reset` | Start episode, returns `session_id` |
| `POST` | `/step` | Submit action, returns reward |
| `GET`  | `/state/{session_id}` | Current state (no advance) |
| `DELETE` | `/close/{session_id}` | End session |
| `GET`  | `/validate` | Self-test all graders |

### Reset example

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "classify_category", "seed": 42}'
```

### Step example

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "action": {"action": "classify", "category": "network"}
  }'
```

---

## Baseline Scores

Evaluated with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Score | Difficulty |
|------|-------|-----------|
| `classify_category` | 0.86 | Easy |
| `suggest_steps` | 0.61 | Medium |
| `escalate_or_resolve` | 0.80 | Medium |
| `draft_response` | 0.64 | Hard |
| **Mean** | **0.73** | — |

---

## Inference Script Output Format

```
[START] task=classify_category env=it-helpdesk-openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=classify(network) reward=1.00 done=false error=null
[STEP] step=2 action=classify(hardware) reward=1.00 done=false error=null
[STEP] step=3 action=classify(software) reward=1.00 done=false error=null
[STEP] step=4 action=classify(network) reward=1.00 done=false error=null
[STEP] step=5 action=classify(access) reward=1.00 done=true error=null
[END] success=true steps=5 score=0.86 rewards=1.00,1.00,1.00,1.00,0.30
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HELPDESK_TASK` | No | all tasks | Run single task only |

---

## Hardware Requirements

- **CPU:** 2 vCPU (no GPU required)
- **Memory:** 2 GB RAM (well within 8 GB limit)
- **Inference time:** ~3-5 minutes for all 4 tasks (20 LLM calls)

---

## License

MIT
