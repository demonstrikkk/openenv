"""
IT Helpdesk OpenEnv — HuggingFace Space FastAPI Server
========================================================
Exposes the environment as a REST API for automated evaluation.
"""

import uuid
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import (
    TASKS, TICKETS_PER_TASK,
    HelpdeskAction, HelpdeskEnv,
    HelpdeskObservation, HelpdeskState,
)

app = FastAPI(
    title       = "IT Helpdesk OpenEnv",
    description = "Real-world IT support ticket triage environment for LLM evaluation.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Session store ─────────────────────────────────────────────────────
_SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL = 1800  # 30 min

def _gc():
    now = time.time()
    expired = [sid for sid, s in _SESSIONS.items() if now - s["ts"] > SESSION_TTL]
    for sid in expired:
        del _SESSIONS[sid]

def _get_env(session_id: str) -> HelpdeskEnv:
    _gc()
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found. Call /reset first.")
    _SESSIONS[session_id]["ts"] = time.time()
    return _SESSIONS[session_id]["env"]


# ── Request / Response schemas ─────────────────────────────────────────
class ResetRequest(BaseModel):
    task_name:  Optional[str] = None
    seed:       Optional[int] = 42
    session_id: Optional[str] = None   # pass to reuse session

class StepRequest(BaseModel):
    session_id: str
    action:     HelpdeskAction

class ResetResponse(BaseModel):
    session_id:  str
    observation: HelpdeskObservation
    task_names:  list

class StepResponse(BaseModel):
    observation: HelpdeskObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any]
    score:       float


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def health():
    return {
        "status":      "ok",
        "environment": "it-helpdesk-openenv",
        "version":     "1.0.0",
        "tasks":       [t["name"] for t in TASKS],
        "tickets_per_task": TICKETS_PER_TASK,
    }

@app.get("/tasks", summary="List available tasks")
def list_tasks():
    return {"tasks": TASKS}


@app.post("/reset", response_model=ResetResponse, summary="Start a new episode")
def reset(req: ResetRequest):
    """
    Start (or restart) an episode.
    Returns a session_id — pass this to /step and /state.
    """
    sid = req.session_id or str(uuid.uuid4())
    env = _SESSIONS[sid]["env"] if sid in _SESSIONS else HelpdeskEnv(seed=req.seed or 42)
    try:
        obs = env.reset(task_name=req.task_name, seed=req.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _SESSIONS[sid] = {"env": env, "ts": time.time()}
    return ResetResponse(
        session_id  = sid,
        observation = obs,
        task_names  = env.task_names,
    )


@app.post("/step", response_model=StepResponse, summary="Take one action")
def step(req: StepRequest):
    """
    Submit an action. Returns next observation, reward, done flag, and info.
    Reward is always in [0.0, 1.0].
    """
    env = _get_env(req.session_id)
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return StepResponse(
        observation = obs,
        reward      = reward,
        done        = done,
        info        = info,
        score       = env.score(),
    )


@app.get("/state/{session_id}", response_model=HelpdeskState, summary="Get current state")
def state(session_id: str):
    """Returns current state without advancing the episode."""
    env = _get_env(session_id)
    return env.state()


@app.delete("/close/{session_id}", summary="Close session")
def close(session_id: str):
    if session_id in _SESSIONS:
        _SESSIONS[session_id]["env"].close()
        del _SESSIONS[session_id]
    return {"closed": session_id}


@app.get("/validate", summary="Self-validation — runs graders on all tasks")
def validate():
    """
    Smoke-tests all 4 tasks with keyword-guided actions.
    All scores must be in [0, 1]. Returns pass/fail per task.
    """
    results = {}
    env = HelpdeskEnv(seed=99)

    SMART_ACTIONS = {
        "classify_category": lambda obs: HelpdeskAction(
            action="classify",
            category=(
                "network"  if any(w in obs.ticket.lower() for w in
                                  ["vpn","wifi","network","dns","packet","ssh","sso","drive"]) else
                "hardware" if any(w in obs.ticket.lower() for w in
                                  ["battery","screen","printer","drive","hard","monitor","fan","cpu"]) else
                "software" if any(w in obs.ticket.lower() for w in
                                  ["excel","outlook","zoom","update","windows","onedrive","teams"]) else
                "access"
            )
        ),
        "suggest_steps": lambda obs: HelpdeskAction(
            action="suggest_steps",
            steps=["Restart the device and check settings",
                   "Run built-in diagnostic or repair tool",
                   "Contact IT support if the issue persists"]
        ),
        "escalate_or_resolve": lambda obs: HelpdeskAction(
            action="decide",
            decision=("escalate_to_level2"
                      if any(w in obs.ticket.lower() for w in
                             ["vpn","network","packet","dns","ssh","sso"])
                      else "resolve")
        ),
        "draft_response": lambda obs: HelpdeskAction(
            action="draft",
            draft=("We have received your ticket and our support team is investigating. "
                   "Please follow the recommended steps and contact IT if the issue persists. "
                   "We will follow up with you shortly.")
        ),
    }

    all_ok = True
    for task in TASKS:
        obs = env.reset(task_name=task["name"])
        rewards, errors = [], []
        while True:
            action = SMART_ACTIONS[task["name"]](obs)
            obs, r, done, info = env.step(action)
            assert 0. <= r <= 1., f"Reward {r} out of [0,1]!"
            rewards.append(r)
            if info.get("last_action_error"):
                errors.append(info["last_action_error"])
            if done: break
        score = env.score()
        ok    = 0. <= score <= 1.
        all_ok = all_ok and ok
        results[task["name"]] = {
            "score":       round(score, 4),
            "rewards_ok":  all(0. <= r <= 1. for r in rewards),
            "pass":        ok,
            "errors":      errors,
        }

    env.close()
    return {"all_pass": all_ok, "tasks": results}
