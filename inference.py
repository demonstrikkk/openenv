"""
IT Helpdesk OpenEnv — Inference Script
=======================================
Uses OpenAI-compatible client to run an LLM against the environment.
Emits mandatory stdout format: [START] / [STEP] / [END]

Environment variables:
  API_BASE_URL   LLM API endpoint  (default: HuggingFace router)
  MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
  HELPDESK_TASK  Single task to run (optional; default: run all 4 tasks)
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from environment import TASKS, HelpdeskAction, HelpdeskEnv

# ── Configuration ─────────────────────────────────────────────────────
API_BASE_URL  = os.getenv("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",    "Qwen/Qwen2.5-72B-Instruct")
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf-no-key")
BENCHMARK     = "it-helpdesk-openenv"
MAX_STEPS     = 8          # safety cap per episode
TEMPERATURE   = 0.1        # low for more deterministic structured output
MAX_TOKENS    = 300
SUCCESS_THRESHOLD = 0.5    # score ≥ this → success=true

# ── OpenAI client ─────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── System prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert IT helpdesk analyst. You will be given an IT support ticket and a specific task.
You MUST respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.
The exact JSON schema required will be shown in the HINT field of each request.

Category definitions:
  network  — connectivity issues: Wi-Fi, VPN, DNS, LAN, switches, routers
  hardware — physical device issues: laptop, monitor, keyboard, battery, printer
  software — application/OS issues: Excel, Outlook, Windows, Zoom, browsers
  access   — authentication/authorization: passwords, MFA, permissions, SSH, SSO
"""

# ── LLM prompt builder ────────────────────────────────────────────────
def build_user_prompt(obs_dict: Dict) -> str:
    task_descriptions = {
        "classify_category":   "Classify this ticket into exactly one category.",
        "suggest_steps":       "Provide exactly 2-4 concrete troubleshooting steps for this ticket.",
        "escalate_or_resolve": "Decide whether to resolve internally or escalate to level-2 support.",
        "draft_response":      "Write a professional, empathetic response (15-60 words) to the user.",
    }
    return (
        f"TICKET: {obs_dict['ticket']}\n\n"
        f"TASK: {task_descriptions.get(obs_dict['task_name'], obs_dict['task_name'])}\n\n"
        f"HINT (JSON schema to use): {obs_dict['hint']}\n\n"
        f"Respond with ONLY the JSON object:"
    )

# ── JSON parser (handles markdown fences + common LLM quirks) ─────────
def parse_json(text: str) -> Dict[str, Any]:
    """Extract first valid JSON object from LLM response."""
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*', '', text).strip()
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    # Fallback: try the whole text
    return json.loads(text)

# ── JSON → HelpdeskAction converter ──────────────────────────────────
def json_to_action(data: Dict, task_name: str) -> HelpdeskAction:
    action_map = {
        "classify_category":   "classify",
        "suggest_steps":       "suggest_steps",
        "escalate_or_resolve": "decide",
        "draft_response":      "draft",
    }
    action_type = action_map.get(task_name, data.get("action","classify"))

    return HelpdeskAction(
        action   = action_type,
        category = data.get("category"),
        steps    = data.get("steps"),
        decision = data.get("decision"),
        draft    = data.get("draft"),
    )

# ── Compact action string for [STEP] log ──────────────────────────────
def action_str(action: HelpdeskAction) -> str:
    if action.action == "classify":
        return f"classify({action.category})"
    if action.action == "suggest_steps":
        steps = "|".join((action.steps or [])[:3])
        return f"steps([{steps[:60]}])"
    if action.action == "decide":
        return f"decide({action.decision})"
    if action.action == "draft":
        return f"draft({(action.draft or '')[:50]})"
    return f"{action.action}()"

# ── Fallback action when LLM parse fails ──────────────────────────────
def fallback_action(task_name: str) -> HelpdeskAction:
    defaults = {
        "classify_category":   HelpdeskAction(action="classify",      category="software"),
        "suggest_steps":       HelpdeskAction(action="suggest_steps",  steps=["Restart device","Contact IT"]),
        "escalate_or_resolve": HelpdeskAction(action="decide",         decision="resolve"),
        "draft_response":      HelpdeskAction(action="draft",
                                              draft="We have received your request and will investigate shortly."),
    }
    return defaults.get(task_name, HelpdeskAction(action="classify", category="software"))

# ── Single LLM call ───────────────────────────────────────────────────
def llm_call(prompt: str) -> Tuple[str, Optional[str]]:
    """Returns (raw_text, error_or_None)."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content or "", None
    except Exception as exc:
        return "", str(exc)

# ── Run one task episode ───────────────────────────────────────────────
def run_task(env: HelpdeskEnv, task_name: str) -> Dict:
    obs    = env.reset(task_name=task_name)
    rewards: List[float] = []
    errors:  List[Optional[str]] = []
    step_n = 0

    # ── [START] ────────────────────────────────────────────────────────
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    last_error: Optional[str] = None

    while step_n < MAX_STEPS:
        obs_dict = obs.dict()
        prompt   = build_user_prompt(obs_dict)

        # LLM call
        raw, api_err = llm_call(prompt)
        action_error: Optional[str] = api_err

        if api_err:
            action = fallback_action(task_name)
        else:
            try:
                data   = parse_json(raw)
                action = json_to_action(data, task_name)
            except Exception as parse_exc:
                action       = fallback_action(task_name)
                action_error = f"json_parse_error: {parse_exc}"

        obs, reward, done, info = env.step(action)
        step_n += 1

        # Prefer env-reported error over parse error
        env_error = info.get("last_action_error")
        final_error = env_error or action_error

        rewards.append(reward)
        errors.append(final_error)
        last_error = final_error

        # ── [STEP] ─────────────────────────────────────────────────────
        print(
            f"[STEP] step={step_n} "
            f"action={action_str(action)} "
            f"reward={reward:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={final_error if final_error else 'null'}",
            flush=True,
        )

        if done:
            break

    score   = env.score()
    success = score >= SUCCESS_THRESHOLD
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # ── [END] ──────────────────────────────────────────────────────────
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_n} "
        f"score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return {
        "task":    task_name,
        "score":   score,
        "success": success,
        "steps":   step_n,
        "rewards": rewards,
    }


# ── Main ──────────────────────────────────────────────────────────────
def main():
    env        = HelpdeskEnv(seed=42)
    target     = os.getenv("HELPDESK_TASK")  # optional single-task override
    task_names = [target] if target else [t["name"] for t in TASKS]

    all_results: List[Dict] = []
    for task_name in task_names:
        result = run_task(env, task_name)
        all_results.append(result)

    env.close()

    # Summary to stderr (not part of mandatory stdout format)
    mean_score = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n# Summary: mean_score={mean_score:.3f} "
          f"tasks_passed={sum(1 for r in all_results if r['success'])}/{len(all_results)}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
