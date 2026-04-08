"""
IT Helpdesk OpenEnv Environment
================================
Real-world task: IT support ticket triage.
4 tasks (easy → hard): classify → suggest_steps → escalate_or_resolve → draft_response
Graders: deterministic keyword-based, always return float in [0.0, 1.0]
Zero ML dependencies at runtime — pure Python + Pydantic.
"""

import re
import random
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ── OpenEnv base (fallback if not installed) ──────────────────────────
try:
    from openenv import Environment as OpenEnvBase
except ImportError:
    try:
        from openenv.core import Environment as OpenEnvBase
    except ImportError:
        class OpenEnvBase:
            """OpenEnv-compliant fallback stub."""
            def reset(self): raise NotImplementedError
            def step(self, action): raise NotImplementedError
            def state(self): raise NotImplementedError
            def close(self): pass


# ── Pydantic models (OpenEnv-compliant) ──────────────────────────────
class HelpdeskObservation(BaseModel):
    ticket:       str
    task_name:    str
    task_id:      int
    step:         int
    total_steps:  int
    hint:         str

class HelpdeskAction(BaseModel):
    action:    Literal["classify", "suggest_steps", "decide", "draft"]
    category:  Optional[Literal["network", "hardware", "software", "access"]] = None
    steps:     Optional[List[str]]  = None
    decision:  Optional[Literal["resolve", "escalate_to_level2"]]             = None
    draft:     Optional[str]        = None

class HelpdeskReward(BaseModel):
    reward:    float = Field(..., ge=0.0, le=1.0)
    rationale: str

class HelpdeskState(BaseModel):
    task_name:     str
    task_id:       int
    step:          int
    total_steps:   int
    score:         float
    done:          bool
    tickets_seen:  int
    last_reward:   float


# ── Task definitions ──────────────────────────────────────────────────
TASKS: List[Dict] = [
    {"id": 0, "name": "classify_category",    "difficulty": "easy",   "action": "classify"},
    {"id": 1, "name": "suggest_steps",         "difficulty": "medium", "action": "suggest_steps"},
    {"id": 2, "name": "escalate_or_resolve",   "difficulty": "medium", "action": "decide"},
    {"id": 3, "name": "draft_response",        "difficulty": "hard",   "action": "draft"},
]

HINTS = {
    "classify_category":  'Respond with JSON: {"action":"classify","category":"<network|hardware|software|access>"}',
    "suggest_steps":      'Respond with JSON: {"action":"suggest_steps","steps":["step1","step2","step3"]}',
    "escalate_or_resolve":'Respond with JSON: {"action":"decide","decision":"<resolve|escalate_to_level2>"}',
    "draft_response":     'Respond with JSON: {"action":"draft","draft":"Your professional response to the user."}',
}

TICKETS_PER_TASK = 5   # steps per episode


# ── Ticket pool (5 per category × 4 categories = 20 tickets) ─────────
TICKET_POOL: List[Dict] = [
    # ── NETWORK ───────────────────────────────────────────────────────
    {
        "id": "n1", "category": "network",
        "description": "VPN keeps disconnecting every 10 minutes during work calls.",
        "gt_steps": ["Update VPN client to latest version",
                     "Toggle network adapter off and on",
                     "Check firewall rules blocking VPN port"],
        "gt_decision": "escalate_to_level2",
        "gt_draft": (
            "We have received your VPN disconnection report and are investigating. "
            "Please update your VPN client and toggle your network adapter. "
            "Our network team will follow up within 2 hours."
        ),
    },
    {
        "id": "n2", "category": "network",
        "description": "Wi-Fi drops connection every 5 minutes on all devices in the office.",
        "gt_steps": ["Restart the Wi-Fi access point",
                     "Check for wireless interference on the 2.4GHz band",
                     "Update access point firmware"],
        "gt_decision": "escalate_to_level2",
        "gt_draft": (
            "Thank you for reporting the Wi-Fi drops. We are investigating the access point. "
            "Please restart the device and check interference. "
            "Our network team will contact you shortly."
        ),
    },
    {
        "id": "n3", "category": "network",
        "description": "Cannot map network drive Z: — access is denied.",
        "gt_steps": ["Verify user has share permissions in Active Directory",
                     "Re-map the drive using net use command",
                     "Restart Windows Credential Manager service"],
        "gt_decision": "resolve",
        "gt_draft": (
            "We have logged your network drive access issue. "
            "Please verify your share permissions and try re-mapping using net use. "
            "IT will confirm access rights within 4 hours."
        ),
    },
    {
        "id": "n4", "category": "network",
        "description": "DNS lookup fails for internal company servers but internet works.",
        "gt_steps": ["Flush DNS cache with ipconfig /flushdns",
                     "Set DNS server to internal DC IP address",
                     "Restart DNS Client service"],
        "gt_decision": "escalate_to_level2",
        "gt_draft": (
            "We have received your DNS resolution issue. "
            "Please flush your DNS cache and verify your DNS settings point to the internal server. "
            "Our network team will investigate further."
        ),
    },
    {
        "id": "n5", "category": "network",
        "description": "Severe packet loss on the office LAN — 40% packets dropped.",
        "gt_steps": ["Run ping test to identify affected segment",
                     "Check switch port for duplex mismatch",
                     "Replace Ethernet cable if cable test fails"],
        "gt_decision": "escalate_to_level2",
        "gt_draft": (
            "Your packet loss report has been escalated to our network team. "
            "Please run a ping test and check your Ethernet cable. "
            "A network engineer will contact you within 1 hour."
        ),
    },
    # ── HARDWARE ──────────────────────────────────────────────────────
    {
        "id": "h1", "category": "hardware",
        "description": "Laptop battery not charging — stuck at 5% even when plugged in.",
        "gt_steps": ["Check power adapter LED and try a different outlet",
                     "Perform a battery reset by holding power button 30 seconds",
                     "Update battery driver in Device Manager"],
        "gt_decision": "resolve",
        "gt_draft": (
            "We have noted your battery charging issue. "
            "Please check your power adapter and perform a battery reset. "
            "A technician will be assigned if hardware replacement is needed."
        ),
    },
    {
        "id": "h2", "category": "hardware",
        "description": "External monitor not detected when connected via HDMI.",
        "gt_steps": ["Try a different HDMI cable",
                     "Press Win+P to cycle display mode to Extend",
                     "Update graphics driver from Device Manager"],
        "gt_decision": "resolve",
        "gt_draft": (
            "Thank you for reporting the monitor issue. "
            "Please try a different HDMI cable and press Win+P to detect the display. "
            "Updating your graphics driver should resolve this."
        ),
    },
    {
        "id": "h3", "category": "hardware",
        "description": "Printer shows online but prints blank pages.",
        "gt_steps": ["Cancel all print jobs and restart print spooler",
                     "Clean printhead from printer maintenance menu",
                     "Run printer diagnostics from manufacturer utility"],
        "gt_decision": "resolve",
        "gt_draft": (
            "We have received your printer issue report. "
            "Please cancel pending jobs, restart the print spooler, and clean the printhead. "
            "Contact IT if blank pages persist after diagnostics."
        ),
    },
    {
        "id": "h4", "category": "hardware",
        "description": "Laptop overheating and throttling CPU during normal use.",
        "gt_steps": ["Clean laptop vents with compressed air",
                     "Check Task Manager for processes causing high CPU load",
                     "Update BIOS and thermal management drivers"],
        "gt_decision": "resolve",
        "gt_draft": (
            "Your overheating issue has been logged. "
            "Please clean the vents with compressed air and check for high CPU processes. "
            "IT will arrange hardware inspection if throttling continues."
        ),
    },
    {
        "id": "h5", "category": "hardware",
        "description": "Hard drive making clicking noise — data still accessible but slow.",
        "gt_steps": ["Immediately backup all critical data",
                     "Run CrystalDiskInfo to check SMART status",
                     "Submit hardware replacement request"],
        "gt_decision": "resolve",
        "gt_draft": (
            "URGENT: Your hard drive clicking noise indicates potential failure. "
            "Please backup your data immediately and run a SMART health check. "
            "IT will prioritize hardware replacement for your device."
        ),
    },
    # ── SOFTWARE ──────────────────────────────────────────────────────
    {
        "id": "s1", "category": "software",
        "description": "Microsoft Excel crashes immediately when opening any .xlsx file.",
        "gt_steps": ["Launch Excel in Safe Mode via excel.exe /safe",
                     "Disable all Excel add-ins from Options > Add-ins",
                     "Run Office Repair from Apps & Features"],
        "gt_decision": "resolve",
        "gt_draft": (
            "We have received your Excel crash report. "
            "Please start Excel in Safe Mode and disable all add-ins. "
            "Running an Office Repair should resolve the issue within 15 minutes."
        ),
    },
    {
        "id": "s2", "category": "software",
        "description": "Outlook stuck in outbox — emails not sending, status shows disconnected.",
        "gt_steps": ["Switch Outlook to Work Offline then back online",
                     "Clear stuck email from Outbox folder",
                     "Recreate Outlook profile in Control Panel > Mail"],
        "gt_decision": "resolve",
        "gt_draft": (
            "Your Outlook sending issue has been received. "
            "Please toggle Work Offline mode and clear the Outbox folder. "
            "IT can recreate your profile if the issue persists."
        ),
    },
    {
        "id": "s3", "category": "software",
        "description": "Blue screen of death after latest Windows Update — error CRITICAL_PROCESS_DIED.",
        "gt_steps": ["Boot into Safe Mode and uninstall recent Windows Update",
                     "Run sfc /scannow and DISM health repair",
                     "Rollback driver that was updated alongside the patch"],
        "gt_decision": "resolve",
        "gt_draft": (
            "We have escalated your BSOD report to our Windows support team. "
            "Please boot in Safe Mode and uninstall the recent update. "
            "IT will deploy a clean image if system files are corrupted."
        ),
    },
    {
        "id": "s4", "category": "software",
        "description": "Zoom microphone not working in meetings — others cannot hear me.",
        "gt_steps": ["Check microphone permissions in Windows Privacy settings",
                     "Select correct microphone in Zoom Audio settings",
                     "Update Zoom to latest version"],
        "gt_decision": "resolve",
        "gt_draft": (
            "Thank you for reporting the Zoom microphone issue. "
            "Please check microphone permissions and select the correct device in Zoom settings. "
            "Updating Zoom to the latest version usually resolves this."
        ),
    },
    {
        "id": "s5", "category": "software",
        "description": "OneDrive sync stuck at 99% for 3 hours — files not uploading.",
        "gt_steps": ["Pause and resume OneDrive sync",
                     "Check available OneDrive cloud storage quota",
                     "Reset OneDrive via onedrive.exe /reset"],
        "gt_decision": "resolve",
        "gt_draft": (
            "We have received your OneDrive sync issue. "
            "Please pause and resume sync, check your storage quota, and reset OneDrive if needed. "
            "IT can assist with quota increases if required."
        ),
    },
    # ── ACCESS ────────────────────────────────────────────────────────
    {
        "id": "a1", "category": "access",
        "description": "Account locked after too many failed password attempts — cannot log in.",
        "gt_steps": ["Unlock account via IT admin console",
                     "Reset temporary password and force change on next login",
                     "Enable MFA to prevent future lockouts"],
        "gt_decision": "resolve",
        "gt_draft": (
            "Your account lockout has been received as high priority. "
            "IT will unlock your account and issue a temporary password within 30 minutes. "
            "Please verify your identity when our team contacts you."
        ),
    },
    {
        "id": "a2", "category": "access",
        "description": "MFA authenticator device lost — cannot log into any company systems.",
        "gt_steps": ["Verify identity via manager confirmation and employee ID",
                     "Generate backup codes from IT admin console",
                     "Enroll new MFA device and revoke old one"],
        "gt_decision": "resolve",
        "gt_draft": (
            "We have received your MFA emergency request. "
            "Please contact IT immediately with your employee ID and manager confirmation. "
            "We will issue backup codes and enroll a new device within 1 hour."
        ),
    },
    {
        "id": "a3", "category": "access",
        "description": "SSH key expired on production server — deployment pipeline broken.",
        "gt_steps": ["Generate new SSH key pair on the user workstation",
                     "Add new public key to authorized_keys on server",
                     "Revoke expired key and update CI/CD configuration"],
        "gt_decision": "escalate_to_level2",
        "gt_draft": (
            "Your SSH key expiry has been escalated to our infrastructure team. "
            "Please generate a new key pair and submit the public key to IT. "
            "A DevOps engineer will update server access within 2 hours."
        ),
    },
    {
        "id": "a4", "category": "access",
        "description": "Permission denied on shared project folder — needed for client presentation today.",
        "gt_steps": ["Verify requester is in correct Active Directory security group",
                     "Add user to folder ACL with Read permission",
                     "Have user log off and back on to refresh token"],
        "gt_decision": "resolve",
        "gt_draft": (
            "Your folder access request is being treated as urgent. "
            "IT will add you to the required security group and update permissions within 15 minutes. "
            "Please log off and back on after we confirm the change."
        ),
    },
    {
        "id": "a5", "category": "access",
        "description": "SAML SSO redirecting to login loop — cannot access internal web apps.",
        "gt_steps": ["Clear browser cookies and cached SAML tokens",
                     "Check IdP session timeout settings in admin portal",
                     "Test SSO with incognito window to isolate browser issue"],
        "gt_decision": "escalate_to_level2",
        "gt_draft": (
            "Your SSO redirect loop has been escalated to our identity team. "
            "Please clear browser cookies and test in an incognito window. "
            "Our identity engineer will investigate the IdP configuration."
        ),
    },
]


# ── Grader utilities ──────────────────────────────────────────────────
_STOP = {"a","an","the","is","are","was","were","be","been","to","of","and",
         "or","in","on","at","for","with","that","this","it","as","by","from",
         "not","but","have","has","had","do","did","will","would","could","should",
         "if","then","when","please","can","you","your","our","we","i","us","my"}

def _words(text: str) -> set:
    return {w for w in re.findall(r'[a-z]+', text.lower())
            if w not in _STOP and len(w) > 2}

def _jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb: return 0.
    return len(wa & wb) / len(wa | wb)

def _coverage(pred: str, gt: str) -> float:
    """Fraction of GT keywords present in prediction."""
    wg = _words(gt)
    if not wg: return 0.
    wp = _words(pred)
    return len(wg & wp) / len(wg)


# ── Per-task graders ──────────────────────────────────────────────────
def grade_classify(action: HelpdeskAction, ticket: Dict) -> Tuple[float, str]:
    gt  = ticket["category"]
    pred = (action.category or "").strip().lower()
    if pred == gt:
        return 1.0, f"Correct: {pred}"
    # Partial credit for semantically adjacent categories
    adjacent = {"network": {"hardware"}, "hardware": {"network"},
                "software": {"access"},  "access": {"software"}}
    if pred in adjacent.get(gt, set()):
        return 0.3, f"Adjacent category: predicted={pred} gt={gt}"
    return 0.0, f"Wrong: predicted={pred} gt={gt}"


def grade_steps(action: HelpdeskAction, ticket: Dict) -> Tuple[float, str]:
    pred_steps = action.steps or []
    if not pred_steps:
        return 0.0, "No steps provided"
    gt_steps = ticket["gt_steps"]
    # Per-step: best Jaccard match against any GT step
    step_scores = []
    for ps in pred_steps[:4]:  # cap at 4
        best = max(_jaccard(ps, gs) for gs in gt_steps)
        step_scores.append(best)
    # Coverage: how many GT steps are covered
    gt_joined   = " ".join(gt_steps)
    pred_joined = " ".join(pred_steps)
    cov = _coverage(pred_joined, gt_joined)
    quality = float(sum(step_scores) / len(step_scores)) if step_scores else 0.
    score   = 0.5 * quality + 0.5 * cov
    bonus   = 0.1 if 2 <= len(pred_steps) <= 4 else 0.   # appropriate length
    return float(min(1., score + bonus)), f"quality={quality:.2f} coverage={cov:.2f}"


def grade_decide(action: HelpdeskAction, ticket: Dict) -> Tuple[float, str]:
    gt   = ticket["gt_decision"]
    pred = (action.decision or "").strip().lower()
    if pred == gt:
        return 1.0, f"Correct: {pred}"
    # Partial: network should escalate; penalise wrong direction more
    if ticket["category"] == "network" and pred == "resolve":
        return 0.0, "Network issues must escalate, not resolve"
    if ticket["category"] != "network" and pred == "escalate_to_level2":
        return 0.2, "Escalation not required for this category"
    return 0.0, f"Wrong decision: predicted={pred} gt={gt}"


_EMPATHY = {"thank","received","investigating","follow","contact","resolve",
            "support","will","team","urgent","priorit","escalat","assigned"}

def grade_draft(action: HelpdeskAction, ticket: Dict) -> Tuple[float, str]:
    pred = (action.draft or "").strip()
    if not pred:
        return 0.0, "No draft provided"
    gt   = ticket["gt_draft"]

    # 1. Semantic overlap with GT (weight 0.50)
    overlap = min(1., _coverage(pred, gt) * 1.5)   # generous scaling

    # 2. Appropriate length (weight 0.20)
    wc = len(pred.split())
    if   15 <= wc <= 80: length = 1.0
    elif  8 <= wc < 15:  length = 0.6
    elif     wc > 80:    length = 0.7
    else:                length = 0.2

    # 3. Empathy / professionalism keywords (weight 0.30)
    pred_l = pred.lower()
    hits   = sum(1 for w in _EMPATHY if w in pred_l)
    emp    = min(1., hits * 0.2)

    score  = 0.50 * overlap + 0.20 * length + 0.30 * emp
    return float(round(min(1., score), 4)), \
           f"overlap={overlap:.2f} length_wc={wc} empathy={emp:.2f}"


GRADERS = {
    "classify_category":   grade_classify,
    "suggest_steps":        grade_steps,
    "escalate_or_resolve":  grade_decide,
    "draft_response":       grade_draft,
}


# ── Main Environment ──────────────────────────────────────────────────
class HelpdeskEnv(OpenEnvBase):
    """
    IT Helpdesk OpenEnv-compliant environment.

    Episode flow per task_name:
      reset(task_name) → HelpdeskObservation
      step(action)     → (obs, reward, done, info)  ×  TICKETS_PER_TASK steps
      state()          → HelpdeskState
      close()

    Tasks (task_id):
      0  classify_category    (easy)    — pick network|hardware|software|access
      1  suggest_steps        (medium)  — list 2-3 troubleshooting steps
      2  escalate_or_resolve  (medium)  — decide resolve|escalate_to_level2
      3  draft_response       (hard)    — write professional user response
    """

    TASK_BY_NAME = {t["name"]: t for t in TASKS}
    TASK_BY_ID   = {t["id"]:   t for t in TASKS}

    def __init__(self, seed: int = 42):
        self._rng   = random.Random(seed)
        self._task  = TASKS[0]
        self._queue: List[Dict] = []
        self._step_n = 0
        self._score  = 0.
        self._done   = False
        self._last_r = 0.
        self._last_e: Optional[str] = None

    # ── OpenEnv interface ─────────────────────────────────────────────
    def reset(self, task_name: Optional[str] = None,
              seed: Optional[int] = None) -> HelpdeskObservation:
        if seed is not None:
            self._rng = random.Random(seed)
        if task_name is not None and task_name not in self.TASK_BY_NAME:
            raise ValueError(f"Unknown task: {task_name}. Valid: {list(self.TASK_BY_NAME)}")

        self._task   = self.TASK_BY_NAME.get(task_name, TASKS[0])
        # Build per-task ticket queue: sample TICKETS_PER_TASK tickets
        cat_tickets  = {cat: [t for t in TICKET_POOL if t["category"] == cat]
                        for cat in ("network","hardware","software","access")}
        ordered: List[Dict] = []
        for cat in ("network","hardware","software","access"):
            pool = cat_tickets[cat]
            self._rng.shuffle(pool)
            ordered.extend(pool[:max(1, TICKETS_PER_TASK // 4)])
        # Pad to exactly TICKETS_PER_TASK
        all_t = list(TICKET_POOL)
        self._rng.shuffle(all_t)
        while len(ordered) < TICKETS_PER_TASK:
            ordered.append(all_t[len(ordered) % len(all_t)])
        self._queue  = ordered[:TICKETS_PER_TASK]
        self._step_n = 0
        self._score  = 0.
        self._done   = False
        self._last_r = 0.
        self._last_e = None
        return self._make_obs()

    def step(self, action: HelpdeskAction) -> Tuple[HelpdeskObservation, float, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode done — call reset() first")

        ticket  = self._queue[self._step_n]
        grader  = GRADERS[self._task["name"]]
        info: Dict[str, Any] = {"last_action_error": None, "ticket_id": ticket["id"]}

        # Validate action type matches task
        expected_action = self._task["action"]
        if action.action != expected_action:
            reward = 0.0
            info["last_action_error"] = (
                f"Wrong action type '{action.action}' for task "
                f"'{self._task['name']}' (expected '{expected_action}')"
            )
        else:
            try:
                reward, rationale = grader(action, ticket)
                info["rationale"]  = rationale
                info["gt_category"] = ticket["category"]
            except Exception as exc:
                reward = 0.0
                info["last_action_error"] = f"Grader error: {exc}"

        self._score  += reward
        self._last_r  = reward
        self._last_e  = info["last_action_error"]
        self._step_n += 1
        self._done    = self._step_n >= len(self._queue)

        obs = self._make_obs() if not self._done else self._make_obs(terminal=True)
        return obs, float(reward), self._done, info

    def state(self) -> HelpdeskState:
        return HelpdeskState(
            task_name    = self._task["name"],
            task_id      = self._task["id"],
            step         = self._step_n,
            total_steps  = TICKETS_PER_TASK,
            score        = round(self._score, 4),
            done         = self._done,
            tickets_seen = self._step_n,
            last_reward  = round(self._last_r, 4),
        )

    def close(self):
        pass

    # ── Internal ──────────────────────────────────────────────────────
    def _make_obs(self, terminal: bool = False) -> HelpdeskObservation:
        if terminal or self._step_n >= len(self._queue):
            ticket_desc = "[Episode complete]"
        else:
            ticket_desc = self._queue[self._step_n]["description"]
        return HelpdeskObservation(
            ticket      = ticket_desc,
            task_name   = self._task["name"],
            task_id     = self._task["id"],
            step        = self._step_n,
            total_steps = TICKETS_PER_TASK,
            hint        = HINTS[self._task["name"]],
        )

    # ── Convenience ───────────────────────────────────────────────────
    @property
    def task_names(self) -> List[str]:
        return [t["name"] for t in TASKS]

    def score(self) -> float:
        """Normalised score in [0, 1]: total_reward / max_possible_reward."""
        max_r = float(TICKETS_PER_TASK)
        return round(self._score / max_r, 4) if max_r > 0 else 0.


# ── Quick smoke-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    env = HelpdeskEnv(seed=42)
    for task in env.task_names:
        obs = env.reset(task_name=task)
        total = 0.
        while True:
            # Simulate a "smart" agent that always picks the correct action type
            if task == "classify_category":
                # keyword-based guess
                d = obs.ticket.lower()
                cat = "network" if any(w in d for w in ["vpn","wifi","network","dns"]) else \
                      "hardware" if any(w in d for w in ["battery","screen","printer","hard"]) else \
                      "software" if any(w in d for w in ["excel","outlook","zoom","update"]) else "access"
                action = HelpdeskAction(action="classify", category=cat)
            elif task == "suggest_steps":
                action = HelpdeskAction(action="suggest_steps",
                    steps=["Restart the device","Check settings","Contact IT if issue persists"])
            elif task == "escalate_or_resolve":
                d = obs.ticket.lower()
                dec = "escalate_to_level2" if any(w in d for w in ["vpn","network","packet","dns","ssh","sso"]) \
                      else "resolve"
                action = HelpdeskAction(action="decide", decision=dec)
            else:
                action = HelpdeskAction(action="draft",
                    draft="We have received your ticket and are investigating. "
                          "Our support team will follow up with you shortly. "
                          "Please contact IT if the issue is urgent.")

            obs, r, done, info = env.step(action)
            total += r
            if done: break

        print(f"Task: {task:<25} score={env.score():.3f}  total_r={total:.2f}")
    env.close()
