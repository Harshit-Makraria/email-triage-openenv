"""
Grading functions for the three email triage tasks.

Each function returns (score: float, feedback: str) where score ∈ [0.0, 1.0].
"""

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr as _spearmanr

    def _spearman(x: List[float], y: List[float]) -> float:
        if len(set(x)) <= 1 or len(set(y)) <= 1:
            return 0.0
        corr, _ = _spearmanr(x, y)
        return float(corr) if not np.isnan(corr) else 0.0

except ImportError:
    def _spearman(x: List[float], y: List[float]) -> float:
        """Fallback Spearman without scipy."""
        n = len(x)
        if n < 2:
            return 0.0

        def _ranks(lst):
            sorted_idx = sorted(range(n), key=lambda i: lst[i])
            r = [0.0] * n
            for rank, idx in enumerate(sorted_idx):
                r[idx] = rank + 1.0
            return r

        rx, ry = _ranks(x), _ranks(y)
        d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
        corr = 1 - (6 * d_sq) / (n * (n * n - 1))
        return float(corr)


# ─────────────────────────────────────────────────────────────
# Component graders
# ─────────────────────────────────────────────────────────────

def score_classification(
    email_actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
) -> Tuple[float, str]:
    """Accuracy of category labels."""
    if not email_actions:
        return 0.0, "No actions submitted"

    correct = 0
    parts: List[str] = []
    for act in email_actions:
        eid = act.get("email_id", "")
        if eid not in ground_truth:
            continue
        gt_cat = ground_truth[eid]["category"]
        ag_cat = act.get("category", "")
        if ag_cat == gt_cat:
            correct += 1
            parts.append(f"{eid}:✓{gt_cat}")
        else:
            parts.append(f"{eid}:✗({ag_cat}≠{gt_cat})")

    total = len(ground_truth)
    score = correct / total if total > 0 else 0.0
    feedback = "Classification — " + ", ".join(parts[:6])
    return round(score, 4), feedback


def score_priority_ranking(
    email_actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
) -> Tuple[float, str]:
    """Spearman rank correlation between agent priorities and ground truth."""
    if not email_actions:
        return 0.0, "No actions submitted"

    eids = sorted(ground_truth.keys())
    gt_ranks = [float(ground_truth[e]["priority"]) for e in eids]

    max_p = len(eids) + 1
    agent_map = {
        a["email_id"]: float(a.get("priority", max_p))
        for a in email_actions
        if "email_id" in a
    }
    ag_ranks = [agent_map.get(e, float(max_p)) for e in eids]

    corr = _spearman(gt_ranks, ag_ranks)
    score = (corr + 1.0) / 2.0  # rescale [-1, 1] → [0, 1]

    # Highlight big mistakes
    mistakes = [
        f"{e}(gt={int(g)},got={int(a)})"
        for e, g, a in zip(eids, gt_ranks, ag_ranks)
        if abs(g - a) > 2
    ]
    feedback = f"Priority ranking — Spearman={corr:.3f}, score={score:.3f}"
    if mistakes:
        feedback += "; errors: " + ", ".join(mistakes[:4])
    return round(score, 4), feedback


def score_actions(
    email_actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
) -> Tuple[float, str]:
    """Correctness of chosen actions (read/archive/delete/respond/flag)."""
    if not email_actions:
        return 0.0, "No actions submitted"

    total = len(ground_truth)
    credit = 0.0
    parts: List[str] = []
    for act in email_actions:
        eid = act.get("email_id", "")
        if eid not in ground_truth:
            continue
        gt_act = ground_truth[eid]["action"]
        ag_act = act.get("action", "")
        if ag_act == gt_act:
            credit += 1.0
            parts.append(f"{eid}:✓{gt_act}")
        elif {ag_act, gt_act} <= {"archive", "read"}:
            # read vs archive are close — half credit
            credit += 0.5
            parts.append(f"{eid}:~({ag_act}≈{gt_act})")
        else:
            parts.append(f"{eid}:✗({ag_act}≠{gt_act})")

    score = credit / total if total > 0 else 0.0
    feedback = "Actions — " + ", ".join(parts[:6])
    return round(score, 4), feedback


def score_responses(
    email_actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
) -> Tuple[float, str]:
    """Quality of response drafts for emails that require one."""
    need_resp = {eid: gt for eid, gt in ground_truth.items() if gt.get("needs_response")}
    if not need_resp:
        return 1.0, "No responses required for this task"

    action_map = {a["email_id"]: a for a in email_actions if "email_id" in a}
    total_score = 0.0
    parts: List[str] = []

    for eid, gt in need_resp.items():
        act = action_map.get(eid, {})
        draft = (act.get("response_draft") or "").strip()

        if not draft or len(draft) < 10:
            parts.append(f"{eid}:✗missing")
            continue

        words = draft.split()
        word_count = len(words)
        draft_lower = draft.lower()
        keywords = gt.get("response_keywords", [])

        # Length component (0–0.30): ≥20 words is full credit
        length_score = min(word_count / 20.0, 1.0) * 0.30

        # Keyword component (0–0.40): fraction of relevant keywords mentioned
        if keywords:
            matched = sum(1 for kw in keywords if kw.lower() in draft_lower)
            keyword_score = (matched / len(keywords)) * 0.40
        else:
            keyword_score = 0.40

        # Format component (0–0.30): greeting + sign-off
        greetings = ["hi", "hello", "dear", "greetings", "good morning", "good afternoon",
                     "thank you for", "thanks for reaching"]
        signoffs = ["regards", "sincerely", "thanks", "best", "cheers",
                    "kind regards", "thank you", "looking forward"]
        has_greet = any(g in draft_lower for g in greetings)
        has_sign = any(s in draft_lower for s in signoffs)
        format_score = (0.15 if has_greet else 0.0) + (0.15 if has_sign else 0.0)

        item_score = length_score + keyword_score + format_score
        total_score += item_score
        kw_hit = matched if keywords else "-"
        parts.append(f"{eid}:{item_score:.2f}(w={word_count},kw={kw_hit}/{len(keywords)})")

    final = total_score / len(need_resp)
    feedback = "Responses — " + ", ".join(parts[:5])
    return round(final, 4), feedback


# ─────────────────────────────────────────────────────────────
# Composite grader
# ─────────────────────────────────────────────────────────────

def grade_task(
    task_id: str,
    email_actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute a weighted composite score for a full set of email actions.

    Returns:
        {
            "total_score": float,           # ∈ [0, 1]
            "component_scores": {...},
            "feedback": str,
        }
    """
    cls_score, cls_fb = score_classification(email_actions, ground_truth)
    pri_score, pri_fb = score_priority_ranking(email_actions, ground_truth)
    act_score, act_fb = score_actions(email_actions, ground_truth)

    components: Dict[str, float] = {
        "classification": cls_score,
        "priority": pri_score,
        "action": act_score,
    }
    feedback_parts = [cls_fb, pri_fb, act_fb]

    if weights.get("response", 0.0) > 0.0:
        resp_score, resp_fb = score_responses(email_actions, ground_truth)
        components["response"] = resp_score
        feedback_parts.append(resp_fb)
    else:
        components["response"] = 0.0

    total = sum(components.get(k, 0.0) * v for k, v in weights.items())
    total = round(max(0.0, min(1.0, total)), 4)

    return {
        "total_score": total,
        "component_scores": components,
        "feedback": " | ".join(feedback_parts),
    }
