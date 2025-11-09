# coach.py
# Generates coaching cues from phase-aware features with LLM (optional) and rule-based fallback.
# Uses shared thresholds/formatting from jwcore.coaching_thresholds.

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Shared thresholds + helpers (single source of truth)
from jwcore.coaching_thresholds import (
    TAU,
    UNITS,
    PHASE_TAG,
    select_top_metrics,
)

CoachLLM = Callable[[str], str]


# =========================
# System / style prompt
# =========================

SYSTEM_PROMPT = (
    "You are a professional freeride-ski and trampoline performance coach "
    "working with advanced athletes. "
    "Your feedback must be biomechanically precise, numeric-data-grounded, "
    "and use technical language (torso angle, hip extension timing, takeoff symmetry, "
    "rotation axis control, arm path, stance width). "
    "Assume the athlete knows fundamentals; focus on specific joint-level deviations "
    "between athlete and pro. "
    "Use only values and metrics provided; do NOT invent numbers or phases. "
    "Include safety considerations where relevant. "
    "Tone: concise, analytical, non-motivational."
)


# =========================
# Utilities
# =========================

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
] + [f"joint_{i}" for i in range(21, 33)]


def _heuristic_tips(features: Dict[str, float], label: Optional[str]) -> List[str]:
    """Tiny safety net if LLM fails or nothing material surfaces."""
    tips: List[str] = []
    airtime = float(features.get("airtime_s", float("nan")))
    pitch = float(features.get("pitch_total_rad", float("nan")))
    knees = float(features.get("knee_angle_landing_deg", float("nan")))

    if math.isfinite(airtime) and airtime < 0.6:
        tips.append("Increase jump height; stronger, more vertical set to add airtime.")
    if math.isfinite(pitch) and abs(pitch) < 0.25:
        tips.append("Commit to full rotation; drive the set more before tucking.")
    if math.isfinite(knees) and knees < 10:
        tips.append("Soften the landing; allow knees to flex and track over toes.")
    if not tips:
        tips = ["Solid base—refine set timing and keep arm path clean."]
    return tips


def _compute_feature_deltas(am: Dict[str, float], pro: Dict[str, float]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    keys = set(am.keys()) | set(pro.keys())
    for k in keys:
        try:
            av = float(am.get(k, 0.0))
            pv = float(pro.get(k, 0.0))
            dv = av - pv
            if math.isfinite(dv):
                deltas[k] = dv
        except Exception:
            continue
    return deltas


def _summarize_joint_diffs_for_prompt(
    joint_diffs: np.ndarray,
    torso_lengths: Optional[np.ndarray] = None,
    top_n: int = 6,
) -> List[str]:
    """
    Token-cheap overview of joint deviations (for strengths / context only).
    joint_diffs: (T, J, 3) amateur - pro in torso space
    torso_lengths: (T,) optional, to scale deltas back into torso units
    """
    if joint_diffs is None or joint_diffs.ndim != 3:
        return []

    T, J, _ = joint_diffs.shape
    dists = np.linalg.norm(joint_diffs[:, :, :2], axis=2)  # (T, J)
    mean_d = np.nanmean(dists, axis=0)  # (J,)

    if torso_lengths is not None and torso_lengths.size == T:
        torso_mean = float(np.nanmean(torso_lengths))
        if math.isfinite(torso_mean) and torso_mean > 1e-6:
            mean_d = mean_d / torso_mean

    pairs: List[tuple[str, float]] = []
    for j in range(min(J, len(JOINT_NAMES))):
        v = float(mean_d[j])
        if math.isfinite(v):
            pairs.append((JOINT_NAMES[j], v * 100.0))  # percent of torso
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)

    return [f"{name}: {val:.1f}% of torso" for name, val in pairs[:max(1, top_n)]]


def _fundamental_for_label(label: Optional[str]) -> str:
    l = (label or "").lower()
    if "front" in l:
        return "Set tall, drive hips forward, then tuck—avoid throwing shoulders down early."
    if "back" in l:
        return "Finish a tall set with eyes forward before dropping shoulders into the backflip."
    if "barani" in l or "off" in l or "axis" in l:
        return "Square shoulders and hips to travel direction before initiating off-axis rotation."
    if "twist" in l or "cork" in l:
        return "Stack hips over core and establish clean flip axis before adding twist."
    return "Maintain stacked posture and soft, aligned landing (knees over toes)."


def _strength_lines(features: Dict[str, float]) -> List[str]:
    out: List[str] = []
    try:
        v = float(features.get("midair_hand_asym_pct_of_torso", float("nan")))
        if math.isfinite(v) and abs(v) < 6.0:
            out.append("Symmetric arm path through midair.")
    except Exception:
        pass

    try:
        sw = float(features.get("landing_stance_width_pct_of_hip", float("nan")))
        if math.isfinite(sw) and 90.0 <= sw <= 110.0:
            out.append("Neutral, centered landing stance.")
    except Exception:
        pass

    if not out:
        out = ["Clean overall line.", "Stable landing mechanics."]
    return out[:2]


def _short_series_block(series: Dict[str, Any]) -> List[str]:
    """
    Render a tiny, token-cheap view of key time-series if present.
    Expect values already downsampled (e.g. from compare_metrics.py).
    """
    if not isinstance(series, dict):
        return []

    keys = [
        "pitch_deg_am",
        "pitch_deg_pro",
        "head_pitch_deg_am",
        "head_pitch_deg_pro",
        "knee_mean_deg_am",
        "knee_mean_deg_pro",
    ]
    lines: List[str] = []
    for k in keys:
        if k in series:
            try:
                arr = [float(x) for x in series[k]]
                if not arr:
                    continue
                # Light compression: 8 samples max
                idx = np.linspace(0, len(arr) - 1, num=min(8, len(arr)), dtype=int)
                samp = [round(arr[i], 1) for i in idx]
                lines.append(f"    {k}: {samp}")
            except Exception:
                continue
    return lines


def _format_prompt(
    features: Dict[str, Any],
    label: Optional[str],
    pro_name: Optional[str],
    pro_features: Optional[Dict[str, float]],
    joint_diffs: Optional[np.ndarray],
    torso_lengths: Optional[np.ndarray],
    top_metrics: List[Dict[str, Any]],
    series: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = [SYSTEM_PROMPT, "", "CONTEXT:"]

    if label:
        lines.append(f"- predicted_label: {label}")
    if pro_name:
        lines.append(f"- pro_reference: {pro_name}")

    # Preselected, thresholded metrics (single source of truth: select_top_metrics)
    lines.append("- top_metrics (preselected; |delta| >= τ):")
    if top_metrics:
        for e in top_metrics:
            key = e["key"]
            av = e.get("athlete")
            pv = e.get("ref")
            d = e.get("delta")
            units = e.get("units") or UNITS.get(key, "")
            tag = e.get("phase_tag") or PHASE_TAG.get(key, "")
            try:
                lines.append(
                    f"    {key}: athlete={float(av):.3f}, pro={float(pv):.3f} "
                    f"Δ={float(d):+ .3f} {units} @{tag}"
                )
            except Exception:
                lines.append(
                    f"    {key}: athlete={av}, pro={pv}, Δ={d} {units} @{tag}"
                )
    else:
        lines.append("    (none above thresholds)")

    # Very small scalar context (non-decisive)
    for k in ("airtime_s", "pitch_total_rad"):
        try:
            v = float(features.get(k, float("nan")))
            if math.isfinite(v):
                lines.append(f"- {k}: {v:.4f}")
        except Exception:
            continue

    # Optional: compact time-series snapshot
    if series:
        ts_lines = _short_series_block(series)
        if ts_lines:
            lines.append("- time_series_samples (normalized 0-100% of trick):")
            lines.extend(ts_lines)

    # Optional: joint-based strengths hints
    if joint_diffs is not None:
        jd_lines = _summarize_joint_diffs_for_prompt(joint_diffs, torso_lengths, top_n=4)
        if jd_lines:
            lines.append("- joint_strengths_hints (for positive notes only):")
            for ln in jd_lines:
                lines.append(f"    {ln}")

    # Task: bullets only, tightly constrained
    lines += [
        "",
        "TASK:",
        "1) If there are at least 2 top_metrics:",
        "   - Output up to 3 bullets.",
        "   - Each bullet must:",
        "       * Target ONE top_metric.",
        "       * Be ≤120 characters.",
        "       * Use precise coaching language (body part + phase + action).",
        "       * End with (evidence: metric_key@phase_tag).",
        "   - Use only provided metric keys and tags.",
        "2) If fewer than 2 top_metrics:",
        "   - Do NOT invent metric-based tips.",
        "   - Output exactly two lines:",
        "       Strengths: <one or two positives, no numbers>",
        "       Fundamental: <one key generic cue based on trick family>",
        "3) No extra sections, headers, or explanations.",
        "",
        "OUTPUT:",
        "Return only the bullets/lines as specified above.",
    ]

    return "\n".join(lines)


# =========================
# Public API
# =========================

def coach(
    features: Dict[str, Any],
    predicted_label: Optional[str] = None,
    pro_name: Optional[str] = None,
    pro_features: Optional[Dict[str, float]] = None,
    llm: Optional[CoachLLM] = None,
) -> Dict[str, Any]:
    """
    Produce coaching tips from phase-aware features.

    Expects `features` to already contain scalar metrics (e.g. from compare_metrics.py),
    and optionally:
      - features["series"]: compact time-series dict
      - features["joint_diffs"], features["torso_lengths"]: for context only

    Behavior:
      - If <2 material deltas (per τ) or no LLM:
          return strengths + one fundamental (rule-based).
      - Else:
          build a strict prompt around top_metrics and call LLM.
    """
    if not isinstance(features, dict):
        return {
            "label": predicted_label or "",
            "pro_reference": pro_name or "",
            "tips": [],
            "source": "none",
        }

    phase_scores = features.get("phase_scores") or {}

    # Optional extras from features
    series = features.get("series") if isinstance(features.get("series"), dict) else None

    joint_diffs = None
    torso_lengths = None
    if "joint_diffs" in features:
        try:
            joint_diffs = np.asarray(features["joint_diffs"])
        except Exception:
            joint_diffs = None
    if "torso_lengths" in features:
        try:
            torso_lengths = np.asarray(features["torso_lengths"]).reshape(-1)
        except Exception:
            torso_lengths = None


    # Phase-aware scores (if provided by compare_metrics)
    phase_scores = features.get("phase_scores") or {}

    # Candidate metrics for coaching focus — all thresholded via select_top_metrics
    candidate_keys = [
        "midair_hand_span_pct_of_torso",
        "midair_hand_asym_pct_of_torso",
        "landing_stance_width_pct_of_hip",
        "pitch_total_rad",
        "ankle_dev_pct_of_torso_max",
        "hand_dev_pct_of_torso_max",
        "leg_axis_diff_deg_apex",
        "head_early_pitch_lead_deg",
    ]

    # Start with a generous set; we'll τ-rank and trim.
    top_metrics = select_top_metrics(
        am_features=features,
        pro_features=pro_features,
        keys=candidate_keys,
        max_items=8,
    )

    def _tau_weighted(m: dict) -> float:
        """Score = |delta| / τ; higher = more materially off."""
        try:
            tau = float(TAU.get(m.get("key", ""), 1.0) or 1.0)
            d = float(m.get("delta", 0.0))
            return abs(d) / max(tau, 1e-6)
        except Exception:
            return 0.0

    # 1) Sort all candidates by |delta| / τ (most material first)
    top_metrics = sorted(top_metrics, key=_tau_weighted, reverse=True)

    # 2) Ensure clearly material head_early_pitch_lead_deg is represented
    HEAD_KEY = "head_early_pitch_lead_deg"
    tau_head = float(TAU.get(HEAD_KEY, 8.0) or 8.0)

    def _is_strong_head() -> bool:
        v = features.get(HEAD_KEY, None)
        try:
            v = float(v)
        except Exception:
            v = None

        ps = phase_scores.get(HEAD_KEY) or {}
        ratio_set = None
        if "set" in ps:
            try:
                ratio_set = float(ps["set"])
            except Exception:
                ratio_set = None

        return (
            (v is not None and abs(v) >= tau_head)
            or (ratio_set is not None and abs(ratio_set) >= 1.0)
        )

    # Check if head metric already present
    head_idx = next((i for i, m in enumerate(top_metrics) if m.get("key") == HEAD_KEY), None)

    # If strong but missing, synthesize and insert a head_early entry
    if head_idx is None and _is_strong_head():
        v = float(features.get(HEAD_KEY, 0.0))
        top_metrics.append({
            "key": HEAD_KEY,
            "athlete": v,
            "ref": 0.0,
            "delta": v,
            "delta_abs": abs(v),
            "units": UNITS.get(HEAD_KEY, "deg"),
            "phase_tag": PHASE_TAG.get(HEAD_KEY, "set@0-25%"),
        })
        top_metrics = sorted(top_metrics, key=_tau_weighted, reverse=True)
        head_idx = next((i for i, m in enumerate(top_metrics) if m.get("key") == HEAD_KEY), None)

    # 3) Enforce top-3, but if strong head metric sits below index 2, bump it in.
    if len(top_metrics) > 3:
        if head_idx is not None and head_idx >= 3 and _is_strong_head():
            weakest_idx = min(range(3), key=lambda i: _tau_weighted(top_metrics[i]))
            # swap head metric into that slot
            top_metrics[weakest_idx], top_metrics[head_idx] = (
                top_metrics[head_idx],
                top_metrics[weakest_idx],
            )
        top_metrics = top_metrics[:3]



    # If no LLM or not enough metrics, use deterministic, metric-aware fallback.
    if llm is None or len(top_metrics) < 2:
        tips: List[str] = []

        # 1) If we have material top_metrics, turn them directly into cues.
        for m in top_metrics[:3]:
            k = m["key"]
            tag = m.get("phase_tag") or PHASE_TAG.get(k, "")
            if k == "ankle_dev_pct_of_torso_max":
                tips.append(f"Square lower legs with torso around apex. (evidence: {k}@{tag})")
            elif k == "hand_dev_pct_of_torso_max":
                tips.append(f"Track hands closer to trunk midair to clean the line. (evidence: {k}@{tag})")
            elif k == "landing_stance_width_pct_of_hip":
                if m["delta"] > 0:
                    tips.append(f"Land with a narrower stance closer to hip width. (evidence: {k}@{tag})")
                else:
                    tips.append(f"Avoid landing too narrow; keep feet nearer hip width. (evidence: {k}@{tag})")
            elif k == "midair_hand_span_pct_of_torso":
                tips.append(f"Bring hands in toward shoulders at midair for tighter control. (evidence: {k}@{tag})")
            elif k == "midair_hand_asym_pct_of_torso":
                tips.append(f"Match left/right arm height to remove asymmetry. (evidence: {k}@{tag})")
            elif k == "leg_axis_diff_deg_apex":
                tips.append(f"Align leg line with torso at apex; avoid bending the chain. (evidence: {k}@{tag})")
            elif k == "head_early_pitch_lead_deg":
                tips.append(f"Delay head throw; keep eyes level until set is complete. (evidence: {k}@{tag})")

        # 2) If nothing came out (no top_metrics), fall back to old strengths+fundamental.
        if not tips:
            strengths = _strength_lines(features)
            fundamental = _fundamental_for_label(predicted_label)
            tips = strengths[:2] + [f"Fundamental: {fundamental}"]

        out = {
            "label": predicted_label or "",
            "pro_reference": pro_name or "",
            "tips": tips[:5],
            "source": "rule",
            "meta": {"top_metrics": top_metrics, "tau": TAU},
        }
        if pro_features:
            out["deltas"] = _compute_feature_deltas(features, pro_features)
        return out


    # LLM path
    prompt = _format_prompt(
        features=features,
        label=predicted_label,
        pro_name=pro_name,
        pro_features=pro_features,
        joint_diffs=joint_diffs,
        torso_lengths=torso_lengths,
        top_metrics=top_metrics,
        series=series,
    )

    raw = ""
    try:
        raw = llm(prompt) if llm else ""
    except Exception:
        raw = ""

    # Parse bullets: non-empty lines, strip list bullets
    lines = [ln.strip(" -•\t") for ln in str(raw).splitlines() if ln.strip()]
    if not lines:
        # If model failed, drop to heuristic fallback
        lines = _heuristic_tips(features, predicted_label)

    out = {
        "label": predicted_label or "",
        "pro_reference": pro_name or "",
        "tips": lines[:5],
        "source": "llm" if llm else "rule",
        "meta": {
            "prompt_len": len(prompt),
            "top_metrics": top_metrics,
            "tau": TAU,
        },
    }
    if pro_features:
        out["deltas"] = _compute_feature_deltas(features, pro_features)
    return out
