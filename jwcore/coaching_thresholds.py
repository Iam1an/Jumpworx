# jwcore/coaching_thresholds.py
# Centralized thresholds + helpers for selecting material coaching metrics.

from __future__ import annotations

from typing import Any, Dict, List, Optional

# =========================
# Thresholds (τ) by metric
# =========================
# Interpretation: |delta| >= TAU[key] => worth talking about.

TAU: Dict[str, float] = {
    # Hands / arms
    "midair_hand_span_pct_of_torso": 10.0,    # % torso; >10% wider/narrower is noticeable
    "midair_hand_asym_pct_of_torso": 8.0,     # % torso; asymmetry

    # Landing stance
    "landing_stance_width_pct_of_hip": 15.0,  # % of hip width

    # Global rotation
    "pitch_total_rad": 0.35,                  # ~20 degrees net diff

    # Apex deviations (torso units as %)
    "ankle_dev_pct_of_torso_max": 18.0,       # foot/ankle line vs pro
    "hand_dev_pct_of_torso_max": 18.0,        # hand track vs pro

    # Shape / axis
    "leg_axis_diff_deg_apex": 8.0,            # degrees; leg line vs torso at apex

    # Set / head timing
    "head_early_pitch_lead_deg": 8.0,         # degrees; early head throw vs pro
}

# =========================
# Units (for display / LLM)
# =========================

UNITS: Dict[str, str] = {
    "midair_hand_span_pct_of_torso": "% torso",
    "midair_hand_asym_pct_of_torso": "% torso",
    "landing_stance_width_pct_of_hip": "% hip_width",
    "pitch_total_rad": "rad",
    "ankle_dev_pct_of_torso_max": "% torso",
    "hand_dev_pct_of_torso_max": "% torso",
    "leg_axis_diff_deg_apex": "deg",
    "head_early_pitch_lead_deg": "deg",
}

# =========================
# Phase tags (for evidence)
# =========================

PHASE_TAG: Dict[str, str] = {
    "midair_hand_span_pct_of_torso": "midair@40-60%",
    "midair_hand_asym_pct_of_torso": "midair@40-60%",
    "landing_stance_width_pct_of_hip": "landing@85-100%",
    "pitch_total_rad": "global@0-100%",
    "ankle_dev_pct_of_torso_max": "apex@midair",
    "hand_dev_pct_of_torso_max": "apex@midair",
    "leg_axis_diff_deg_apex": "apex@midair",
    "head_early_pitch_lead_deg": "set@0-25%",
}

# =========================
# Helpers
# =========================

def material_delta(metric: str, delta: float) -> bool:
    """Return True if |delta| is above the configured τ for this metric."""
    tau = TAU.get(metric, 0.0)
    try:
        return abs(float(delta)) >= float(tau)
    except Exception:
        return False


def _extract_pair(
    am_features: Dict[str, Any],
    pro_features: Optional[Dict[str, Any]],
    key: str,
) -> Optional[Dict[str, float]]:
    """
    Extract athlete/ref values and delta for a metric key.

    Priority:
      1) If both am_features[key] and pro_features[key] exist → use those.
      2) Else if am_features has 'delta_<key>' → interpret as delta vs ref=0.
      3) Else if only am_features[key] exists → delta vs ref=0 (rare, but safe).
    """
    if key not in am_features and (not pro_features or key not in pro_features):
        # Try delta_* form
        dk = f"delta_{key}"
        if dk in am_features:
            try:
                d = float(am_features[dk])
                return {"athlete": d, "ref": 0.0, "delta": d}
            except Exception:
                return None
        return None

    try:
        av = float(am_features.get(key, 0.0))
    except Exception:
        return None

    if pro_features and key in pro_features:
        try:
            pv = float(pro_features[key])
        except Exception:
            pv = 0.0
    else:
        pv = 0.0

    try:
        dv = float(av - pv)
    except Exception:
        return None

    return {"athlete": av, "ref": pv, "delta": dv}


def select_top_metrics(
    am_features: Dict[str, Any],
    pro_features: Optional[Dict[str, Any]],
    keys: List[str],
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    """
    From the given candidate metric keys, pick up to `max_items` that:
      - have valid athlete/ref values,
      - exceed their τ threshold in absolute delta,
      - and are sorted by |delta| descending.

    Returns list of dicts:
      {
        "key": str,
        "athlete": float,
        "ref": float,
        "delta": float,
        "delta_abs": float,
        "units": str,
        "phase_tag": str,
      }
    """
    items: List[Dict[str, Any]] = []

    for key in keys:
        pair = _extract_pair(am_features, pro_features, key)
        if not pair:
            continue

        delta = pair["delta"]
        if not material_delta(key, delta):
            continue

        units = UNITS.get(key, "")
        phase_tag = PHASE_TAG.get(key, "")

        items.append({
            "key": key,
            "athlete": pair["athlete"],
            "ref": pair["ref"],
            "delta": delta,
            "delta_abs": abs(delta),
            "units": units,
            "phase_tag": phase_tag,
        })

    # Sort by magnitude of delta (largest first)
    items.sort(key=lambda e: e["delta_abs"], reverse=True)

    return items[:max_items]


__all__ = [
    "TAU",
    "UNITS",
    "PHASE_TAG",
    "material_delta",
    "select_top_metrics",
]
