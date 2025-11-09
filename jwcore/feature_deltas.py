# jwcore/feature_deltas.py
from __future__ import annotations
from typing import Dict
import math

STANCE_KEY = "stance_width_norm_mean"
HAND_KEY   = "hand_span_norm_tail_mean"

def _safe_num(x) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")

def stance_hand_deltas(am: Dict[str, float], pro: Dict[str, float]) -> Dict[str, float]:
    """
    Compute deltas for stance width and hand span (amateur - pro) and % vs pro.
    Returns a flat dict with explicit keys that the LLM can latch onto.

    Keys produced:
      - stance_width_delta
      - stance_width_pct
      - stance_width_amateur
      - stance_width_pro
      - hand_span_delta
      - hand_span_pct
      - hand_span_amateur
      - hand_span_pro
    """
    out: Dict[str, float] = {}

    def delta(name: str, prefix: str):
        av = _safe_num(am.get(name, float("nan")))
        pv = _safe_num(pro.get(name, float("nan")))
        if not (math.isfinite(av) and math.isfinite(pv)) or pv == 0.0:
            d = float("nan"); pct = float("nan")
        else:
            d = av - pv
            pct = d / pv
        out[f"{prefix}_delta"]    = d
        out[f"{prefix}_pct"]      = pct
        out[f"{prefix}_amateur"]  = av
        out[f"{prefix}_pro"]      = pv

    delta(STANCE_KEY, "stance_width")
    delta(HAND_KEY,   "hand_span")
    return out
