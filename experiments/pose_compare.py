# jwcore/pose_compare.py
# Thin wrapper: loads two NPZs, runs compare_metrics_from_xyz, returns results.

from __future__ import annotations
from typing import Any, Dict
import numpy as np
from jwcore.posetrack_io import load_posetrack_npz
from jwcore.compare_metrics import compare_metrics_from_xyz


def compare_posetracks_npz(
    amateur_npz: str,
    pro_npz: str,
    align_feature: str = "ankle_y",
    include_time_series: bool = True,
) -> Dict[str, Any]:
    """
    Compare two pose tracks (amateur vs pro).

    Returns:
      dict with keys:
        - scalars: compact coaching metrics (stance width, hand span, etc.)
        - series: optional short time-series traces for key metrics
        - align_info: alignment summary (dtw_cost, shift, length)
    """
    am_P, _, _, _ = load_posetrack_npz(amateur_npz)
    pr_P, _, _, _ = load_posetrack_npz(pro_npz)

    result = compare_metrics_from_xyz(am_P, pr_P, align_feature=align_feature)

    if not include_time_series:
        result.pop("series", None)

    return result


__all__ = ["compare_posetracks_npz"]
