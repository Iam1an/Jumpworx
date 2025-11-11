#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import argparse

import numpy as np
import cv2

# ---------------------------------------------------------------------
# Make jwcore importable when this script is in ./scripts/
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from jwcore.posetrack_io import load_posetrack_npz
from jwcore.compare_metrics import compare_metrics_from_xyz
from jwcore.phase_segmentation import (
    segment_phases_with_airtime_v2,
    fine_labels_from_phases,
    Phases,
)


# ================= MediaPipe-style indices =================
NOSE = 0
L_EYE = 2
R_EYE = 5
L_EAR = 7
R_EAR = 8
L_SHO = 11
R_SHO = 12
L_ELB = 13
R_ELB = 14
L_WRI = 15
R_WRI = 16
L_HIP = 23
R_HIP = 24
L_KNE = 25
R_KNE = 26
L_ANK = 27
R_ANK = 28

KEY_NAMES = {
    L_SHO: "L_SHO",
    R_SHO: "R_SHO",
    L_ELB: "L_ELB",
    R_ELB: "R_ELB",
    L_WRI: "L_HAND",
    R_WRI: "R_HAND",
    L_HIP: "L_HIP",
    R_HIP: "R_HIP",
    L_KNE: "L_KNE",
    R_KNE: "R_KNE",
    L_ANK: "L_ANK",
    R_ANK: "R_ANK",
    NOSE: "NOSE",
    L_EYE: "L_EYE",
    R_EYE: "R_EYE",
    L_EAR: "L_EAR",
    R_EAR: "R_EAR",
}

KEYS_BASE8 = [L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]
KEYS_EXPAND = [L_ELB, R_ELB, L_WRI, R_WRI]

SKELETON_EDGES = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (12, 14), (14, 16), (11, 13), (13, 15),
    (24, 26), (26, 28), (23, 25), (25, 27),
    (7, 8), (8, 5), (7, 2),
]

# ================= Scoring hyperparams =================
D0_JOINT = 0.15
A0_PITCH = 20.0
A0_KNEE = 25.0
A0_HEAD = 15.0

W_JOINT = 1.00
W_PITCH = 0.60
W_KNEE = 0.50
W_HEAD = 0.60

# ================= Utilities =================

def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def _interp_nan_1d(y):
    y = np.asarray(y, float)
    if not np.isnan(y).any():
        return y
    idx = np.arange(len(y))
    m = np.isfinite(y)
    if not m.any():
        return np.zeros_like(y)
    y[~m] = np.interp(idx[~m], idx[m], y[m])
    return y

# ================= Series helpers =================

def series_ankle_y(arr):
    pair = arr[:, [L_ANK, R_ANK], 1]
    finite_any = np.isfinite(pair).any(axis=1)
    out = np.full(arr.shape[0], np.nan, float)
    with np.errstate(invalid="ignore"):
        out[finite_any] = np.nanmean(pair[finite_any], axis=1)
    return _interp_nan_1d(out)

def series_hip_y(arr):
    pair = arr[:, [L_HIP, R_HIP], 1]
    finite_any = np.isfinite(pair).any(axis=1)
    out = np.full(arr.shape[0], np.nan, float)
    with np.errstate(invalid="ignore"):
        out[finite_any] = np.nanmean(pair[finite_any], axis=1)
    return _interp_nan_1d(out)

def apex_index(arr):
    y = series_ankle_y(arr)
    if np.isnan(y).any():
        fin = y[np.isfinite(y)]
        if fin.size:
            y = np.where(np.isfinite(y), y, np.nanmax(fin) + 1e3)
    return int(np.argmin(y))

def angle_deg(a, b, c):
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return np.nan
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def torso_pitch_deg_frame(frame_xy):
    need = (L_HIP, R_HIP, L_SHO, R_SHO)
    if not all(np.isfinite(frame_xy[i]).all() for i in need):
        return np.nan
    hip_mid = 0.5 * (frame_xy[L_HIP] + frame_xy[R_HIP])
    sh_mid = 0.5 * (frame_xy[L_SHO] + frame_xy[R_SHO])
    v = sh_mid - hip_mid
    up = np.array([0.0, -1.0], dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.nan
    cosang = float(np.clip(np.dot(v / n, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def torso_pitch_deg_series(xy_seq):
    T = xy_seq.shape[0]
    out = np.full(T, np.nan, float)
    for i in range(T):
        out[i] = torso_pitch_deg_frame(xy_seq[i])
    return _interp_nan_1d(out)

def knee_angles_deg_frame(frame_xy):
    L = angle_deg(frame_xy[L_HIP], frame_xy[L_KNE], frame_xy[L_ANK])
    R = angle_deg(frame_xy[R_HIP], frame_xy[R_KNE], frame_xy[R_ANK])
    vals = [v for v in (L, R) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan

def knee_flex_deg_series(xy_seq, side="L"):
    T = xy_seq.shape[0]
    out = np.full(T, np.nan, float)
    for i in range(T):
        if side == "L":
            out[i] = angle_deg(xy_seq[i][L_HIP], xy_seq[i][L_KNE], xy_seq[i][L_ANK])
        else:
            out[i] = angle_deg(xy_seq[i][R_HIP], xy_seq[i][R_KNE], xy_seq[i][R_ANK])
    return _interp_nan_1d(out)

def head_midpoint(xy):
    if np.isfinite(xy[L_EAR]).all() and np.isfinite(xy[R_EAR]).all():
        return 0.5 * (xy[L_EAR] + xy[R_EAR])
    if np.isfinite(xy[L_EYE]).all() and np.isfinite(xy[R_EYE]).all():
        return 0.5 * (xy[L_EYE] + xy[R_EYE])
    if np.isfinite(xy[NOSE]).all():
        return xy[NOSE]
    return np.array([np.nan, np.nan], dtype=np.float32)

def head_pitch_deg(xy):
    hm = head_midpoint(xy)
    nose = xy[NOSE] if np.isfinite(xy[NOSE]).all() else None
    if nose is None or not np.isfinite(hm).all():
        return np.nan
    v = nose - hm
    up = np.array([0.0, -1.0], dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.nan
    cosang = float(np.clip(np.dot(v / n, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def head_pitch_deg_series(xy_seq):
    T = xy_seq.shape[0]
    out = np.full(T, np.nan, float)
    for i in range(T):
        out[i] = head_pitch_deg(xy_seq[i])
    return _interp_nan_1d(out)

def joint_l2_series(An, Bn, j):
    return np.linalg.norm(An[:, j, :2] - Bn[:, j, :2], axis=1)

def pelvis_center_torso_scale(frame_xy):
    hips = frame_xy[[L_HIP, R_HIP], :]
    shs = frame_xy[[L_SHO, R_SHO], :]
    if not (np.isfinite(hips).all() and np.isfinite(shs).all()):
        return frame_xy.copy(), 1.0
    pelvis = hips.mean(axis=0)
    shmid = shs.mean(axis=0)
    span = float(np.linalg.norm(shmid - pelvis))
    scale = 1.0 if span < 1e-6 else (1.0 / span)
    return (frame_xy - pelvis) * scale, scale

def normalize_pose_sequence(arr_xyz):
    T = arr_xyz.shape[0]
    out = np.empty((T, arr_xyz.shape[1], 2), dtype=np.float32)
    for t in range(T):
        xy = arr_xyz[t, :, :2]
        xy_n, _ = pelvis_center_torso_scale(xy)
        out[t] = xy_n
    return out

# ================= Airtime heuristic =================

def estimate_airtime_from_ankle(arr_xyz, fps):
    if not fps or fps <= 1e-3:
        return None, None, None
    y = series_ankle_y(arr_xyz)
    if y.size < 5 or not np.isfinite(y).any():
        return None, None, None
    fin = y[np.isfinite(y)]
    y_max = float(np.nanmax(fin))
    y_min = float(np.nanmin(fin))
    rng = y_max - y_min
    if rng < 1e-3:
        return None, None, None
    thr = y_max - 0.2 * rng
    air_mask = y < thr
    if not air_mask.any():
        return None, None, None
    apex = int(np.nanargmin(y))
    best_start = best_end = None
    cur_start = None
    for i, is_air in enumerate(air_mask):
        if is_air:
            if cur_start is None:
                cur_start = i
        elif cur_start is not None:
            cur_end = i - 1
            if best_start is None:
                best_start, best_end = cur_start, cur_end
            else:
                prev_mid = 0.5 * (best_start + best_end)
                this_mid = 0.5 * (cur_start + cur_end)
                if abs(this_mid - apex) < abs(prev_mid - apex):
                    best_start, best_end = cur_start, cur_end
            cur_start = None
    if cur_start is not None:
        cur_end = len(air_mask) - 1
        if best_start is None:
            best_start, best_end = cur_start, cur_end
        else:
            prev_mid = 0.5 * (best_start + best_end)
            this_mid = 0.5 * (cur_start + cur_end)
            if abs(this_mid - apex) < abs(prev_mid - apex):
                best_start, best_end = cur_start, cur_end
    if best_start is None or best_end is None or best_end <= best_start:
        return None, None, None
    airtime = (best_end - best_start) / float(fps)
    return int(best_start), int(best_end), float(airtime)

# ================= DTW helpers =================

def _dtw_path_1d(a, b):
    a = _interp_nan_1d(np.asarray(a, float))
    b = _interp_nan_1d(np.asarray(b, float))
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf, float)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = (ai - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = min(
            ((i - 1, j), (i, j - 1), (i - 1, j - 1)),
            key=lambda ij: D[ij]
        )
    path.reverse()
    return path

def build_index_map_dtw(arrA, arrB, feature="ankle_y"):
    if feature == "ankle_y":
        sA, sB = series_ankle_y(arrA), series_ankle_y(arrB)
    elif feature == "hip_y":
        sA, sB = series_hip_y(arrA), series_hip_y(arrB)
    elif feature == "pitch":
        sA = torso_pitch_deg_series(arrA[:, :, :2])
        sB = torso_pitch_deg_series(arrB[:, :, :2])
    else:
        raise ValueError(f"Unknown align_feature {feature}")
    path = _dtw_path_1d(sA, sB)
    m = {}
    for i, j in path:
        if i not in m:
            m[i] = j
        else:
            if abs(sB[j] - sA[i]) < abs(sB[m[i]] - sA[i]):
                m[i] = j
    return m

# ================= Coordinate mapping =================

def _guess_coords_mode(frame_pts):
    pts = frame_pts[:, :2]
    finite = np.isfinite(pts)
    if not finite.any():
        return "pixels"
    vals = pts[finite]
    return "normalized" if np.nanmax(vals) <= 5.0 else "pixels"

def _pelvis_and_shoulders(frame_xy):
    need = (L_HIP, R_HIP, L_SHO, R_SHO)
    if not all(np.isfinite(frame_xy[i]).all() for i in need):
        return None, None
    pelvis = 0.5 * (frame_xy[L_HIP] + frame_xy[R_HIP])
    sh_mid = 0.5 * (frame_xy[L_SHO] + frame_xy[R_SHO])
    return pelvis, sh_mid

def _pretransform_stabilize(frame_xy, stabilize):
    if not stabilize:
        return frame_xy.copy(), 1.0, np.array([0.0, 0.0], np.float32)
    pelvis, sh_mid = _pelvis_and_shoulders(frame_xy)
    if pelvis is None or sh_mid is None:
        return frame_xy.copy(), 1.0, np.array([0.0, 0.0], np.float32)
    span = float(np.linalg.norm(sh_mid - pelvis))
    scale = 1.0 if span < 1e-6 else (1.0 / span)
    return (frame_xy - pelvis) * scale, scale, pelvis

def _fit_bbox_transform(all_xy, W, H, margin=40):
    xs, ys = [], []
    for xy in all_xy:
        if xy is None:
            continue
        fx = np.isfinite(xy[:, 0])
        fy = np.isfinite(xy[:, 1])
        if fx.any():
            xs.append(xy[fx, 0])
        if fy.any():
            ys.append(xy[fy, 1])
    if not xs or not ys:
        return 1.0, 0.0, 0.0
    xmin = float(np.nanmin([np.nanmin(x) for x in xs]))
    xmax = float(np.nanmax([np.nanmax(x) for x in xs]))
    ymin = float(np.nanmin([np.nanmin(y) for y in ys]))
    ymax = float(np.nanmax([np.nanmax(y) for y in ys]))
    bw = max(1e-6, xmax - xmin)
    bh = max(1e-6, ymax - ymin)
    s = min((W - 2 * margin) / bw, (H - 2 * margin) / bh)
    tx = margin - s * xmin
    ty = margin - s * ymin
    return float(s), float(tx), float(ty)

def world_to_canvas(
    frame_pts,
    W,
    H,
    *,
    coords_mode="auto",
    invert_y=False,
    fit_params=None,
    per_frame_fit=False,
    margin=40,
    stabilize=False,
):
    pts = frame_pts[:, :2].astype(np.float32).copy()
    mode = _guess_coords_mode(pts) if coords_mode == "auto" else coords_mode
    if mode == "normalized":
        pts[:, 0] *= W
        pts[:, 1] *= H
    if stabilize:
        pts, _, _ = _pretransform_stabilize(pts, True)
    if per_frame_fit and np.isfinite(pts).any():
        xmin = float(np.nanmin(pts[:, 0]))
        xmax = float(np.nanmax(pts[:, 0]))
        ymin = float(np.nanmin(pts[:, 1]))
        ymax = float(np.nanmax(pts[:, 1]))
        bw = max(1e-6, xmax - xmin)
        bh = max(1e-6, ymax - ymin)
        s = min((W - 2 * margin) / bw, (H - 2 * margin) / bh)
        tx = margin - s * xmin
        ty = margin - s * ymin
        pts[:, 0] = s * pts[:, 0] + tx
        pts[:, 1] = s * pts[:, 1] + ty
    elif fit_params is not None:
        s, tx, ty = fit_params
        pts[:, 0] = s * pts[:, 0] + tx
        pts[:, 1] = s * pts[:, 1] + ty
    if invert_y:
        pts[:, 1] = (H - 1) - pts[:, 1]
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts

# ================= Drawing helpers =================

def _alpha_blend(color, a):
    a = float(max(0.0, min(1.0, a)))
    return (int(color[0] * a), int(color[1] * a), int(color[2] * a))

def draw_skeleton(img, pts_px, vis=None, base=(255, 255, 255), thickness=3, radius=4):
    for a, b in SKELETON_EDGES:
        pa, pb = pts_px[a], pts_px[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        col = base
        if vis is not None:
            va = vis[a] if np.isfinite(vis[a]) else 0.0
            vb = vis[b] if np.isfinite(vis[b]) else 0.0
            col = _alpha_blend(base, float(min(max(va, 0.0), max(vb, 0.0))))
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 col, thickness, cv2.LINE_AA)
    for i, p in enumerate(pts_px):
        if not np.isfinite(p).all():
            continue
        col = base
        if vis is not None and np.isfinite(vis[i]):
            col = _alpha_blend(base, float(max(vis[i], 0.0)))
        cv2.circle(img, (int(p[0]), int(p[1])), radius, col, -1, cv2.LINE_AA)

def draw_ground_line(img, y_px, color=(180, 180, 180)):
    y = int(round(y_px))
    cv2.line(img, (0, y), (img.shape[1] - 1, y), color, 1, cv2.LINE_AA)

def resize_keep_height(img, H):
    h, w = img.shape[:2]
    if h == H:
        return img
    scale = H / float(h)
    W = int(round(w * scale))
    if scale < 1.0:
        interp = cv2.INTER_AREA      # downscale: smooth
    else:
        interp = cv2.INTER_LANCZOS4  # upscale: sharp
    return cv2.resize(img, (W, H), interpolation=interp)

def put_text_with_shadow(img, text, org, font, scale, color, thick,
                         shadow=(0, 0, 0), offset=2):
    if not text:
        return
    x, y = org
    cv2.putText(img, text, (x + offset, y + offset),
                font, scale, shadow, max(1, thick - 1), cv2.LINE_AA)
    cv2.putText(img, text, (x, y),
                font, scale, color, thick, cv2.LINE_AA)

# ================= HUD bottom =================

def fmt_val(v, fmt=".2f"):
    if v is None:
        return "N/A"
    if isinstance(v, float) and (not np.isfinite(v)):
        return "N/A"
    try:
        return f"{v:{fmt}}"
    except Exception:
        return str(v)

def hud_lines_from_metrics(m):
    def g(key, fmt, unit=""):
        v = m.get(key, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "N/A"
        return f"{v:{fmt}}{unit}"

    return [
        f"frames: {m.get('frames')}",
        f"apex: {m.get('apex_frame')}",
        f"height_delta: {g('height_range_pixels', '.1f', ' px')}",
        f"pitch@apex: {g('torso_pitch_deg_at_apex', '.1f', ' deg')}",
        f"min_knee: {g('min_knee_angle_deg', '.1f', ' deg')}",
        f"air: {g('airtime_seconds', '.2f', ' s')}",
    ]

def put_hud_bottom(
    img,
    metrics,
    corner="left",
    margin=18,
    line_h=30,
    line_scale=1.0,
    line_thick=2,
    color=(255, 255, 255),
    bottom_offset=0,
):
    """
    Bottom HUD:
      - Metrics only (no labels).
      - Supports left or right alignment.
      - Sits above bottom_offset (e.g. sparkline band).
    """
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = hud_lines_from_metrics(metrics)

    y = H - bottom_offset - margin

    for line in lines:
        if not line:
            continue
        if corner == "left":
            x_line = margin
        else:
            (tw, _), _ = cv2.getTextSize(line, font, line_scale, line_thick)
            x_line = max(margin, W - margin - tw)
        put_text_with_shadow(
            img, line, (x_line, y),
            font, line_scale, color, line_thick
        )
        y -= line_h

# ================= Sparkline =================

def draw_sparkline_panel(img, series, t_idx, height=80, margin=6):
    """
    Draw a clean sparkline at the very bottom (no text).
    Only invoked on Amateur side.
    """
    H, W = img.shape[:2]
    if height <= 0 or height >= H:
        return

    y0 = H - height
    cv2.rectangle(img, (0, y0), (W - 1, H - 1), (0, 0, 0), -1)

    s = np.asarray(series, float)
    if not np.isfinite(s).any():
        return

    s = s.copy()
    s[~np.isfinite(s)] = np.nanmin(s[np.isfinite(s)])

    vmin = float(np.min(s))
    vmax = float(np.max(s))
    denom = (vmax - vmin) if (vmax > vmin) else 1.0

    N = len(s)
    if N < 2:
        return

    px = np.linspace(margin, W - margin - 1, N).astype(int)
    py = (y0 + height - 1 - ((s - vmin) / denom) * (height - 2 * margin)).astype(int)

    # sparkline
    for i in range(1, N):
        cv2.line(
            img,
            (int(px[i - 1]), int(py[i - 1])),
            (int(px[i]), int(py[i])),
            (200, 200, 255),
            2,
            cv2.LINE_AA,
        )

    # current frame marker
    i = max(0, min(N - 1, int(t_idx)))
    cv2.line(
        img,
        (int(px[i]), y0),
        (int(px[i]), H - 1),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

# ================= Top block =================

def draw_top_bg(img, height_px=230):
    H, W = img.shape[:2]
    h = max(10, min(H, int(height_px)))
    cv2.rectangle(img, (0, 0), (W - 1, h), (0, 0, 0), -1)
    return h

def put_top_block_with_title_4rows(
    img,
    title,
    r1,
    r2,
    r3,
    r4=None,
    top_bg_height=230,
    margin_top=22,
    title_scale=1.7,
    title_thick=3,
    r_scale=1.15,
    r_thick=2,
    r4_scale=1.25,
    r4_thick=3,
    color=(255, 255, 255),
):
    """
    Top overlay:
      - Centered title
      - 3 shared metrics rows
      - Optional row4 (used only on Amateur: frame + overall score)
      - Auto-shrinks to fit width
    """
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    bg_h = draw_top_bg(img, top_bg_height)
    margin_x = 40

    def fit_and_center(text, base_scale, thick, y):
        if not text:
            return None, y, base_scale, 0
        (tw, th), _ = cv2.getTextSize(text, font, base_scale, thick)
        max_width = max(40, W - 2 * margin_x)
        scale = base_scale
        if tw > max_width:
            scale = base_scale * (max_width / float(tw))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x = (W - tw) // 2
        x = max(margin_x, min(x, W - margin_x - tw))
        yb = y + th
        return x, yb, scale, th

    y_cur = max(0, min(bg_h - 1, margin_top))

    # Title
    if title:
        x, yb, s_used, th = fit_and_center(title, title_scale, title_thick, y_cur)
        if x is not None:
            put_text_with_shadow(img, title, (x, yb),
                                 font, s_used, color, title_thick)
            y_cur = yb + int(th * 0.35)

    # Rows 1–3
    for row, base_scale, thick, gap_mul in [
        (r1, r_scale, r_thick, 0.35),
        (r2, r_scale, r_thick, 0.30),
        (r3, r_scale, r_thick, 0.30),
    ]:
        if row:
            x, yb, s_used, th = fit_and_center(row, base_scale, thick, y_cur)
            if x is not None:
                put_text_with_shadow(img, row, (x, yb),
                                     font, s_used, color, thick)
                y_cur = yb + int(th * gap_mul)

    # Row 4 (Amateur only)
    if r4:
        x, yb, s_used, th = fit_and_center(r4, r4_scale, r4_thick, y_cur)
        if x is not None:
            put_text_with_shadow(img, r4, (x, yb),
                                 font, s_used, color, r4_thick)

# ================= Clip metrics (with airtime) =================

def metrics_for_clip(arr_xyz, fps, labels=None, seg=None):
    y_ank = series_ankle_y(arr_xyz)
    apex = apex_index(arr_xyz)
    finite_y = y_ank[np.isfinite(y_ank)]
    if finite_y.size:
        height_range = float(np.max(finite_y) - np.min(finite_y))
    else:
        height_range = np.nan

    k = 5
    pitches = [
        torso_pitch_deg_frame(arr_xyz[t, :, :2])
        for t in range(max(0, apex - k), min(arr_xyz.shape[0], apex + k + 1))
    ]
    pitch_apex = float(np.nanmean(pitches)) if pitches else np.nan

    half = max(1, arr_xyz.shape[0] // 2)
    kne = [knee_angles_deg_frame(arr_xyz[t, :, :2]) for t in range(half)]
    min_knee = float(np.nanmin(kne)) if kne else np.nan

    takeoff_idx = getattr(seg, "takeoff_idx", None) if seg is not None else None
    landing_idx = getattr(seg, "landing_idx", None) if seg is not None else None
    airtime = getattr(seg, "airtime_seconds", None) if seg is not None else None

    if (takeoff_idx is None or landing_idx is None) and labels is not None:
        try:
            if takeoff_idx is None and "takeoff" in labels:
                takeoff_idx = labels.index("takeoff")
            if landing_idx is None and "landing" in labels:
                landing_idx = labels.index("landing")
        except Exception:
            pass

    if (
        airtime is None
        and fps
        and takeoff_idx is not None
        and landing_idx is not None
        and landing_idx > takeoff_idx
    ):
        airtime = (landing_idx - takeoff_idx) / float(fps)

    if airtime is None:
        est_to, est_ld, est_air = estimate_airtime_from_ankle(arr_xyz, fps or 30.0)
        if est_air is not None:
            if takeoff_idx is None:
                takeoff_idx = est_to
            if landing_idx is None:
                landing_idx = est_ld
            airtime = est_air

    return {
        "frames": int(arr_xyz.shape[0]),
        "fps": float(fps) if fps else None,
        "apex_frame": int(apex),
        "height_range_pixels": height_range,
        "torso_pitch_deg_at_apex": pitch_apex,
        "min_knee_angle_deg": min_knee,
        "takeoff_idx": int(takeoff_idx) if takeoff_idx is not None else None,
        "landing_idx": int(landing_idx) if landing_idx is not None else None,
        "airtime_seconds": float(airtime) if airtime is not None else None,
    }

# ================= Main =================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_a")
    ap.add_argument("--video_b")
    ap.add_argument("--npz_a", required=True)
    ap.add_argument("--npz_b", required=True)
    ap.add_argument("--label_a", default="Amateur")
    ap.add_argument("--label_b", default="Pro")

    ap.add_argument("--align", choices=["apex", "dtw", "takeoff", "landing"], default="apex")
    ap.add_argument("--align_feature", choices=["ankle_y", "hip_y", "pitch"], default="ankle_y")
    ap.add_argument("--seconds_before", type=float, default=1.0)
    ap.add_argument("--seconds_after", type=float, default=1.0)

    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default="viz/compare_side_by_side.mp4")
    ap.add_argument("--metrics_out", default="viz/compare_metrics.json")
    ap.add_argument("--metrics_csv", default="viz/compare_metrics.csv")

    ap.add_argument("--phase_json_a")
    ap.add_argument("--phase_json_b")
    ap.add_argument("--use_segmenter", action="store_true")

    ap.add_argument("--output_fps", type=float, default=None)

    ap.add_argument("--blank_canvas", action="store_true")
    ap.add_argument("--canvas_width", type=int, default=1280)
    ap.add_argument("--canvas_height", type=int, default=720)

    ap.add_argument("--color_a", default="red")
    ap.add_argument("--color_b", default="blue")
    ap.add_argument("--no_hud", action="store_true")

    # NEW: accept hud_corner for backwards compatibility / control
    ap.add_argument("--hud_corner", choices=["left", "right"], default="left")

    ap.add_argument("--coords", choices=["auto", "pixels", "normalized"], default="auto")
    ap.add_argument("--invert_y", action="store_true")
    ap.add_argument("--fit", choices=["none", "window", "frame"], default="none")
    ap.add_argument("--margin_px", type=int, default=40)
    ap.add_argument("--stabilize_pelvis", action="store_true")

    ap.add_argument("--draw_ground", action="store_true")
    ap.add_argument("--metrics_panel", action="store_true")
    ap.add_argument("--include_hands", action="store_true")
    ap.add_argument("--include_head", action="store_true")

    ap.add_argument("--score_ema_alpha", type=float, default=0.0)
    ap.add_argument("--top_bg_height", type=int, default=230)

    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    def dbg(*msg):
        if args.debug:
            print(*msg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)

    # colors
    def parse_color(s: str):
        named = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "orange": (0, 165, 255),
        }
        if not s:
            return (255, 255, 255)
        s2 = s.strip().lower()
        if s2 in named:
            return named[s2]
        if s2.startswith("#") and len(s2) == 7:
            r = int(s2[1:3], 16)
            g = int(s2[3:5], 16)
            b = int(s2[5:7], 16)
            return (b, g, r)
        if "," in s2:
            try:
                r, g, b = [int(x) for x in s2.split(",")]
                return (b, g, r)
            except Exception:
                pass
        return (255, 255, 255)

    colA = parse_color(args.color_a)
    colB = parse_color(args.color_b)

    # Load pose-tracks
    arrA, visA, fpsA_npz, _ = load_posetrack_npz(args.npz_a)
    arrB, visB, fpsB_npz, _ = load_posetrack_npz(args.npz_b)
    dbg("[DEBUG] Loaded npz A:", args.npz_a, "shape:", arrA.shape, "fps_npz:", fpsA_npz)
    dbg("[DEBUG] Loaded npz B:", args.npz_b, "shape:", arrB.shape, "fps_npz:", fpsB_npz)

    # Phase labels (optional)
    def load_phase_json(path):
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    labelsA = load_phase_json(args.phase_json_a)
    labelsB = load_phase_json(args.phase_json_b)

    segA = segB = None

    # If requested, derive phase labels from the unified phase_segmentation engine
    # for any side that does not already have explicit phase JSON.
    if args.use_segmenter and (labelsA is None or labelsB is None):
        if labelsA is None:
            phases_a, _ = segment_phases_with_airtime_v2(
                arrA,
                fpsA_npz or 30.0,
                return_debug=False,
            )
            if phases_a.airtime is not None:
                segA = phases_a
                labelsA = fine_labels_from_phases(phases_a, arrA.shape[0])

        if labelsB is None:
            phases_b, _ = segment_phases_with_airtime_v2(
                arrB,
                fpsB_npz or 30.0,
                return_debug=False,
            )
            if phases_b.airtime is not None:
                segB = phases_b
                labelsB = fine_labels_from_phases(phases_b, arrB.shape[0])


    # Video / canvas setup
    if not args.blank_canvas:
        if not args.video_a or not args.video_b:
            print("[ERROR] Must provide --video_a and --video_b when not using --blank_canvas")
            return

        capA = cv2.VideoCapture(args.video_a)
        capB = cv2.VideoCapture(args.video_b)
        okA, okB = capA.isOpened(), capB.isOpened()
        dbg("[DEBUG] video_a:", args.video_a, "opened:", okA)
        dbg("[DEBUG] video_b:", args.video_b, "opened:", okB)
        if not (okA and okB):
            print("[ERROR] Failed to open one of the videos.")
            if capA:
                capA.release()
            if capB:
                capB.release()
            return

        fpsA_vid = capA.get(cv2.CAP_PROP_FPS) or 30.0
        fpsB_vid = capB.get(cv2.CAP_PROP_FPS) or 30.0
        fpsA = fpsA_npz or fpsA_vid
        fpsB = fpsB_npz or fpsB_vid
        WA = int(capA.get(cv2.CAP_PROP_FRAME_WIDTH))
        HA = int(capA.get(cv2.CAP_PROP_FRAME_HEIGHT))
        WB = int(capB.get(cv2.CAP_PROP_FRAME_WIDTH))
        HB = int(capB.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        capA = capB = None
        fpsA = fpsA_npz or 30.0
        fpsB = fpsB_npz or 30.0
        WA = WB = args.canvas_width
        HA = HB = args.canvas_height
        dbg("[DEBUG] Using blank canvas:", (WA, HA))

    def window_by_center(center_idx, fps_clip, T, before_s, after_s):
        left = max(0, center_idx - int(before_s * fps_clip))
        right = min(T - 1, center_idx + int(after_s * fps_clip))
        return left, right

    # Alignment
    if args.align == "apex":
        apexA = apex_index(arrA)
        apexB = apex_index(arrB)
        winA = window_by_center(apexA, fpsA, arrA.shape[0],
                                args.seconds_before, args.seconds_after)
        winB = window_by_center(apexB, fpsB, arrB.shape[0],
                                args.seconds_before, args.seconds_after)
        lenA = winA[1] - winA[0] + 1
        lenB = winB[1] - winB[0] + 1
        T = max(lenA, lenB)
        idxsA = np.clip(np.arange(T), 0, lenA - 1) + winA[0]
        idxsB = np.clip(np.arange(T), 0, lenB - 1) + winB[0]


    elif args.align in ("takeoff", "landing"):
        def event_from_phase(labels, event_name):
            """If labels were loaded from JSON and contain an explicit event, use it."""
            if not labels:
                return None
            try:
                return int(labels.index(event_name))
            except ValueError:
                return None

        # 1) Try explicit labels (from JSON or derived)
        evA = event_from_phase(labelsA, args.align)
        evB = event_from_phase(labelsB, args.align)

        # 2) Fallback: use unified phase_segmentation engine if requested
        if evA is None and args.use_segmenter:
            if segA is not None:
                # segA came from earlier segment_phases_with_airtime_v2 call
                evA = segA.takeoff_idx if args.align == "takeoff" else segA.landing_idx
            else:
                phases_a, _ = segment_phases_with_airtime_v2(
                    arrA,
                    fpsA,
                    return_debug=False,
                )
                if phases_a.airtime is not None:
                    segA = phases_a
                    evA = segA.takeoff_idx if args.align == "takeoff" else segA.landing_idx

        if evB is None and args.use_segmenter:
            if segB is not None:
                evB = segB.takeoff_idx if args.align == "takeoff" else segB.landing_idx
            else:
                phases_b, _ = segment_phases_with_airtime_v2(
                    arrB,
                    fpsB,
                    return_debug=False,
                )
                if phases_b.airtime is not None:
                    segB = phases_b
                    evB = segB.takeoff_idx if args.align == "takeoff" else segB.landing_idx

        # 3) Final fallback: use apex alignment if phase detection failed
        if evA is None:
            evA = apex_index(arrA)
        if evB is None:
            evB = apex_index(arrB)

        # 4) Build windows and aligned index arrays
        winA = window_by_center(
            evA,
            fpsA,
            arrA.shape[0],
            args.seconds_before,
            args.seconds_after,
        )
        winB = window_by_center(
            evB,
            fpsB,
            arrB.shape[0],
            args.seconds_before,
            args.seconds_after,
        )
        lenA = winA[1] - winA[0] + 1
        lenB = winB[1] - winB[0] + 1
        T = max(lenA, lenB)
        idxsA = np.clip(np.arange(T), 0, lenA - 1) + winA[0]
        idxsB = np.clip(np.arange(T), 0, lenB - 1) + winB[0]

    else:  # dtw
        mapA2B = build_index_map_dtw(arrA, arrB, feature=args.align_feature)
        apexA = apex_index(arrA)
        leftA = max(0, apexA - int(args.seconds_before * fpsA))
        rightA = min(arrA.shape[0] - 1, apexA + int(args.seconds_after * fpsA))
        idxsA = np.arange(leftA, rightA + 1)
        if not mapA2B:
            idxsB = np.full_like(idxsA, fill_value=apex_index(arrB))
        else:
            ia = np.array(sorted(mapA2B.keys()), dtype=np.int32)
            jb = np.array([mapA2B[int(i)] for i in ia], dtype=np.float32)
            jb = np.maximum.accumulate(jb)
            jb_full = np.interp(
                idxsA.astype(np.float32),
                ia.astype(np.float32),
                jb
            ).round().astype(np.int32)
            idxsB = np.clip(jb_full, 0, arrB.shape[0] - 1)
        T = len(idxsA)

    dbg("[DEBUG] Align mode:", args.align, "feature:", args.align_feature)
    dbg("[DEBUG] idxsA len:", len(idxsA), "min/max:", int(idxsA.min()), int(idxsA.max()))
    dbg("[DEBUG] idxsB len:", len(idxsB), "min/max:", int(idxsB.min()), int(idxsB.max()))
    dbg("[DEBUG] Using T:", T)

    fps_out = args.output_fps if (args.output_fps and args.output_fps > 0) else float(max(fpsA, fpsB))
    dbg("[DEBUG] Output fps:", fps_out)

    # Clip-level metrics (for HUD / airtime)
    # If requested, ensure we have Phases objects from the unified segmenter
    # (segA/segB may already be set earlier from phase label logic).
    if args.use_segmenter:
        if segA is None:
            phases_a, _ = segment_phases_with_airtime_v2(
                arrA,
                fpsA,
                return_debug=False,
            )
            if phases_a.airtime is not None:
                segA = phases_a

        if segB is None:
            phases_b, _ = segment_phases_with_airtime_v2(
                arrB,
                fpsB,
                return_debug=False,
            )
            if phases_b.airtime is not None:
                segB = phases_b

    mA = metrics_for_clip(arrA, fpsA, labelsA, segA)
    mB = metrics_for_clip(arrB, fpsB, labelsB, segB)


    # Window & normalize for scoring
    Awin = arrA[idxsA]
    Bwin = arrB[idxsB]
    An = normalize_pose_sequence(Awin)
    Bn = normalize_pose_sequence(Bwin)

    joints = list(KEYS_BASE8) + (KEYS_EXPAND if args.include_hands else [])
    delta = {KEY_NAMES[j]: joint_l2_series(An, Bn, j) for j in joints}

    pitchA = torso_pitch_deg_series(An)
    pitchB = torso_pitch_deg_series(Bn)
    kneeLA = knee_flex_deg_series(An, "L")
    kneeLB = knee_flex_deg_series(Bn, "L")
    kneeRA = knee_flex_deg_series(An, "R")
    kneeRB = knee_flex_deg_series(Bn, "R")

    d_pitch = np.abs(pitchA - pitchB)
    d_knee = 0.5 * (np.abs(kneeLA - kneeLB) + np.abs(kneeRA - kneeRB))

    if args.include_head:
        headA = head_pitch_deg_series(An)
        headB = head_pitch_deg_series(Bn)
        d_head = np.abs(headA - headB)
    else:
        d_head = None

    # per-frame similarity
    joint_keys = [KEY_NAMES[j] for j in joints]
    S_joint = np.ones(len(An), dtype=np.float32)
    if joint_keys:
        mat = np.stack([delta[k] for k in joint_keys], axis=1)
        with np.errstate(invalid="ignore"):
            Sj = np.exp(-((mat / D0_JOINT) ** 2))
            m = np.isfinite(Sj)
            valid = m.any(axis=1)
            S_joint = np.zeros(Sj.shape[0], dtype=np.float32)
            S_joint[valid] = np.nanmean(Sj[valid], axis=1)

    S_pitch = np.exp(-((d_pitch / A0_PITCH) ** 2))
    S_knee = np.exp(-((d_knee / A0_KNEE) ** 2))
    if d_head is not None:
        S_head = np.exp(-((d_head / A0_HEAD) ** 2))
    else:
        S_head = None

    w_sum = W_JOINT + W_PITCH + W_KNEE + (W_HEAD if S_head is not None else 0.0)
    sim = (
        W_JOINT * S_joint
        + W_PITCH * S_pitch
        + W_KNEE * S_knee
        + ((W_HEAD * S_head) if S_head is not None else 0.0)
    ) / max(1e-6, w_sum)
    score_raw = 100.0 * np.clip(sim, 0.0, 1.0)

    # EMA smoothing
    alpha = float(max(0.0, min(1.0, args.score_ema_alpha)))
    if alpha > 0.0:
        score_disp = np.copy(score_raw)
        ema = score_disp[0] if (len(score_disp) and np.isfinite(score_disp[0])) else 0.0
        for i in range(len(score_disp)):
            x = score_disp[i] if np.isfinite(score_disp[i]) else ema
            ema = alpha * x + (1.0 - alpha) * ema
            score_disp[i] = ema
    else:
        score_disp = score_raw

    # Phase-aware compare_metrics
    cm = compare_metrics_from_xyz(
        arrA,
        arrB,
        align_feature=(args.align_feature if args.align == "dtw" else "ankle_y"),
    )
    scalars = cm.get("scalars", {})
    phase_scores = cm.get("phase_scores", {})

    def _phase_score_to_weight(ps_val):
        if ps_val is None:
            return 1.0
        try:
            x = float(ps_val)
        except Exception:
            return 1.0
        if not np.isfinite(x):
            return 1.0
        return float(np.exp(-abs(x)))

    mean_disp = float(np.nanmean(score_disp)) if np.isfinite(np.nanmean(score_disp)) else 0.0
    S_frame = mean_disp / 100.0

    pitch_midair_ps = phase_scores.get("pitch_profile", {}).get("midair", None)
    head_set_ps = phase_scores.get("head_early_pitch_lead_deg", {}).get("set", None)

    S = (
        0.6 * S_frame
        + 0.25 * _phase_score_to_weight(pitch_midair_ps)
        + 0.15 * _phase_score_to_weight(head_set_ps)
    )
    overall_score = 100.0 * max(0.0, min(1.0, S))

    # Top shared metric rows
    r1 = f"Midair hand span: {fmt_val(scalars.get('midair_hand_span_pct_of_torso'), '.0f')}% torso"
    r2 = (
        f"Landing stance: {fmt_val(scalars.get('landing_stance_width_pct_of_hip'), '.0f')}% hip | "
        f"Ankle dev max: {fmt_val(scalars.get('ankle_dev_pct_of_torso_max'), '.0f')}% torso"
    )
    r3 = (
        f"Hand dev max: {fmt_val(scalars.get('hand_dev_pct_of_torso_max'), '.0f')}% torso | "
        f"Leg axis diff@apex: {fmt_val(scalars.get('leg_axis_diff_deg_apex'), '.1f')} deg"
    )

    # Fit mode for skeleton drawing
    per_frame_fit = args.fit == "frame"
    fitA = fitB = None
    if args.fit == "window":
        allA = []
        for i in idxsA:
            xy = arrA[int(i), :, :2]
            if args.stabilize_pelvis:
                xy, _, _ = _pretransform_stabilize(xy, True)
            allA.append(xy)
        allB = []
        for i in idxsB:
            xy = arrB[int(i), :, :2]
            if args.stabilize_pelvis:
                xy, _, _ = _pretransform_stabilize(xy, True)
            allB.append(xy)
        fitA = _fit_bbox_transform(allA, WA, HA, margin=args.margin_px)
        fitB = _fit_bbox_transform(allB, WB, HB, margin=args.margin_px)

    writer = None
    wrote_frames = 0
    sparkline_height = 80 if args.metrics_panel else 0

    for k in range(T):
        iA = int(idxsA[k])
        iB = int(idxsB[k])

        if not args.blank_canvas:
            capA.set(cv2.CAP_PROP_POS_FRAMES, iA)
            capB.set(cv2.CAP_PROP_POS_FRAMES, iB)
            okA, frameA = capA.read()
            okB, frameB = capB.read()
            if not (okA and okB):
                dbg("[DEBUG] Frame read failed at k=", k, "iA=", iA, "iB=", iB)
                break
        else:
            frameA = np.zeros((HA, WA, 3), dtype=np.uint8)
            frameB = np.zeros((HB, WB, 3), dtype=np.uint8)

        ptsA = world_to_canvas(
            arrA[iA], frameA.shape[1], frameA.shape[0],
            coords_mode=args.coords,
            invert_y=args.invert_y,
            fit_params=fitA,
            per_frame_fit=per_frame_fit,
            margin=args.margin_px,
            stabilize=args.stabilize_pelvis,
        )
        ptsB = world_to_canvas(
            arrB[iB], frameB.shape[1], frameB.shape[0],
            coords_mode=args.coords,
            invert_y=args.invert_y,
            fit_params=fitB,
            per_frame_fit=per_frame_fit,
            margin=args.margin_px,
            stabilize=args.stabilize_pelvis,
        )

        vA = visA[iA] if visA is not None else None
        vB = visB[iB] if visB is not None else None

        draw_skeleton(frameA, ptsA, vA, base=colA, thickness=3, radius=4)
        draw_skeleton(frameB, ptsB, vB, base=colB, thickness=3, radius=4)

        if args.draw_ground:
            anklesA = ptsA[[L_ANK, R_ANK], 1]
            anklesB = ptsB[[L_ANK, R_ANK], 1]
            if np.isfinite(anklesA).any():
                draw_ground_line(frameA, float(np.nanpercentile(anklesA, 90)))
            if np.isfinite(anklesB).any():
                draw_ground_line(frameB, float(np.nanpercentile(anklesB, 90)))

        # Sparkline ONLY on Amateur
        if args.metrics_panel:
            draw_sparkline_panel(frameA, score_disp, k, height=sparkline_height)

        # Bottom HUDs (no labels)
        if not args.no_hud:
            put_hud_bottom(
                frameA,
                mA,
                corner=args.hud_corner,
                color=colA,
                bottom_offset=(sparkline_height + 10) if args.metrics_panel else 0,
            )
            put_hud_bottom(
                frameB,
                mB,
                corner=args.hud_corner,
                color=colB,
                bottom_offset=0,
            )

        # Per-frame + Overall score: Amateur only
        frame_score = float(score_disp[k]) if np.isfinite(score_disp[k]) else float("nan")
        row4_am = (
            f"Frame score: {fmt_val(frame_score, '.1f')}/100 | "
            f"Overall score: {fmt_val(overall_score, '.1f')}/100"
        )

        put_top_block_with_title_4rows(
            frameA,
            args.label_a,
            r1,
            r2,
            r3,
            row4_am,
            top_bg_height=args.top_bg_height,
            color=colA,
        )
        put_top_block_with_title_4rows(
            frameB,
            args.label_b,
            r1,
            r2,
            r3,
            None,
            top_bg_height=args.top_bg_height,
            color=colB,
        )

        # Stack and write
        A = resize_keep_height(frameA, int(args.height))
        B = resize_keep_height(frameB, int(args.height))
        combo = np.hstack([A, B])

        if writer is None:
            Hc, Wc = combo.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, float(fps_out), (Wc, Hc))
            if hasattr(cv2, "VIDEOWRITER_PROP_QUALITY"):
                writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
            dbg("[DEBUG] Initializing writer with size:", (Wc, Hc), "fps_out:", fps_out)

        writer.write(combo)
        wrote_frames += 1

    if writer is not None:
        writer.release()
    if capA is not None:
        capA.release()
    if capB is not None:
        capB.release()

    if wrote_frames == 0:
        print("[ERROR] No frames were written. Check videos/indices.")
    else:
        print(f"[INFO] Wrote {wrote_frames} frames to {args.out}")

    # Metrics JSON
    delta_summary = {
        **{
            k: {
                "max": float(np.nanmax(delta[k])) if np.isfinite(delta[k]).any() else float("nan"),
                "mean": float(np.nanmean(delta[k])) if np.isfinite(delta[k]).any() else float("nan"),
            }
            for k in delta.keys()
        },
        "Score": {
            "max": float(np.nanmax(score_disp)) if np.isfinite(score_disp).any() else float("nan"),
            "mean": float(np.nanmean(score_disp)) if np.isfinite(score_disp).any() else float("nan"),
        },
    }

    metrics = {
        "A": {"label": args.label_a, **mA},
        "B": {"label": args.label_b, **mB},
        "deltas": delta_summary,
        "compare_metrics": _to_jsonable(cm),
        "viz": {
            "align_mode": args.align,
            "align_feature": args.align_feature,
            "frames_used": int(T),
            "frames_written": int(wrote_frames),
            "overall_score": float(overall_score),
        },
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("[INFO] Wrote metrics JSON:", args.metrics_out)

    # Metrics CSV
    with open(args.metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        delta_keys = list(delta.keys()) + ["Score"]
        header = (
            ["label_A"]
            + list(mA.keys())
            + ["label_B"]
            + list(mB.keys())
            + [f"{k}_max" for k in delta_keys]
            + [f"{k}_mean" for k in delta_keys]
        )
        w.writerow(header)
        row = (
            [args.label_a]
            + [mA[k] for k in mA.keys()]
            + [args.label_b]
            + [mB[k] for k in mB.keys()]
            + [delta_summary[k]["max"] for k in delta_keys]
            + [delta_summary[k]["mean"] for k in delta_keys]
        )
        w.writerow(row)
    print("[INFO] Wrote metrics CSV:", args.metrics_csv)


if __name__ == "__main__":
    main()
