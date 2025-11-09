#!/usr/bin/env python3
"""
Compare two cached pose tracks with:
  - timebase normalization
  - optional per-frame Procrustes
  - optional torso normalization
  - optional preview rendering (slow-mo supported)
  - optional airtime windowing (+ padding)
  - optional integer time-shift search

This version adds --airtime-pad-ms to extend the airtime window on both sides.
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime

# --- Timebase normalization helper (from jwcore, if available) ---
try:
    from jwcore.timebase import resample_to_fps
except Exception:
    def resample_to_fps(kps_xyz, visibility, src_fps: float, target_fps: float):
        return kps_xyz, visibility, src_fps

# --- Procrustes alignment (try jwcore, else inline implementation) ---
try:
    from jwcore.utils_alignment import rigid_procrustes as _rigid_procrustes
except Exception:
    def _rigid_procrustes(src_xy, dst_xy, weights=None, eps=1e-8):
        src = np.asarray(src_xy, dtype=np.float32)
        dst = np.asarray(dst_xy, dtype=np.float32)
        J = src.shape[0]
        if weights is None:
            w = np.ones((J,), dtype=np.float32)
        else:
            w = np.clip(np.asarray(weights, dtype=np.float32), 0.0, 1.0)
        ws = w.sum() + eps
        w = w / ws
        cs = (src * w[:, None]).sum(axis=0)
        cd = (dst * w[:, None]).sum(axis=0)
        X = src - cs
        Y = dst - cd
        C = (X.T * w) @ Y
        U, _, Vt = np.linalg.svd(C)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        denom = (np.sum((X**2) * w[:, None]) + 1e-8)
        s = np.trace(R @ C) / denom
        aligned = (s * (X @ R.T)) + cd
        return aligned

# --- Joint indices (MediaPipe) ---
LS, RS, LH, RH = 11, 12, 23, 24
LK, RK = 25, 26
LA, RA = 27, 28

# --- Local helpers ---
def _load_posetrack_npz(path: str):
    d = np.load(path)
    kps = d["kps_xyz"].astype(np.float32)
    vis = d["visibility"].astype(np.float32) if "visibility" in d else np.ones(kps.shape[:2], dtype=np.float32)
    fps = float(d["fps"][0]) if "fps" in d else 0.0
    return kps, vis, fps

def _resolve_npz_path(arg: str, cache_dir: str):
    if os.path.isfile(arg):
        return arg
    candidate = os.path.join(cache_dir, f"{arg}.posetrack.npz")
    if os.path.isfile(candidate):
        return candidate
    if arg.endswith(".posetrack.npz"):
        candidate2 = os.path.join(cache_dir, arg)
        if os.path.isfile(candidate2):
            return candidate2
    raise FileNotFoundError(f"Could not resolve npz for '{arg}'. Tried as file and as stem in '{cache_dir}'.")

def _torso_scale_per_frame(XY: np.ndarray, VIS: np.ndarray, eps=1e-6):
    T = XY.shape[0]
    s = np.full((T,), np.nan, dtype=np.float32)
    ls, rs, lh, rh = XY[:, LS], XY[:, RS], XY[:, LH], XY[:, RH]
    vls, vrs, vlh, vrh = VIS[:, LS], VIS[:, RS], VIS[:, LH], VIS[:, RH]
    def dist(a,b): return np.linalg.norm(a-b, axis=-1)
    valid_ls_lh = ((vls>0.2)&(vlh>0.2)&np.isfinite(ls).all(1)&np.isfinite(lh).all(1))
    valid_rs_rh = ((vrs>0.2)&(vrh>0.2)&np.isfinite(rs).all(1)&np.isfinite(rh).all(1))
    valid_ls_rs = ((vls>0.2)&(vrs>0.2)&np.isfinite(ls).all(1)&np.isfinite(rs).all(1))
    d1 = np.where(valid_ls_lh, dist(ls,lh), np.nan)
    d2 = np.where(valid_rs_rh, dist(rs,rh), np.nan)
    d3 = np.where(valid_ls_rs, dist(ls,rs), np.nan)
    stack = np.stack([d1,d2,d3], axis=1)
    has_any = np.isfinite(stack).any(axis=1)
    if np.any(has_any):
        s[has_any] = np.nanmean(stack[has_any], axis=1)
    s[s < eps] = np.nan
    missing = int(np.sum(~has_any))
    if missing > 0:
        print(f"[norm] torso scale unavailable in {missing} frame(s); those frames are ignored in N-MPJPE.")
    return s

def _simple_mpjpe(A_xy: np.ndarray, B_xy: np.ndarray, visA: np.ndarray, visB: np.ndarray) -> float:
    assert A_xy.shape == B_xy.shape
    w = np.clip(visA, 0, 1) * np.clip(visB, 0, 1)
    finite = (np.isfinite(A_xy[...,0]) & np.isfinite(A_xy[...,1]) &
              np.isfinite(B_xy[...,0]) & np.isfinite(B_xy[...,1]))
    w = w * finite.astype(np.float32)
    dists = np.linalg.norm(A_xy - B_xy, axis=-1)
    wsum = np.sum(w) + 1e-8
    return float(np.sum(dists * w) / wsum)

def _normalized_mpjpe(A_xy, B_xy, visA, visB) -> float:
    assert A_xy.shape == B_xy.shape
    w = np.clip(visA, 0, 1) * np.clip(visB, 0, 1)
    finite = (np.isfinite(A_xy[...,0]) & np.isfinite(A_xy[...,1]) &
              np.isfinite(B_xy[...,0]) & np.isfinite(B_xy[...,1]))
    w = w * finite.astype(np.float32)
    s = _torso_scale_per_frame(B_xy, visB)
    valid_s = np.isfinite(s)
    dists = np.linalg.norm(A_xy - B_xy, axis=-1)
    s_expand = np.where(valid_s[:, None], s[:, None], np.nan)
    ndists = dists / s_expand
    w = np.where(valid_s[:, None], w, 0.0)
    wsum = np.sum(w) + 1e-8
    return float(np.nansum(ndists * w) / wsum)

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _parse_joint_list(s: str):
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()!=""]
    except Exception:
        return [LS, RS, LH, RH]

def _align_per_frame(A_xy, B_xy, visA, visB, joint_ids):
    T, _, _ = A_xy.shape
    A2 = A_xy.copy()
    Jidx = np.array(joint_ids, dtype=np.int32)
    for t in range(T):
        a = A_xy[t, Jidx, :]
        b = B_xy[t, Jidx, :]
        va = visA[t, Jidx]; vb = visB[t, Jidx]
        finite = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        w = (np.clip(va,0,1) * np.clip(vb,0,1)) * finite.astype(np.float32)
        if np.sum(w > 0.2) < 2:
            continue
        try:
            a_aligned = _rigid_procrustes(a, b, weights=w)
            ac = a.mean(axis=0); aa = a - ac
            aac = a_aligned.mean(axis=0); bb = a_aligned - aac
            C = aa.T @ bb
            U, _, Vt = np.linalg.svd(C)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            s = np.trace(R @ C) / (np.sum(aa**2) + 1e-8)
            tvec = aac - (s * (R @ ac))
            A2[t] = (s * (A_xy[t] @ R.T)) + tvec
        except Exception:
            pass
    return A2, B_xy

# ---------- Airtime detection (NaN-safe ankle Y) ----------
def _detect_airtime_from_ankles(kps_xyz: np.ndarray, fps: float, vA: np.ndarray=None,
                                ema_alpha=0.2, min_air_ms=150):
    T = kps_xyz.shape[0]
    if T < 5 or fps <= 0:
        return -1, -1

    def _robust_mean_y(jidx_pair):
        yy = kps_xyz[:, jidx_pair, 1]  # (T,2)
        vis = None
        if vA is not None:
            vis = vA[:, jidx_pair]
        finite = np.isfinite(yy)
        if vis is not None:
            finite = finite & (vis > 0.2)
        has_any = finite.any(axis=1)
        y = np.full((T,), np.nan, dtype=np.float32)
        if np.any(has_any):
            y[has_any] = np.nanmean(np.where(finite[has_any], yy[has_any], np.nan), axis=1)
        return y

    y = _robust_mean_y([LA, RA])
    if not np.isfinite(y).any():
        y = _robust_mean_y([LK, RK])
    if not np.isfinite(y).any():
        y = _robust_mean_y([LH, RH])
    if not np.isfinite(y).any():
        return -1, -1

    idx_f = np.where(np.isfinite(y), np.arange(T), -1)
    np.maximum.accumulate(idx_f, out=idx_f)
    y_ff = y[idx_f.clip(0, T-1)]
    idx_b = np.where(np.isfinite(y), np.arange(T), T)
    np.minimum.accumulate(idx_b[::-1], out=idx_b[::-1])
    y_bf = y[idx_b.clip(0, T-1)]
    y = np.where(np.isfinite(y), y, 0.5*(y_ff + y_bf))

    dy = np.diff(y, prepend=y[0])
    ema = 0.0
    dys = np.empty_like(dy)
    for i, vi in enumerate(dy):
        ema = (1-ema_alpha)*ema + ema_alpha*vi
        dys[i] = ema

    med = np.median(dys)
    mad = 1.4826*np.median(np.abs(dys - med)) + 1e-6
    up_th = med - 1.0*mad
    dn_th = med + 1.0*mad

    take = int(np.argmax(dys < up_th))
    if take <= 0 or take >= T-2:
        return -1, -1
    peak = int(np.nanargmin(y[take:])) + take
    after = np.where(dys[peak:] > dn_th)[0]
    land = int(after[0] + peak) if after.size else -1

    min_air = int(round((min_air_ms/1000.0) * fps))
    if land <= take or (land - take) < max(2, min_air):
        return -1, -1
    return take, land

# -------- Preview rendering --------
def _mp_pose_edges():
    return [
        (11,12), (11,23), (12,24), (23,24),
        (11,13), (13,15), (12,14), (14,16),
        (23,25), (25,27), (24,26), (26,28),
        (15,19), (16,20), (27,31), (28,32)
    ]

def _render_preview(A_xy, B_xy, visA, visB, out_path, fps_preview=24, max_frames=None, title="Amateur (blue) vs Pro (orange)"):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.animation import FFMpegWriter, PillowWriter
    T = A_xy.shape[0]; edges = _mp_pose_edges()
    F = min(T, max_frames) if max_frames else T
    if F < 2:
        print("[preview] Not enough frames to render."); return
    all_xy = np.concatenate([A_xy[:F].reshape(-1,2), B_xy[:F].reshape(-1,2)], axis=0)
    finite = np.isfinite(all_xy[:,0]) & np.isfinite(all_xy[:,1])
    if not np.any(finite):
        print("[preview] No finite coordinates to render."); return
    xmin, xmax = np.min(all_xy[finite,0]), np.max(all_xy[finite,0])
    ymin, ymax = np.min(all_xy[finite,1]), np.max(all_xy[finite,1])
    pad_x = 0.05*(xmax - xmin + 1e-6); pad_y = 0.05*(ymax - ymin + 1e-6)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(xmin - pad_x, xmax + pad_x); ax.set_ylim(ymax + pad_y, ymin - pad_y)
    ax.set_aspect('equal', adjustable='box'); ax.set_title(title)
    lines_A = [ax.plot([], [], lw=2, alpha=0.9)[0] for _ in edges]
    lines_B = [ax.plot([], [], lw=2, alpha=0.9)[0] for _ in edges]
    pts_A = ax.plot([], [], 'o', ms=3, alpha=0.9)[0]; pts_B = ax.plot([], [], 'o', ms=3, alpha=0.9)[0]
    def _frame_to_xy(seq, t):
        xy = seq[t]; m = np.isfinite(xy[:,0]) & np.isfinite(xy[:,1]); return xy, m
    def init():
        for ln in lines_A + lines_B: ln.set_data([], [])
        pts_A.set_data([], []); pts_B.set_data([], [])
        return lines_A + lines_B + [pts_A, pts_B]
    def animate(t):
        xa, ma = _frame_to_xy(A_xy, t); xb, mb = _frame_to_xy(B_xy, t)
        for k, (i,j) in enumerate(edges):
            lines_A[k].set_data([xa[i,0], xa[j,0]], [xa[i,1], xa[j,1]]) if (ma[i] and ma[j]) else lines_A[k].set_data([], [])
            lines_B[k].set_data([xb[i,0], xb[j,0]], [xb[i,1], xb[j,1]]) if (mb[i] and mb[j]) else lines_B[k].set_data([], [])
        pts_A.set_data(xa[ma,0], xa[ma,1]); pts_B.set_data(xb[mb,0], xb[mb,1])
        return lines_A + lines_B + [pts_A, pts_B]
    interval_ms = 1000.0 / max(1, int(fps_preview))
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=F, interval=interval_ms, blit=True)
    out_dir = os.path.dirname(out_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    try:
        if ext == ".gif":
            ani.save(out_path, writer=PillowWriter(fps=fps_preview))
        else:
            if animation.writers.is_available("ffmpeg"):
                ani.save(out_path, writer=FFMpegWriter(fps=fps_preview, bitrate=1800), dpi=150)
            else:
                print("[preview] FFmpeg not available; saving GIF instead.")
                out_path = os.path.splitext(out_path)[0] + ".gif"
                ani.save(out_path, writer=PillowWriter(fps=fps_preview))
        print(f"[preview] wrote {out_path}")
    except Exception as e:
        print(f"[preview] failed to write preview: {e}")
    finally:
        plt.close(fig)

# ---------- Scoring / shift helpers ----------
def _score_pair(A_xy, B_xy, visA, visB, do_procrustes: bool, pr_joints: list, norm_mode: str):
    if do_procrustes:
        A_xy, B_xy = _align_per_frame(A_xy, B_xy, visA, visB, pr_joints)
    return _normalized_mpjpe(A_xy, B_xy, visA, visB) if norm_mode == "torso" else _simple_mpjpe(A_xy, B_xy, visA, visB)

def _apply_shift(A_xy, visA, B_xy, visB, shift: int):
    if shift == 0:
        T = min(A_xy.shape[0], B_xy.shape[0])
        return A_xy[:T], visA[:T], B_xy[:T], visB[:T]
    if shift > 0:
        T = min(A_xy.shape[0]-shift, B_xy.shape[0])
        if T <= 1: return None
        return A_xy[shift:shift+T], visA[shift:shift+T], B_xy[:T], visB[:T]
    s = -shift
    T = min(A_xy.shape[0], B_xy.shape[0]-s)
    if T <= 1: return None
    return A_xy[:T], visA[:T], B_xy[s:s+T], visB[s:s+T]

def main():
    ap = argparse.ArgumentParser(description="Compare pose tracks: resample, optional Procrustes, normalization, preview, airtime windowing (with padding), and shift search.")
    ap.add_argument("--amateur", required=True)
    ap.add_argument("--pro", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--target-fps", type=float, default=60.0)
    ap.add_argument("--no-resample", action="store_true")

    ap.add_argument("--procrustes", action="store_true")
    ap.add_argument("--procrustes-joints", default="11,12,23,24")
    ap.add_argument("--norm", choices=["none", "torso"], default="none")

    # Preview
    ap.add_argument("--preview-out", default="")
    ap.add_argument("--preview-fps", type=int, default=24)
    ap.add_argument("--preview-frames", type=int, default=0)
    ap.add_argument("--preview-slowmo", type=float, default=1.0,
                    help="Playback slowdown multiplier for preview only (e.g., 2.0 = 2x slower)")

    # Windowing & shift
    ap.add_argument("--window", choices=["full", "airtime"], default="full")
    ap.add_argument("--airtime-pad-ms", type=float, default=0.0,
                    help="Pad airtime window on both sides (milliseconds). Example: 200 adds ~0.2s before/after.")
    ap.add_argument("--shift-range", type=int, default=0)

    ap.add_argument("--out-dir", default="viz/compare")
    args = ap.parse_args()

    pathA = _resolve_npz_path(args.amateur, args.cache_dir)
    pathB = _resolve_npz_path(args.pro, args.cache_dir)
    kpsA, visA, fpsA = _load_posetrack_npz(pathA)
    kpsB, visB, fpsB = _load_posetrack_npz(pathB)

    print(f"[load] amateur: {os.path.basename(pathA)}  fps={fpsA:.3f}  shape={kpsA.shape}")
    print(f"[load] pro     : {os.path.basename(pathB)}  fps={fpsB:.3f}  shape={kpsB.shape}")

    used_fpsA, used_fpsB = fpsA, fpsB
    if not args.no_resample and args.target_fps > 0:
        if abs(fpsA - args.target_fps) > 1e-3:
            kpsA, visA, _ = resample_to_fps(kpsA, visA, fpsA, args.target_fps)
            print(f"[timebase] resampled amateur {fpsA:.3f} -> {args.target_fps:.1f} fps"); used_fpsA = args.target_fps
        if abs(fpsB - args.target_fps) > 1e-3:
            kpsB, visB, _ = resample_to_fps(kpsB, visB, fpsB, args.target_fps)
            print(f"[timebase] resampled pro      {fpsB:.3f} -> {args.target_fps:.1f} fps"); used_fpsB = args.target_fps

    T = min(kpsA.shape[0], kpsB.shape[0])
    if T < 2: raise ValueError("Not enough overlapping frames after (optional) resampling.")
    kpsA = kpsA[:T]; visA = visA[:T]
    kpsB = kpsB[:T]; visB = visB[:T]

    crop_slice = slice(0, T); take, land = -1, -1
    if args.window == "airtime":
        take, land = _detect_airtime_from_ankles(kpsB, used_fpsB, visB)
        if 0 <= take < land <= T-1:
            pad_frames = int(round((args.airtime_pad_ms/1000.0) * used_fpsB))
            start = max(0, take - pad_frames)
            end = min(T-1, land + pad_frames)
            crop_slice = slice(start, end+1)
            dur = (end - start) / max(1.0, used_fpsB)
            print(f"[window] pro airtime [{take}:{land}], pad={pad_frames}f -> slice [{start}:{end}] (≈ {dur:.3f}s)")
        else:
            print("[window] airtime detection failed; using full overlap.")
    kpsA = kpsA[crop_slice]; visA = visA[crop_slice]
    kpsB = kpsB[crop_slice]; visB = visB[crop_slice]

    A_xy_full = kpsA[..., :2]
    B_xy_full = kpsB[..., :2]

    pr_joints = _parse_joint_list(args.procrustes_joints)
    best = {"shift": 0, "value": None, "frames": 0}
    for s in range(-abs(args.shift_range), abs(args.shift_range)+1):
        shifted = _apply_shift(A_xy_full, visA, B_xy_full, visB, s)
        if shifted is None: continue
        A_xy, vA, B_xy, vB = shifted
        if A_xy.shape[0] < 2: continue
        val = _score_pair(A_xy, B_xy, vA, vB, args.procrustes, pr_joints, args.norm)
        if best["value"] is None or val < best["value"]:
            best.update({"shift": s, "value": float(val), "frames": int(A_xy.shape[0])})

    if best["shift"] != 0:
        A_xy_full, visA, B_xy_full, visB = _apply_shift(A_xy_full, visA, B_xy_full, visB, best["shift"])
        print(f"[shift] selected shift={best['shift']} frame(s)  (frames compared: {best['frames']})")

    aligned = bool(args.procrustes)
    metric_name = "nmpjpe_px_per_torso" if args.norm == "torso" else "mpjpe_px"
    final_value = best["value"] if best["value"] is not None else _score_pair(
        A_xy_full, B_xy_full, visA, visB, args.procrustes, pr_joints, args.norm
    )
    T_final = best["frames"] if best["frames"] > 0 else min(A_xy_full.shape[0], B_xy_full.shape[0])

    print(f"[result] {metric_name} = {final_value:.4f}  over {T_final} frame(s)  "
          f"(fps used: amateur {used_fpsA:.1f}, pro {used_fpsB:.1f}, "
          f"procrustes={'on' if aligned else 'off'}, norm={args.norm}, window={args.window}, shift={best['shift']})")

    out_dir = args.out_dir; _ensure_dir(out_dir)
    stemA = os.path.splitext(os.path.basename(pathA))[0]
    stemB = os.path.splitext(os.path.basename(pathB))[0]
    out_json = os.path.join(out_dir, f"compare_{stemA}_vs_{stemB}.json")
    summary = {
        "amateur_file": os.path.basename(pathA),
        "pro_file": os.path.basename(pathB),
        "fps_amateur_used": used_fpsA,
        "fps_pro_used": used_fpsB,
        "frames_compared": int(T_final),
        "metric": metric_name,
        "value": float(final_value),
        "resampled": (abs(used_fpsA - fpsA) > 1e-3) or (abs(used_fpsB - fpsB) > 1e-3),
        "procrustes": bool(aligned),
        "procrustes_joints": args.procrustes_joints if aligned else "",
        "norm": args.norm,
        "window": args.window,
        "airtime_frames": [int(take), int(land)],
        "airtime_pad_ms": float(args.airtime_pad_ms),
        "shift_selected": int(best["shift"]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": "Airtime window padded on both sides via --airtime-pad-ms."
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {out_json}")

    # Preview (slow-mo applied as FPS reduction)
    if args.preview_out:
        frames_lim = args.preview_frames if args.preview_frames and args.preview_frames > 0 else None
        slow = max(args.preview_slowmo, 1e-3)
        eff_fps = max(1, int(round(args.preview_fps / slow)))
        # If Procrustes is on, align preview frames as in scoring
        if aligned:
            A_prev, _ = _align_per_frame(A_xy_full.copy(), B_xy_full, visA, visB, pr_joints)
        else:
            A_prev = A_xy_full
        _render_preview(
            A_xy=A_prev[:T_final], B_xy=B_xy_full[:T_final], visA=visA[:T_final], visB=visB[:T_final],
            out_path=args.preview_out, fps_preview=eff_fps, max_frames=frames_lim,
            title=f"{stemA} vs {stemB}  (Procrustes={'on' if aligned else 'off'}, norm={args.norm}, window={args.window}, shift={best['shift']}, slowmo={slow}x)"
        )

if __name__ == "__main__":
    main()
