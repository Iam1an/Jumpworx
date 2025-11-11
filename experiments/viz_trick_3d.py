"""
Chunk 3b — 3D Plotly Skeleton Visualizer
----------------------------------------
Interactive, orbitable 3D skeleton animation of a trick video.
- Prefers MediaPipe pose_world_landmarks (meters).
- Falls back to normalized pelvis-centered coords if world not available.
- Optionally marks takeoff/landing frames from phase_segmentation v2.
- Exports an interactive HTML you can open in any browser.

Usage:
  python viz_trick_3d.py --video ./dataset/Backflip/BACKFLIP_01.mov --html_out ./viz/3d/BACKFLIP_01_3d.html

Options:
  --show              # also opens a browser window
  --downsample 2      # keep every 2nd frame for speed
  --flip_x            # flip X axis if your capture looks mirrored
  --flip_z            # flip Z axis if depth looks inverted
  --mark_phases       # draw vertical planes at takeoff / landing (needs phase_segmentation.py)

Requires:
  pip install mediapipe opencv-python numpy plotly
"""

import os, sys, argparse, webbrowser
import numpy as np
import cv2

# Plotly (no separate server; writes a self-contained HTML)
import plotly.graph_objects as go

# MediaPipe
try:
    import mediapipe as mp
except Exception as e:
    print("ERROR: mediapipe is required. Run: pip install mediapipe opencv-python numpy plotly")
    sys.exit(1)

# Optional airtime segmentation (for takeoff/landing markers)
try:
    from phase_segmentation import segment_phases_with_airtime_v2
    _HAS_SEG = True
except Exception:
    _HAS_SEG = False

mp_pose = mp.solutions.pose

# Landmark indices (for fallback normalization)
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24

# -----------------------------
# MediaPipe extraction (2D+Z and WORLD 3D)
# -----------------------------
def extract_pose_sequences(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,           # higher accuracy for world landmarks
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    world_xyz = []   # (T,33,3) if available; meters
    img_xyz   = []   # (T,33,3) x_px,y_px,z_rel for fallback normalization
    vis       = []   # (T,33)   visibility

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        # pixel-space + relative z (for fallback)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            img_pts = np.array([[p.x * w, p.y * h, p.z] for p in lm], dtype=np.float32)
            v = np.array([p.visibility for p in lm], dtype=np.float32)
        else:
            img_pts = np.full((33, 3), np.nan, dtype=np.float32)
            v = np.zeros((33,), dtype=np.float32)

        img_xyz.append(img_pts)
        vis.append(v)

        # world-space (preferred)
        if res.pose_world_landmarks:
            wl = res.pose_world_landmarks.landmark
            wpts = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)  # meters
        else:
            wpts = np.full((33, 3), np.nan, dtype=np.float32)
        world_xyz.append(wpts)

    cap.release()
    pose.close()

    world_xyz = np.asarray(world_xyz) if len(world_xyz) else None
    img_xyz   = np.asarray(img_xyz)   if len(img_xyz)   else None
    vis       = np.asarray(vis)       if len(vis)       else None
    return world_xyz, img_xyz, vis, fps

# -----------------------------
# Fallback normalization (pelvis translate + torso scale; NO rotation)
# -----------------------------
def forward_fill_nan(arr):
    arr = arr.copy()
    for t in range(1, arr.shape[0]):
        bad = ~np.isfinite(arr[t])
        if bad.any():
            arr[t][bad] = arr[t-1][bad]
    return arr

def normalize_prerot(img_xyz, vis=None, min_conf=0.4, eps=1e-6):
    kps = img_xyz.copy().astype(np.float32)
    if vis is not None:
        kps[vis < float(min_conf)] = np.nan
    kps = forward_fill_nan(kps)

    pelvis = 0.5 * (kps[:, L_HIP, :] + kps[:, R_HIP, :])  # (T,3)
    shctr  = 0.5 * (kps[:, L_SHO, :] + kps[:, R_SHO, :])

    kps -= pelvis[:, None, :]
    torso_len = np.linalg.norm(shctr, axis=1) + eps
    scale_vals = torso_len[np.isfinite(torso_len)]
    scale = np.median(scale_vals) if scale_vals.size else 1.0
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    kps /= scale
    return kps  # ~meters-ish, arbitrary scale

# -----------------------------
# Build a single polyline for bones (with NaN gaps) for fast animation
# -----------------------------
def make_bone_polyline(xyz_frame, connections):
    xs, ys, zs = [], [], []
    for i, j in connections:
        a = xyz_frame[i]; b = xyz_frame[j]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            xs.extend([a[0], b[0], np.nan])
            ys.extend([a[1], b[1], np.nan])
            zs.extend([a[2], b[2], np.nan])
    return np.array(xs), np.array(ys), np.array(zs)

# -----------------------------
# Compute phases for optional markers (using normalized coords)
# -----------------------------
def compute_phases_for_markers(world_xyz, img_xyz, vis, fps):
    if not _HAS_SEG:
        return None, None
    # prefer world for segmentation if fully valid; else use normalized img coords
    use_xyz = world_xyz
    if use_xyz is None or not np.isfinite(use_xyz).any():
        use_xyz = normalize_prerot(img_xyz, vis)
    phases = segment_phases_with_airtime_v2(use_xyz, fps, require_precontact=True, min_precontact_ms=200)
    return phases.takeoff_idx, phases.landing_idx

# -----------------------------
# Main: build Plotly animated figure
# -----------------------------
def build_3d_figure(xyz_seq, fps, takeoff=None, landing=None, flip_x=False, flip_z=False, title="3D Trick Viewer"):
    T, J, _ = xyz_seq.shape
    if flip_x: xyz_seq[..., 0] *= -1.0
    if flip_z: xyz_seq[..., 2] *= -1.0

    connections = list(mp_pose.POSE_CONNECTIONS)

    # Initial frame
    x0 = xyz_seq[0, :, 0]; y0 = xyz_seq[0, :, 1]; z0 = xyz_seq[0, :, 2]
    bx0, by0, bz0 = make_bone_polyline(xyz_seq[0], connections)

    # Joints
    joints = go.Scatter3d(
        x=x0, y=y0, z=z0,
        mode="markers",
        marker=dict(size=4, color="royalblue"),
        name="joints"
    )
    # Bones (single polyline with NaN breaks)
    bones = go.Scatter3d(
        x=bx0, y=by0, z=bz0,
        mode="lines",
        line=dict(width=4, color="lightgray"),
        name="bones"
    )

    frames = []
    for t in range(T):
        x = xyz_seq[t, :, 0]; y = xyz_seq[t, :, 1]; z = xyz_seq[t, :, 2]
        bx, by, bz = make_bone_polyline(xyz_seq[t], connections)

        # Frame data order must match initial traces order
        data = [
            dict(type="scatter3d", x=x, y=y, z=z),     # joints
            dict(type="scatter3d", x=bx, y=by, z=bz), # bones
        ]
        frames.append(go.Frame(data=data, name=str(t)))

    # Scene layout — Plotly uses Z as "up" by default; MediaPipe world has Y-up.
    # We'll relabel axes to be clear and keep aspect equal.
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        width=900, height=700,
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=1000/max(1,fps), redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
            x=0.05, y=0.0, xanchor="left", yanchor="bottom"
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                        label=str(k)) for k in range(T)],
            transition=dict(duration=0), x=0.15, y=0.0, currentvalue=dict(prefix="Frame: ")
        )]
    )

    fig = go.Figure(data=[joints, bones], layout=layout, frames=frames)

    # Optional takeoff/landing vertical planes: render as translucent surfaces spanning the current data range
    if takeoff is not None or landing is not None:
        # bounds
        xmin, xmax = np.nanmin(xyz_seq[...,0]), np.nanmax(xyz_seq[...,0])
        ymin, ymax = np.nanmin(xyz_seq[...,1]), np.nanmax(xyz_seq[...,1])
        zmin, zmax = np.nanmin(xyz_seq[...,2]), np.nanmax(xyz_seq[...,2])

        def add_plane(frame_idx, color, name):
            # vertical plane perpendicular to Y axis at a specific FRAME -> approximate by freezing torso root position
            root = xyz_seq[frame_idx, [L_HIP, R_HIP], :].mean(axis=0)
            # Plane perpendicular to Y, crossing root point. We'll draw a quad in X-Z at Y=root[1].
            X = [xmin, xmax, xmax, xmin]
            Z = [zmin, zmin, zmax, zmax]
            Y = [root[1]] * 4
            fig.add_trace(go.Mesh3d(
                x=X, y=Y, z=Z,
                color=color, opacity=0.15, name=name,
                i=[0], j=[1], k=[2],  # minimal faces; plotly will triangulate; quad will show as translucent slab
                showscale=False
            ))

        if takeoff is not None:
            add_plane(int(takeoff), "green", "Takeoff")
        if landing is not None:
            add_plane(int(landing), "red", "Landing")

    return fig

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=str, help="Path to video file")
    ap.add_argument("--html_out", type=str, default=None, help="Where to save the interactive HTML")
    ap.add_argument("--show", action="store_true", help="Open in a browser after saving")
    ap.add_argument("--downsample", type=int, default=1, help="Keep every Nth frame for speed (default 1 = all)")
    ap.add_argument("--flip_x", action="store_true", help="Flip X axis (mirror) if needed")
    ap.add_argument("--flip_z", action="store_true", help="Flip Z axis (depth) if needed")
    ap.add_argument("--mark_phases", action="store_true", help="Overlay takeoff/landing planes (needs phase_segmentation)")
    args = ap.parse_args()

    world_xyz, img_xyz, vis, fps = extract_pose_sequences(args.video)
    if world_xyz is not None and np.isfinite(world_xyz).any():
        xyz = world_xyz.copy()
    else:
        if img_xyz is None:
            raise RuntimeError("No pose extracted.")
        xyz = normalize_prerot(img_xyz, vis)

    # Downsample
    ds = max(1, int(args.downsample))
    if ds > 1:
        xyz = xyz[::ds]

    # Optional phases
    takeoff = landing = None
    if args.mark_phases:
        t_idx, l_idx = compute_phases_for_markers(world_xyz, img_xyz, vis, fps)
        if t_idx is not None and l_idx is not None:
            takeoff = int(t_idx // ds)
            landing = int(l_idx // ds)

    # Build the figure
    os.makedirs("./viz/3d", exist_ok=True)
    html_out = args.html_out or os.path.join("./viz/3d", os.path.splitext(os.path.basename(args.video))[0] + "_3d.html")
    fig = build_3d_figure(
        xyz, fps=max(1e-6, fps/ds),
        takeoff=takeoff, landing=landing,
        flip_x=args.flip_x, flip_z=args.flip_z,
        title=f"3D Trick Viewer - {os.path.basename(args.video)}"
    )
    fig.write_html(html_out, include_plotlyjs="cdn", auto_play=False)
    print(f"[save] 3D viewer -> {html_out}")

    if args.show:
        webbrowser.open("file://" + os.path.abspath(html_out))

if __name__ == "__main__":
    main()
