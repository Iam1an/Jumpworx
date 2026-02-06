# jwcore/io_cache.py
import os, hashlib
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None


def _hash(path: str, mc: int) -> str:
    base = os.path.abspath(path).encode()
    return hashlib.sha1(base + f"_mc{mc}".encode()).hexdigest()[:16]


def _to_float_scalar(x) -> float:
    """Accept scalar or small array; return a real Python float."""
    return float(x[0] if np.ndim(x) > 0 else x)


def _to_int_scalar(x) -> int:
    """Accept scalar or small array; return a real Python int."""
    return int(x[0] if np.ndim(x) > 0 else x)


def _extract_pose(video_path: str, model_complexity: int = 1) -> tuple[np.ndarray, float, int, int]:
    if cv2 is None or mp is None:
        missing = [n for n, m in [("opencv-python", cv2), ("mediapipe", mp)] if m is None]
        raise ImportError(
            f"Video dependencies not installed: {', '.join(missing)}. "
            'Install them with: pip install -e ".[video]"'
        )

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=int(model_complexity),
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    kps = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        else:
            pts = np.full((33, 2), np.nan, dtype=np.float32)
        kps.append(pts)

    pose.close()
    cap.release()

    if not kps:
        raise RuntimeError("No readable frames.")

    return np.asarray(kps, dtype=np.float32), fps, w, h


def load_pose(video_path: str, cache_dir: str = "./cache", model_complexity: int = 1) -> tuple[np.ndarray, float, int, int]:
    """
    Returns (kps_pix:(T,33,2), fps:float, w:int, h:int).
    Extracts only once per video+model_complexity; otherwise loads from cache.

    Raises ImportError if opencv-python or mediapipe are not installed.
    Install with: pip install -e ".[video]"
    """
    if cv2 is None or mp is None:
        missing = [n for n, m in [("opencv-python", cv2), ("mediapipe", mp)] if m is None]
        raise ImportError(
            f"Video dependencies not installed: {', '.join(missing)}. "
            'Install them with: pip install -e ".[video]"'
        )
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{_hash(video_path, model_complexity)}.npz"
    cache_path = os.path.join(cache_dir, key)

    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=False)
        kps = d["kps_pix"]
        fps = _to_float_scalar(d["fps"]) if "fps" in d else 0.0
        w = _to_int_scalar(d["w"]) if "w" in d else 0
        h = _to_int_scalar(d["h"]) if "h" in d else 0
        return kps, fps, w, h

    kps_pix, fps, w, h = _extract_pose(video_path, model_complexity)
    np.savez_compressed(
        cache_path,
        kps_pix=kps_pix,
        fps=np.array([fps], dtype=np.float32),
        w=np.array([w], dtype=np.int32),
        h=np.array([h], dtype=np.int32),
    )
    return kps_pix, fps, w, h
