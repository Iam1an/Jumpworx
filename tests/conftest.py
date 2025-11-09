import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Lightweight defaults for test_core_modules.py expectations ---
import os
import numpy as np
import pytest

@pytest.fixture
def kps():
    """
    Minimal (T,33,3) array of zeros so geometry functions can run.
    Keeps semantics identical to our z-dominant pipeline (z=0 for tests).
    """
    T, J = 10, 33
    return np.zeros((T, J, 3), dtype=np.float32)

@pytest.fixture
def video_path(tmp_path):
    """
    Provide a usable video path for IO tests:
      1) Prefer a real repo asset if present.
      2) Else, synthesize a tiny MP4 with OpenCV.
      3) Else, return a stable placeholder; tests that actually open the file
         should skip/fail gracefully based on their own guards.
    """
    # 1) Known repo candidates (pick the first that exists)
    candidates = [
        "videos/training/TRICK15_BACKFLIP.mov",
        "viz/airtime_overlay/BACKFLIP_01_overlay.mp4",
        "videos/comparison_white.mp4",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # 2) Try to synthesize a tiny video (requires OpenCV)
    try:
        import cv2
        h, w = 64, 64
        out = tmp_path / "dummy.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(str(out), fourcc, 10, (w, h))
        for _ in range(5):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            wr.write(frame)
        wr.release()
        return str(out)
    except Exception:
        pass

    # 3) Fallback placeholder path (some tests only need a string)
    return "videos/training/TRICK15_BACKFLIP.mov"
