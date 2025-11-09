# test_core_modules.py
import os
import numpy as np
from jwcore.io_cache import load_pose
from jwcore.normalize import normalize_prerot, height_series_up

def test_io_cache(video_path):
    print(f"▶ Testing io_cache on {video_path}")
    kps, fps, w, h = load_pose(video_path, cache_dir="./cache", model_complexity=0)
    print(f"Frames: {len(kps)}, FPS: {fps}, Size: {w}x{h}")
    assert kps.shape[1:] == (33, 2), "Expected 33 keypoints per frame"
    assert fps > 0
    return kps, fps, w, h

def test_normalize(kps):
    print("▶ Testing normalize_prerot and height_series_up")
    kps_norm = normalize_prerot(kps)
    assert np.isfinite(kps_norm).any(), "Normalization failed"
    heights = height_series_up(kps_norm)
    print(f"Height series: mean={np.nanmean(heights):.3f}, min={np.nanmin(heights):.3f}, max={np.nanmax(heights):.3f}")
    return kps_norm, heights

if __name__ == "__main__":
    video = "./videos/Amateur.mp4"  # change this if needed
    assert os.path.exists(video), f"Video not found: {video}"
    kps, fps, w, h = test_io_cache(video)
    kps_norm, heights = test_normalize(kps)
    print("\n✅ All core tests passed successfully.")
