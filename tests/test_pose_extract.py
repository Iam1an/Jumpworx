import os, json, numpy as np
from jwcore.pose_extract import process_file

def test_process_file(tmp_path):
    # minimal fake npz with pixel coords + fps
    kps_xy = np.zeros((5,33,2), dtype=np.float32)
    fps = np.array(30.0, dtype=np.float32)
    npz_path = tmp_path / "fake.posetrack.npz"
    np.savez_compressed(npz_path, kps_xy=kps_xy, fps=fps)

    out = tmp_path / "fake.json"
    process_file(str(npz_path), str(out), fps_override=None, strict=True, verbose=False)
    assert out.exists()
    obj = json.loads(out.read_text())
    assert "features" in obj and isinstance(obj["features"], dict)
