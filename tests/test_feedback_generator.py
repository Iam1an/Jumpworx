import pytest

# Module lives in experiments/feedback_generator.py, not jwcore/.
# Skip until it is promoted or the test is rewritten.
pytest.skip(
    "jwcore.feedback_generator does not exist (module is in experiments/)",
    allow_module_level=True,
)

import json, numpy as np
from pathlib import Path
from jwcore.feedback_generator import compute_breakdown_from_npz, summarize_deltas_from_path

def _fake_npz(path: Path):
    # 5 frames of zeros as a smoke test
    kps_xy = np.zeros((5,33,2), dtype=np.float32)
    fps = np.array(30.0, dtype=np.float32)
    np.savez_compressed(path, kps_xy=kps_xy, fps=fps, size=np.array([720,1280], dtype=np.int32))

def test_compute_and_summarize(tmp_path: Path):
    stu = tmp_path / "s.npz"
    pro = tmp_path / "p.npz"
    _fake_npz(stu); _fake_npz(pro)
    brk, path, diag = compute_breakdown_from_npz(str(stu), str(pro))
    assert isinstance(brk.overall, float)
    deltas = summarize_deltas_from_path(str(stu), str(pro), path)
    assert isinstance(deltas, dict)
